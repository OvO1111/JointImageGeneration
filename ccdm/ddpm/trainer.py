import importlib
import logging
import os
import pathlib
import pprint
import random
import shutil
import traceback

from medpy.metric.binary import dc, precision, recall
from dataclasses import dataclass
from random import randint
from types import FunctionType
from typing import Callable, Dict, Any, cast
from typing import Union, Optional, List, Tuple, Text, BinaryIO

import ignite.distributed as idist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from tqdm import tqdm
from PIL import Image
from ignite.contrib.handlers import ProgressBar, WandBLogger
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Frequency, ConfusionMatrix, mIoU, IoU, DiceCoefficient
from ignite.utils import setup_logger
from torch import Tensor
from torch import nn
import tensorboardX
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid  # save_image

from datasets.pipelines.transforms import build_transforms
from .models import DenoisingModel, build_model, FrozenBERTEmbedder
from .models.condition_encoder import ConditionEncoder, _build_feature_cond_encoder
from .models.one_hot_categorical import OneHotCategoricalBCHW
from .optimizer import build_optimizer
from .polyak import PolyakAverager
from .utils import WithStateDict, archive_code, expanduservars, worker_init_fn, _onehot_to_color_image, calc_batched_generalised_energy_distance, \
    batched_hungarian_matching, _add_number_to_image, _make_image_from_text

__all__ = ["run_train", "Trainer", "load"]

LOGGER = logging.getLogger(__name__)
Model = Union[DenoisingModel, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _flatten(m: Model) -> DenoisingModel:
    if isinstance(m, DenoisingModel):
        return m
    elif isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        return cast(DenoisingModel, m.module)
    else:
        raise TypeError("type(m) should be one of (DenoisingModel, DataParallel, DistributedDataParallel)")


def _loader_subset(loader: DataLoader, num_images: int, randomize=False) -> DataLoader:
    dataset = loader.dataset
    lng = len(dataset)
    fixed_indices = range(0, lng - lng % num_images, lng // num_images)
    if randomize:
        overlap = True
        fixed_indices_set = set(fixed_indices)
        maxatt = 5
        cnt = 0
        while overlap and cnt < maxatt:
            indices = [randint(0, lng - 1) for _ in range(0, num_images)]
            overlap = len(set(indices).intersection(fixed_indices_set)) > 0
            cnt += 1
    else:
        indices = fixed_indices
    return DataLoader(
        Subset(dataset, indices),
        batch_size=loader.batch_size,
        shuffle=False
    )


@torch.no_grad()
def grid_of_predictions(model: Model, feature_cond_encoder: ConditionEncoder, loader: DataLoader, num_predictions: int, cond_vis_fn: FunctionType,
                        params: dict) -> Tensor:
    model.eval()
    if feature_cond_encoder:
        feature_cond_encoder.eval()

    labels_: List[Tensor] = []
    predictions_: List[Tensor] = []
    conditions_: List[Tensor] = []
    xts_: List[Tensor] = []
    contexts_: List[str] = []

    for batch in tqdm(loader):
        if "cityscapes" in params["dataset_file"]:  # there must be cleaner way to handle this
            # mm cityscapes returns 3
            images_b, labels_b, text_b = batch["image"], batch["mask"], None
        else:
            # cityscapes and friends return 2
            images_b, labels_b, text_b = batch["image"], batch["mask"], batch.get("text", None)

        context_b = None if text_b is None else batch["context"]
        images_b = images_b.to(idist.device())
        condition_b_enc = images_b
        context_b = feature_cond_encoder(context_b) if feature_cond_encoder else None

        labels_b = labels_b.to(idist.device())

        if "cityscapes" in params["dataset_file"]:
            # mm cityscapes
            label_shape = (labels_b.shape[0], model.diffusion.num_classes, *labels_b.shape[3:])
        else:
            # cityscapes and friends
            label_shape = (labels_b.shape[0], model.diffusion.num_classes, *labels_b.shape[2:])

        predictions_b = []
        xt_b = []
        for _ in range(num_predictions):
            x = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=labels_b.device)).sample()
            y = model(x, condition_b_enc, None, context=context_b)["diffusion_out"]
            prediction = _onehot_to_color_image(y, params)
            predictions_b.append(prediction)
            xt_b.append(_onehot_to_color_image(x, params))
            print(prediction.shape, make_grid(prediction, nrow=8).shape)
            # Image.fromarray(make_grid(prediction, nrow=8).cpu().permute(1, 2, 0).numpy().astype(np.uint8)).save(f"./train/layers/val_{_}.png")
            
        predictions_b = torch.stack(predictions_b, dim=1) if num_predictions > 1 else predictions_b[0]
        xt_b = torch.stack(xt_b, dim=1) if num_predictions > 1 else xt_b[0]
        if "cityscapes" in params["dataset_file"]:
            labels_.append(_onehot_to_color_image(labels_b[:, 0], params))
        else:
            labels_.append(_onehot_to_color_image(labels_b, params))

        predictions_.append(predictions_b)
        xts_.append(xt_b)
        conditions_.append(cond_vis_fn(images_b))
        contexts_.extend([None for _ in range(labels_b.shape[0])] if text_b is None else text_b)

    # labels = torch.cat(labels_, dim=0).cpu()
    # predictions = torch.cat(predictions_, dim=0).cpu()
    # conditions = torch.cat(conditions_, dim=0).cpu()
    # xts = torch.cat(xts_, dim=0).cpu()
    outdict = dict(gt = torch.cat(labels_, dim=0).cpu(),
                   pred = torch.cat(predictions_, dim=0).cpu(),
                   cond = torch.cat(conditions_, dim=0).cpu(),
                   xt = torch.cat(xts_, dim=0).cpu())
    keys = list(outdict.keys())
    if contexts_ is not None and contexts_[0] is not None:
        outdict["context"] = contexts_
    
    # n_slices = 24
    # if params["dims"] == 3 and outdict["gt"].shape[1] > n_slices:
    #     random_slice = random.randint(0, outdict["gt"].shape[1]-n_slices-1)
    #     random_slice = torch.arange(random_slice, random_slice + n_slices)
    for k in keys:
        v = outdict[k][0:1, :]
        outdict[k] = v.view((-1,) + v.shape[2:])

    # grid = torch.cat([
    #     conditions[:, :, None].expand((-1, -1, 3, -1, -1)),
    #     labels.expand((-1, -1, 3, -1, -1)),
    #     predictions.expand((-1, -1, 3, -1, -1))
    # ],
    #     dim=1
    # )
    # return torch.reshape(grid, (-1,) + grid.shape[2:]), n
    return outdict, min(8, outdict["gt"].shape[0])


# @torch.no_grad()
# def compute_ged(loader, model, num_samples, average_feature_cond_encoder):
#     model.eval()
#     if average_feature_cond_encoder:
#         average_feature_cond_encoder.eval()

#     ged = 0
#     hm_iou = 0
#     sim_samples = 0
#     cnt = 0
#     for batch in loader:
#         LOGGER.info(f"{cnt}...")
#         image, labels, likelihoods = batch["image"], batch["mask"], None

#         image = image.to(idist.device())
#         feature_condition = None
#         if average_feature_cond_encoder:
#             feature_condition = average_feature_cond_encoder(image)
#             feature_condition = feature_condition.repeat_interleave(num_samples, dim=0)

#         image = image.repeat_interleave(num_samples, dim=0)

#         x = OneHotCategoricalBCHW(logits=torch.zeros(labels[:, 0].repeat_interleave(num_samples, dim=0).shape, device=labels.device)).sample().to(
#             idist.device())

#         prediction = model(x, image, feature_condition=feature_condition, context=None)['diffusion_out']
#         prediction = prediction.reshape(labels.shape[0], -1, *labels.shape[2:])

#         labels = labels.to(idist.device())
#         num_classes = labels.shape[2]
#         labels = labels.argmax(dim=2)

#         batch_ged, _, similarity_samples = calc_batched_generalised_energy_distance(labels.cpu().numpy(), prediction.argmax(dim=2).cpu().numpy(),
#                                                                                     num_classes)

#         ged += np.sum(batch_ged)
#         sim_samples += np.sum(similarity_samples)

#         lcm = np.lcm(num_samples, labels.shape[1])

#         hm_labels = labels.repeat_interleave(lcm // labels.shape[1], dim=1).cpu().numpy()
#         predictions = prediction.repeat_interleave(lcm // num_samples, dim=1).argmax(dim=2).cpu().numpy()

#         batch_hm_iou = batched_hungarian_matching(hm_labels, predictions, num_classes)

#         hm_iou += np.sum(batch_hm_iou)

#         cnt += len(batch_ged)

#     ged = ged / cnt
#     sim_samples = sim_samples / cnt
#     hm_iou = hm_iou / cnt

#     return ged, sim_samples, hm_iou


@dataclass
class Trainer:
    polyak: PolyakAverager
    optimizer: torch.optim.Optimizer
    lr_scheduler: Union[torch.optim.lr_scheduler.LambdaLR, None]
    class_weights: torch.Tensor
    save_debug_state: Callable[[Engine, Dict[str, Any]], None]
    feature_cond_polyak: PolyakAverager
    params: dict
    writer: tensorboardX.SummaryWriter

    @property
    def flat_model(self):
        """View of the model without DataParallel wrappers."""
        return _flatten(self.model)
    
    # @property
    # def frozen_bert_embedder(self):
    #     return FrozenBERTEmbedder()

    @property
    def model(self):
        return self.polyak.model

    @property
    def average_model(self):
        return self.polyak.average_model

    @property
    def feature_cond_encoder(self):
        return self.feature_cond_polyak.model if self.feature_cond_polyak else None

    @property
    def average_feature_cond_encoder(self):
        return self.feature_cond_polyak.average_model if self.feature_cond_polyak else None

    @property
    def time_steps(self):
        return self.flat_model.time_steps

    @property
    def diffusion_model(self):
        return self.flat_model.diffusion

    def train_step(self, engine: Engine, batch) -> dict:

        image, x0, text = batch["image"], batch["mask"], batch.get("text", None)
        context = None if text is None else batch.get("context")

        self.model.train()
        if self.params['feature_cond_encoder']['train']:
            self.feature_cond_encoder.train()

        self.shape = x0.shape[1:]

        device = idist.device()
        image = image.to(device, non_blocking=True)
        x0 = x0.to(device, non_blocking=True)

        condition = image
        context = self.feature_cond_encoder(context) if self.feature_cond_polyak else None
        # if isinstance(feature_condition, dict):
        #     for name, feature in feature_condition.items():
        #         feature_condition[name] = feature_condition[name].contiguous()
        # elif feature_condition is not None:
        #     feature_condition = feature_condition.contiguous()

        batch_size = x0.shape[0]

        # Sample a random step and generate gaussian noise
        # t = torch.randint(1, self.time_steps + 1, size=(batch_size,), device=device)
        t = torch.multinomial(torch.arange(self.time_steps + 1, device=device) ** 1.5, batch_size)
        xt = self.diffusion_model.q_xt_given_x0(x0, t).sample()

        # Estimate the noise with the model
        ################ added context for text constraint ###################
        ret = self.model(xt.contiguous(), condition.contiguous(), None, t, context=context)
        x0pred = ret["diffusion_out"]
        x_res = x0.argmax(dim=1)
        pred_res = x0pred.argmax(1)

        prob_xtm1_given_xt_x0 = self.diffusion_model.theta_post(xt, x0, t)
        prob_xtm1_given_xt_x0pred = self.diffusion_model.theta_post_prob(xt, x0pred, t)

        # Look for nan and inf in loss and save debug state if found
        mask = self.class_weights[x0.argmax(dim=1)]
        
        loss_diffusion = nn.functional.kl_div(
            torch.log(torch.clamp(prob_xtm1_given_xt_x0pred, min=1e-12)),
            prob_xtm1_given_xt_x0,
            reduction='none'
        )
        loss_diffusion = loss_diffusion.sum(dim=1) * mask
        loss_ce = nn.functional.cross_entropy(x0pred, x0.argmax(1), reduction="none")
        # loss_diffusion = ((x0pred - x0) ** 2).float()
        # loss_diffusion = loss_diffusion * mask
        self._check_loss(loss_diffusion, locals())

        loss_kl = torch.sum(loss_diffusion) / batch_size
        loss_ce = torch.sum(loss_ce) / batch_size
        loss = loss_kl + loss_ce

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()
        else:
            lr = self.optimizer.defaults['lr']
        self.polyak.update()
        if self.feature_cond_polyak:
            self.feature_cond_polyak.update()

        if engine.state.epoch > 0 and engine.state.epoch % self.params["train_vis_freq"] == 0 and engine.state.iteration == 1:
            # save tensorboard results
            xt = _onehot_to_color_image(xt, self.params)[0]
            x = _onehot_to_color_image(x0, self.params)[0]
            pred = _onehot_to_color_image(x0pred, self.params)[0]
            cond = image if self.params["dims"] == 2 else image[0].permute(1, 0, 2, 3)
            
            # n_slices = 10
            # if self.params["dims"] == 3 and pred.shape[1] > n_slices:
            #     random_slice = random.randint(0, pred.shape[1]-n_slices-1)
            #     random_slice = torch.arange(random_slice, random_slice + n_slices)
            #     x = x[random_slice]
            #     pred = pred[random_slice]
            #     cond = cond[random_slice]
            n = min(8, pred.shape[0] if self.params["dims"] == 3 else batch_size)
            
            self.writer.add_image("train/xt", _add_number_to_image(make_grid(xt, nrow=n), t, nrow=pred.shape[0] / n, ncol=n), engine.state.iteration)
            self.writer.add_image("train/gt", make_grid(x, nrow=n), engine.state.iteration)
            self.writer.add_image("train/pred", make_grid(pred, nrow=n), engine.state.iteration)
            self.writer.add_image("train/cond", make_grid(cond, nrow=n), engine.state.iteration)
            if context is not None:
                self.writer.add_image("train/context", _make_image_from_text(make_grid(xt, row=n), text, nrow=pred.shape[0] / n, ncol=n), engine.state.iteration)
            
        self.writer.add_scalar("train/diffusion_kl_loss", loss_kl.item(), engine.state.iteration)
        self.writer.add_scalar("train/diffusion_ce_loss", loss_ce.item(), engine.state.iteration)
        # self.writer.add_scalar("train/ce_loss", loss_ce.item(), engine.state.iteration)
        self.writer.add_scalar("train/lr", lr, engine.state.iteration)
        self.writer.add_scalars("train/dice", {f"label_{im}": dc(x_res.data.cpu().numpy() == im, pred_res.data.cpu().numpy() == im) for im in range(1, x0.shape[1])}, engine.state.iteration)
        self.writer.add_scalars("train/precision", {f"label_{im}": precision(x_res.data.cpu().numpy() == im, pred_res.data.cpu().numpy() == im) for im in range(1, x0.shape[1])}, engine.state.iteration)
        self.writer.add_scalars("train/recall", {f"label_{im}": recall(x_res.data.cpu().numpy() == im, pred_res.data.cpu().numpy() == im) for im in range(1, x0.shape[1])}, engine.state.iteration)
        return {"num_items": batch_size,
                "loss": loss.item(),
                "lr": lr}

    def _debug_state(self, locals: dict, debug_names: Optional[List[str]] = None) -> Dict[str, Any]:

        if debug_names is None:
            debug_names = [
                "image", "t", "x0", "xt", "x0pred",
                "prob_xtm1_given_xt_x0", "prob_xtm1_given_xt_x0pred", "loss"
            ]

        to_save = self.objects_to_save(locals["engine"])
        debug_tensors = {k: locals[k] for k in debug_names}
        to_save["tensors"] = WithStateDict(**debug_tensors)
        return to_save

    def _check_loss(self, loss: Tensor, locals: dict) -> None:

        invalid_values = []

        if torch.isnan(loss).any():
            LOGGER.error("nan found in loss!!")
            invalid_values.append("nan")

        if torch.isinf(loss).any():
            LOGGER.error("inf found in loss!!")
            invalid_values.append("inf")

        if (loss.sum(dim=1) < -1e-3).any():
            LOGGER.error("negative KL divergence in loss!!")
            invalid_values.append("neg")

        if invalid_values:
            LOGGER.error("Saving debug state...")
            self.save_debug_state(locals["engine"], self._debug_state(locals))
            raise ValueError(f"Invalid value {invalid_values} found in loss. Debug state has been saved.")

    @torch.no_grad()
    def test_step(self, engine: Engine, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        image, label, text = batch["image"], batch["mask"], batch.get("text", None)
        context = None if text is None else batch.get("context")
        image = image.to(idist.device())
        label = label.to(idist.device())

        label_shape = (label.shape[0], self.flat_model.diffusion.num_classes, *label.shape[2:])
        x = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=label.device)).sample()

        label = label.argmax(dim=1)
        prediction = self.predict(x, image, context=context)
        if prediction.ndim == label.ndim:
            prediction = nn.functional.one_hot(prediction, self.flat_model.diffusion.num_classes)
            if prediction.ndim == 4:
                prediction = prediction.permute(0, 3, 1, 2)
            if prediction.ndim == 5:
                prediction = prediction.permute(0, 4, 1, 2, 3)

        return {'y': label, 'y_pred': prediction}

    @torch.no_grad()
    def predict(self, xt: Tensor, image: Tensor, label_ref: Optional[Tensor] = None, context=None) -> Tensor:
        self.average_model.eval()
        condition = image

        feature_condition = None
        if self.feature_cond_polyak:
            self.average_feature_cond_encoder.eval()
            context = self.average_feature_cond_encoder(context)

        ret = self.average_model(xt, condition, feature_condition=None, context=context)
        return ret["diffusion_out"]

    def objects_to_save(self, engine: Optional[Engine] = None, weights_only: bool = False) -> Dict[str, Any]:
        to_save: Dict[str, Any] = {
            "model": self.flat_model.unet,
        }

        if self.feature_cond_encoder and self.params['feature_cond_encoder']['train']:
            to_save["feature_cond_encoder"] = self.feature_cond_encoder

        to_save["average_model"] = _flatten(self.average_model).unet
        if self.average_feature_cond_encoder and self.params['feature_cond_encoder']['train']:
            to_save["average_feature_cond_encoder"] = self.average_feature_cond_encoder

        if not weights_only:
            to_save["optimizer"] = self.optimizer
            to_save["scheduler"] = self.lr_scheduler

            if engine is not None:
                to_save["engine"] = engine

        return to_save


def build_engine(trainer: Trainer,
                 output_path: str,
                 train_loader: DataLoader,
                 validation_loader: DataLoader,
                 cond_vis_fn: FunctionType,
                 num_classes: int,
                 ignore_class: int,
                 train_ids_to_class_names: dict,
                 params: dict) -> Engine:
    engine = Engine(trainer.train_step)
    frequency_metric = Frequency(output_transform=lambda x: x["num_items"])
    frequency_metric.attach(engine, "imgs/s", Events.ITERATION_COMPLETED)
    GpuInfo().attach(engine, "gpu")
    ProgressBar(persist=True).attach(engine)

    validation_freq = params["validation_freq"] if "validation_freq" in params else 5000
    save_freq = params["save_freq"] if "save_freq" in params else 1000
    display_freq = params['display_freq'] if "display_freq" in params else 500
    n_validation_predictions = params["n_validation_predictions"] if "n_validation_predictions" in params else 4
    n_validation_images = params["n_validation_images"] if "n_validation_images" in params else 5

    engine_test = Engine(trainer.test_step)
    cm = ConfusionMatrix(num_classes=num_classes)
    LOGGER.info(f"Ignore class {ignore_class} in metric evaluation...")
    # IoU(cm, ignore_index=ignore_class).attach(engine_test, "IoU")
    # mIoU(cm, ignore_index=ignore_class).attach(engine_test, "mIoU")
    DiceCoefficient(cm, ignore_index=ignore_class).attach(engine_test, "Dice")

    # engine_train = Engine(trainer.test_step)
    # cm_train = ConfusionMatrix(num_classes=num_classes)
    # IoU(cm_train, ignore_index=ignore_class).attach(engine_train, "IoU")
    # mIoU(cm_train, ignore_index=ignore_class).attach(engine_train, "mIoU")

    if idist.get_local_rank() == 0:
        ProgressBar(persist=True).attach(engine_test)

        if params["wandb"]:
            tb_logger = WandBLogger(project=params["wandb_project"], entity='cdm', config=params)

            tb_logger.attach_output_handler(
                engine,
                Events.ITERATION_COMPLETED(every=50),
                tag="training",
                output_transform=lambda x: x,
                metric_names=["imgs/s"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

            tb_logger.attach_output_handler(
                engine_test,
                Events.EPOCH_COMPLETED,
                tag="testing",
                metric_names=["mIoU", "IoU"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

        checkpoint_handler = ModelCheckpoint(
            output_path,
            "model",
            n_saved=3,
            require_empty=False,
            score_function=None,
            score_name=None
        )

        # checkpoint_best_hmiou = ModelCheckpoint(
        #     output_path,
        #     "best",
        #     n_saved=3,
        #     require_empty=False,
        #     score_function=lambda engine: engine.state.metrics['HMIoU'],
        #     score_name='HM-IoU',
        #     global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        # )

        # checkpoint_best_ged = ModelCheckpoint(
        #     output_path,
        #     "best",
        #     n_saved=3,
        #     require_empty=False,
        #     score_function=lambda engine: -engine.state.metrics['GED'],
        #     score_name='GED',
        #     global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        # )
        # checkpoint_best_miou = ModelCheckpoint(
        #     output_path,
        #     "best",
        #     n_saved=3,
        #     require_empty=False,
        #     score_function=lambda engine: engine.state.metrics['mIoU'],
        #     score_name='mIoU',
        #     global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        # )
        checkpoint_best_dice = ModelCheckpoint(
            output_path,
            "best",
            n_saved=3,
            require_empty=False,
            score_function=lambda engine: engine.state.metrics['Dice'].mean().item(),
            score_name='Dice',
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )

    @engine.on(Events.EPOCH_COMPLETED(every=1))
    def epoch_completed_and_set_epoch(engine: Engine):
        if isinstance(engine.state.dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            engine.state.dataloader.sampler.set_epoch(engine.state.epoch)
            LOGGER.info("DDP sampler: set_epoch=%d (iter=%d) completed rank=%d ",
                        engine.state.epoch,
                        engine.state.iteration,
                        idist.get_local_rank())

    # Display some info every 100 iterations
    @engine.on(Events.EPOCH_COMPLETED(every=1))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_info(engine: Engine):
        LOGGER.info(
            "epoch=%d, iter=%d, speed=%.2fimg/s, loss=%.4g, lr=%.6g,  gpu:0 util=%.2f%%",
            engine.state.epoch,
            engine.state.iteration,
            engine.state.metrics["imgs/s"],
            engine.state.output["loss"],
            engine.state.output["lr"],
            engine.state.metrics["gpu:0 util(%)"]
        )

    # Save model every save_freq iterations
    @engine.on(Events.ITERATION_COMPLETED(every=save_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_model(engine: Engine):
        checkpoint_handler(engine, trainer.objects_to_save(engine, weights_only=False))

    # Generate and save a few segmentations every validation_freq iterations
    @engine.on(Events.EPOCH_COMPLETED(every=validation_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_qualitative_results(_: Engine, num_images=n_validation_images, num_predictions=n_validation_predictions):
        LOGGER.info("Generating images...")
        loader = _loader_subset(validation_loader, num_images, randomize=True)
        grid, n = grid_of_predictions(_flatten(trainer.average_model), trainer.average_feature_cond_encoder, loader, num_predictions, cond_vis_fn,
                                   params)
        # loader = _loader_subset(validation_loader, num_images, randomize=True)
        # grid_shuffle = grid_of_predictions(_flatten(trainer.average_model), trainer.average_feature_cond_encoder, loader, num_predictions,
        #                                    cond_vis_fn, params)
        # grid = torch.concat([grid, grid_shuffle], dim=0)
        # filename = os.path.join(output_path, f"images_{engine.state.iteration:06}.png")
        # LOGGER.info("Saving images to %s...", filename)
        # if params["dims"] == 3:
        #     img = save_image(grid, filename, nrow=n)  # nrow specifies the #columns
        # elif params["dims"] == 2:
        #     img = save_image(grid, filename, nrow=3)
        
        trainer.writer.add_image("val/gt", make_grid(grid["gt"], nrow=n), engine.state.iteration)
        trainer.writer.add_image("val/pred", make_grid(grid["pred"], nrow=n), engine.state.iteration)
        trainer.writer.add_image("val/cond", make_grid(grid["cond"], nrow=n), engine.state.iteration)
        # trainer.writer.add_image("val/xt", make_grid(grid["xt"], nrow=n), engine.state.iteration)
        if grid.get("context", None) is not None:
            trainer.writer.add_image("val/context", _make_image_from_text(make_grid(grid["xt"], nrow=n), [grid["context"][0]], nrow=grid["pred"].shape[0] / n, ncol=n), engine.state.iteration)

        # if params["wandb"]:
        #     images = wandb.Image(img, caption=f"Iteration {engine.state.iteration}")
        #     wandb.log({"examples": images})

    # Compute the GED score every validation_freq iterations
    @engine.on(Events.EPOCH_COMPLETED(every=validation_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def test(_: Engine):
        # if 'lidc' in params["dataset_file"]:
        #     LOGGER.info("GED computation...")
        #     ged, sim_samples, hm_iou = compute_ged(validation_loader, _flatten(trainer.average_model), params["samples"],
        #                                            trainer.average_feature_cond_encoder)
        #     LOGGER.info("mean GED %.3f, mean sim-samples %.3f, m-IoU %.2f", ged, sim_samples, hm_iou)
        #     # if params["wandb"]:
        #     #     wandb.log({"GED": ged,
        #     #                "Diversity samples": sim_samples,
        #     #                "HM-IoU": hm_iou})

        #     engine.state.metrics['GED'] = ged
        #     engine.state.metrics['HMIoU'] = hm_iou

        #     checkpoint_best_ged(engine, trainer.objects_to_save(engine, weights_only=False))
        #     checkpoint_best_hmiou(engine, trainer.objects_to_save(engine, weights_only=False))
        # else:
        LOGGER.info("val dice computation...")
        engine_test.run(validation_loader, max_epochs=1)
            
        LOGGER.info(f"val dice score: {engine_test.state.metrics['Dice'].data.cpu().numpy().tolist()}")
        trainer.writer.add_scalars("val/dice", {f"{train_ids_to_class_names.get(im+1, f'label_{im+1}')}": engine_test.state.metrics["Dice"][im].item() for im in range(len(engine_test.state.metrics["Dice"]))}, 
                                   engine.state.iteration)
        checkpoint_best_dice(engine_test, trainer.objects_to_save(engine, weights_only=False))

    # Save the best models by mIoU score (runs once every len(validation_loader))
    # @engine.on(Events.EPOCH_COMPLETED(every=validation_freq))
    # @idist.one_rank_only(rank=0, with_barrier=True)
    # def log_dice(engine_test: Engine):
    #     LOGGER.info(f"val dice score: {engine_test.state.metrics['Dice'].data.cpu().numpy().tolist()}")
    #     if params["wandb"]:
    #         wandb.log({"mIoU_val": engine_test.state.metrics["mIoU"]})
    #     trainer.writer.add_scalars("val/dice", {f"label_{im+1}": engine_test.state.metrics["Dice"][im].item() for im in range(len(engine_test.state.metrics["Dice"]))}, 
    #                                engine.state.iteration)
    #     checkpoint_best_dice(engine_test, trainer.objects_to_save(engine, weights_only=False))

    # @engine.on(Events.ITERATION_COMPLETED(every=validation_freq))
    # def train_mIoU(_: Engine):
    #     LOGGER.info("train mIoU computation...")
    #     train_loader_ss = _loader_subset(train_loader, 6, randomize=False)
    #     engine_train.run(train_loader_ss, max_epochs=1)

    # @engine_train.on(Events.EPOCH_COMPLETED)
    # @idist.one_rank_only(rank=0, with_barrier=True)
    # def log_train_mIoU(engine_train: Engine):
    #     if 'cityscapes' in params["dataset_file"]:
    #         LOGGER.info("train mIoU score: %.4g", engine_train.state.metrics["mIoU"])
    #         if params["wandb"]:
    #             wandb.log({"mIoU_train": engine_train.state.metrics["mIoU"]})

    return engine


# taken from torchvision utils
def save_image(tensor: Union[torch.Tensor, List[torch.Tensor]],
               fp: Union[Text, pathlib.Path, BinaryIO],
               format: Optional[str] = None,
               **kwargs) -> Image:
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    return im


def load(filename: str, trainer: Trainer, engine: Engine, weights_only: bool = False):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    to_load = trainer.objects_to_save(engine, weights_only)
    ModelCheckpoint.load_objects(to_load, state)


def _build_model(params: dict, input_shapes: List[Tuple[int, int, int]], cond_encoded_shape, dims=3) -> Model:
    model: Model = build_model(
        time_steps=params["time_steps"],
        schedule=params["beta_schedule"],
        schedule_params=params.get("beta_schedule_params", None),
        cond_encoded_shape=cond_encoded_shape,
        input_shapes=input_shapes,
        backbone=params["backbone"],
        backbone_params=params[params["backbone"]],
        dataset_file=params['dataset_file'],
        step_T_sample=params.get('evaluation_vote_strategy', None),
        feature_cond_encoder=params['feature_cond_encoder'] if params['feature_cond_encoder']['type'] != 'none' else None,
        dims=dims,
    ).to(idist.device())

    # Wrap the model in DataParallel or DistributedDataParallel for parallel processing
    if params["distributed"]:
        local_rank = idist.get_local_rank()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif params["multigpu"]:
        model = nn.DataParallel(model)

    return model


def _build_datasets(params: dict) -> Tuple[DataLoader, DataLoader, torch.Tensor, int, dict]:
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_ids_to_class_names = dataset_module.train_ids_to_class_names()

    if (dataset_file == 'datasets.cityscapes') and all(['dataset_pipeline_train' in params,
                                                        'dataset_pipeline_train_settings' in params,
                                                        'dataset_pipeline_val' in params,
                                                        'dataset_pipeline_val_settings' in params]):

        transforms_names_train = params["dataset_pipeline_train"]
        transforms_settings_train = params["dataset_pipeline_train_settings"]
        transforms_dict_train = build_transforms(transforms_names_train, transforms_settings_train, num_classes=20)

        transforms_names_val = params["dataset_pipeline_val"]
        transforms_settings_val = params["dataset_pipeline_val_settings"]
        transforms_dict_val = build_transforms(transforms_names_val, transforms_settings_val, num_classes=20)

        args = {'transforms_dict_train': transforms_dict_train}
        train_dataset = dataset_module.training_dataset(**args)  # type: ignore

        validation_dataset = dataset_module.validation_dataset(max_size=params['dataset_val_max_size'],
                                                               transforms_dict_val=transforms_dict_val)  # type: ignore
    else:
        train_dataset = dataset_module.training_dataset()  # type: ignore
        validation_dataset = dataset_module.validation_dataset(max_size=params['dataset_val_max_size'])  # type: ignore
    num_classes = dataset_module.get_num_classes()

    LOGGER.info("%d images in dataset '%s'", len(train_dataset), dataset_file)
    LOGGER.info("%d images in validation dataset '%s'", len(validation_dataset), dataset_file)

    # If there is no 'get_weights' function in the dataset module, create a tensor full of ones.
    get_weights = getattr(dataset_module, 'get_weights', lambda: torch.ones(num_classes))
    class_weights = get_weights()

    #  worker_init_function not used: each worker has the same random seed every epoch
    #  workers (threads) are re-initialized every epoch and their seeding is the same every time
    #  see https://discuss.pytorch.org/t/does-getitem-of-dataloader-reset-random-seed/8097
    #  https://github.com/pytorch/pytorch/issues/5059

    if params['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        rank=idist.get_local_rank(),
                                                                        num_replicas=params['num_gpus'])
        batch_size = params['batch_size'] // params['num_gpus']  # batch_size of each process

    else:
        train_sampler = None
        batch_size = params['batch_size']  # if single_gpu or non-DDP

    dataset_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                drop_last=True,
                                pin_memory=True,
                                sampler=train_sampler,
                                shuffle=train_sampler is None,
                                num_workers=params["mp_loaders"],
                                worker_init_fn=worker_init_fn)

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=max(1, batch_size // params["samples"]),
                                   num_workers=params["mp_loaders"],
                                   shuffle=False,
                                   worker_init_fn=worker_init_fn)

    return dataset_loader, validation_loader, class_weights, dataset_module.get_ignore_class(), train_ids_to_class_names, num_classes


def _build_debug_checkpoint(output_path: str) -> ModelCheckpoint:
    return ModelCheckpoint(output_path, "debug_state", require_empty=False, score_function=None, score_name=None)


def maybe_make_dir_or_dirs(path, destory_on_exist=False):
    if os.path.exists(path):
        if destory_on_exist:
            try:
                if os.path.isdir(path): shutil.rmtree(path)
                elif os.path.isfile(path): os.remove(path)
                else: shutil.rmtree(path, onerror=None)
            except Exception as e:
                print(traceback.print_exception(e))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def run_train(local_rank: int, params: dict, exp_name: str):
    # Create output folder and archive the current code and the parameters there
    dims = params["dims"]
    output_path = expanduservars(params['output_path'])
    if exp_name is None and params["exp_name"] is None:
        exp_name = "version_static"
    exp_name = params["exp_name"] if exp_name is None else exp_name
    output_path = maybe_make_dir_or_dirs(os.path.join(output_path, exp_name),
                                         destory_on_exist=True if exp_name in ["local_test", "version_static"] else False)
    writer = tensorboardX.SummaryWriter(log_dir=os.path.join(output_path, "tensorboard_logs"),)
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True,
                 distributed_rank=local_rank)
    # params["train_vis_freq"] = params["train_vis_freq"] // params["batch_size"]
    
    os.makedirs(output_path, exist_ok=True)
    LOGGER.info("experiment dir: %s", output_path)
    archive_code(output_path)

    LOGGER.info("Training params:\n%s", pprint.pformat(params))

    # num_gpus = torch.cuda.device_count()
    LOGGER.info("%d GPUs available", torch.cuda.device_count())

    cudnn.benchmark = params['cudnn']['benchmark']  # this set to true usually slightly accelerates training
    LOGGER.info(f'*** setting cudnn.benchmark to {cudnn.benchmark} ***')
    LOGGER.info(f"*** cudnn.enabled {cudnn.enabled}")
    LOGGER.info(f"*** cudnn.deterministic {cudnn.deterministic}")

    # Load the datasets
    train_loader, validation_loader, class_weights, ignore_class, train_ids_to_class_names, num_classes = _build_datasets(params)

    # Build the model, optimizer, trainer and training engine
    input_shapes = [i.shape for i in train_loader.dataset[0].values() if hasattr(i, 'shape')]
    LOGGER.info("Input shapes: " + str(input_shapes))

    feature_cond_encoder, cond_vis_fn = _build_feature_cond_encoder(params)
    average_feature_cond_encoder, _ = _build_feature_cond_encoder(params)
    feature_cond_polyak = PolyakAverager(feature_cond_encoder, average_feature_cond_encoder) if feature_cond_encoder else None

    cond_encoded_shape = input_shapes[0]
    LOGGER.info("Encoded condition shape: " + str(cond_encoded_shape))

    assert len(class_weights) == num_classes, f"len(class_weights) != num_classes: {len(class_weights)} != {num_classes}"

    model, average_model = [_build_model(params, input_shapes, cond_encoded_shape, dims) for _ in range(2)]
    average_model.load_state_dict(model.state_dict())
    polyak = PolyakAverager(model, average_model, alpha=params["polyak_alpha"])

    optimizer_staff = build_optimizer(params, model, feature_cond_encoder, train_loader, debug=False)
    optimizer = optimizer_staff['optimizer']
    lr_scheduler = optimizer_staff['lr_scheduler']

    debug_checkpoint = _build_debug_checkpoint(output_path)
    trainer = Trainer(polyak, optimizer, lr_scheduler, class_weights.to(idist.device()), debug_checkpoint,
                      feature_cond_polyak, params, writer)
    engine = build_engine(trainer, output_path, train_loader, validation_loader, cond_vis_fn,
                          num_classes=num_classes, ignore_class=ignore_class, train_ids_to_class_names=train_ids_to_class_names,
                          params=params)

    # Load a model (if requested in params.yml) to continue training from it
    load_from = params.get('load_from', None)
    if load_from is not None:
        load_from = expanduservars(load_from)
        load(load_from, trainer=trainer, engine=engine)
        optimizer.param_groups[0]['capturable'] = True

    # Run the training engine for the requested number of epochs
    engine.run(train_loader, max_epochs=params["max_epochs"])
