import importlib
import logging
import os
import pathlib
import pprint
import random
import shutil
import traceback
import SimpleITK

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
        
        for im in range(image.shape[0]):
            foldername = maybe_make_dir_or_dirs(f"{self.params['save_path']}/{batch['casename'][im]}")

            Image.fromarray(make_grid(label, nrow=8).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save(os.path.join(foldername, "gt.png"))
            Image.fromarray(make_grid(prediction, nrow=8).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save(os.path.join(foldername, "pred.png"))

            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(label[im].cpu().numpy().astype(np.uint8)), os.path.join(foldername, "gt.nii.gz"))
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(prediction[im].cpu().numpy().astype(np.uint8)), os.path.join(foldername, "pred.nii.gz"))

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


def build_engine(trainer: Trainer,
                 output_path: str,
                 train_loader: DataLoader,
                 validation_loader: DataLoader,
                 cond_vis_fn: FunctionType,
                 num_classes: int,
                 ignore_class: int,
                 train_ids_to_class_names: dict,
                 params: dict) -> Engine:
    engine = Engine(trainer.test_step)
    frequency_metric = Frequency(output_transform=lambda x: x["num_items"])
    frequency_metric.attach(engine, "imgs/s", Events.ITERATION_COMPLETED)
    GpuInfo().attach(engine, "gpu")
    ProgressBar(persist=True).attach(engine)

    cm = ConfusionMatrix(num_classes=num_classes)
    LOGGER.info(f"Ignore class {ignore_class} in metric evaluation...")
    DiceCoefficient(cm, ignore_index=ignore_class).attach(engine, "Dice")

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


def run_eval(local_rank: int, params: dict, exp_name: str):
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
