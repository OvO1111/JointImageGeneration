import sys, os
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as f

import pathlib as pb
import nibabel as nib
import numpy as np

from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter, GlobalSummaryWriter

sys.path.append("/mnt/workspace/dailinrui/code/multimodal/trajectory_generation/ccdm")
from train.utils import default, identity, maybe_mkdir, get_cls_from_pkg
from train.ccdm import CategoricalDiffusionModel, OneHotCategoricalBCHW
from train.loss import DiffusionKLLoss, CrossEntropyLoss, LPIPS


writer = lambda path: SummaryWriter(path)


class Trainer:
    legends = ["background", "spleen", "kidney_left", "kidney_right", "liver", "stomach", "pancreas", "small_bowel",
               "duodenum", "colon", "uniary_bladder", "colorectal_cancer"]
    writer = SummaryWriter("/mnt/workspace/dailinrui/data/pretrained/ccdm/leftover_collect")
    def __init__(self, spec, /, 
                 val_device=None,
                 batch_size=24,
                 lr=1e-3,
                 max_epochs=100,
                 timesteps=1000,
                 snapshot_path=None,
                 restore_path=None) -> None:
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.timesteps = timesteps
        self.snapshot_path = maybe_mkdir(snapshot_path)
        
        self.best = 0
        self.spec = spec
        self.read_spec(spec)
        
        self.model_path = os.path.join(self.snapshot_path, "model")
        self.tensorboard_path = os.path.join(self.snapshot_path, "logs")

        self.val_proc = None
        
        self.train_dl = DataLoader(self.train_ds, self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        self.val_dl = DataLoader(self.val_ds, batch_size=1, pin_memory=True, num_workers=1)
        
        self.optimizer = AdamW(self.model.parameters(), self.lr / self.batch_size)
        self.lr_scheduler = LinearLR(self.optimizer, start_factor=1, total_iters=self.max_epochs)
        
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.val_device = torch.device("cuda", int(val_device)) if val_device is not None else self.device
        self.lpips = LPIPS(1, 1, ndim=self.model.unet.dims, model_path="", net_backbone="colon_segmentation", is_training=True)
        self.model, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler, self.lpips =\
            self.accelerator.prepare(self.model, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler, self.lpips)
        
        self.loss = {#CrossEntropyLoss(): 0.5,
                     DiffusionKLLoss(attn_weight=torch.tensor(self.train_ds.cls_weight, device=self.device),
                                     diffusion_model=self.model.diffusion_model): 0.5,
                     self.lpips: 0.}
        
        
    def read_spec(self, specs):
        dataset_spec, model_spec, encoder_spec = specs["dataset"], specs['model'], specs['encoder']
        
        self.train_ds = get_cls_from_pkg(dataset_spec["train"])
        self.val_ds = get_cls_from_pkg(dataset_spec["validation"])
        
        self.x_encoder = default(get_cls_from_pkg(encoder_spec["data_encoder"]), identity)
        self.condition_encoder = default(get_cls_from_pkg(encoder_spec["condition_encoder"]), identity)
        self.context_encoder = default(get_cls_from_pkg(encoder_spec["context_encoder"]), identity)
        
        # self.__dict__.update(specs["trainer"]["params"])
        self.model = get_cls_from_pkg(model_spec,
                                      num_classes=len(self.legends),
                                      num_timesteps=self.timesteps,
                                      condition_channels=getattr(self.condition_encoder, "in_channels", 0))
        
    def train(self):
        self.model.train()
        train_it = tqdm(range(self.max_epochs), desc="train progress")
        for ep in train_it:
            if ep % 1 == 0:
                if self.val_proc is not None:
                    self.val_proc.join()  
                self.val_proc = torch.multiprocessing.get_context("spawn").Process(target=self.val, args=(ep, self.val_device))
                self.val_proc.start()
                
            self._train(ep, train_it)
            self.lr_scheduler.step()
                
        self._save(os.path.join(self.model_path),
                   lpips_model=self.lpips.state_dict())
        self._save(os.path.join(self.model_path, "last.ckpt"),
                   model=self.model.state_dict())
            
    def _train(self, epoch, iterator):
        for itr, batch in enumerate(self.train_dl):
            loss_log = {k.__class__.__name__: 0 for k in self.loss.keys()}
            itr_loss = {k.__class__.__name__: None for k in self.loss.keys()}
            x0, condition, context = map(lambda attr: None if not batch.__contains__(attr) else batch[attr].to(self.device), ["mask", "image", "context"])
            
            x0 = self.x_encoder(x0)
            t = torch.multinomial(torch.arange(self.timesteps + 1, device=self.device) ** 1.5, self.batch_size)
            xt = self.model.diffusion_model.q_xt_given_x0(x0, t).sample()
            ret = self.model(xt.contiguous(),
                             self.condition_encoder(None), None, t,
                             context=self.context_encoder(context))
            x0_pred = ret["diffusion_out"]
            
            for fn, p in self.loss.items():
                loss_log[fn.__class__.__name__] = fn(x0=x0, xt=xt, t=t, x0_pred=x0_pred)
                itr_loss[fn.__class__.__name__] = p * loss_log[fn.__class__.__name__]
            itr_loss["TotalLoss"] = sum(itr_loss.values())
            
            self.optimizer.zero_grad()
            self.accelerator.backward(itr_loss["TotalLoss"])
            self.optimizer.step()
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.get_last_lr()[0]
                self.lr_scheduler.step()
            else: lr = self.optimizer.defaults['lr']
            
            global_step = itr + len(self.train_dl) * epoch
            self.writer.add_scalar("train/lr", lr, global_step)
            self.writer.add_scalars("train/loss",
                                    {k: round(v.item(), 2) for k, v in itr_loss.items()},
                                    global_step)
            iterator.set_postfix(itr=global_step, **{k: f"{v.item():.2f} -> {itr_loss[k].item():.2f}" for k, v in loss_log.items()})
    
    def val(self, ep, device):
        model = self.model.eval()
        val_it = tqdm(enumerate(self.val_dl), total=len(self.val_dl), desc='validation progress')
        val_loss = {k.__class__.__name__: 0 for k in self.loss.keys()}
        
        for itr, batch in val_it:
            x0, condition, context = map(lambda attr: None if not batch.__contains__(attr) else batch[attr].to(device), ["mask", "image", "context"])
            
            x0 = self.x_encoder(x0)
            x0_shape = (x0.shape[0], model.diffusion_model.num_classes, *x0.shape[2:])
            xt = OneHotCategoricalBCHW(logits=torch.zeros(x0_shape, device=self.device)).sample()
            ret = model(xt.argmax(1),
                        self.condition_encoder(None),
                        feature_condition=None,
                        context=self.context_encoder(context))
            x0_pred = ret["diffusion_out"]
            
            for fn, p in self.loss.items():
                val_loss[fn.__class__.__name__] += fn(x0=x0, xt=xt, t=self.timesteps, x0_pred=x0_pred, is_training=False).item()
            val_loss["TotalLoss"] += sum(val_loss.values())
            
            val_it.set_postfix(**{k: v / (itr + 1) for k, v in val_loss.items()})
            
        for k, v in val_loss.items(): self.writer.add_scalar(f'val/unnormalized_{k}', v / len(self.val_dl), ep)
            
        if abs(new_best := val_loss["LPIPS"]) < self.best:
            print(f"best lpips for epoch {ep}: {new_best:.2f}")
            self.best = new_best
            self._save(os.path.join(self.model_path, f"best_model_lpips.ckpt"),
                       best_lpips=new_best, model=model.state_dict(), optimizer=self.optimizer.state_dict())
    
    @staticmethod
    def _save(_save_path, /, **_save_dict):
        torch.save(_save_dict, _save_path)
        
    def _load(self, _load_path):
        obj = torch.load(_load_path, map_location="cpu")
        self.model.load_state_dict(obj["model"])
        self.optimizer.load_state_dict(obj["optimizer"])
        self.best = obj["current_best_dice"]
    
    def test(self, input_folder_or_file, output_folder_or_file):
        pass
            
    def _test(self, tensor):
        pass


if __name__ == "__main__":
    spec = OmegaConf.to_container(OmegaConf.load("./run/train_ruijin_ccdm.yaml"))
    trainer = Trainer(spec, **spec["trainer"]["params"])
    trainer.train()
            