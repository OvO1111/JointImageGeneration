import os
import torch

from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
import torch.distributed
from train.utils import get_cls_from_pkg

def main(spec):
    torch.distributed.init_process_group("nccl")
    spec = OmegaConf.to_container(OmegaConf.load(spec))
    trainer = get_cls_from_pkg(spec["trainer"], spec=spec)
    trainer.writer = SummaryWriter(trainer.tensorboard_path)
    trainer.train()
    

if __name__ == "__main__":
    main("./run/train_ruijin_ccdm.yaml")