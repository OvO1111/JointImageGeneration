from typing import Callable

import os
import torch
import importlib
import numpy as np
from einops import rearrange
from matplotlib.cm import get_cmap
from torchvision.utils import make_grid

def default(expr, defeval=None):
    return defeval if expr is None else expr


def identity(inputs, *args, **kwargs):
    return inputs


def maybe_mkdir(d):
    os.makedirs(d, exist_ok=True)
    return d


def get_cls_from_pkg(pkg, /, **kwargs):
    if pkg is None: return None
    if isinstance(pkg, dict):
        pkg, _kwargs = pkg["target"], pkg["params"]
        if _kwargs is not None: kwargs.update(_kwargs)
    pkg, attr = '.'.join(pkg.split('.')[:-1]), pkg.split('.')[-1]
    try:
        cls = getattr(importlib.import_module(pkg), attr)
        if isinstance(cls, Callable):
            cls = cls(**kwargs)
    except AttributeError as e:
        print(e)
        return None
    return cls


def check_loss(loss: torch.Tensor):
    if torch.isnan(loss).any():
        print("nan found in loss!!")

    if torch.isinf(loss).any():
        print("inf found in loss!!")

    if (loss.sum(1) < -1e-3).any():
        print(f"negative KL divergence {loss.sum()} in loss!!")
        

def visualize(image: torch.Tensor, n: int=11):
    if len(image.shape) == 5:
        h = image.shape[2]
        if h > 8:
            image = image[:, :, ::h // 8]
        image = rearrange(image, "b c h w d -> (b h) c w d")
    image = make_grid(image, nrow=8, normalize=image.dtype == torch.float32)

    if image.dtype == torch.long:
        cmap = get_cmap("viridis")
        rgb = torch.tensor([(0, 0, 0)] + [cmap(i)[:-1] for i in np.arange(0.3, n) / n])
        colored_mask = rgb[image]
    
        return colored_mask
    
    else:
        return image
        
        
class dummy_context:
    def __enter__(self, *args, **kwargs):
        ...
    
    def __exit__(self):
        ...