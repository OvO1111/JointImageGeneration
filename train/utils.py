from typing import Callable

import os
import torch
import importlib

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
    except Exception as e:
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
        
        
class dummy_context:
    def __enter__(self, *args, **kwargs):
        ...
    
    def __exit__(self):
        ...