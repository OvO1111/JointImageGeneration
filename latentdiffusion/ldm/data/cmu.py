import os, os.path as path, yaml, pathlib as pb
import json, torchio as tio, torchvision as tv, shutil, nibabel as nib
import re, SimpleITK as sitk, scipy.ndimage as ndimage, numpy as np, multiprocessing as mp

import torch

import re
import random
from tqdm import tqdm
from omegaconf import OmegaConf
from functools import reduce, partial
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset
from einops import rearrange
from torchvision.transforms import RandomCrop, Resize

def maybe_mkdir(p, destory_on_exist=False):
    if path.exists(p) and destory_on_exist:
        shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)
    return pb.Path(p)
    
    
def min_max_norm(image, min=-1000, max=1000):
    image = (image - min) / (max - min)
    return image


def write_split(basefolder, *splits):
    with open(os.path.join(basefolder, "splits.json"), "w") as f:
        json.dump(dict(zip(["train", "val", "test"], splits)), f, indent=4)
        

def use_split(basefolder):
    with open(os.path.join(basefolder, "splits.json")) as f:
        splits = json.load(f)
    return dict(train=splits.get("train"), val=splits.get("val"), test=splits.get("test"))
    
    
class AutoencoderDataset(Dataset):
    def __init__(self, split="train"):
        self.base = os.path.join(os.environ.get("nnUNet_raw"), "Dataset004CMU")
        self.data_keys = ["{:04d}".format(int(re.findall(r'\d+', r)[1])) for r in os.listdir(os.path.join(self.base, "imagesTr"))]

        self.load_fn = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))

        self.split = split
        random.shuffle(self.data_keys)
        if not os.path.exists(os.path.join(self.base, "splits.json")):
            self.train_keys = self.data_keys[:round(len(self.data_keys) * 0.8)]
            self.val_keys = self.data_keys[round(len(self.data_keys) * 0.8):]
            write_split(self.base, self.train_keys, self.val_keys)
        else:
            _ = use_split(self.base)
            self.train_keys, self.val_keys = _["train"], _["val"]
        
        self.split_keys = self.train_keys if split == "train" else self.val_keys
        # self.data = {_: dict(
        #     image=self.load_fn(os.path.join(self.base, "imagesTr", f"Dataset004CMU_{_}_0000.nii.gz")),
        #     pseudo=self.load_fn(os.path.join(self.base, "pimagesTr", f"Dataset004CMU_{_}_0000.nii.gz")),
        #     mask=self.load_fn(os.path.join(self.base, "labelsTr", f"Dataset004CMU_{_}.nii.gz")),
        #     spacing=sitk.ReadImage(os.path.join(self.base, "labelsTr", f"Dataset004CMU_{_}.nii.gz")).GetSpacing(),
        # ) for _ in tqdm(self.split_keys, desc="loading data cache")}
        self.data = {}
        self.transform = Resize((512, 512), antialias=True)

    def __len__(self):
        return len(self.split_keys)

    def __getitem__(self, idx):
        index = self.split_keys[idx]
        item = self.data.get(index,
                             dict(image=self.load_fn(os.path.join(self.base, "imagesTr", f"Dataset004CMU_{index}_0000.nii.gz")),
                                  pseudo=self.load_fn(os.path.join(self.base, "pimagesTr", f"Dataset004CMU_{index}_0000.nii.gz")),
                                  mask=self.load_fn(os.path.join(self.base, "labelsTr", f"Dataset004CMU_{index}.nii.gz")),
                                  background=self.load_fn(os.path.join(self.base, "background", f"Dataset004CMU_{index}_0000.nii.gz")),
                                  spacing=sitk.ReadImage(os.path.join(self.base, "labelsTr", f"Dataset004CMU_{index}.nii.gz")).GetSpacing(),))
        image = min_max_norm(item.get("image"))
        pseudo_image = min_max_norm(item.get("pseudo"))
        mask = item.get("mask")
        h1, h2 = np.where(np.any(mask, axis=(1, 2)))[0][[0, -1]]
        random_slice = random.randint(h1, h2 + 1)
        
        # image = np.pad(image, ((0, 0), (0, 0), (0, 0)), mode="edge")
        # pseudo_image = np.pad(pseudo_image, ((0, 0), (0, 0), (0, 0)), mode="edge")
        image, pseudo_image = torch.tensor(image), torch.tensor(pseudo_image)
        
        random_slice = slice(random_slice, random_slice + 1)
        sample = torch.cat([image[random_slice][None],
                            pseudo_image[random_slice][None],], dim=0)
        sample = self.transform(sample)
        sample = dict(target=rearrange(sample[0], "c h w -> h w c"), 
                      image=rearrange(sample[1], "c h w -> h w c"),
                      wholeimage=image)
        
        # background = item.get("background")
        # random_slice = slice(random_slice, random_slice + 1)
        # sample = torch.cat([image[random_slice][None],
        #                     background[random_slice][None],
        #                     mask[random_slice][None]], dim=0)
        # sample = self.transform(sample)
        # sample = dict(image=rearrange(sample[1:], "b 1 h w -> h w b"), 
        #               target=rearrange(sample[0], "c h w -> h w c"))
        return sample


if __name__ == "__main__":
    c = AutoencoderDataset()
    c[0]