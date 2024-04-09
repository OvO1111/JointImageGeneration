
import re
import json
import random
import pickle

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torchio as tio
from einops import rearrange
import nibabel
import SimpleITK as sitk
from datetime import datetime
from functools import reduce, partial

import numpy as np
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ddpm.models import FrozenBERTEmbedder

import torch
import torch.nn.functional as f
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as tf
from torch.utils.data.dataset import Subset
from ddpm.models import FrozenBERTEmbedder
from datasets.ruijin_config import abd_organ_classes


def conserve_only_certain_labels(label, designated_labels=[1, 2, 3, 5, 6, 10, 55, 56, 57, 104]):
    # 6: stomach, 57: colon
    if designated_labels is None:
        return label.long()
    label_ = torch.zeros_like(label, dtype=torch.long)
    for il, l in enumerate(designated_labels):
        label_[label == l] = il + 1
    return label_


def window_norm(image, window_pos=60, window_width=360):
    window_min = window_pos - window_width // 2
    image = (image - window_min) / window_width
    image[image < 0] = 0
    image[image > 1] = 1
    return image


def resize_by_slice(im, target_slice_size=(128, 128)):
    im = f.interpolate(rearrange(im, "c h w d -> d c h w"), 
                       size=target_slice_size, antialias=True, mode="nearest")
    return rearrange(im, "d c h w -> c h w d")


def load_fn(n, load_type="image"):
    if load_type == "image":
        return tio.ScalarImage(tensor=nibabel.load(n).dataobj[:][None])
    elif load_type == "mask":
        return tio.LabelMap(tensor=nibabel.load(n).dataobj[:].astype(np.uint8)[None])
    
    
def load_or_write_split(basefolder, **splits):
    splits_file = os.path.join(basefolder, "splits.json")
    if os.path.exists(splits_file):
        with open(splits_file, "r") as f:
            splits = json.load(f)
    else:
        with open(splits_file, "w") as f:
            json.dump(splits, f, indent=4)
    splits = list(splits.get(_) for _ in ["train", "val", "test"])
    return splits


class PretrainDataset(Dataset):
    num_classes = 11  # not including crc mask
    # pretrain: CT cond on report -> totalseg organ mask
    def __init__(self, split="train", use_summary=False, max_size=None, cache_len=None, z=64):
        with open('/mnt/data/oss_beijing/dailinrui/data/ruijin/records/dataset_crc_v2.json', 'rt') as f:
            self.data = json.load(f)
            self.data_keys = list(self.data.keys())
            # self.data_keys.remove('RJ202302171638320174')  # which is to cause a "direction mismatch" in ct and mask
            
        self.use_summary = use_summary
        self.base_folder = "/mnt/data/oss_beijing/dailinrui/data/ruijin"
        # stage 1: train controlnet
        conserve_only_colon_label = partial(conserve_only_certain_labels, designated_labels=[1, 2, 3, 5, 6, 10, 55, 56, 57, 104])
        self.mask_transform = tio.Lambda(conserve_only_colon_label)
        # self.volume_transform = tio.Lambda(window_norm)
        self.joined_transform = tio.Compose((
            # tio.CropOrPad((512, 512, z), mask_name="crc_mask", labels=[1,]),
            tio.Resize((128, 128, z)),
            # tio.OneOf(spatial_transformations)
        ))
        
        self.split = split
        train_portion = .8
        self.train_keys = self.data_keys[:round(len(self.data_keys) * train_portion)]
        self.val_keys = self.data_keys[round(len(self.data_keys) * train_portion):]
        self.train_keys, self.val_keys, _ = load_or_write_split("/mnt/workspace/dailinrui/data/pretrained/ccdm/",
                                                                train=self.train_keys, val=self.val_keys)
        
        if max_size is not None:
            self.train_keys = self.train_keys[:max_size]
            self.val_keys = self.val_keys[:max_size]
        self.split_keys = self.train_keys if self.split == "train" else self.val_keys
        
        if cache_len is None: cache_len = 0
        self.text_feature_cache = np.load(
            "/mnt/workspace/dailinrui/data/pretrained/ccdm/bert-ernie-health_extracted_features.npz"
        )
        self.text_feature_cache = {m: self.text_feature_cache[m] for m in tqdm(self.split_keys, desc="text feature preload")}
        self.data_cache = {m: {#"ct": load_fn(self.data[m]["ct"]), 
                               "totalseg": load_fn(self.data[m]["totalseg"], load_type="mask"), 
                               "crcseg": load_fn(self.data[m]["crcseg"], load_type="mask")} 
                           for im, m in tqdm(enumerate(self.split_keys), 
                                             desc=f"{self.split} image preload", 
                                             total=min(len(self.split_keys), cache_len)) 
                           if im < cache_len}

    def __len__(self):
        return len(self.split_keys)
    
    def _preload(self, keys, cache):
        load_fn_im = lambda n: tio.ScalarImage(tensor=sitk.GetArrayFromImage(sitk.ReadImage(n))[None])
        load_fn_msk = lambda n: tio.LabelMap(tensor=sitk.GetArrayFromImage(sitk.ReadImage(n)).astype(np.uint8)[None])
        cache.update({m: {"ct": load_fn_im(self.data[m]["ct"]), 
                          "totalseg": load_fn_msk(self.data[m]["totalseg"]), 
                          "crcseg": load_fn_msk(self.data[m]["crcseg"])}
                          for _, m in tqdm(enumerate(keys), desc="data preload", total=len(keys))})
        return cache

    def __getitem__(self, idx):
        key = self.split_keys[idx]
        item = self.data[key]  # {pacs, date, data, cond={<date>:<string>}}
        report = item.get("report", item.get("text", None))
        
        # if len(cond) == 0: report = ""
        # else:
        #     report = re.sub(r"\\+[a-z]", "", reduce(lambda x, y: x + y, [f"{v}" for _, v in cond.items()], ""), 0, re.MULTILINE)
        #     report = re.sub(r" {2,}", " ", report, 0, re.MULTILINE)
        # mask_name = Path(self.base_folder, "totalseg_8k", data.parts[-3], data.parts[-2].split("_")[0], data.parts[-1].split(".")[0], "all.nii.gz")
        
        if self.data_cache.__contains__(key):
            # image = self.data_cache[key]["ct"]
            mask = self.data_cache[key]["totalseg"]
            crc_mask = self.data_cache[key]["crcseg"]
        else:
            # image = tio.ScalarImage(item["ct"])
            mask = tio.LabelMap(item["totalseg"])
            crc_mask = tio.LabelMap(item["crcseg"])
        context = rearrange(self.text_feature_cache[key][0], "l c -> c l")
        spacing = torch.tensor(sitk.ReadImage(item["totalseg"]).GetSpacing())
            
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        # if self.volume_transform is not None:
        #     image = self.volume_transform(image)
        
        subject = tio.Subject(
            # image=image,
            mask=mask, crc_mask=crc_mask
        )
        if self.joined_transform is not None:
            subject = self.joined_transform(subject)
            
        # image = subject.image.data
        mask = subject.mask.data
        crc_mask = subject.crc_mask.data
            
        # for stage-1 training
        # random_slice = np.random.randint(self.z)
        # # random_slice = slice(random_slice, random_slice + 1)
        # image = image[..., random_slice]
        # mask = mask[0, ..., random_slice]
        # crc_mask = crc_mask[0, ..., random_slice]
        mask[crc_mask > 0] = PretrainDataset.num_classes
        mask = f.one_hot(mask.long(), num_classes=PretrainDataset.num_classes + 1)
        mask = rearrange(mask, "1 h w d c -> c d h w")
        # image = rearrange(image, "1 h w d -> 1 d h w")
        
        image = mask[0:1].float()
        image[...] = 0
        
        return {"image": image,
                "mask": mask,
                "text": report,
                "context": context,
                "spacing": spacing,
                "casename": key}
    
    def _preload_text_features(self, 
                               save_to="/mnt/workspace/dailinrui/data/pretrained/ccdm",
                               bert_ckpt="/mnt/data/oss_beijing/dailinrui/data/pretrained_weights/bert-ernie-health"):
        _frozen_text_embedder = FrozenBERTEmbedder(ckpt_path=bert_ckpt)
        feats = {}
        for c in tqdm(self.data_keys):
            feats[c] = _frozen_text_embedder([self.data[c]["text"]]).cpu().numpy()
        np.savez(os.path.join(save_to, bert_ckpt.split('/')[-1] + "_nogpt_extracted_features.npz"), **feats)
        
        
def training_dataset(toy=False):
    return PretrainDataset(max_size=None, split="train", cache_len=000)


def validation_dataset(max_size=None):
    return PretrainDataset(max_size=None, split="val", cache_len=000)


def get_ignore_class():
    return 0

def get_weights(*args, **kwargs):
    raw = torch.ones(get_num_classes())
    raw[-1] = 1
    return raw

def get_num_classes():
    return PretrainDataset.num_classes + 1

def train_ids_to_class_names():
    return {ic: c.label_name for ic, c in enumerate(abd_organ_classes)}


if __name__ == "__main__":
    x = PretrainDataset(cache_len=0)
    x._preload_text_features()