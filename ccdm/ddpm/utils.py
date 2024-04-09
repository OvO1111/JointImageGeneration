
import os
import shutil
from typing import Union

import numpy as np
import torch
import math
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch import nn
from torchvision.transforms import ToPILImage

__all__ = [
    'ParallelType',
    'expanduservars',
    'archive_code',
    'WithStateDict',
    'worker_init_fn',
    '_onehot_to_color_image'
]

ParallelType = Union[nn.DataParallel, nn.parallel.DistributedDataParallel]


class WithStateDict(nn.Module):
    """Wrapper to provide a `state_dict` method to a single tensor."""

    def __init__(self, **tensors):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value)
        # self.tensor = nn.Parameter(tensor, requires_grad=False)


def expanduservars(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


def archive_code(path: str) -> None:
    shutil.copy("params.yml", path)
    # Copy the current code to the output folder.
    os.system(f"git ls-files -z | xargs -0 tar -czf {os.path.join(path, 'code.tar.gz')}")


def to_numpy(tensor):
    """Tensor to numpy, calls .cpu() if necessary"""
    with torch.no_grad():
        if tensor.device.type == 'cuda':
            tensor = tensor.cpu()
        return tensor.numpy()


def pil_from_bchw_tensor_label(x, save=False, name=''):
    # debugging function
    # from datasets.ade20k_config import decode_target_to_color
    from datasets.cityscapes_config import decode_target_to_color
    # x B,C,H,W
    assert len(x.shape) == 4
    b, c, h, w = x.shape

    if c == 1:  # integer valued labels
        x_int = x.squeeze(1).long()
    else:  # assume onehot otherwise
        x_int = torch.argmax(x, dim=1, keepdim=True).squeeze(1).long()

    # x_int is B1HW -> x_rgb is B,H,W,3
    x_rgb = decode_target_to_color(x_int)  # BCHW
    x_rgb_numpy = to_numpy(x_rgb).astype(np.uint8)

    # only show 1st image in the batch
    x_pil = Image.fromarray(x_rgb_numpy[0])

    if save:
        # x_pil.show()
        x_pil.save(name)
    return x_pil

def pil_from_bchw_tensor_image(x, save=False, name=''):
    # debugging function
    # from datasets.ade20k_config import decode_target_to_color
    # x B,C,H,W
    assert len(x.shape) == 4
    b, c, h, w = x.shape
    x_pil = ToPILImage()(x[0])
    return x_pil


def worker_init_fn(_):
    # used for seeding dataloader workers in case mp_loader > 1
    # ensures numpy's randomness across epochs
    np.random.seed(torch.initial_seed() % 2**32)


def _onehot_to_color_image(source, params):
    target = _onehot_to_cityscape_color_image(source)
    return target


def _add_number_to_image(image: np.ndarray, text, nrow, ncol):
    if not isinstance(nrow, int):
        nrow = math.ceil(nrow)
    if not isinstance(ncol, int):
        ncol = math.ceil(ncol)
    img = Image.fromarray((image.data.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
    font = ImageFont.truetype('/mnt/data/oss_beijing/dailinrui/data/resources/fonts/truetype/Arial-Unicode-Bold.ttf', size=15)
    d = ImageDraw.Draw(img)
    text = torch.nn.functional.pad(text, (0, nrow * ncol - len(text)), "constant", text[-1])
    index = 0
    row_step = img.size[1] // nrow
    col_step = img.size[0] // ncol
    for col in range(0, nrow):
        for row in range(0, ncol):
            d.text((row_step * row, col_step * col), str(text[index].cpu().item()), fill="black", font=font)
            index += 1
    ys = torch.tensor(np.asarray(img)).permute(2, 0, 1)
    return ys


def _make_image_from_text(image: np.ndarray, text: list, nrow, ncol):
    if not isinstance(nrow, int):
        nrow = math.ceil(nrow)
    if not isinstance(ncol, int):
        ncol = math.ceil(ncol)
    image[...] = 1
    img = Image.fromarray((image.data.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
    font_size = 10
    font = ImageFont.truetype('/mnt/data/oss_beijing/dailinrui/data/resources/fonts/truetype/Arial-Unicode-Bold.ttf', size=font_size)
    d = ImageDraw.Draw(img)
    if len(text) < nrow * ncol:
        text.extend([text[-1]] * (nrow * ncol - len(text)))
    index = 0
    row_step = img.size[1] // nrow
    col_step = img.size[0] // ncol
    for col in range(0, nrow):
        for row in range(0, ncol):
            # make text fit in the grid image
            fulltext = text[index] if len(text[index]) < 80 else text[index][:80] + "..."
            nc = round(len(fulltext) / math.ceil(font.getlength(fulltext) / col_step))
            lines = "\n".join(fulltext[start:start + nc] for start in range(0, len(fulltext), nc))
            d.text((row_step * row, col_step * col), lines, fill="black", font=font)
            index += 1
    ys = torch.tensor(np.asarray(img)).permute(2, 0, 1)
    return ys


def _onehot_to_cityscape_color_image(arr: Tensor):
    from datasets.cityscapes_config import decode_target_to_crc_color
    if len(arr.shape) == 4:
        B, C, H, W = arr.size()
        if C != 1:
            arr = arr.argmax(dim=1, keepdim=True)
        arr = decode_target_to_crc_color(arr)
        arr = arr.permute(0, 4, 2, 3, 1)
        arr = torch.squeeze(arr)

    elif len(arr.shape) == 3:
        C, H, W = arr.size()
        B = 1
        if C != 1:
            arr = arr.argmax(dim=0, keepdim=True)
        arr = decode_target_to_crc_color(arr)
        arr = arr.permute((3, 1, 2, 0))
        arr = torch.squeeze(arr)
        assert arr.shape == (C, H, W), f"{arr.shape} {C} {H} {W}"
    else:
        if len(arr.shape) == 5:
            # 3D case
            B, C, H, W, D = arr.size()
            if C != 1:
                arr = arr[0].argmax(dim=0, keepdim=True)
            arr = decode_target_to_crc_color(arr)
            arr = arr.squeeze(0).permute(0, 3, 1, 2)
        else:
            raise NotImplementedError

    if B == 1:
        arr = torch.unsqueeze(arr, dim=0)
    if arr.shape == (B, 3, H, W):
        return arr / 255
    arr = arr.squeeze(0)
    if arr.shape == (H, 3, W, D):
        return arr[None] / 255
    else:
        raise NotImplementedError(f"{arr.shape} {B} {C} {H} {W}")


# LIDC
def iou(x, y, axis=-1):
    iou_ = (x & y).sum(axis) / (x | y).sum(axis)
    iou_[np.isnan(iou_)] = 1.
    return iou_


# exclude background
def batched_distance(x, y):
    try:
        per_class_iou = iou(x[:, :, None], y[:, None, :], axis=-2)
    except MemoryError:
        raise NotImplementedError

    return 1 - per_class_iou[..., 1:].mean(-1)


def calc_batched_generalised_energy_distance(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.reshape(*samples_dist_0.shape[:2], -1)
    samples_dist_1 = samples_dist_1.reshape(*samples_dist_1.shape[:2], -1)

    eye = np.eye(num_classes)

    samples_dist_0 = eye[samples_dist_0].astype(np.bool)
    samples_dist_1 = eye[samples_dist_1].astype(np.bool)
    
    cross = np.mean(batched_distance(samples_dist_0, samples_dist_1), axis=(1,2))
    diversity_0 = np.mean(batched_distance(samples_dist_0, samples_dist_0), axis=(1,2))
    diversity_1 = np.mean(batched_distance(samples_dist_1, samples_dist_1), axis=(1,2))
    return 2 * cross - diversity_0 - diversity_1, diversity_0, diversity_1


def batched_hungarian_matching(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.reshape((*samples_dist_0.shape[:2], -1))
    samples_dist_1 = samples_dist_1.reshape((*samples_dist_1.shape[:2], -1))

    eye = np.eye(num_classes)

    samples_dist_0 = eye[samples_dist_0].astype(np.bool)
    samples_dist_1 = eye[samples_dist_1].astype(np.bool)
    
    cost_matrix = batched_distance(samples_dist_0, samples_dist_1)

    h_scores = []
    for i in range(samples_dist_0.shape[0]):
        h_scores.append((1-cost_matrix[i])[linear_sum_assignment(cost_matrix[i])].mean())

    return h_scores