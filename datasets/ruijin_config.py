
from collections import namedtuple

import torch
import numpy as np

OrganClass = namedtuple("OrganClass", ["label_name", "totalseg_id", "color"])
abd_organ_classes = [
    OrganClass("unlabeled", 0, (255, 255, 255)),
    OrganClass("spleen", 1, (0, 80, 100)),
    OrganClass("kidney_left", 2, (119, 11, 32)),
    OrganClass("kidney_right", 3, (119, 11, 32)),
    OrganClass("liver", 5, (250, 170, 30)),
    OrganClass("stomach", 6, (220, 220, 0)),
    OrganClass("pancreas", 10, (107, 142, 35)),
    OrganClass("small_bowel", 55, (0, 255, 0)),
    OrganClass("duodenum", 56, (70, 130, 180)),
    OrganClass("colon", 57, (0, 0, 255)),
    OrganClass("uniary_bladder", 104, (90, 115, 69)),
    OrganClass("crc", -1, (255, 0, 0))
]