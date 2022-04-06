import scipy.io as sio
import torch
import os
import numpy as np
from PIL import Image
from scipy.special import comb
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
import dill
import pickle


mean, std = [0.485, 0.456, 0.406], [1.0, 1.0, 1.0]

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/std[0], 1/std[1], 1/std[2] ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def pad(img, size_max=256):
    """
    Pads images to the specified size (height x width).
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return TF.pad(
        img,
        (pad_left, pad_top, pad_right, pad_bottom),
        fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))

transformations = transforms.Compose([
   transforms.Resize(size=255, max_size=256),
   transforms.Lambda(pad),
   transforms.CenterCrop((224, 224)),
   transforms.RandomHorizontalFlip(),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])])

# Small change to datasets.ImageFolders, so it also returns folder name
class ImageFolderWithName(datasets.ImageFolder):
    def __init__(self, return_fn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_fn = return_fn

    def __getitem__(self, i):
        img, label = super(ImageFolderWithName, self).__getitem__(i)
        if not self.return_fn:
            return img, label
        else:
            return img, label, self.imgs[i]

from sampler import BalancedBatchSampler

def main_train():

    data = ImageFolderWithName(return_fn=False, root='CUB_100_train/images', transform=transformations)

    return data

def main_test():
    data = ImageFolderWithName(return_fn=False, root='CUB_100_test/images', transform=transformations)

    return data
