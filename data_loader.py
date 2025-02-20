from __future__ import print_function, division

from typing import List, Tuple

import torch
from skimage import io, transform
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class HorizontalFlip:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, sample, *args, **kwargs):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]
        if random.random() > self.p:
            image = F.hflip(image)
            label = F.hflip(label)
        return {"imidx": imidx, "image": image, "label": label}


class RandomColorJitter:
    def __init__(
        self, saturation_factor, hue_factor, contrast_factor, brightness_factor
    ):
        self.saturation_factor = saturation_factor
        self.hue_factor = hue_factor
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor

    def __call__(self, sample, *args, **kwargs):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]
        r = random.random()
        if r < 1 / 5:
            image = F.adjust_brightness(image, self.brightness_factor)
        if 1 / 5 <= r < 2 / 5:
            image = F.adjust_contrast(image, self.contrast_factor)
        if 2 / 5 <= r < 3 / 5:
            image = F.adjust_saturation(image, self.saturation_factor)
        if 3 / 5 <= r < 4 / 5:
            image = F.adjust_hue(image, self.hue_factor)

        return {"imidx": imidx, "image": image, "label": label}


class GaussianBlur:
    def __init__(
        self,
        p,
        kernel_size: List[int] = (3, 3),
        sigma: Tuple[float, float] = (0.1, 2.0),
    ):
        self.p = p
        assert sigma[0] < sigma[1]
        assert len(sigma) == 2
        self.sigma = sigma
        self.kernel_size = kernel_size

    def __call__(self, sample, *args, **kwargs):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]
        sigma = self.sigma[0] + (random.random() * (self.sigma[1] - self.sigma[0]))
        F.gaussian_blur(image, self.kernel_size, [sigma, sigma])
        return {"imidx": imidx, "image": image, "label": label}


class RescaleT:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

        img = transform.resize(
            image, (self.output_size, self.output_size), mode="constant"
        )
        lbl = transform.resize(
            label,
            (self.output_size, self.output_size),
            mode="constant",
            order=0,
            preserve_range=True,
        )

        return {"imidx": imidx, "image": img, "label": lbl}


class RandomCrop:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w]
        label = label[top : top + new_h, left : left + new_w]

        return {"imidx": imidx, "image": image, "label": label}


class ToTensorLab:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        imidx, image, label = sample["imidx"], sample["image"], sample["label"]

        tmp_lbl = np.zeros(label.shape)

        if np.max(label) < 1e-6:
            label = label
        else:
            label = label / np.max(label)

        tmp_img = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)
        if image.shape[2] == 1:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmp_lbl[:, :, 0] = label[:, :, 0]

        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_lbl = label.transpose((2, 0, 1))

        return {
            "imidx": torch.from_numpy(imidx),
            "image": torch.from_numpy(tmp_img),
            "label": torch.from_numpy(tmp_lbl.copy()),
        }


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.image_name_list[idx])
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        if len(self.label_name_list) == 0:
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        label = np.zeros(label_3.shape[0:2])
        if len(label_3.shape) == 3:
            label = label_3[:, :, 0]
        elif len(label_3.shape) == 2:
            label = label_3

        if len(image.shape) == 3 and len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        elif len(image.shape) == 2 and len(label.shape) == 2:
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {"imidx": imidx, "image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
