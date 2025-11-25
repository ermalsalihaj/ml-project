"""
data_loader.py

Handles:
- Loading SOD dataset (images + masks)
- Resizing, normalizing, basic augmentations
- Train/val/test DataLoaders
"""

import os
from typing import Tuple, List

from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF


class SaliencyDataset(Dataset):
    """
    Custom Dataset for Salient Object Detection (SOD).

    Expects:
        image_dir: folder with RGB images
        mask_dir:  folder with corresponding 1-channel masks
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: int = 128,
        augment: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        self.image_paths = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

        self.mask_paths = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        )

        assert len(self.image_paths) == len(self.mask_paths), \
            "Number of images and masks must be equal."

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image_mask(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 1-channel

        return image, mask

    def _resize(self, image: Image.Image, mask: Image.Image):
        image = image.resize((self.img_size, self.img_size))
        mask = mask.resize((self.img_size, self.img_size))
        return image, mask

    def _augment(self, image: Image.Image, mask: Image.Image):
        # Horizontal flip
        if np.random.rand() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Simple brightness jitter
        if np.random.rand() < 0.3:
            factor = 0.7 + 0.6 * np.random.rand()  # [0.7, 1.3]
            image = TF.adjust_brightness(image, factor)

        return image, mask

    def __getitem__(self, idx: int):
        image, mask = self._load_image_mask(idx)
        image, mask = self._resize(image, mask)

        if self.augment:
            image, mask = self._augment(image, mask)

        # To tensor + normalize to [0,1]
        image = TF.to_tensor(image)  # [3, H, W], already /255
        mask = TF.to_tensor(mask)    # [1, H, W], already /255

        # Ensure mask is binary-ish (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask


def create_dataloaders(root_dir, img_size=128, batch_size=8):
    """
    Correct 70/15/15 split of DUTS dataset.
    """

    image_dir = os.path.join(root_dir, "images")
    mask_dir  = os.path.join(root_dir, "masks")

    if not os.path.exists(image_dir):
        raise ValueError(f"Image folder not found: {image_dir}")
    if not os.path.exists(mask_dir):
        raise ValueError(f"Mask folder not found: {mask_dir}")

    dataset = SaliencyDataset(image_dir=image_dir, mask_dir=mask_dir, img_size=img_size)

    total = len(dataset)
    train_size = int(0.70 * total)
    val_size   = int(0.15 * total)
    test_size  = total - train_size - val_size  # leftover

    print(f"Total images: {total}")
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_subset, val_subset, test_subset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("This file defines SaliencyDataset and create_dataloaders().")
