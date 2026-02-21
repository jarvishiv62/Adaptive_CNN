# data/dataset.py
# LOL Dataset loader with augmentations for CATKC-Net training

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LOLDataset(Dataset):
    """
    Dataset loader for the LOL (Low-Light) dataset.
    Returns paired (low-light input, normal-light ground truth) image tensors.
    """

    def __init__(self, low_dir, high_dir, image_size=256, augment=True, split='train'):
        """
        Args:
            low_dir   : Path to low-light images folder
            high_dir  : Path to normal-light (ground truth) images folder
            image_size: Resize images to this size (square)
            augment   : Apply data augmentation (only for training)
            split     : 'train', 'val', or 'test'
        """
        self.low_dir    = low_dir
        self.high_dir   = high_dir
        self.image_size = image_size
        self.augment    = augment and (split == 'train')
        self.split      = split

        # Collect all image filenames (sorted for reproducibility)
        self.filenames = sorted([
            f for f in os.listdir(low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        assert len(self.filenames) > 0, f"No images found in {low_dir}"
        print(f"[{split.upper()}] Loaded {len(self.filenames)} image pairs from {low_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load images
        low_path  = os.path.join(self.low_dir, fname)
        high_path = os.path.join(self.high_dir, fname)

        low_img  = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')

        # Resize
        low_img  = low_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        high_img = high_img.resize((self.image_size, self.image_size), Image.BICUBIC)

        # Data Augmentation (applied identically to both low and high)
        if self.augment:
            low_img, high_img = self._augment(low_img, high_img)

        # Convert to tensor and normalize to [0, 1]
        low_tensor  = TF.to_tensor(low_img)    # Shape: (3, H, W), values in [0,1]
        high_tensor = TF.to_tensor(high_img)   # Shape: (3, H, W), values in [0,1]

        return {
            'low'      : low_tensor,
            'high'     : high_tensor,
            'filename' : fname
        }

    def _augment(self, low, high):
        """Apply the same random augmentations to both images."""

        # Random horizontal flip
        if config.USE_HFLIP and random.random() > 0.5:
            low  = TF.hflip(low)
            high = TF.hflip(high)

        # Random rotation (±10 degrees)
        if config.USE_ROTATION:
            angle = random.uniform(-config.ROTATION_DEGREE, config.ROTATION_DEGREE)
            low  = TF.rotate(low, angle)
            high = TF.rotate(high, angle)

        # Random crop and resize back
        if config.USE_CROP:
            i, j, h, w = transforms.RandomCrop.get_params(
                low, output_size=(int(self.image_size * 0.9), int(self.image_size * 0.9))
            )
            low  = TF.resized_crop(low,  i, j, h, w, (self.image_size, self.image_size))
            high = TF.resized_crop(high, i, j, h, w, (self.image_size, self.image_size))

        return low, high


def get_dataloaders(
    train_low_dir=config.TRAIN_LOW_DIR,
    train_high_dir=config.TRAIN_HIGH_DIR,
    test_low_dir=config.TEST_LOW_DIR,
    test_high_dir=config.TEST_HIGH_DIR,
    image_size=config.IMAGE_SIZE,
    batch_size=config.BATCH_SIZE,
    val_split=config.VAL_SPLIT,
    seed=config.RANDOM_SEED
):
    """
    Creates train, validation, and test DataLoaders.

    Train: ~400 images with augmentation
    Val:   ~80 images without augmentation (split from train)
    Test:  15 images without augmentation

    Returns:
        train_loader, val_loader, test_loader
    """
    # Full train dataset (no augment, we split first, then enable augment)
    full_train = LOLDataset(train_low_dir, train_high_dir, image_size, augment=False, split='train')

    # Split into train and validation
    total = len(full_train)
    val_size   = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)

    # Enable augment on training portion by wrapping
    train_dataset = LOLDataset(train_low_dir, train_high_dir, image_size, augment=True,  split='train')
    val_dataset   = LOLDataset(train_low_dir, train_high_dir, image_size, augment=False, split='val')
    test_dataset  = LOLDataset(test_low_dir,  test_high_dir,  image_size, augment=False, split='test')

    # Use same split indices for train/val
    train_dataset.filenames = [full_train.filenames[i] for i in train_subset.indices]
    val_dataset.filenames   = [full_train.filenames[i] for i in val_subset.indices]

    print(f"\nDataset sizes — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing dataset loader...")
    train_loader, val_loader, test_loader = get_dataloaders()

    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  low  shape : {batch['low'].shape}")
    print(f"  high shape : {batch['high'].shape}")
    print(f"  dtype      : {batch['low'].dtype}")
    print(f"  min/max    : {batch['low'].min():.3f} / {batch['low'].max():.3f}")
    print(f"  filename   : {batch['filename'][0]}")
    print("\nDataset loader working correctly!")