# data/dataset.py — FIXED VERSION
# Fix: num_workers now reads from config (0 on Windows, 4 on Linux)
# Fix: persistent_workers only enabled when num_workers > 0

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class LOLDataset(Dataset):
    def __init__(self, low_dir, high_dir, image_size=256, augment=True, split='train'):
        self.low_dir    = low_dir
        self.high_dir   = high_dir
        self.image_size = image_size
        self.augment    = augment and (split == 'train')
        self.split      = split

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
        low_img  = Image.open(os.path.join(self.low_dir,  fname)).convert('RGB')
        high_img = Image.open(os.path.join(self.high_dir, fname)).convert('RGB')

        low_img  = low_img.resize((self.image_size, self.image_size), Image.BICUBIC)
        high_img = high_img.resize((self.image_size, self.image_size), Image.BICUBIC)

        if self.augment:
            low_img, high_img = self._augment(low_img, high_img)

        return {
            'low'      : TF.to_tensor(low_img),
            'high'     : TF.to_tensor(high_img),
            'filename' : fname
        }

    def _augment(self, low, high):
        if config.USE_HFLIP and random.random() > 0.5:
            low, high = TF.hflip(low), TF.hflip(high)

        if config.USE_ROTATION:
            angle = random.uniform(-config.ROTATION_DEGREE, config.ROTATION_DEGREE)
            low, high = TF.rotate(low, angle), TF.rotate(high, angle)

        if config.USE_CROP:
            crop_size = int(self.image_size * 0.9)
            i, j, h, w = transforms.RandomCrop.get_params(low, output_size=(crop_size, crop_size))
            low  = TF.resized_crop(low,  i, j, h, w, (self.image_size, self.image_size))
            high = TF.resized_crop(high, i, j, h, w, (self.image_size, self.image_size))

        return low, high


def get_dataloaders(
    train_low_dir  = config.TRAIN_LOW_DIR,
    train_high_dir = config.TRAIN_HIGH_DIR,
    test_low_dir   = config.TEST_LOW_DIR,
    test_high_dir  = config.TEST_HIGH_DIR,
    image_size     = config.IMAGE_SIZE,
    batch_size     = config.BATCH_SIZE,
    val_split      = config.VAL_SPLIT,
    seed           = config.RANDOM_SEED
):
    # Build full dataset for splitting
    full_train = LOLDataset(train_low_dir, train_high_dir, image_size, augment=False, split='train')

    total      = len(full_train)
    val_size   = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=generator)

    # Separate datasets with correct augment flag
    train_dataset = LOLDataset(train_low_dir, train_high_dir, image_size, augment=True,  split='train')
    val_dataset   = LOLDataset(train_low_dir, train_high_dir, image_size, augment=False, split='val')
    test_dataset  = LOLDataset(test_low_dir,  test_high_dir,  image_size, augment=False, split='test')

    train_dataset.filenames = [full_train.filenames[i] for i in train_subset.indices]
    val_dataset.filenames   = [full_train.filenames[i] for i in val_subset.indices]

    print(f"\nDataset — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"num_workers = {config.NUM_WORKERS}  (0 = Windows-safe, 4 = Linux/Colab)\n")

    # FIX: persistent_workers only valid when num_workers > 0
    pw = (config.NUM_WORKERS > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == 'cuda'),
        drop_last=True,
        persistent_workers=pw
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == 'cuda'),
        persistent_workers=pw
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == 'cuda'),
        persistent_workers=pw
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    batch = next(iter(train_loader))
    print(f"Batch — low: {batch['low'].shape}, high: {batch['high'].shape}")
    print(f"Range: [{batch['low'].min():.3f}, {batch['low'].max():.3f}]")
    print("Dataset loader OK!")