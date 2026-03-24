# data/dataset.py — v4 (Combined LOL-v1 + LOL-v2 support)
#
# Changes from v3:
#   ✓ CombinedDataset class merges multiple dataset sources
#   ✓ get_dataloaders() auto-detects LOL-v1 and LOL-v2 and combines them
#   ✓ Falls back to single dataset if only one is present
#   ✓ Test set uses LOL-v2 only (cleaner benchmark)

import os
import re
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config


class LOLDataset(Dataset):
    VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    def __init__(self, low_dir, high_dir, split='train', augment=True, source_tag=''):
        self.low_dir    = low_dir
        self.high_dir   = high_dir
        self.split      = split
        self.augment    = augment and (split == 'train')
        self.source_tag = source_tag

        self.pairs = self._load_pairs()

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No image pairs found!\n"
                f"  low_dir : {low_dir}\n"
                f"  high_dir: {high_dir}\n"
            )

        print(f"[{split.upper():5s}] Loaded {len(self.pairs):4d} pairs from {low_dir}  [{source_tag}]")

    def _load_pairs(self):
        if not os.path.isdir(self.low_dir):
            raise FileNotFoundError(f"low_dir not found: {self.low_dir}")
        if not os.path.isdir(self.high_dir):
            raise FileNotFoundError(f"high_dir not found: {self.high_dir}")

        def extract_id(fname):
            stem = os.path.splitext(fname)[0]
            nums = re.findall(r'\d+', stem)
            return nums[-1] if nums else stem

        low_files  = {
            extract_id(f): f
            for f in os.listdir(self.low_dir)
            if os.path.splitext(f)[1].lower() in self.VALID_EXTS
        }
        high_files = {
            extract_id(f): f
            for f in os.listdir(self.high_dir)
            if os.path.splitext(f)[1].lower() in self.VALID_EXTS
        }

        common = sorted(set(low_files.keys()) & set(high_files.keys()))
        return [
            (
                os.path.join(self.low_dir,  low_files[uid]),
                os.path.join(self.high_dir, high_files[uid]),
                f"{self.source_tag}_{low_files[uid]}" if self.source_tag else low_files[uid]
            )
            for uid in common
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        low_path, high_path, filename = self.pairs[idx]
        low  = Image.open(low_path).convert('RGB')
        high = Image.open(high_path).convert('RGB')
        low, high = self._apply_transforms(low, high)
        return {'low': low, 'high': high, 'filename': filename}

    def _apply_transforms(self, low, high):
        size = config.IMAGE_SIZE
        low  = TF.resize(low,  [size, size], interpolation=TF.InterpolationMode.BICUBIC)
        high = TF.resize(high, [size, size], interpolation=TF.InterpolationMode.BICUBIC)

        if self.augment:
            if getattr(config, 'USE_HFLIP', True) and random.random() > 0.5:
                low  = TF.hflip(low)
                high = TF.hflip(high)

            if getattr(config, 'USE_VFLIP', True) and random.random() > 0.5:
                low  = TF.vflip(low)
                high = TF.vflip(high)

            if getattr(config, 'USE_ROTATION', True):
                angle = random.uniform(-config.ROTATION_DEGREE, config.ROTATION_DEGREE)
                low   = TF.rotate(low,  angle)
                high  = TF.rotate(high, angle)

            if getattr(config, 'USE_CROP', True):
                crop_size = int(size * 0.9)
                i, j, h, w = T.RandomCrop.get_params(low, (crop_size, crop_size))
                low  = TF.resized_crop(low,  i, j, h, w, [size, size])
                high = TF.resized_crop(high, i, j, h, w, [size, size])

            if getattr(config, 'USE_COLOR_JITTER', True) and random.random() > 0.5:
                jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
                low = jitter(low)

            if getattr(config, 'USE_GAMMA_AUG', True) and random.random() > 0.5:
                gamma = random.uniform(0.8, 1.2)
                low   = TF.adjust_gamma(low, gamma)

        low  = TF.to_tensor(low)
        high = TF.to_tensor(high)
        return low, high


def _try_load(low_dir, high_dir, split, augment, tag):
    """Try to load a dataset, return None if folders don't exist."""
    try:
        return LOLDataset(low_dir, high_dir, split=split, augment=augment, source_tag=tag)
    except FileNotFoundError:
        return None


def get_dataloaders():
    """
    Auto-detects available datasets and combines them:
      - LOL-v1 : data/LOL/train/low + data/LOL/train/high
      - LOL-v2 : data/LOL-v2/train/low + data/LOL-v2/train/high
    Uses whichever are present. Test set = LOL-v2 only (or LOL-v1 fallback).
    """

    # ── Collect all available training datasets ───────────────────────
    train_sources = []

    lolv1 = _try_load("data/LOL/train/low", "data/LOL/train/high",
                      split='train', augment=True, tag='LOLv1')
    if lolv1:
        train_sources.append(lolv1)

    lolv2 = _try_load("data/LOL-v2/train/low", "data/LOL-v2/train/high",
                      split='train', augment=True, tag='LOLv2')
    if lolv2:
        train_sources.append(lolv2)

    if not train_sources:
        raise FileNotFoundError(
            "No training data found! Make sure at least one of these exists:\n"
            "  data/LOL/train/low\n"
            "  data/LOL-v2/train/low"
        )

    # ── Combine datasets ─────────────────────────────────────────────
    if len(train_sources) > 1:
        full_train = ConcatDataset(train_sources)
        print(f"Combined: {len(full_train)} total training pairs (LOL-v1 + LOL-v2)")
    else:
        full_train = train_sources[0]

    # ── Val split ────────────────────────────────────────────────────
    n_total = len(full_train)
    n_val   = max(1, int(n_total * config.VAL_SPLIT))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_train, [n_train, n_val], generator=generator)

    # ── Test set: prefer LOL-v2, fall back to LOL-v1 ────────────────
    test_dataset = None
    test_dataset = _try_load("data/LOL-v2/test/low", "data/LOL-v2/test/high",
                             split='test', augment=False, tag='LOLv2')
    if test_dataset is None:
        test_dataset = _try_load("data/LOL/test/low", "data/LOL/test/high",
                                 split='test', augment=False, tag='LOLv1')
    if test_dataset is None:
        raise FileNotFoundError("No test data found!")

    print(f"Dataset — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Settings — size: {config.IMAGE_SIZE}×{config.IMAGE_SIZE}, batch: {config.BATCH_SIZE}, workers: {config.NUM_WORKERS}")
    augs = []
    if config.USE_HFLIP:        augs.append('hflip')
    if config.USE_VFLIP:        augs.append('vflip')
    if config.USE_ROTATION:     augs.append('rot')
    if config.USE_CROP:         augs.append('crop')
    if config.USE_COLOR_JITTER: augs.append('jitter')
    if config.USE_GAMMA_AUG:    augs.append('gamma')
    print(f"Augmentation: {'+'.join(augs) if augs else 'none'}")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = config.EVAL_BATCH_SIZE,
        shuffle     = False,
        num_workers = config.NUM_WORKERS,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing combined dataset loader...")
    train_loader, val_loader, test_loader = get_dataloaders()
    batch = next(iter(train_loader))
    print(f"  low shape : {batch['low'].shape}")
    print(f"  high shape: {batch['high'].shape}")
    print(f"  filenames : {batch['filename'][:3]}")
    print("Dataset loader OK!")