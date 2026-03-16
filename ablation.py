# ablation.py — v4
# Runs ablation experiments A1, A2, A3, A4.
# A4 is already trained — only run A1, A2, A3 from here.
#
# Usage:
#   python ablation.py --exp A1
#   python ablation.py --exp A2
#   python ablation.py --exp A3
#   python ablation.py --exp all   ← runs all 4

import os
import argparse
import config
from data.dataset import get_dataloaders
from trainer import Trainer


def run_A1(train_loader, val_loader):
    print("\n" + "="*60)
    print("  A1: Baseline — Wang & Hu 2020 (static kernels, MSE only)")
    print("="*60)
    from models.base_model import BaselineModel
    from losses.composite_loss import get_loss_function
    model   = BaselineModel()
    loss_fn = get_loss_function('mse_only')
    trainer = Trainer(model, loss_fn, train_loader, val_loader, experiment_name='A1_baseline')
    return trainer.train()


def run_A2(train_loader, val_loader):
    print("\n" + "="*60)
    print("  A2: Parallel kernels (no CAM), MSE only")
    print("="*60)
    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function
    model   = CATKCNet(use_attention=False)
    loss_fn = get_loss_function('mse_only')
    trainer = Trainer(model, loss_fn, train_loader, val_loader, experiment_name='A2_parallel_only')
    return trainer.train()


def run_A3(train_loader, val_loader):
    print("\n" + "="*60)
    print("  A3: Parallel kernels + CAM, MSE only")
    print("="*60)
    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function
    model   = CATKCNet(use_attention=True)
    loss_fn = get_loss_function('mse_only')
    trainer = Trainer(model, loss_fn, train_loader, val_loader, experiment_name='A3_cam_mse')
    return trainer.train()


def run_A4(train_loader, val_loader):
    print("\n" + "="*60)
    print("  A4: Full model — Parallel + CAM + Composite loss")
    print("="*60)
    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function
    model   = CATKCNet(use_attention=True)
    loss_fn = get_loss_function('composite')
    trainer = Trainer(model, loss_fn, train_loader, val_loader, experiment_name='A4_full_model')
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'A1', 'A2', 'A3', 'A4'])
    args = parser.parse_args()

    train_loader, val_loader, _ = get_dataloaders()
    results = {}

    if args.exp in ['all', 'A1']: results['A1'] = run_A1(train_loader, val_loader)
    if args.exp in ['all', 'A2']: results['A2'] = run_A2(train_loader, val_loader)
    if args.exp in ['all', 'A3']: results['A3'] = run_A3(train_loader, val_loader)
    if args.exp in ['all', 'A4']: results['A4'] = run_A4(train_loader, val_loader)

    if results:
        print("\n" + "="*60)
        print("  ABLATION TRAINING SUMMARY — Best Val PSNR")
        print("="*60)
        for exp, psnr in results.items():
            print(f"  {exp}: {psnr:.4f} dB")
        print("="*60)
        print("\nNext: python evaluate.py --ablation")