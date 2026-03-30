# ablation.py
# Runs all 4 ablation experiments sequentially: A1, A2, A3, A4
# Each experiment trains a model variant and saves its best checkpoint.
#
# Usage:
#   python ablation.py              ← Run all experiments
#   python ablation.py --exp A4     ← Run only one experiment

import os
import argparse
import torch

import config
from data.dataset import get_dataloaders
from trainer import Trainer


def run_A1(train_loader, val_loader):
    """
    A1: Baseline — Wang & Hu 2020 (static kernel assignment, MSE loss)
    This is the model we are improving upon.
    """
    print("\n" + "="*60)
    print("  ABLATION A1: Baseline (Wang & Hu 2020)")
    print("="*60)

    from models.base_model import BaselineModel
    from losses.composite_loss import get_loss_function

    model   = BaselineModel()
    loss_fn = get_loss_function('mse_only')

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        experiment_name='A1_baseline'
    )
    best_psnr = trainer.train()
    return best_psnr


def run_A2(train_loader, val_loader):
    """
    A2: Parallel Kernels (no attention, equal weights) + MSE loss
    Isolates benefit of multi-scale vs. static assignment.
    """
    print("\n" + "="*60)
    print("  ABLATION A2: Parallel Kernels (no CAM), MSE Loss")
    print("="*60)

    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function

    model   = CATKCNet(use_attention=False)
    loss_fn = get_loss_function('mse_only')

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        experiment_name='A2_parallel_only'
    )
    best_psnr = trainer.train()
    return best_psnr


def run_A3(train_loader, val_loader):
    """
    A3: Parallel Kernels + CAM (Channel Attention) + MSE loss only
    Isolates benefit of attention mechanism itself.
    """
    print("\n" + "="*60)
    print("  ABLATION A3: Parallel Kernels + CAM, MSE Loss Only")
    print("="*60)

    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function

    model   = CATKCNet(use_attention=True)
    loss_fn = get_loss_function('mse_only')

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        experiment_name='A3_cam_mse'
    )
    best_psnr = trainer.train()
    return best_psnr


def run_A4(train_loader, val_loader):
    """
    A4: FULL MODEL — Parallel Kernels + CAM + Composite Loss (MSE + SSIM + Perceptual)
    This is our complete proposed model.
    """
    print("\n" + "="*60)
    print("  ABLATION A4: Full Model (CATKC-Net) — Our Proposal")
    print("="*60)

    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function

    model   = CATKCNet(use_attention=True)
    loss_fn = get_loss_function('composite')

    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        experiment_name='A4_full_model'
    )
    best_psnr = trainer.train()
    return best_psnr


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CATKC-Net Ablation Experiments")
    parser.add_argument(
        '--exp', type=str, default='all',
        choices=['all', 'A1', 'A2', 'A3', 'A4'],
        help="Which ablation experiment to run (default: all)"
    )
    args = parser.parse_args()

    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders()

    results = {}

    if args.exp in ['all', 'A1']:
        results['A1'] = run_A1(train_loader, val_loader)

    if args.exp in ['all', 'A2']:
        results['A2'] = run_A2(train_loader, val_loader)

    if args.exp in ['all', 'A3']:
        results['A3'] = run_A3(train_loader, val_loader)

    if args.exp in ['all', 'A4']:
        results['A4'] = run_A4(train_loader, val_loader)

    # Final summary
    print("\n" + "="*60)
    print("  ABLATION TRAINING SUMMARY — Best Val PSNR")
    print("="*60)
    for exp, psnr in results.items():
        print(f"  {exp}: {psnr:.4f} dB")
    print("="*60)

    print("\nAll experiments done! Run evaluate.py --ablation to get test metrics.")
    print(f"Visualize training: tensorboard --logdir={config.LOG_DIR}")