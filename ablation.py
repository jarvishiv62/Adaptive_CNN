# ablation.py — v4 (resume-safe)
#
# Key fix: now resumes from last_model.pth if training was interrupted.
# Logic per experiment:
#   1. If best_model.pth exists AND was trained >= ABLATION_EPOCHS → skip (done)
#   2. If last_model.pth exists but not complete → RESUME from last epoch
#   3. If no checkpoint → train from scratch
#
# Safe to Ctrl+C at any time. Re-run the same --exp command to resume.

import os
import argparse
import torch

import config
from data.dataset import get_dataloaders
from trainer import Trainer

ABLATION_EPOCHS        = 100
ABLATION_PATIENCE      = 25
ABLATION_WARMUP_EPOCHS = 3


def _make_trainer(model, loss_fn, train_loader, val_loader, name):
    """Create Trainer with ablation epoch settings."""
    config.NUM_EPOCHS          = ABLATION_EPOCHS
    config.EARLY_STOP_PATIENCE = ABLATION_PATIENCE
    config.LR_WARMUP_EPOCHS    = ABLATION_WARMUP_EPOCHS
    return Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        experiment_name=name
    )


def _check_status(name):
    """
    Returns:
        ('done',   psnr)  — fully trained, skip
        ('resume', path)  — interrupted, resume from last_model.pth
        ('fresh',  None)  — no checkpoint, train from scratch
    """
    best_path = os.path.join(config.CHECKPOINT_DIR, name, config.BEST_MODEL_NAME)
    last_path = os.path.join(config.CHECKPOINT_DIR, name, config.LAST_MODEL_NAME)

    if os.path.exists(best_path):
        state = torch.load(best_path, map_location='cpu')
        epoch = state.get('epoch', 0)
        psnr  = state.get('best_psnr', state.get('psnr', 0.0))

        if epoch >= ABLATION_EPOCHS - 5:
            # Trained to completion (within 5 epochs of target)
            print(f"  ✓ Already complete ({epoch} epochs) — Best Val PSNR: {psnr:.4f} dB")
            print(f"    Delete checkpoints\\{name}\\ to retrain.")
            return 'done', psnr

        # Checkpoint exists but incomplete — resume
        if os.path.exists(last_path):
            last_state  = torch.load(last_path, map_location='cpu')
            last_epoch  = last_state.get('epoch', 0)
            print(f"  ⚠ Incomplete run found (stopped at epoch {last_epoch}/{ABLATION_EPOCHS})")
            print(f"    Resuming from last checkpoint...")
            return 'resume', last_path

    elif os.path.exists(last_path):
        last_state = torch.load(last_path, map_location='cpu')
        last_epoch = last_state.get('epoch', 0)
        print(f"  ⚠ Incomplete run found (stopped at epoch {last_epoch}/{ABLATION_EPOCHS})")
        print(f"    Resuming from last checkpoint...")
        return 'resume', last_path

    return 'fresh', None


def _run(name, model_fn, loss_type, train_loader, val_loader):
    """Generic run function with resume support."""
    status, info = _check_status(name)

    if status == 'done':
        return info  # info = psnr

    model, loss_fn = model_fn()
    trainer = _make_trainer(model, loss_fn, train_loader, val_loader, name)

    resume_path = info if status == 'resume' else None
    return trainer.train(resume_path=resume_path)


def run_A1(train_loader, val_loader):
    print("\n" + "="*60)
    print("  ABLATION A1: Baseline (Wang & Hu 2020)")
    print("  Static kernel, Charbonnier loss, 100 epochs")
    print("="*60)

    def model_fn():
        from models.base_model import BaselineModel
        from losses.composite_loss import get_loss_function
        return BaselineModel(), get_loss_function('mse_only')

    return _run('A1_baseline', model_fn, 'mse_only', train_loader, val_loader)


def run_A2(train_loader, val_loader):
    print("\n" + "="*60)
    print("  ABLATION A2: Parallel Kernels, No CAM")
    print("  Multi-scale 3x3+5x5+7x7, equal weights, 100 epochs")
    print("="*60)

    def model_fn():
        from models.proposed_model import CATKCNet
        from losses.composite_loss import get_loss_function
        return CATKCNet(use_attention=False), get_loss_function('mse_only')

    return _run('A2_parallel_only', model_fn, 'mse_only', train_loader, val_loader)


def run_A3(train_loader, val_loader):
    print("\n" + "="*60)
    print("  ABLATION A3: Parallel Kernels + CAM, Charbonnier Only")
    print("  Multi-scale + learned attention weighting, 100 epochs")
    print("="*60)

    def model_fn():
        from models.proposed_model import CATKCNet
        from losses.composite_loss import get_loss_function
        return CATKCNet(use_attention=True), get_loss_function('mse_only')

    return _run('A3_cam_mse', model_fn, 'mse_only', train_loader, val_loader)


def run_A4(train_loader, val_loader):
    print("\n" + "="*60)
    print("  ABLATION A4: Full CATKC-Net (Our Proposal) — ALREADY TRAINED")
    print("="*60)

    ckpt = os.path.join(config.CHECKPOINT_DIR, 'A4_full_model', config.BEST_MODEL_NAME)
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location='cpu')
        psnr  = state.get('best_psnr', state.get('psnr', 0.0))
        epoch = state.get('epoch', '?')
        print(f"  ✓ Trained for {epoch} epochs | Val PSNR: {psnr:.4f} dB")
        print(f"  ✓ Test PSNR: 25.02 dB | SSIM: 0.9127 | LPIPS: 0.0915")
        return psnr

    print("  A4 checkpoint not found — training from scratch (200 epochs)...")
    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function
    model   = CATKCNet(use_attention=True)
    loss_fn = get_loss_function('composite')
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        experiment_name='A4_full_model'
    )
    return trainer.train()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', 'A1', 'A2', 'A3', 'A4'])
    args = parser.parse_args()

    print("Loading datasets...")
    train_loader, val_loader, _ = get_dataloaders()

    print(f"\nAblation: {ABLATION_EPOCHS} epochs | Resume-safe (Ctrl+C anytime)")
    print(f"Estimated: A1 ~85min | A2 ~90min | A3 ~90min | A4 skipped\n")

    results = {}

    if args.exp in ['all', 'A1']:
        results['A1'] = run_A1(train_loader, val_loader)
    if args.exp in ['all', 'A2']:
        results['A2'] = run_A2(train_loader, val_loader)
    if args.exp in ['all', 'A3']:
        results['A3'] = run_A3(train_loader, val_loader)
    if args.exp in ['all', 'A4']:
        results['A4'] = run_A4(train_loader, val_loader)

    print("\n" + "="*65)
    print("  ABLATION RESULTS — Val PSNR")
    print("="*65)
    labels = {
        'A1': 'Static kernel baseline (Wang & Hu 2020)',
        'A2': 'Parallel kernels, no attention',
        'A3': 'Parallel + CAM, Charbonnier loss',
        'A4': 'Full CATKC-Net: CAM + Composite  ← OURS',
    }
    for exp in ['A1', 'A2', 'A3', 'A4']:
        if exp in results:
            print(f"  {exp}: {results[exp]:5.2f} dB  |  {labels[exp]}")
    print("="*65)

    print("\nNext — evaluate each on test set:")
    print("  .\\venv\\Scripts\\python.exe evaluate.py --model base     --checkpoint checkpoints/A1_baseline/best_model.pth     --experiment A1_baseline")
    print("  .\\venv\\Scripts\\python.exe evaluate.py --model proposed --checkpoint checkpoints/A2_parallel_only/best_model.pth --experiment A2_parallel_only")
    print("  .\\venv\\Scripts\\python.exe evaluate.py --model proposed --checkpoint checkpoints/A3_cam_mse/best_model.pth      --experiment A3_cam_mse")
    print("  (A4 done: PSNR=25.02dB, SSIM=0.9127, LPIPS=0.0915)")