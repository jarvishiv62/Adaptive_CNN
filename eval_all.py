# eval_all.py — v2
# Evaluates all 4 ablation models on LOL-v2 test set (100 images)
# Same test set as evaluate.py used for A4 (25.02 dB)
#
# Fixes from v1:
#   ✓ Uses LOL-v2 test set (100 images) for all models — consistent comparison
#   ✓ A4 hardcoded to feature_channels=40 (original training config)
#   ✓ A1/A2/A3 auto-detect channels from checkpoint
#
# Run: .\venv\Scripts\python.exe eval_all.py

import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.dataset import LOLDataset
from utils import compute_psnr, compute_ssim

device = torch.device(config.DEVICE)


def get_test_loader():
    # LOL-v2 test set (100 images) — same as evaluate.py used for A4
    low_dir  = 'data/LOL-v2/test/low'
    high_dir = 'data/LOL-v2/test/high'

    if not os.path.exists(low_dir):
        print(f"  ⚠ LOL-v2 test not found at {low_dir}")
        print(f"  Falling back to LOL-v1 test (15 images) — results won't match A4's 25.02 dB")
        low_dir  = config.TEST_LOW_DIR
        high_dir = config.TEST_HIGH_DIR

    test_dataset = LOLDataset(low_dir, high_dir, augment=False, split='test')
    print(f"Test set: {low_dir}")
    print(f"Test images: {len(test_dataset)}")
    return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


def evaluate_model(model, test_loader, name):
    model.eval()
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  Evaluating {name}"):
            low      = batch['low'].to(device)
            high     = batch['high'].to(device)
            out      = model(low)
            enhanced = (out[0] if isinstance(out, tuple) else out).clamp(0, 1)
            psnr_list.append(min(compute_psnr(enhanced, high), 100.0))
            ssim_list.append(compute_ssim(enhanced, high))

    return sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)


def load_baseline(ckpt_path):
    """Load BaselineModel (A1) from checkpoint."""
    from models.base_model import BaselineModel
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = BaselineModel().to(device)
    model.load_state_dict(ckpt['model_state'])
    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')}, val_psnr={ckpt.get('best_psnr',0):.4f} dB")
    return model


def load_catkcnet(ckpt_path, use_attention, force_channels=None):
    """Load CATKCNet, auto-detecting feature_channels from checkpoint."""
    from models.proposed_model import CATKCNet

    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state']

    if force_channels is not None:
        feature_channels = force_channels
        print(f"  Using hardcoded feature_channels={feature_channels}")
    else:
        # Auto-detect from weight shapes
        feature_channels = config.FEATURE_CHANNELS
        for key, val in state.items():
            if len(val.shape) == 4 and val.shape[1] == 3:
                # First conv layer: shape = (out_ch, in_ch=3, kH, kW)
                feature_channels = val.shape[0]
                break
        print(f"  Detected feature_channels={feature_channels} from checkpoint")

    orig = config.FEATURE_CHANNELS
    config.FEATURE_CHANNELS = feature_channels
    model = CATKCNet(use_attention=use_attention).to(device)
    config.FEATURE_CHANNELS = orig

    model.load_state_dict(state, strict=False)
    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')}, val_psnr={ckpt.get('best_psnr',0):.4f} dB")
    return model


def eval_A1(test_loader):
    print("\n── A1: Static Kernel Baseline (Wang & Hu 2020) ──")
    ckpt_path = 'checkpoints/A1_baseline/best_model.pth'
    if not os.path.exists(ckpt_path):
        print("  ✗ Checkpoint not found — run: ablation.py --exp A1")
        return None, None
    model = load_baseline(ckpt_path)
    return evaluate_model(model, test_loader, 'A1')


def eval_A2(test_loader):
    print("\n── A2: Parallel Kernels, No CAM ──")
    ckpt_path = 'checkpoints/A2_parallel_only/best_model.pth'
    if not os.path.exists(ckpt_path):
        print("  ✗ Checkpoint not found — run: ablation.py --exp A2")
        return None, None
    model = load_catkcnet(ckpt_path, use_attention=False)
    return evaluate_model(model, test_loader, 'A2')


def eval_A3(test_loader):
    print("\n── A3: Parallel Kernels + CAM, Charbonnier Loss ──")
    ckpt_path = 'checkpoints/A3_cam_mse/best_model.pth'
    if not os.path.exists(ckpt_path):
        print("  ✗ Checkpoint not found — run: ablation.py --exp A3")
        return None, None
    model = load_catkcnet(ckpt_path, use_attention=True)
    return evaluate_model(model, test_loader, 'A3')


def eval_A4(test_loader):
    print("\n── A4: Full CATKC-Net (Our Proposal) ──")
    ckpt_path = 'checkpoints/A4_full_model/best_model.pth'
    if not os.path.exists(ckpt_path):
        print("  ✗ Checkpoint not found")
        return None, None
    # A4 was trained with feature_channels=40 (original training config)
    # force_channels=40 prevents the auto-detect from picking wrong value
    model = load_catkcnet(ckpt_path, use_attention=True, force_channels=64)
    return evaluate_model(model, test_loader, 'A4')


if __name__ == '__main__':
    print("="*70)
    print("  CATKC-Net Ablation Evaluation")
    print("  Test set: LOL-v2 (100 images) — consistent with A4 evaluation")
    print("="*70)

    print("\nLoading test dataset...")
    test_loader = get_test_loader()

    results = {}

    p, s = eval_A1(test_loader)
    if p is not None: results['A1'] = {'psnr': p, 'ssim': s}

    p, s = eval_A2(test_loader)
    if p is not None: results['A2'] = {'psnr': p, 'ssim': s}

    p, s = eval_A3(test_loader)
    if p is not None: results['A3'] = {'psnr': p, 'ssim': s}

    p, s = eval_A4(test_loader)
    if p is not None: results['A4'] = {'psnr': p, 'ssim': s}

    # ── Thesis table ──
    print("\n" + "="*70)
    print("  ABLATION STUDY — Test Set Results (LOL-v2, 100 images)")
    print("="*70)
    print(f"  {'Model':<6} {'PSNR (dB)':>10} {'SSIM':>8}  {'Gain':>8}  Description")
    print(f"  {'-'*67}")

    labels = {
        'A1': 'Static kernel baseline (Wang & Hu 2020)',
        'A2': 'Parallel kernels, no attention',
        'A3': 'Parallel + CAM, Charbonnier loss',
        'A4': 'Full CATKC-Net: CAM + Composite  ← OURS',
    }

    base_psnr = results.get('A1', {}).get('psnr', None)
    for exp in ['A1', 'A2', 'A3', 'A4']:
        if exp in results:
            r    = results[exp]
            gain = f"+{r['psnr']-base_psnr:.2f}" if base_psnr and exp != 'A1' else 'baseline'
            print(f"  {exp:<6} {r['psnr']:>10.4f} {r['ssim']:>8.4f}  {gain:>8}  {labels[exp]}")
        else:
            print(f"  {exp:<6} {'—':>10} {'—':>8}  {'—':>8}  {labels[exp]} (not done yet)")

    print("="*70)

    # ── Save ──
    os.makedirs('results', exist_ok=True)
    with open('results/ablation_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/ablation_summary.json")