# utils.py
# Helper functions for CATKC-Net:
#   - PSNR computation
#   - SSIM computation
#   - Image saving utilities
#   - Attention weight visualization
#   - Metric tracking

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_psnr(pred, target, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.

    Args:
        pred   : (B, C, H, W) or (C, H, W) tensor in [0, 1]
        target : Same shape as pred
        max_val: Maximum pixel value (1.0 for normalized images)

    Returns:
        psnr: scalar float (average over batch if batched)
    """
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse.item() == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val) - 10 * torch.log10(mse)
    return psnr.item()


def compute_psnr_batch(pred, target):
    """Compute per-image PSNR for a batch, return list of values."""
    psnr_list = []
    for i in range(pred.shape[0]):
        psnr_list.append(compute_psnr(pred[i], target[i]))
    return psnr_list


def compute_ssim(pred, target, window_size=11, channel=3):
    """
    Compute SSIM between pred and target.

    Args:
        pred, target: (B, C, H, W) tensors in [0, 1]

    Returns:
        ssim_val: scalar float (average SSIM)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    def gaussian_kernel(size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    _1d = gaussian_kernel(window_size)
    _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)
    window = _2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(pred.device)

    pad = window_size // 2

    mu_x = F.conv2d(pred,   window, padding=pad, groups=channel)
    mu_y = F.conv2d(target, window, padding=pad, groups=channel)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy   = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred,     window, padding=pad, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=pad, groups=channel) - mu_y_sq
    sigma_xy   = F.conv2d(pred * target,   window, padding=pad, groups=channel) - mu_xy

    num   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denom = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    return (num / denom).mean().item()


# ─────────────────────────────────────────────
# Image Saving
# ─────────────────────────────────────────────

def save_comparison_image(low, enhanced, ground_truth, save_path, title=None):
    """
    Save a side-by-side comparison: Low | Enhanced | Ground Truth

    Args:
        low, enhanced, ground_truth: (C, H, W) or (1, C, H, W) tensors in [0, 1]
        save_path : Path to save the image
        title     : Optional title string
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    def to_np(t):
        if t.dim() == 4:
            t = t.squeeze(0)
        return t.permute(1, 2, 0).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [to_np(low), to_np(enhanced), to_np(ground_truth)]
    labels = ['Input (Low-Light)', 'Ours (Enhanced)', 'Ground Truth']

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_ablation_comparison(outputs_dict, save_path):
    """
    Save a row comparison of ablation outputs: A1 | A2 | A3 | A4 | GT

    Args:
        outputs_dict: dict with keys 'input', 'A1', 'A2', 'A3', 'A4', 'gt'
                      each value is a (C, H, W) tensor
        save_path : Output path
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    def to_np(t):
        if t.dim() == 4:
            t = t.squeeze(0)
        return np.clip(t.permute(1, 2, 0).cpu().numpy(), 0, 1)

    keys   = ['input', 'A1', 'A2', 'A3', 'A4', 'gt']
    labels = ['Input', 'A1 (Base)', 'A2 (Parallel)', 'A3 (+CAM)', 'A4 (Full)', 'Ground Truth']

    valid_keys   = [k for k in keys if k in outputs_dict]
    valid_labels = [labels[keys.index(k)] for k in valid_keys]

    fig, axes = plt.subplots(1, len(valid_keys), figsize=(5 * len(valid_keys), 5))
    if len(valid_keys) == 1:
        axes = [axes]

    for ax, key, label in zip(axes, valid_keys, valid_labels):
        ax.imshow(to_np(outputs_dict[key]))
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Ablation Study — Visual Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# Attention Weight Visualization
# ─────────────────────────────────────────────

def visualize_attention_weights(weights_dict, save_path):
    """
    Visualize attention weights (w3, w5, w7) as grouped bar charts.

    Args:
        weights_dict: dict {image_name: np.array of shape (3,)}
                      e.g., {'textured.png': [0.7, 0.2, 0.1], 'smooth.png': [0.1, 0.2, 0.7]}
        save_path   : Output path for the plot
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    n_images = len(weights_dict)
    x = np.arange(n_images)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n_images * 1.5), 5))

    w3_vals = [v[0] for v in weights_dict.values()]
    w5_vals = [v[1] for v in weights_dict.values()]
    w7_vals = [v[2] for v in weights_dict.values()]

    bars1 = ax.bar(x - width, w3_vals, width, label='w₃ (3×3 kernel)', color='#4C72B0', alpha=0.85)
    bars2 = ax.bar(x,         w5_vals, width, label='w₅ (5×5 kernel)', color='#55A868', alpha=0.85)
    bars3 = ax.bar(x + width, w7_vals, width, label='w₇ (7×7 kernel)', color='#C44E52', alpha=0.85)

    ax.set_xlabel('Test Images', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title('CAM Attention Weights — w₃, w₅, w₇ per Image', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(weights_dict.keys()), rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, label='Uniform weight (1/3)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention weights visualization saved to: {save_path}")


def plot_metrics_bar(results_dict, metric='PSNR', save_path='results/metrics_bar.png'):
    """
    Plot bar chart comparing methods on a given metric.

    Args:
        results_dict: dict {method_name: metric_value}
                      e.g., {'A1 (Baseline)': 17.2, 'A4 (Ours)': 18.9, ...}
        metric      : 'PSNR' or 'SSIM'
        save_path   : Output path
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    methods = list(results_dict.keys())
    values  = list(results_dict.values())
    colors  = ['#95B3D7'] * (len(methods) - 1) + ['#E05C5C']  # Highlight our method

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, values, color=colors, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    unit = ' (dB)' if metric == 'PSNR' else ''
    ax.set_ylabel(f'{metric}{unit}', fontsize=12)
    ax.set_title(f'{metric} Comparison — Methods', fontsize=14, fontweight='bold')
    ax.set_ylim(min(values) - 0.5, max(values) + 0.8)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Metrics bar chart saved to: {save_path}")


# ─────────────────────────────────────────────
# Training Utilities
# ─────────────────────────────────────────────

class AverageMeter:
    """Tracks running average of a metric during training."""
    def __init__(self, name=''):
        self.name  = name
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=30, min_delta=0.001, mode='max'):
        """
        Args:
            patience : Epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode     : 'max' for PSNR/SSIM, 'min' for loss
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def tensor_to_image(tensor):
    """Convert (C, H, W) tensor in [0,1] to numpy (H, W, C) uint8."""
    img = tensor.detach().cpu().clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def create_dirs(*dirs):
    """Create multiple directories."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing utilities...")
    import config

    # Test metrics
    pred   = torch.rand(4, 3, 256, 256)
    target = torch.rand(4, 3, 256, 256)
    psnr = compute_psnr(pred, target)
    ssim = compute_ssim(pred, target)
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")

    # Test AverageMeter
    meter = AverageMeter('PSNR')
    for v in [20.0, 21.0, 19.5]:
        meter.update(v)
    print(f"  AverageMeter: {meter}")

    # Test EarlyStopping
    es = EarlyStopping(patience=3, mode='max')
    for score in [20.0, 20.5, 20.3, 20.2, 20.1]:
        stopped = es(score)
        print(f"  Score: {score}, Stopped: {stopped}")

    print("\nUtils working correctly!")