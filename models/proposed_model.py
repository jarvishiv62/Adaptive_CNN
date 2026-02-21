# models/proposed_model.py
# CATKC-Net: Content-Adaptive Trapezoidal Kernel CNN
# Our proposed model that replaces static kernel assignment with
# dynamic Channel Attention Module (CAM) for multi-scale kernel fusion.
#
# Key difference from base paper:
#   Base paper: Static kernel per channel (determined offline via MSD)
#   Ours      : Parallel 3×3/5×5/7×7 with learned attention weights

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.attention import MultiScaleParallelConv, ChannelAttentionModule


class CharacteristicActivationLayer(nn.Module):
    """
    Layer 1: Characteristic Activation Layer
    Extracts initial features (shared with base paper).
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HighDimMappingLayer(nn.Module):
    """
    Layer 2: High-Dimensional Mapping Layer
    Multiple stacked conv layers for better feature representation.
    Includes residual connections for better gradient flow.
    """
    def __init__(self, channels=64, n_layers=5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)   # Residual connection
        return x


class ImageGenerationLayer(nn.Module):
    """
    Layer 3: Image Generation Layer (Deconvolution)
    Reconstructs noise map from feature maps.
    """
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()    # Noise map in [0, 1]
        )

    def forward(self, x):
        return self.deconv(x)


class CATKCNet(nn.Module):
    """
    CATKC-Net: Content-Adaptive Trapezoidal Kernel CNN (Proposed Model)

    Architecture Overview:
    ┌─────────────────────────────────────────────────────┐
    │  Input (B, 3, H, W)                                 │
    │        ↓                                            │
    │  MultiScaleParallelConv ← OUR KEY INNOVATION        │
    │  ┌──────────────────────────────────────────────┐   │
    │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
    │  │  │ Conv 3×3 │  │ Conv 5×5 │  │ Conv 7×7 │   │   │
    │  │  └──────────┘  └──────────┘  └──────────┘   │   │
    │  │         ↓            ↓            ↓           │   │
    │  │  ┌────────────────────────────────────────┐  │   │
    │  │  │    Channel Attention Module (CAM)       │  │   │
    │  │  │  GAP → FC(64) → ReLU → FC(3) → Softmax │  │   │
    │  │  │    w3   w5   w7  (learned weights)      │  │   │
    │  │  └────────────────────────────────────────┘  │   │
    │  │         ↓  Weighted Sum Fusion               │   │
    │  └──────────────────────────────────────────────┘   │
    │        ↓  (B, 64, H, W)                             │
    │  CharacteristicActivationLayer                      │
    │        ↓                                            │
    │  HighDimMappingLayer (with residuals)               │
    │        ↓                                            │
    │  ImageGenerationLayer → Noise Map (B, 3, H, W)      │
    │        ↓                                            │
    │  Enhanced = clamp(Input - Noise Map, 0, 1)          │
    └─────────────────────────────────────────────────────┘

    This is used for:
        A3: With CAM, MSE loss only
        A4: With CAM, Composite loss (full proposed model)
    """

    def __init__(
        self,
        in_channels=3,
        feature_channels=64,
        n_layers=5,
        cam_hidden_dim=64,
        cam_dropout=0.1,
        use_attention=True     # If False → equal-weight fusion (ablation A2)
    ):
        super(CATKCNet, self).__init__()

        self.use_attention = use_attention

        # ── Our core innovation: Parallel Multi-Scale Conv + CAM ──
        if use_attention:
            self.multi_scale = MultiScaleParallelConv(
                in_channels=in_channels,
                out_channels=feature_channels,
                hidden_dim=cam_hidden_dim,
                dropout=cam_dropout
            )
        else:
            # A2: Parallel kernels with equal weights (no attention)
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_channels), nn.ReLU(inplace=True)
            )
            self.conv5x5 = nn.Sequential(
                nn.Conv2d(in_channels, feature_channels, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm2d(feature_channels), nn.ReLU(inplace=True)
            )
            self.conv7x7 = nn.Sequential(
                nn.Conv2d(in_channels, feature_channels, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(feature_channels), nn.ReLU(inplace=True)
            )

        # ── Remaining layers (same as base paper) ──
        self.char_act = CharacteristicActivationLayer(
            in_channels=feature_channels,
            out_channels=feature_channels
        )
        self.high_dim = HighDimMappingLayer(channels=feature_channels, n_layers=n_layers)
        self.img_gen  = ImageGenerationLayer(in_channels=feature_channels, out_channels=in_channels)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) — weak contrast (low-light) input image in [0, 1]

        Returns:
            enhanced  : (B, 3, H, W) — enhanced output in [0, 1]
            noise_map : (B, 3, H, W) — predicted noise map
            weights   : (B, 3) or None — attention weights [w3, w5, w7]
                        (None if use_attention=False)
        """
        # ── Dynamic multi-scale feature extraction ──
        if self.use_attention:
            feat, weights = self.multi_scale(x)    # (B, 64, H, W), (B, 3)
        else:
            # Equal-weight fusion (ablation A2)
            out3 = self.conv3x3(x)
            out5 = self.conv5x5(x)
            out7 = self.conv7x7(x)
            feat    = (out3 + out5 + out7) / 3.0   # Simple average
            weights = None

        # ── Subsequent processing ──
        feat = self.char_act(feat)
        feat = self.high_dim(feat)

        # ── Noise map prediction ──
        noise_map = self.img_gen(feat)

        # ── Final enhancement: Enhanced = Input - Noise Map ──
        enhanced = torch.clamp(x - noise_map, 0.0, 1.0)

        return enhanced, noise_map, weights

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode = "with CAM" if self.use_attention else "parallel only (no CAM)"
        print(f"CATKCNet ({mode}) — Total trainable parameters: {total:,}")
        return total

    def get_attention_weights_for_batch(self, x):
        """
        Helper: Run a batch through the model and return attention weights.
        Useful for visualization.

        Returns:
            weights_np: numpy array of shape (B, 3) — [w3, w5, w7] per image
        """
        self.eval()
        with torch.no_grad():
            _, _, weights = self.forward(x)
        if weights is not None:
            return weights.cpu().numpy()
        return None


# ─────────────────────────────────────────────
# Factory functions for ablation experiments
# ─────────────────────────────────────────────

def build_model_A2():
    """A2: Parallel kernels with equal weights, no attention."""
    return CATKCNet(use_attention=False)

def build_model_A3():
    """A3: Parallel kernels + CAM (attention). Train with MSE only."""
    return CATKCNet(use_attention=True)

def build_model_A4():
    """A4: Full model — Parallel + CAM + Composite loss. Same as A3 but different loss."""
    return CATKCNet(use_attention=True)


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing CATKC-Net (Proposed Model)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test full model (A4)
    model = CATKCNet(use_attention=True).to(device)
    model.count_parameters()

    x = torch.rand(4, 3, 256, 256).to(device)   # Simulated low-light input
    enhanced, noise_map, weights = model(x)

    print(f"\n  Input shape     : {x.shape}")
    print(f"  Enhanced shape  : {enhanced.shape}")
    print(f"  Noise map shape : {noise_map.shape}")
    print(f"  Weights shape   : {weights.shape}")
    print(f"  Sample weights  : {weights[0].detach().cpu().numpy()}")
    print(f"  Weights sum     : {weights[0].sum().item():.4f}  (should be ~1.0)")
    print(f"  Enhanced range  : [{enhanced.min():.3f}, {enhanced.max():.3f}]")

    # Test A2 variant
    print("\n--- Testing A2 (no attention) ---")
    model_a2 = build_model_A2().to(device)
    model_a2.count_parameters()
    enhanced_a2, _, w_a2 = model_a2(x)
    print(f"  Weights (A2): {w_a2}  (should be None)")

    print("\nAdaptive CNN working correctly!")