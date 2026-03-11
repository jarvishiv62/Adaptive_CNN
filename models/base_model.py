# models/base_model.py
# Baseline model: Wang & Hu (2020) — static kernel assignment
#
# Paper: "An Improved Enhancement Algorithm Based on CNN Applicable
#         for Weak Contrast Images" — IEEE Access 2020
#
# Key difference from CATKC-Net:
#   - Kernel selection is STATIC (based on offline MSD computation)
#   - One fixed kernel per channel group
#   - No channel attention, no multi-scale fusion
#   - MSE loss only
#
# This is A1 in the ablation study.

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ResidualBlock(nn.Module):
    """Standard residual block with two 3×3 convolutions."""

    def __init__(self, channels, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.residual_scale * self.block(x))


class BaselineModel(nn.Module):
    """
    Wang & Hu 2020 baseline: Static trapezoidal kernel CNN.

    Architecture:
        Input (3ch) → Conv3×3 → [N residual blocks] → Conv3×3 → Output (3ch)

    Uses a single fixed 3×3 kernel throughout (static assignment).
    No attention mechanism, no multi-scale fusion.
    """

    def __init__(
        self,
        in_channels      = None,
        out_channels     = None,
        feature_channels = None,
        n_residual       = None,
        residual_scale   = None,
    ):
        super().__init__()
        in_channels      = in_channels      or config.IN_CHANNELS
        out_channels     = out_channels     or config.OUT_CHANNELS
        feature_channels = feature_channels or config.FEATURE_CHANNELS
        n_residual       = n_residual       or config.N_RESIDUAL_LAYERS
        residual_scale   = residual_scale   if residual_scale is not None else config.RESIDUAL_SCALE

        # Encoder
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        # Residual body (static 3×3 only)
        self.body = nn.Sequential(
            *[ResidualBlock(feature_channels, residual_scale) for _ in range(n_residual)]
        )

        # Decoder
        self.tail = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        out  = self.tail(feat)
        # Residual learning: predict enhancement delta
        out  = torch.clamp(x + out, 0.0, 1.0)
        return out


if __name__ == "__main__":
    print("Testing BaselineModel...")
    model = BaselineModel()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"  Input : {x.shape}")
    print(f"  Output: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")
    print("BaselineModel OK!")