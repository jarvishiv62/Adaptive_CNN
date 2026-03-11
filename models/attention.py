# models/attention.py
# Channel Attention Module (CAM) — core innovation of CATKC-Net
#
# Architecture:
#   MultiScaleParallelConv: runs 3×3, 5×5, 7×7 convolutions in parallel
#   ChannelAttentionModule: GAP → FC → ReLU → FC → Softmax → weights [w3, w5, w7]
#   SpatialAttentionModule: optional spatial refinement after channel attention

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class MultiScaleParallelConv(nn.Module):
    """
    Runs 3 parallel convolutions (3×3, 5×5, 7×7) on the same input.
    Returns concatenated features and individual branch outputs.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f3 = self.branch3(x)
        f5 = self.branch5(x)
        f7 = self.branch7(x)
        # concat for downstream CAM input
        fused = torch.cat([f3, f5, f7], dim=1)   # [B, 3*C, H, W]
        return fused, f3, f5, f7


class ChannelAttentionModule(nn.Module):
    """
    SE-style channel attention that produces soft weights [w3, w5, w7]
    for blending the three parallel convolution branches.

    Input : concatenated features [B, 3*C, H, W]
    Output: weighted fused features [B, C, H, W] + weights [B, 3]
    """

    def __init__(self, feature_channels, hidden_dim=None, dropout=0.1):
        super().__init__()
        in_dim     = feature_channels * 3
        hidden_dim = hidden_dim or in_dim // 4

        self.gap = nn.AdaptiveAvgPool2d(1)   # Global Average Pooling

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 3),        # 3 weights: w3, w5, w7
        )

    def forward(self, fused, f3, f5, f7):
        B, _, H, W = f3.shape

        # GAP over concatenated features → [B, 3C]
        gap_out = self.gap(fused).view(B, -1)

        # Soft weights via FC + Softmax → [B, 3]
        weights = F.softmax(self.fc(gap_out), dim=1)

        w3 = weights[:, 0].view(B, 1, 1, 1)
        w5 = weights[:, 1].view(B, 1, 1, 1)
        w7 = weights[:, 2].view(B, 1, 1, 1)

        # Weighted blend → [B, C, H, W]
        out = w3 * f3 + w5 * f5 + w7 * f7

        return out, weights


class SpatialAttentionModule(nn.Module):
    """
    Simple spatial attention: learns where in the image to focus.
    Applied after channel attention for additional refinement.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        attn_map = torch.sigmoid(self.conv(spatial))
        return x * attn_map


class CAMBlock(nn.Module):
    """
    Full attention block combining:
        MultiScaleParallelConv → ChannelAttentionModule → (optional) SpatialAttentionModule

    This is the core building block of CATKC-Net.
    """

    def __init__(
        self,
        in_channels,
        feature_channels = None,
        hidden_dim       = None,
        dropout          = None,
        use_spatial      = None,
    ):
        super().__init__()
        feature_channels = feature_channels or config.FEATURE_CHANNELS
        hidden_dim       = hidden_dim       or config.CAM_HIDDEN_DIM
        dropout          = dropout          if dropout is not None else config.CAM_DROPOUT
        use_spatial      = use_spatial      if use_spatial is not None else config.USE_SPATIAL_ATTN

        self.ms_conv = MultiScaleParallelConv(in_channels, feature_channels)
        self.cam     = ChannelAttentionModule(feature_channels, hidden_dim, dropout)
        self.spatial = SpatialAttentionModule() if use_spatial else None

        # Project back to in_channels if needed
        self.proj = (
            nn.Conv2d(feature_channels, in_channels, kernel_size=1, bias=False)
            if feature_channels != in_channels else nn.Identity()
        )

    def forward(self, x):
        fused, f3, f5, f7 = self.ms_conv(x)
        out, weights       = self.cam(fused, f3, f5, f7)

        if self.spatial is not None:
            out = self.spatial(out)

        out = self.proj(out)

        # Residual connection
        out = out + x

        return out, weights


if __name__ == "__main__":
    print("Testing attention modules...")

    x = torch.randn(4, 64, 256, 256)

    ms = MultiScaleParallelConv(64, 64)
    fused, f3, f5, f7 = ms(x)
    print(f"  MultiScaleParallelConv: fused={fused.shape}, f3={f3.shape}")

    cam = ChannelAttentionModule(64)
    out, w = cam(fused, f3, f5, f7)
    print(f"  CAM out: {out.shape}, weights: {w.shape}")
    print(f"  Weight sum (should be 1.0): {w[0].sum().item():.4f}")

    block = CAMBlock(64)
    out, w = block(x)
    print(f"  CAMBlock out: {out.shape}, weights: {w[0].detach().numpy()}")

    print("Attention modules OK!")