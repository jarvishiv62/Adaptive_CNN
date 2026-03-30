# models/base_model.py — REDESIGNED v4
# Updated to use residual addition (same as proposed model) for fair comparison.
# The only difference from proposed model: static kernel assignment (no CAM).

import torch
import torch.nn as nn
import numpy as np


class StaticTrapezoidalConvBlock(nn.Module):
    """
    Wang & Hu (2020) static kernel assignment.
    R→5×5, G→7×7, B→3×3 (default for natural images where G has highest MSD).
    """
    def __init__(self, in_channels=3, out_channels=64, kernel_assignment=None):
        super().__init__()
        if kernel_assignment is None:
            kernel_assignment = {0: 5, 1: 7, 2: 3}  # R:5x5, G:7x7, B:3x3
        self.kernel_assignment = kernel_assignment

        self.channel_convs = nn.ModuleDict()
        for ch_idx, k_size in kernel_assignment.items():
            self.channel_convs[str(ch_idx)] = nn.Sequential(
                nn.Conv2d(1, out_channels, kernel_size=k_size, padding=k_size//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.fusion = nn.Conv2d(out_channels * in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        feats = []
        for ch_idx in sorted(self.kernel_assignment.keys()):
            ch_input = x[:, ch_idx:ch_idx+1, :, :]
            feats.append(self.channel_convs[str(ch_idx)](ch_input))
        return self.fusion(torch.cat(feats, dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class BaselineModel(nn.Module):
    """
    Wang & Hu (2020) re-implementation with residual output formulation.
    Uses static kernel assignment; everything else matches proposed model.
    """
    def __init__(self, kernel_assignment=None):
        super().__init__()
        self.trapezoidal = StaticTrapezoidalConvBlock(3, 64, kernel_assignment)

        self.char_act = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.high_dim = nn.ModuleList([ResidualBlock(64) for _ in range(5)])

        self.output_conv1 = nn.Conv2d(64, 32, 3, padding=1, bias=True)
        self.output_act   = nn.LeakyReLU(0.2, inplace=True)
        self.output_conv2 = nn.Conv2d(32, 3,  3, padding=1, bias=True)
        self.tanh         = nn.Tanh()

        nn.init.xavier_normal_(self.output_conv2.weight, gain=0.01)
        nn.init.zeros_(self.output_conv2.bias)

    def forward(self, x):
        feat = self.trapezoidal(x)
        feat = self.char_act(feat)
        for block in self.high_dim:
            feat = block(feat)

        residual = self.tanh(self.output_act(self.output_conv1(feat)))
        residual = self.tanh(self.output_conv2(
            self.output_act(self.output_conv1(feat))
        )) * 0.5

        enhanced = torch.clamp(x + residual, 0.0, 1.0)
        return enhanced, residual

    def count_parameters(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"BaselineModel — params: {n:,}")
        return n


if __name__ == "__main__":
    model = BaselineModel()
    model.count_parameters()
    x = torch.rand(2, 3, 128, 128)
    enhanced, residual = model(x)
    print(f"Output: {enhanced.shape}, residual mean: {residual.mean():.4f}")
    print("BaselineModel OK!")