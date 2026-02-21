# models/base_model.py
# Re-implementation of Wang & Hu (2020) baseline:
# "An Improved Enhancement Algorithm Based on CNN Applicable for Weak Contrast Images"
# IEEE Access, DOI: 10.1109/ACCESS.2020.2964816
#
# Key features:
#   - Trapezoidal convolution kernel concept
#   - Static kernel assignment per RGB channel based on Mean Square Deviation (MSD)
#   - Channels are assigned 3×3, 5×5, or 7×7 kernel based on offline MSD values
#   - Output: Enhanced = Input - Predicted Noise Map

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_msd(image_tensor):
    """
    Compute Mean Square Deviation (MSD) per channel for a batch of images.
    Used offline to determine static kernel assignment.

    MSD = mean of squared deviations from mean pixel value per channel.

    Args:
        image_tensor: (B, 3, H, W) tensor in [0,1]

    Returns:
        msd: (3,) tensor — one MSD value per channel (R, G, B)
    """
    channel_msd = []
    for c in range(3):
        ch = image_tensor[:, c, :, :]           # (B, H, W)
        mean_val = ch.mean()
        msd = ((ch - mean_val) ** 2).mean()
        channel_msd.append(msd.item())
    return channel_msd


def assign_kernel_by_msd(msd_values):
    """
    Assign kernel size to each channel based on MSD ranking.

    Logic (from Wang & Hu 2020):
        - Channel with highest MSD → 7×7 (captures broad, high-variance features)
        - Channel with middle MSD  → 5×5
        - Channel with lowest MSD  → 3×3 (fine-grained, low-variance)

    Args:
        msd_values: list of 3 floats [msd_R, msd_G, msd_B]

    Returns:
        assignment: dict {channel_idx: kernel_size}
    """
    sorted_indices = np.argsort(msd_values)   # Ascending order
    kernel_map = {sorted_indices[0]: 3,       # Lowest MSD → 3×3
                  sorted_indices[1]: 5,       # Middle MSD → 5×5
                  sorted_indices[2]: 7}       # Highest MSD → 7×7
    return kernel_map


class CharacteristicActivationLayer(nn.Module):
    """
    Layer 1: Characteristic Activation Layer
    Extracts initial features from input using 3×3 conv + ReLU.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class HighDimMappingLayer(nn.Module):
    """
    Layer 2: High-Dimensional Mapping Layer
    Maps features to high-dimensional space for better representation.
    Multiple stacked conv layers to increase representational capacity.
    """
    def __init__(self, channels=64, n_layers=5):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class StaticTrapezoidalConvBlock(nn.Module):
    """
    Static Trapezoidal Convolution Block (core of the base paper).

    Each RGB channel is processed by a different kernel size
    based on pre-computed MSD values (assigned statically).

    Default assignment (for natural images, G typically has highest variance):
        R → 5×5, G → 7×7, B → 3×3
    (This is configurable via kernel_assignment parameter)
    """
    def __init__(self, in_channels=3, out_channels=64, kernel_assignment=None):
        super().__init__()

        # Default static assignment (can be overridden)
        if kernel_assignment is None:
            kernel_assignment = {0: 5, 1: 7, 2: 3}  # R:5×5, G:7×7, B:3×3

        self.kernel_assignment = kernel_assignment

        # One convolution per channel with assigned kernel size
        self.channel_convs = nn.ModuleDict()
        for ch_idx, k_size in kernel_assignment.items():
            self.channel_convs[str(ch_idx)] = nn.Sequential(
                nn.Conv2d(1, out_channels, kernel_size=k_size, padding=k_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Combine channel features
        self.fusion = nn.Conv2d(out_channels * in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            fused: (B, out_channels, H, W) combined feature map
        """
        channel_features = []
        for ch_idx in sorted(self.kernel_assignment.keys()):
            ch_input = x[:, ch_idx:ch_idx+1, :, :]          # (B, 1, H, W)
            ch_feat  = self.channel_convs[str(ch_idx)](ch_input)  # (B, C_out, H, W)
            channel_features.append(ch_feat)

        concat = torch.cat(channel_features, dim=1)          # (B, 3*C_out, H, W)
        fused  = self.fusion(concat)                          # (B, C_out, H, W)
        return fused


class ImageGenerationLayer(nn.Module):
    """
    Layer 3: Image Generation Layer (Deconvolution / Reconstruction)
    Reconstructs the noise map from feature maps using transposed convolution.
    """
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()  # Output noise map in [0,1]
        )

    def forward(self, x):
        return self.deconv(x)


class BaselineModel(nn.Module):
    """
    Complete Baseline Model: Wang & Hu (2020) re-implementation.

    Pipeline:
        Input (weak contrast image)
            ↓
        StaticTrapezoidalConvBlock  ← Different kernel per channel (static MSD)
            ↓
        CharacteristicActivationLayer
            ↓
        HighDimMappingLayer
            ↓
        ImageGenerationLayer         → Noise Map
            ↓
        Enhanced = Input - Noise Map  ← Final enhancement formula

    This is the A1 baseline in our ablation study.
    """

    def __init__(self, kernel_assignment=None):
        super().__init__()

        # Static trapezoidal convolution (core of base paper)
        self.trapezoidal = StaticTrapezoidalConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_assignment=kernel_assignment
        )

        # Remaining layers
        self.char_act = CharacteristicActivationLayer(in_channels=64, out_channels=64)
        self.high_dim = HighDimMappingLayer(channels=64, n_layers=5)
        self.img_gen  = ImageGenerationLayer(in_channels=64, out_channels=3)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) weak contrast (low-light) input image

        Returns:
            enhanced  : (B, 3, H, W) enhanced output, clamped to [0, 1]
            noise_map : (B, 3, H, W) predicted noise map
        """
        # Feature extraction with static trapezoidal kernels
        feat = self.trapezoidal(x)     # (B, 64, H, W)

        # Subsequent processing layers
        feat = self.char_act(feat)     # (B, 64, H, W)
        feat = self.high_dim(feat)     # (B, 64, H, W)

        # Predict noise map
        noise_map = self.img_gen(feat) # (B, 3, H, W)

        # Final enhancement: Enhanced = Input - Noise Map
        enhanced = torch.clamp(x - noise_map, 0.0, 1.0)

        return enhanced, noise_map

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"BaselineModel — Total trainable parameters: {total:,}")
        return total


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Baseline Model (Wang & Hu 2020)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaselineModel().to(device)
    model.count_parameters()

    x = torch.randn(2, 3, 256, 256).to(device)
    enhanced, noise_map = model(x)

    print(f"  Input shape     : {x.shape}")
    print(f"  Enhanced shape  : {enhanced.shape}")
    print(f"  Noise map shape : {noise_map.shape}")
    print(f"  Enhanced range  : [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    print("\nBaseline Model working correctly!")