# models/proposed_model.py — REDESIGNED v4
#
# ROOT CAUSE OF STUCK AT 8.38 dB:
# The "Enhanced = Input - NoiseMap" formulation is wrong for LOL dataset.
# LOL images are genuinely DARK (mean pixel ~0.1-0.2). Ground truth is BRIGHT (mean ~0.5-0.7).
# To go from dark to bright, you need to ADD light, not subtract noise.
# A noise map initialized near 0 → output ≈ input → PSNR vs bright GT = ~8 dB.
# The model gets stuck because subtracting anything makes it worse.
#
# FIX: Replace noise subtraction with DIRECT enhancement output.
# The network directly predicts the enhanced image (like an encoder-decoder).
# The final layer uses Sigmoid so output is in [0,1].
# This lets the network freely learn to brighten, sharpen, and restore.
#
# Secondary fix: Add skip connection from input to output so the network
# learns the RESIDUAL (enhancement delta) rather than the full image.
# This is standard practice in image restoration (DnCNN, DPED, etc.)

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.attention import MultiScaleParallelConv


class CharacteristicActivationLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Standard residual block for high-dim mapping."""
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


class HighDimMappingLayer(nn.Module):
    def __init__(self, channels=64, n_layers=5):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class EnhancementOutputLayer(nn.Module):
    """
    REDESIGNED output layer.

    Instead of predicting a noise map to subtract, we predict
    a RESIDUAL enhancement delta in range [-0.5, 0.5] and ADD it to input.

    Enhanced = clamp(Input + Residual, 0, 1)

    Why residual (not direct output)?
    - Network only needs to learn the CORRECTION, not reconstruct the full image
    - Gradients flow more easily (smaller target values)
    - Natural identity initialization: residual=0 → output=input

    Initialization:
    - Final conv weights → near zero (via xavier with small gain)
    - Final conv bias   → zero
    - This means residual ≈ 0 at start → output ≈ input
    - PSNR at start will be ~8 dB (input vs GT), which is correct —
      the model will then LEARN to push it higher by adding brightness.
    """
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels, 32, 3, padding=1, bias=True)
        self.act    = nn.LeakyReLU(0.2, inplace=True)
        self.conv2  = nn.Conv2d(32, out_channels, 3, padding=1, bias=True)
        self.tanh   = nn.Tanh()   # Output in [-1, 1]; we scale to [-0.5, 0.5]

        # Initialize final layer to near-zero so residual starts small
        nn.init.xavier_normal_(self.conv2.weight, gain=0.01)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        x = self.act(self.conv1(x))
        residual = self.tanh(self.conv2(x)) * 0.5   # Scale to [-0.5, 0.5]
        return residual


class CATKCNet(nn.Module):
    """
    CATKC-Net v4: Content-Adaptive Trapezoidal Kernel CNN.

    Pipeline:
        Input (low-light, dark image)
            ↓
        MultiScaleParallelConv + CAM  ← Our innovation
            ↓
        CharacteristicActivationLayer
            ↓
        HighDimMappingLayer (residual blocks)
            ↓
        EnhancementOutputLayer → Residual ∈ [-0.5, 0.5]
            ↓
        Enhanced = clamp(Input + Residual, 0, 1)

    The network learns to ADD brightness/detail to dark images.
    Positive residual = brightening. Negative = tone-down highlights.
    """

    def __init__(
        self,
        in_channels      = 3,
        feature_channels = 64,
        n_layers         = 5,
        cam_hidden_dim   = 64,
        cam_dropout      = 0.1,
        use_attention    = True
    ):
        super(CATKCNet, self).__init__()
        self.use_attention = use_attention

        if use_attention:
            self.multi_scale = MultiScaleParallelConv(
                in_channels=in_channels,
                out_channels=feature_channels,
                hidden_dim=cam_hidden_dim,
                dropout=cam_dropout
            )
        else:
            # A2: equal-weight parallel kernels, no attention
            self.conv3x3 = nn.Sequential(
                nn.Conv2d(in_channels, feature_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(feature_channels), nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv5x5 = nn.Sequential(
                nn.Conv2d(in_channels, feature_channels, 5, padding=2, bias=False),
                nn.BatchNorm2d(feature_channels), nn.LeakyReLU(0.2, inplace=True)
            )
            self.conv7x7 = nn.Sequential(
                nn.Conv2d(in_channels, feature_channels, 7, padding=3, bias=False),
                nn.BatchNorm2d(feature_channels), nn.LeakyReLU(0.2, inplace=True)
            )

        self.char_act   = CharacteristicActivationLayer(feature_channels, feature_channels)
        self.high_dim   = HighDimMappingLayer(feature_channels, n_layers)
        self.output_layer = EnhancementOutputLayer(feature_channels, in_channels)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) low-light input in [0, 1]
        Returns:
            enhanced : (B, 3, H, W) enhanced image in [0, 1]
            residual : (B, 3, H, W) enhancement map in [-0.5, 0.5]
            weights  : (B, 3) attention weights [w3, w5, w7], or None
        """
        if self.use_attention:
            feat, weights = self.multi_scale(x)
        else:
            feat    = (self.conv3x3(x) + self.conv5x5(x) + self.conv7x7(x)) / 3.0
            weights = None

        feat     = self.char_act(feat)
        feat     = self.high_dim(feat)
        residual = self.output_layer(feat)

        # ADD residual to input (network learns to brighten dark images)
        enhanced = torch.clamp(x + residual, 0.0, 1.0)

        return enhanced, residual, weights

    def count_parameters(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CATKCNet ({'CAM' if self.use_attention else 'no CAM'}) — params: {n:,}")
        return n


def build_model_A2(): return CATKCNet(use_attention=False)
def build_model_A3(): return CATKCNet(use_attention=True)
def build_model_A4(): return CATKCNet(use_attention=True)


if __name__ == "__main__":
    import math
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CATKCNet(use_attention=True).to(device)
    model.count_parameters()

    # Simulate a real LOL scenario: dark input (mean ~0.15), bright GT (mean ~0.5)
    x  = torch.rand(2, 3, 128, 128).to(device) * 0.3        # Dark input
    gt = torch.rand(2, 3, 128, 128).to(device) * 0.5 + 0.3  # Bright GT

    with torch.no_grad():
        enhanced, residual, weights = model(x)

    res_mean   = residual.mean().item()
    res_range  = (residual.min().item(), residual.max().item())
    input_mean = x.mean().item()
    enh_mean   = enhanced.mean().item()

    mse_input    = ((x - gt)**2).mean().item()
    mse_enhanced = ((enhanced - gt)**2).mean().item()
    psnr_input    = -10*math.log10(mse_input + 1e-10)
    psnr_enhanced = -10*math.log10(mse_enhanced + 1e-10)

    print(f"\nInput mean    : {input_mean:.3f}  (dark)")
    print(f"Enhanced mean : {enh_mean:.3f}  (should be ≈ input at init)")
    print(f"Residual mean : {res_mean:.4f}  (should be ≈ 0 at init)")
    print(f"Residual range: [{res_range[0]:.4f}, {res_range[1]:.4f}]")
    print(f"PSNR (input vs GT)   : {psnr_input:.1f} dB")
    print(f"PSNR (enhanced vs GT): {psnr_enhanced:.1f} dB  (≈ same as input at init)")
    print(f"Weights: {weights[0].detach().cpu().numpy()}")

    if abs(res_mean) < 0.05:
        print("\n✓ Model initialized correctly — ready to train!")
    else:
        print("\n✗ Residual not near zero at init")