# models/attention.py
# Channel Attention Module (CAM) for CATKC-Net
# Squeeze-and-Excitation style attention that learns optimal
# fusion weights (w3, w5, w7) for each kernel output per RGB channel.

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module (CAM) for dynamic kernel weight fusion.

    Architecture:
        Input: Concatenation of three kernel outputs → shape (B, 3*C, H, W)
        1. Global Average Pooling      → (B, 3*C, 1, 1)
        2. Squeeze (reshape)           → (B, 3*C)
        3. FC(3*C → hidden_dim) + ReLU → (B, hidden_dim)
        4. Dropout for regularization  → (B, hidden_dim)
        5. FC(hidden_dim → n_kernels)  → (B, n_kernels)  [n_kernels = 3]
        6. Softmax                     → (B, 3)  — weights sum to 1

    The weights w3, w5, w7 ∈ [0, 1] are then used to fuse the
    three parallel kernel outputs with a weighted sum.

    Why this design:
    - Lightweight: only ~200 parameters extra
    - Interpretable: weights directly tell us which kernel the network prefers
    - End-to-end learnable: no offline statistics needed
    """

    def __init__(self, in_channels, n_kernels=3, hidden_dim=64, dropout=0.1):
        """
        Args:
            in_channels : Number of channels in each kernel output (e.g., 64)
            n_kernels   : Number of parallel kernels (default: 3 → 3×3, 5×5, 7×7)
            hidden_dim  : FC hidden layer dimension (default: 64)
            dropout     : Dropout probability for regularization (default: 0.1)
        """
        super(ChannelAttentionModule, self).__init__()

        self.n_kernels   = n_kernels
        self.in_channels = in_channels
        total_channels   = in_channels * n_kernels

        # Global Average Pooling (spatial squeeze)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected attention network
        self.attention_fc = nn.Sequential(
            nn.Linear(total_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, n_kernels),
        )

        # Softmax to get normalized weights (sum to 1)
        self.softmax = nn.Softmax(dim=1)

        # Initialize weights for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize FC layers with small weights for stable early training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, kernel_outputs):
        """
        Args:
            kernel_outputs: List of 3 tensors, each of shape (B, C, H, W)
                           [out_3x3, out_5x5, out_7x7]

        Returns:
            fused   : Attention-weighted sum of kernel outputs, shape (B, C, H, W)
            weights : Attention weights (w3, w5, w7), shape (B, 3)
                      Useful for visualization
        """
        assert len(kernel_outputs) == self.n_kernels, \
            f"Expected {self.n_kernels} kernel outputs, got {len(kernel_outputs)}"

        B, C, H, W = kernel_outputs[0].shape

        # Concatenate along channel dimension: (B, 3*C, H, W)
        concat = torch.cat(kernel_outputs, dim=1)

        # Global Average Pooling: (B, 3*C, 1, 1) → (B, 3*C)
        squeezed = self.gap(concat).view(B, -1)

        # FC layers to get raw scores: (B, 3)
        scores = self.attention_fc(squeezed)

        # Softmax to get weights that sum to 1: (B, 3)
        weights = self.softmax(scores)

        # Weighted fusion: sum over kernels
        # weights[:, k] has shape (B,), need to broadcast to (B, C, H, W)
        fused = sum(
            weights[:, k].view(B, 1, 1, 1) * kernel_outputs[k]
            for k in range(self.n_kernels)
        )

        return fused, weights

    def get_weight_names(self):
        """Returns human-readable names for attention weights."""
        sizes = [3, 5, 7][:self.n_kernels]
        return [f"w{s}x{s}" for s in sizes]


class MultiScaleParallelConv(nn.Module):
    """
    Parallel Multi-Scale Convolution Block.

    Applies 3×3, 5×5, 7×7 convolutions in parallel on the input,
    then uses CAM to dynamically fuse the outputs.

    This replaces the static MSD-based kernel assignment in the base paper.
    """

    def __init__(self, in_channels, out_channels, hidden_dim=64, dropout=0.1):
        """
        Args:
            in_channels  : Input channels
            out_channels : Output channels (same for all three parallel convolutions)
            hidden_dim   : CAM hidden layer size
            dropout      : CAM dropout rate
        """
        super(MultiScaleParallelConv, self).__init__()

        self.out_channels = out_channels

        # Three parallel convolutions with same-size output (via padding)
        # padding = kernel_size // 2 ensures output H,W = input H,W
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Channel Attention Module for fusion
        self.cam = ChannelAttentionModule(
            in_channels=out_channels,
            n_kernels=3,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x : Input tensor (B, C_in, H, W)

        Returns:
            fused   : Attention-fused output (B, C_out, H, W)
            weights : Attention weights (B, 3) — [w3, w5, w7]
        """
        # Apply parallel convolutions
        out3 = self.conv3x3(x)
        out5 = self.conv5x5(x)
        out7 = self.conv7x7(x)

        # Fuse with channel attention
        fused, weights = self.cam([out3, out5, out7])

        return fused, weights


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Channel Attention Module...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C_in, C_out, H, W = 4, 3, 64, 256, 256

    # Test CAM alone
    cam = ChannelAttentionModule(in_channels=C_out, n_kernels=3, hidden_dim=64).to(device)
    dummy_outputs = [torch.randn(B, C_out, H, W).to(device) for _ in range(3)]
    fused, weights = cam(dummy_outputs)
    print(f"  CAM input  : 3 × (B={B}, C={C_out}, H={H}, W={W})")
    print(f"  CAM fused  : {fused.shape}")
    print(f"  CAM weights: {weights.shape}  (sum = {weights.sum(dim=1).mean():.4f})")
    print(f"  Sample weights: {weights[0].detach().cpu().numpy()}")

    # Test MultiScale block
    msc = MultiScaleParallelConv(in_channels=C_in, out_channels=C_out).to(device)
    x = torch.randn(B, C_in, H, W).to(device)
    out, w = msc(x)
    print(f"\n  MultiScaleParallelConv output: {out.shape}")
    print(f"  Attention weights: {w[0].detach().cpu().numpy()}")

    # Count parameters
    total_params = sum(p.numel() for p in msc.parameters() if p.requires_grad)
    print(f"\n  Total parameters in MultiScaleParallelConv: {total_params:,}")
    print("\nChannel Attention Module working correctly!")