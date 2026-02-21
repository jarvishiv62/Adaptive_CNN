# losses/composite_loss.py
# Composite Loss Function for CATKC-Net
#
# L_total = λ_mse * L_mse + λ_ssim * (1 - SSIM) + λ_perc * L_perc
#
# Components:
#   MSE Loss        (λ=0.5) : Pixel-level fidelity
#   SSIM Loss       (λ=0.3) : Structural similarity
#   Perceptual Loss (λ=0.2) : VGG16 feature-level quality

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────
# 1. SSIM Loss
# ─────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    SSIM measures luminance, contrast, and structural similarity.
    Loss = 1 - SSIM  (so minimizing loss maximizes similarity)

    Implementation follows the standard SSIM formula using sliding windows.
    """

    def __init__(self, window_size=11, sigma=1.5, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel     = channel
        self.sigma       = sigma
        self.register_buffer('window', self._create_window(window_size, sigma, channel))

    def _gaussian_kernel(self, size, sigma):
        """Create 1D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, sigma, channel):
        """Create 2D Gaussian window for SSIM computation."""
        _1d = self._gaussian_kernel(window_size, sigma)
        _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)    # (W, W)
        window = _2d.unsqueeze(0).unsqueeze(0)         # (1, 1, W, W)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, x, y):
        """Compute SSIM between x and y."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2

        window = self.window.to(x.device)

        mu_x    = F.conv2d(x, window, padding=pad, groups=self.channel)
        mu_y    = F.conv2d(y, window, padding=pad, groups=self.channel)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy   = mu_x * mu_y

        sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=self.channel) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=self.channel) - mu_y_sq
        sigma_xy   = F.conv2d(x * y, window, padding=pad, groups=self.channel) - mu_xy

        num   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denom = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

        ssim_map = num / denom
        return ssim_map.mean()

    def forward(self, pred, target):
        """
        Args:
            pred   : (B, 3, H, W) predicted enhanced image
            target : (B, 3, H, W) ground truth normal-light image

        Returns:
            loss: 1 - SSIM (scalar)
        """
        return 1.0 - self._ssim(pred, target)


# ─────────────────────────────────────────────
# 2. Perceptual Loss (VGG16 Features)
# ─────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG16 feature maps.
    Computes MSE between VGG16 relu3_3 features of pred and target.

    Why VGG16?
    - Pre-trained on ImageNet — captures rich perceptual features
    - Features at relu3_3 capture mid-level textures
    - Helps avoid blurring artifacts from MSE-only training

    VGG16 feature layers used: relu3_3 (default, layer index 15)
    """

    def __init__(self, feature_layer=15):
        super(PerceptualLoss, self).__init__()

        # Load pre-trained VGG16
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:feature_layer + 1])

        # Freeze VGG weights (we don't train VGG)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # ImageNet normalization (VGG expects normalized input)
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize_for_vgg(self, x):
        """Normalize [0,1] images to ImageNet stats for VGG."""
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """
        Args:
            pred   : (B, 3, H, W) predicted enhanced image in [0,1]
            target : (B, 3, H, W) ground truth in [0,1]

        Returns:
            loss: MSE between VGG features (scalar)
        """
        pred_norm   = self.normalize_for_vgg(pred)
        target_norm = self.normalize_for_vgg(target)

        pred_features   = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)

        return F.mse_loss(pred_features, target_features)


# ─────────────────────────────────────────────
# 3. Composite Loss (Main Loss Function)
# ─────────────────────────────────────────────

class CompositeLoss(nn.Module):
    """
    Composite Loss Function for CATKC-Net.

    L_total = λ_mse * L_mse + λ_ssim * (1 - SSIM) + λ_perc * L_perc

    Default weights:
        λ_mse  = 0.5  — dominant pixel-level fidelity
        λ_ssim = 0.3  — structural preservation
        λ_perc = 0.2  — perceptual quality

    Used in:
        A3: MSE only  (λ_ssim=0, λ_perc=0)
        A4: Full composite loss

    Args:
        lambda_mse  : Weight for MSE loss
        lambda_ssim : Weight for SSIM loss
        lambda_perc : Weight for Perceptual loss
        use_ssim    : Include SSIM term (False for A3 ablation)
        use_perc    : Include Perceptual term (False for A3 ablation)
    """

    def __init__(
        self,
        lambda_mse  = config.LAMBDA_MSE,
        lambda_ssim = config.LAMBDA_SSIM,
        lambda_perc = config.LAMBDA_PERC,
        use_ssim    = True,
        use_perc    = True
    ):
        super(CompositeLoss, self).__init__()

        self.lambda_mse  = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_perc = lambda_perc
        self.use_ssim    = use_ssim
        self.use_perc    = use_perc

        self.mse_loss  = nn.MSELoss()

        if use_ssim:
            self.ssim_loss = SSIMLoss(channel=3)

        if use_perc:
            self.perc_loss = PerceptualLoss(feature_layer=15)

        print(f"CompositeLoss initialized:")
        print(f"  MSE  : λ={lambda_mse}")
        print(f"  SSIM : λ={lambda_ssim if use_ssim else 0} ({'enabled' if use_ssim else 'disabled'})")
        print(f"  Perc : λ={lambda_perc if use_perc else 0} ({'enabled' if use_perc else 'disabled'})")

    def forward(self, pred, target):
        """
        Args:
            pred   : (B, 3, H, W) predicted enhanced image
            target : (B, 3, H, W) ground truth normal-light image

        Returns:
            total_loss : Scalar combined loss
            loss_dict  : Dict with individual loss values (for logging)
        """
        # MSE Loss (always included)
        l_mse = self.mse_loss(pred, target)
        total = self.lambda_mse * l_mse

        loss_dict = {'mse': l_mse.item()}

        # SSIM Loss
        if self.use_ssim:
            l_ssim = self.ssim_loss(pred, target)
            total  = total + self.lambda_ssim * l_ssim
            loss_dict['ssim'] = l_ssim.item()

        # Perceptual Loss
        if self.use_perc:
            l_perc = self.perc_loss(pred, target)
            total  = total + self.lambda_perc * l_perc
            loss_dict['perceptual'] = l_perc.item()

        loss_dict['total'] = total.item()

        return total, loss_dict


def get_loss_function(loss_type='composite'):
    """
    Factory function to get loss for ablation experiments.

    Args:
        loss_type: 'mse_only' (for A1, A2, A3) or 'composite' (for A4)
    """
    if loss_type == 'mse_only':
        return CompositeLoss(use_ssim=False, use_perc=False)
    elif loss_type == 'composite':
        return CompositeLoss(use_ssim=True, use_perc=True)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'mse_only' or 'composite'.")


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Composite Loss Function...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W = 2, 3, 256, 256
    pred   = torch.rand(B, C, H, W).to(device)
    target = torch.rand(B, C, H, W).to(device)

    print("\n--- Testing MSE-only loss ---")
    loss_mse = get_loss_function('mse_only').to(device)
    total, ld = loss_mse(pred, target)
    print(f"  Total: {total.item():.4f} | Components: {ld}")

    print("\n--- Testing Composite loss ---")
    loss_comp = get_loss_function('composite').to(device)
    total, ld = loss_comp(pred, target)
    print(f"  Total: {total.item():.4f} | Components: {ld}")

    print("\nComposite Loss working correctly!")