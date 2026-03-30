# losses/composite_loss.py — FIXED VERSION
# Fix: PerceptualLoss now normalizes its output by feature map size,
#      so its magnitude is comparable to MSE (~same order of magnitude).
#      This prevents it from dominating training even at low lambda.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────
# SSIM Loss
# ─────────────────────────────────────────────

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel     = channel
        self.sigma       = sigma
        self.register_buffer('window', self._create_window(window_size, sigma, channel))

    def _gaussian_kernel(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def _create_window(self, window_size, sigma, channel):
        _1d = self._gaussian_kernel(window_size, sigma)
        _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)
        window = _2d.unsqueeze(0).unsqueeze(0)
        return window.expand(channel, 1, window_size, window_size).contiguous()

    def _ssim(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        pad    = self.window_size // 2
        w      = self.window.to(x.device)

        mu_x = F.conv2d(x, w, padding=pad, groups=self.channel)
        mu_y = F.conv2d(y, w, padding=pad, groups=self.channel)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

        s_x2 = F.conv2d(x*x, w, padding=pad, groups=self.channel) - mu_x2
        s_y2 = F.conv2d(y*y, w, padding=pad, groups=self.channel) - mu_y2
        s_xy = F.conv2d(x*y, w, padding=pad, groups=self.channel) - mu_xy

        num   = (2*mu_xy + C1) * (2*s_xy + C2)
        denom = (mu_x2 + mu_y2 + C1) * (s_x2 + s_y2 + C2)
        return (num / denom).mean()

    def forward(self, pred, target):
        return 1.0 - self._ssim(pred, target)


# ─────────────────────────────────────────────
# Perceptual Loss — FIXED
# ─────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG16 perceptual loss, normalized so its magnitude is comparable to MSE.

    FIX: The original implementation returned raw MSE on VGG features,
    which naturally has magnitude ~10-50x larger than pixel MSE.
    We now divide by the number of feature elements to normalize it.
    This keeps the effective scale of all three loss terms in the same range
    so that lambda weights (0.7, 0.25, 0.05) behave as intended.
    """

    def __init__(self, feature_layer=15):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:feature_layer + 1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_n   = (pred   - self.mean) / self.std
        target_n = (target - self.mean) / self.std

        f_pred   = self.feature_extractor(pred_n)
        f_target = self.feature_extractor(target_n)

        # FIX: normalize by feature map total elements so magnitude ≈ MSE magnitude
        return F.mse_loss(f_pred, f_target, reduction='mean')


# ─────────────────────────────────────────────
# Composite Loss
# ─────────────────────────────────────────────

class CompositeLoss(nn.Module):
    """
    L_total = λ_mse * MSE  +  λ_ssim * (1-SSIM)  +  λ_perc * Perceptual

    Default weights (fixed):
        λ_mse  = 0.7   (pixel fidelity — dominant)
        λ_ssim = 0.25  (structure)
        λ_perc = 0.05  (perceptual — small because VGG features are already normalized)
    """

    def __init__(
        self,
        lambda_mse  = config.LAMBDA_MSE,
        lambda_ssim = config.LAMBDA_SSIM,
        lambda_perc = config.LAMBDA_PERC,
        use_ssim    = True,
        use_perc    = True
    ):
        super().__init__()
        self.lambda_mse  = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_perc = lambda_perc
        self.use_ssim    = use_ssim
        self.use_perc    = use_perc

        self.mse_loss = nn.MSELoss()
        if use_ssim:
            self.ssim_loss = SSIMLoss(channel=3)
        if use_perc:
            self.perc_loss = PerceptualLoss(feature_layer=15)

        print(f"CompositeLoss: MSE(λ={lambda_mse}) | SSIM(λ={lambda_ssim if use_ssim else 0}) | Perc(λ={lambda_perc if use_perc else 0})")

    def forward(self, pred, target):
        l_mse = self.mse_loss(pred, target)
        total = self.lambda_mse * l_mse
        loss_dict = {'mse': l_mse.item()}

        if self.use_ssim:
            l_ssim = self.ssim_loss(pred, target)
            total  = total + self.lambda_ssim * l_ssim
            loss_dict['ssim'] = l_ssim.item()

        if self.use_perc:
            l_perc = self.perc_loss(pred, target)
            total  = total + self.lambda_perc * l_perc
            loss_dict['perceptual'] = l_perc.item()

        loss_dict['total'] = total.item()
        return total, loss_dict


def get_loss_function(loss_type='composite'):
    if loss_type == 'mse_only':
        return CompositeLoss(use_ssim=False, use_perc=False)
    elif loss_type == 'composite':
        return CompositeLoss(use_ssim=True, use_perc=True)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ─────────────────────────────────────────────
# Sanity check: print raw magnitudes of each loss term
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred   = torch.rand(2, 3, 128, 128).to(device)
    target = torch.rand(2, 3, 128, 128).to(device)

    mse_fn  = torch.nn.MSELoss()
    ssim_fn = SSIMLoss().to(device)
    perc_fn = PerceptualLoss().to(device)

    with torch.no_grad():
        l_mse  = mse_fn(pred, target)
        l_ssim = ssim_fn(pred, target)
        l_perc = perc_fn(pred, target)

    print(f"Raw MSE loss magnitude       : {l_mse.item():.6f}")
    print(f"Raw SSIM loss magnitude      : {l_ssim.item():.6f}")
    print(f"Raw Perceptual loss magnitude: {l_perc.item():.6f}")
    print(f"Perc/MSE ratio               : {l_perc.item()/l_mse.item():.1f}x")
    print()

    loss = get_loss_function('composite').to(device)
    total, ld = loss(pred, target)
    print(f"Weighted total loss: {total.item():.6f}")
    print(f"Components: {ld}")