# losses/composite_loss.py — v5 (No perceptual loss — SSIM only composite)
#
# Problem: VGG perceptual loss raw value ~1.3-1.4 crashes training every time
# composite kicks in. At 128×128 resolution, perceptual features are too coarse
# and create gradient conflicts with Charbonnier.
#
# Fix: Remove perceptual loss entirely. Use Charbonnier + SSIM + Brightness only.
# SSIM is stable and improves structure. Perceptual can be added back after epoch 50
# if needed, but for LOL at 128px it's not worth the instability.

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=3):
        super().__init__()
        self.window_size = window_size
        self.channel     = channel
        self.register_buffer('window', self._build_window())

    def _build_window(self):
        def gaussian_kernel(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32) - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            return g / g.sum()
        _1d = gaussian_kernel(self.window_size)
        _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)
        return _2d.unsqueeze(0).unsqueeze(0).expand(
            self.channel, 1, self.window_size, self.window_size
        ).contiguous()

    def forward(self, pred, target):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad     = self.window_size // 2
        w       = self.window

        mu_x    = F.conv2d(pred,   w, padding=pad, groups=self.channel)
        mu_y    = F.conv2d(target, w, padding=pad, groups=self.channel)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy   = mu_x * mu_y

        sg_x  = F.conv2d(pred   * pred,   w, padding=pad, groups=self.channel) - mu_x_sq
        sg_y  = F.conv2d(target * target, w, padding=pad, groups=self.channel) - mu_y_sq
        sg_xy = F.conv2d(pred   * target, w, padding=pad, groups=self.channel) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sg_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sg_x + sg_y + C2))
        return 1.0 - ssim_map.mean()


class BrightnessLoss(nn.Module):
    def forward(self, pred, target):
        return torch.abs(pred.mean() - target.mean())


class CompositeLoss(nn.Module):
    """
    Epoch-aware loss — NO perceptual loss (removed due to instability at 128px).

    Warmup  (epochs 1–N): Charbonnier + Brightness
    Full    (epochs N+1+): Charbonnier + SSIM + Brightness

    Formula: L = 0.5*Charb + 0.3*(1-SSIM) + 0.05*|mean(pred)-mean(target)|
    """
    def __init__(self):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim_loss   = SSIMLoss()
        self.bright_loss = BrightnessLoss()

        self.current_epoch  = 1
        self.warmup_epochs  = getattr(config, 'LOSS_WARMUP_EPOCHS', 8)
        self.use_brightness = getattr(config, 'BRIGHTNESS_LOSS', True)
        self.lambda_bright  = getattr(config, 'LAMBDA_BRIGHT', 0.05)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, pred, target):
        charb  = self.charbonnier(pred, target)
        bright = self.bright_loss(pred, target) if self.use_brightness else torch.tensor(0.0, device=pred.device)

        if self.current_epoch <= self.warmup_epochs:
            total = config.LAMBDA_MSE * charb
            if self.use_brightness:
                total = total + self.lambda_bright * bright
            loss_dict = {
                'charbonnier': charb.item(),
                'bright':      bright.item(),
                '[warmup]':    1.0,
            }
        else:
            ssim  = self.ssim_loss(pred, target)
            total = config.LAMBDA_MSE  * charb + \
                    config.LAMBDA_SSIM * ssim
            if self.use_brightness:
                total = total + self.lambda_bright * bright
            loss_dict = {
                'charbonnier': charb.item(),
                'ssim':        ssim.item(),
                'bright':      bright.item(),
            }

        return total, loss_dict


class MSEOnlyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.charbonnier = CharbonnierLoss()

    def forward(self, pred, target):
        loss = self.charbonnier(pred, target)
        return loss, {'charbonnier': loss.item()}


def get_loss_function(loss_type='composite'):
    if loss_type == 'composite':
        return CompositeLoss()
    elif loss_type == 'mse_only':
        return MSEOnlyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")