# models/proposed_model.py — v7
# Fix: gate_init = 0.0 → sigmoid(0) = 0.5, much stronger corrections from epoch 1
# The -2.0 gate was too conservative — corrections only 12% strength, learning too slow

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CAM(nn.Module):
    def __init__(self, channels, reduction=8, dropout=0.0):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.gap(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale


class MultiScaleParallelConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(in_ch, out_ch, 7, padding=3, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.act   = nn.GELU()
        self.cam   = CAM(out_ch, reduction=8, dropout=config.CAM_DROPOUT)
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x):
        w   = F.softmax(self.fusion_weights, dim=0)
        out = w[0] * self.conv3(x) + w[1] * self.conv5(x) + w[2] * self.conv7(x)
        out = self.cam(self.act(self.bn(out)))
        return out, w.detach()


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class CATKCNet(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        C = config.FEATURE_CHANNELS
        self.use_attention     = use_attention
        self.residual_learning = getattr(config, 'RESIDUAL_LEARNING', True)

        self.enc1 = ConvBlock(config.IN_CHANNELS, C)
        self.enc2 = ConvBlock(C,     C * 2)
        self.enc3 = ConvBlock(C * 2, C * 4)
        self.pool = nn.MaxPool2d(2, 2)

        if use_attention:
            self.bottleneck = MultiScaleParallelConv(C * 4, C * 8)
        else:
            self.bottleneck_conv = ConvBlock(C * 4, C * 8)

        self.up3  = nn.ConvTranspose2d(C * 8, C * 4, 2, stride=2)
        self.dec3 = ConvBlock(C * 8, C * 4)
        self.up2  = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)
        self.dec2 = ConvBlock(C * 4, C * 2)
        self.up1  = nn.ConvTranspose2d(C * 2, C, 2, stride=2)
        self.dec1 = ConvBlock(C * 2, C)

        self.final = nn.Conv2d(C, config.OUT_CHANNELS, 1)

        # v7: gate_init = 0.0 → sigmoid(0) = 0.5
        # Network can make 50% strength corrections from epoch 1
        # Zero-init final conv means correction magnitude starts near 0
        # so output ≈ input + 0.5 * ~0 ≈ input at epoch 0
        # But gradients flow much stronger → faster learning
        gate_init = getattr(config, 'RESIDUAL_GATE_INIT', 0.0)
        self.residual_gate = nn.Parameter(torch.tensor(float(gate_init)))

        # Zero-init final conv — this is what keeps output≈input at init
        # The gate alone isn't enough; final conv weights must start at 0
        nn.init.zeros_(self.final.weight)
        if self.final.bias is not None:
            nn.init.zeros_(self.final.bias)

        self._attention_weights = None

    def get_param_groups(self, base_lr):
        """
        Return param groups with gate at 10x higher LR.
        Use this in trainer instead of model.parameters() for faster gate learning.
        """
        gate_params  = [self.residual_gate]
        other_params = [p for n, p in self.named_parameters() if n != 'residual_gate']
        return [
            {'params': other_params, 'lr': base_lr},
            {'params': gate_params,  'lr': base_lr * 10},
        ]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        if self.use_attention:
            b_out, attn_w = self.bottleneck(self.pool(e3))
            self._attention_weights = attn_w
        else:
            b_out  = self.bottleneck_conv(self.pool(e3))
            attn_w = None

        d3 = self.dec3(torch.cat([self.up3(b_out), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3),    e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2),    e1], dim=1))

        correction = self.final(d1)
        gate_val   = torch.sigmoid(self.residual_gate)

        if self.residual_learning:
            out = torch.clamp(x + gate_val * correction, 0.0, 1.0)
        else:
            out = torch.sigmoid(correction)

        return out, correction, attn_w


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)