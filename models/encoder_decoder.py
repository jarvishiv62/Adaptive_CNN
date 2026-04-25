# models/encoder_decoder.py
# Minimal Encoder-Decoder Architecture for Image Enhancement

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """Basic encoder block with conv + relu + pool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x  # Return pooled and skip connection


class DecoderBlock(nn.Module):
    """Basic decoder block with upsample + conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch due to pooling
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MinimalEncoderDecoder(nn.Module):
    """
    Minimal Encoder-Decoder Architecture for Low-Light Enhancement
    
    Architecture:
        Input (128×128) → Encoder (64→32→16) → Latent (16×16) → Decoder (32→64→128) → Output
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_channels)
        self.enc2 = EncoderBlock(base_channels, base_channels*2)
        self.enc3 = EncoderBlock(base_channels*2, base_channels*4)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec3 = DecoderBlock(base_channels*8, base_channels*4)
        self.dec2 = DecoderBlock(base_channels*4, base_channels*2)
        self.dec1 = DecoderBlock(base_channels*2, base_channels)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, x):
        # Encoder
        x1, skip1 = self.enc1(x)  # 64×64
        x2, skip2 = self.enc2(x1)  # 32×32  
        x3, skip3 = self.enc3(x2)  # 16×16
        
        # Bottleneck
        x = self.bottleneck(x3)  # 16×16
        
        # Decoder
        x = self.dec3(x, skip3)  # 32×32
        x = self.dec2(x, skip2)  # 64×64
        x = self.dec1(x, skip1)  # 128×128
        
        # Output
        enhanced = self.output(x)
        
        return enhanced, None  # Return None for residual to match interface


class LightweightEncoderDecoder(nn.Module):
    """
    Ultra-lightweight encoder-decoder for comparison
    Uses depthwise separable convolutions to reduce parameters
    """
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),  # 64×64
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),  # 32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),  # 16×16
            nn.ReLU(inplace=True)
        )
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2),  # 32×32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),  # 64×64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, out_channels, 2, stride=2),  # 128×128
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, None


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test minimal encoder-decoder
    model1 = MinimalEncoderDecoder().to(device)
    model1.count_parameters = lambda: sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"MinimalEncoderDecoder params: {model1.count_parameters():,}")
    
    # Test lightweight version
    model2 = LightweightEncoderDecoder().to(device)
    model2.count_parameters = lambda: sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"LightweightEncoderDecoder params: {model2.count_parameters():,}")
    
    # Test forward pass
    x = torch.rand(2, 3, 128, 128).to(device)
    
    with torch.no_grad():
        out1, _ = model1(x)
        out2, _ = model2(x)
    
    print(f"Input shape: {x.shape}")
    print(f"MinimalEncoderDecoder output: {out1.shape}")
    print(f"LightweightEncoderDecoder output: {out2.shape}")
    print("Encoder-Decoder models working!")
