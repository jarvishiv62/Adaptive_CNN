# CATKC-Net Architecture Diagram

## Overview
CATKC-Net (Content-Adaptive Trapezoidal Kernel CNN) is a deep learning model for low-light image enhancement. The architecture uses adaptive kernel selection with channel attention to improve image quality.

## Architecture Flow

```
Input Image (Low-light)
       ↓ (3×H×W)
┌─────────────────────────────────────────────────────────────┐
│                Multi-Scale Parallel Conv                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │ 3×3 Conv│  │ 5×5 Conv│  │ 7×7 Conv│                      │
│  │ (64 ch) │  │ (64 ch) │  │ (64 ch) │                      │
│  └─────────┘  └─────────┘  └─────────┘                      │
│       ↓             ↓             ↓                         │
│       └─────────────┼─────────────┘                         │
│                     ↓                                       │
│            Channel Attention Module                         │
│         (Learns weights w3, w5, w7)                         │
│                     ↓                                       │
│              Weighted Fusion                                │
│         w3·out3 + w5·out5 + w7·out7                         │
└─────────────────────────────────────────────────────────────┘
       ↓ (64×H×W)
┌─────────────────────────────────────────────────────────────┐
│           Characteristic Activation Layer                   │
│              Conv3×3 + BN + LeakyReLU                       │
└─────────────────────────────────────────────────────────────┘
       ↓ (64×H×W)
┌─────────────────────────────────────────────────────────────┐
│              High-Dimensional Mapping                       │
│            (5 Residual Blocks)                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Residual Block:                                     │    │
│  │ Conv3×3 → BN → LeakyReLU → Conv3×3 → BN → + Skip    │    │
│  │                ↓ LeakyReLU                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                    ×5                                       │
└─────────────────────────────────────────────────────────────┘
       ↓ (64×H×W)
┌─────────────────────────────────────────────────────────────┐
│               Enhancement Output Layer                      │
│  Conv3×3 (64→32) → LeakyReLU → Conv3×3 (32→3) → Tanh        │
│           (Output: Residual map ∈ [-0.5, 0.5])              │
└─────────────────────────────────────────────────────────────┘
       ↓ (3×H×W)
┌─────────────────────────────────────────────────────────────┐
│                    Residual Addition                        │
│            Enhanced = Input + Residual                      │
│                  Clamp to [0, 1]                            │
└─────────────────────────────────────────────────────────────┘
       ↓ (3×H×W)
Output Image (Enhanced)
```

## Key Components

### 1. Multi-Scale Parallel Convolution
- **3×3 Conv**: Captures fine details and edges
- **5×5 Conv**: Medium-scale features and textures  
- **7×7 Conv**: Large-scale structures and context
- Each outputs 64 feature maps with same spatial dimensions

### 2. Channel Attention Module (CAM)
- **Input**: Concatenated features from all three kernels (192 channels)
- **Global Average Pooling**: Spatial squeeze to (B, 192)
- **FC Layers**: 192 → 64 → 3 (with ReLU and Dropout)
- **Softmax**: Normalized weights [w3, w5, w7] that sum to 1
- **Fusion**: Weighted sum of kernel outputs

### 3. High-Dimensional Mapping
- **5 Residual Blocks**: Deep feature learning with skip connections
- Each block: Conv3×3 → BN → LeakyReLU → Conv3×3 → BN → Add → LeakyReLU
- Preserves spatial dimensions while increasing receptive field

### 4. Enhancement Output Layer
- **Conv3×3**: 64 → 32 channels
- **Conv3×3**: 32 → 3 channels (RGB)
- **Tanh**: Maps residual to [-0.5, 0.5] range
- **Initialization**: Final conv weights near zero for stable training

### 5. Residual Formulation
- **Enhanced = Input + Residual**
- Network learns enhancement delta, not full image
- Natural identity initialization: residual=0 → output=input
- Clamp ensures valid pixel range [0, 1]

## Training Details

### Dataset
- **LOL Dataset**: Low-light image enhancement benchmark
- **Training pairs**: Low-light input + Normal-light ground truth
- **Image size**: 128×128 (configurable)
- **Augmentation**: Horizontal flip, rotation (±10°), random crop

### Loss Function
Composite loss with three components:
```
L_total = λ_mse·MSE + λ_ssim·(1-SSIM) + λ_perc·Perceptual
```
- **MSE Loss** (λ=0.7): Pixel-level fidelity
- **SSIM Loss** (λ=0.05): Structural similarity
- **Perceptual Loss** (λ=0.003): VGG16 feature matching

### Training Configuration
- **Batch size**: 8
- **Epochs**: 200
- **Learning rate**: 1e-3 with cosine annealing
- **Optimizer**: Adam (weight decay=1e-5)
- **Mixed precision**: AMP for faster training
- **Early stopping**: Patience=30 epochs

## Ablation Studies

The project includes 4 ablation configurations:

1. **A1_baseline**: Static kernel assignment (Wang & Hu 2020)
2. **A2_parallel_only**: Parallel kernels without CAM
3. **A3_cam_mse**: Parallel + CAM, MSE loss only
4. **A4_full**: Full model with CAM + Composite loss

## Model Parameters
- **CATKCNet (with CAM)**: ~1.2M parameters
- **Baseline Model**: ~1.1M parameters
- **CAM overhead**: Only ~200 extra parameters

## Innovation
- **Content-adaptive kernel selection** via learned attention weights
- **Multi-scale feature extraction** in parallel
- **Residual formulation** for stable training on dark images
- **Composite loss** balancing pixel, structural, and perceptual quality
