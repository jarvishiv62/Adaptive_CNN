# Adaptive CNN: Content-Adaptive Trapezoidal Kernel CNN

**Dynamic Channel Attention-Driven Adaptive Kernel CNN for Weak Contrast Image Enhancement**  
B.Tech Final Year Research Project

---

## Overview

We extend Wang & Hu (2020) by replacing their **static kernel assignment** (based on offline MSD computation) with a **dynamic Channel Attention Module (CAM)** that learns optimal multi-scale kernel fusion at runtime.

| Component | Base Paper | Ours (CATKC-Net) |
|-----------|-----------|-----------------|
| Kernel selection | Static (offline MSD) | Dynamic (learned CAM) |
| Kernel processing | One kernel per channel | Parallel 3×3+5×5+7×7 |
| Loss function | MSE only | MSE + SSIM + Perceptual |
| Attention | None | SE-style Channel Attention |

---

## Project Structure

```
Adaptive CNN/
├── data/
│   ├── LOL/                    ← Download dataset here
│   │   ├── train/low/
│   │   ├── train/high/
│   │   ├── test/low/
│   │   └── test/high/
│   └── dataset.py              ← LOL dataset loader
├── models/
│   ├── attention.py            ← Channel Attention Module (CAM)
│   ├── base_model.py           ← Wang & Hu 2020 baseline
│   └── proposed_model.py       ← CATKC-Net (our model)
├── losses/
│   └── composite_loss.py       ← MSE + SSIM + Perceptual loss
├── checkpoints/                ← Saved weights (auto-created)
├── results/                    ← Output images (auto-created)
├── logs/                       ← TensorBoard logs (auto-created)
├── config.py                   ← All hyperparameters
├── trainer.py                  ← Training loop
├── evaluate.py                 ← Evaluation + metrics
├── ablation.py                 ← Run ablation experiments
├── utils.py                    ← Metrics, visualization helpers
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/Adaptive_CNN.git
cd Adaptive_CNN
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download LOL Dataset

Go to https://daooshee.github.io/BMVC2018website/ and download the LOL dataset.
Extract to `data/LOL/` so you have:
```
data/LOL/train/low/   ← ~485 low-light images
data/LOL/train/high/  ← ~485 normal-light images
data/LOL/test/low/    ← 15 test images
data/LOL/test/high/   ← 15 test images
```

### 3. Train Full Model (A4)

```bash
python trainer.py
```

### 4. Run All Ablation Experiments

```bash
python ablation.py --exp all
# Or run one at a time:
python ablation.py --exp A1
python ablation.py --exp A4
```

### 5. Evaluate on Test Set

```bash
# Evaluate specific model
python evaluate.py --model proposed --checkpoint checkpoints/A4_full_model/best_model.pth --experiment A4_full

# Evaluate all ablations and compare
python evaluate.py --ablation
```

### 6. Monitor Training

```bash
tensorboard --logdir=logs/
# Open browser: http://localhost:6006
```

---

## Ablation Study

| ID | Configuration | Loss |
|----|--------------|------|
| A1 | Static kernel (base paper) | MSE |
| A2 | Parallel kernels, equal weights | MSE |
| A3 | Parallel kernels + CAM | MSE |
| A4 | Parallel kernels + CAM (Full) | MSE+SSIM+Perceptual |

---

## Architecture

### Channel Attention Module (CAM)

```
3×3 Conv ─┐
5×5 Conv ─┼─→ [CAM] GAP → FC(64) → ReLU → FC(3) → Softmax → [w3, w5, w7]
7×7 Conv ─┘         ↓
                Weighted Sum → Fused Features
```

### Loss Function

```
L_total = 0.5 × L_mse  +  0.3 × (1 - SSIM)  +  0.2 × L_perceptual
```

---

## Expected Results (LOL-v1 Test Set)

| Method | PSNR (dB) | SSIM | LPIPS |
|--------|-----------|------|-------|
| A1 (Baseline) | ~17.5 | ~0.72 | ~0.30 |
| A2 (+Parallel) | ~17.9 | ~0.74 | ~0.28 |
| A3 (+CAM) | ~18.3 | ~0.76 | ~0.25 |
| A4 (Full - Ours) | ~18.8 | ~0.78 | ~0.22 |

*Actual numbers will vary based on training run. Fill in from your experiments.*

---

## Key Files to Understand

1. `models/attention.py` — The core innovation: CAM + MultiScaleParallelConv
2. `models/proposed_model.py` — Full Adaptive CNN architecture
3. `losses/composite_loss.py` — Combined loss function
4. `config.py` — All hyperparameters (change things here)

---

## Citation

Base paper:
```
@article{wang2020improved,
  title={An Improved Enhancement Algorithm Based on CNN Applicable for Weak Contrast Images},
  author={Wang, Jiao and Hu, Yanzhu},
  journal={IEEE Access},
  year={2020},
  doi={10.1109/ACCESS.2020.2964816}
}
```

---

## Team

B.Tech Final Year Project — R. E. C. BIJNOR
Team of 4 | Duration: 6 months

---

## License

MIT License — free to use for educational and research purposes.