# config.py
# Central configuration file for CATKC-Net
# All hyperparameters, paths, and settings are defined here.

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_ROOT       = "data/LOL"
TRAIN_LOW_DIR   = os.path.join(DATA_ROOT, "train/low")
TRAIN_HIGH_DIR  = os.path.join(DATA_ROOT, "train/high")
TEST_LOW_DIR    = os.path.join(DATA_ROOT, "test/low")
TEST_HIGH_DIR   = os.path.join(DATA_ROOT, "test/high")

CHECKPOINT_DIR  = "checkpoints"
RESULTS_DIR     = "results"
LOG_DIR         = "logs"

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
IMAGE_SIZE      = 256          # Resize images to 256×256
VAL_SPLIT       = 0.17         # ~80 images from 485 train images for validation
RANDOM_SEED     = 42

# Augmentation
USE_HFLIP       = True
USE_ROTATION    = True
ROTATION_DEGREE = 10
USE_CROP        = True

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
BATCH_SIZE      = 8
NUM_EPOCHS      = 200
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
GRAD_CLIP_NORM  = 1.0

# Learning rate scheduler
LR_SCHEDULER    = "cosine"     # Options: "cosine", "plateau"
LR_MIN          = 1e-6         # For CosineAnnealingLR
LR_PATIENCE     = 10           # For ReduceLROnPlateau
LR_FACTOR       = 0.5          # For ReduceLROnPlateau

# Early stopping
EARLY_STOP_PATIENCE = 30       # Stop if val PSNR doesn't improve for N epochs

# ─────────────────────────────────────────────
# LOSS FUNCTION WEIGHTS
# ─────────────────────────────────────────────
LAMBDA_MSE      = 0.5
LAMBDA_SSIM     = 0.3
LAMBDA_PERC     = 0.2

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
IN_CHANNELS     = 3            # RGB input
OUT_CHANNELS    = 3            # RGB output
KERNEL_SIZES    = [3, 5, 7]    # Parallel kernel sizes
CAM_HIDDEN_DIM  = 64           # FC hidden dim in Channel Attention Module
CAM_DROPOUT     = 0.1          # Dropout in CAM

# ─────────────────────────────────────────────
# LOGGING & CHECKPOINTING
# ─────────────────────────────────────────────
LOG_INTERVAL    = 10           # Log every N batches
SAVE_INTERVAL   = 10           # Save checkpoint every N epochs
BEST_MODEL_NAME = "best_model.pth"
LAST_MODEL_NAME = "last_model.pth"

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
EVAL_BATCH_SIZE = 1            # Evaluate one image at a time for accurate metrics

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# ABLATION EXPERIMENT CONFIGS
# ─────────────────────────────────────────────
ABLATION_CONFIGS = {
    "A1_baseline": {
        "description": "Static kernel assignment (Wang & Hu 2020)",
        "model": "base_model",
        "loss": "mse_only",
    },
    "A2_parallel_only": {
        "description": "Parallel kernels with equal weights, no attention",
        "model": "parallel_no_attention",
        "loss": "mse_only",
    },
    "A3_cam_mse": {
        "description": "Parallel kernels + CAM, MSE loss only",
        "model": "proposed",
        "loss": "mse_only",
    },
    "A4_full": {
        "description": "Full model: Parallel + CAM + Composite Loss",
        "model": "proposed",
        "loss": "composite",
    },
}