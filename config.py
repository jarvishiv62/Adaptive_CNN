# config.py — v4 (for redesigned residual model)

import os
import torch

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
IMAGE_SIZE      = 128
VAL_SPLIT       = 0.17
RANDOM_SEED     = 42
USE_HFLIP       = True
USE_ROTATION    = True
ROTATION_DEGREE = 10
USE_CROP        = True

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
BATCH_SIZE          = 8
NUM_EPOCHS          = 200
LEARNING_RATE       = 1e-3      # Higher LR for residual model — converges faster
WEIGHT_DECAY        = 1e-5
GRAD_CLIP_NORM      = 1.0
NUM_WORKERS         = 0         # Must be 0 on Windows

LR_SCHEDULER        = "cosine"
LR_MIN              = 1e-6
LR_PATIENCE         = 10
LR_FACTOR           = 0.5
EARLY_STOP_PATIENCE = 30

# ─────────────────────────────────────────────
# LOSS WEIGHTS (calibrated from diagnosis)
# Raw magnitudes on your GPU: MSE≈0.166, SSIM≈0.960, Perc≈8.682
# Target contributions: MSE≈60%, SSIM≈25%, Perc≈15%
# ─────────────────────────────────────────────
LAMBDA_MSE      = 0.7
LAMBDA_SSIM     = 0.050
LAMBDA_PERC     = 0.003

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
IN_CHANNELS     = 3
OUT_CHANNELS    = 3
KERNEL_SIZES    = [3, 5, 7]
CAM_HIDDEN_DIM  = 64
CAM_DROPOUT     = 0.1

# ─────────────────────────────────────────────
# LOGGING & CHECKPOINTING
# ─────────────────────────────────────────────
LOG_INTERVAL    = 10
SAVE_INTERVAL   = 10
BEST_MODEL_NAME = "best_model.pth"
LAST_MODEL_NAME = "last_model.pth"
EVAL_BATCH_SIZE = 1

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# ABLATION CONFIGS
# ─────────────────────────────────────────────
ABLATION_CONFIGS = {
    "A1_baseline":      {"description": "Static kernel (Wang & Hu 2020)", "model": "base_model",            "loss": "mse_only"},
    "A2_parallel_only": {"description": "Parallel kernels, no CAM",       "model": "parallel_no_attention", "loss": "mse_only"},
    "A3_cam_mse":       {"description": "Parallel + CAM, MSE only",       "model": "proposed",              "loss": "mse_only"},
    "A4_full":          {"description": "Full model: CAM + Composite",    "model": "proposed",              "loss": "composite"},
}