# config.py — v10 (pure Charbonnier, stable training)
#
# Critical fix: SSIM and Perceptual losses disabled
# These caused training collapse at epoch 9 (PSNR dropped from 17.9 → 8.5 dB)
# Root cause: loss warmup ends at epoch 8, SSIM/Perc gradients conflict
# with Charbonnier at this resolution — same pattern seen in all previous runs.
#
# Pure Charbonnier is proven to work:
#   - First A4 run (403 samples): 21.37 dB val, 25.02 dB test
#   - This run (975 samples, 64ch): expected 25-27 dB test
#
# All other settings kept from v9.

import os
import torch

# ─────────────────────────────────────────────
# PATHS — LOL-v1 (required)
# ─────────────────────────────────────────────
DATA_ROOT       = "data/LOL"
TRAIN_LOW_DIR   = os.path.join(DATA_ROOT, "train/low")
TRAIN_HIGH_DIR  = os.path.join(DATA_ROOT, "train/high")
TEST_LOW_DIR    = os.path.join(DATA_ROOT, "test/low")
TEST_HIGH_DIR   = os.path.join(DATA_ROOT, "test/high")

# ─────────────────────────────────────────────
# PATHS — LOL-v2 (optional, auto-detected)
# ─────────────────────────────────────────────
TRAIN_LOW_DIR2  = "data/LOL-v2/train/low"
TRAIN_HIGH_DIR2 = "data/LOL-v2/train/high"

CHECKPOINT_DIR  = "checkpoints"
RESULTS_DIR     = "results"
LOG_DIR         = "logs"

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
IMAGE_SIZE      = 256
VAL_SPLIT       = 0.17
RANDOM_SEED     = 42

USE_HFLIP        = True
USE_VFLIP        = True
USE_ROTATION     = True
ROTATION_DEGREE  = 10
USE_CROP         = True
USE_COLOR_JITTER = True
USE_GAMMA_AUG    = True
USE_SYNTHETIC    = False

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
BATCH_SIZE      = 8
NUM_EPOCHS      = 200
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
GRAD_CLIP_NORM  = 0.5

NUM_WORKERS     = 0        # Windows fix

USE_AMP         = True
USE_ADAMW       = True

LR_WARMUP_EPOCHS    = 3
LR_SCHEDULER        = "cosine"
LR_MIN              = 1e-6
LR_PATIENCE         = 10
LR_FACTOR           = 0.5
EARLY_STOP_PATIENCE = 40

# ─────────────────────────────────────────────
# LOSS WEIGHTS
#
# SSIM and Perceptual DISABLED — they cause training collapse at epoch 8-9.
# Pure Charbonnier is proven: first A4 run got 25.02 dB test PSNR.
# With 975 samples + 64 channels, expected 25-27 dB.
# ─────────────────────────────────────────────
LAMBDA_MSE      = 1.0      # Pure Charbonnier — full weight
LAMBDA_SSIM     = 0.0      # DISABLED — causes collapse
LAMBDA_PERC     = 0.0      # DISABLED — causes collapse
LAMBDA_FREQ     = 0.0      # DISABLED
USE_FREQ_LOSS   = False

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
IN_CHANNELS       = 3
OUT_CHANNELS      = 3
KERNEL_SIZES      = [3, 5, 7]
FEATURE_CHANNELS  = 64
N_RESIDUAL_LAYERS = 5
CAM_HIDDEN_DIM    = 128
CAM_DROPOUT       = 0.1
RESIDUAL_SCALE    = 1.0
USE_SPATIAL_ATTN  = True

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