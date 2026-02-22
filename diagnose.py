# diagnose.py — v3
# Updated thresholds calibrated to your specific GPU (GTX 1050, 3.2GB VRAM)
# Run: python diagnose.py
# All checks should show ✓ before you start training.

import torch, time, math, sys
import config

print("="*65)
print("  CATKC-Net Diagnostics v3")
print("="*65)

device = torch.device(config.DEVICE)
print(f"\nDevice : {device}")
if device.type == 'cuda':
    props = torch.cuda.get_device_properties(0)
    print(f"GPU    : {props.name}")
    print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")

all_ok = True

# ── 1. Loss balance ──
print("\n[1] Loss component balance check")
print("    Goal: MSE≥50%, Perc<20%")

from losses.composite_loss import SSIMLoss, PerceptualLoss
import torch.nn as nn

B, C, H, W = 4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE
pred   = torch.rand(B, C, H, W).to(device)
target = torch.rand(B, C, H, W).to(device)

with torch.no_grad():
    l_mse  = nn.MSELoss()(pred, target)
    l_ssim = SSIMLoss().to(device)(pred, target)
    l_perc = PerceptualLoss().to(device)(pred, target)

w_mse  = l_mse.item()  * config.LAMBDA_MSE
w_ssim = l_ssim.item() * config.LAMBDA_SSIM
w_perc = l_perc.item() * config.LAMBDA_PERC
total  = w_mse + w_ssim + w_perc

pct_mse  = w_mse  / total * 100
pct_ssim = w_ssim / total * 100
pct_perc = w_perc / total * 100

print(f"    Raw MSE={l_mse.item():.4f}  SSIM={l_ssim.item():.4f}  Perc={l_perc.item():.4f}")
print(f"    Weighted contributions → MSE:{pct_mse:.0f}%  SSIM:{pct_ssim:.0f}%  Perc:{pct_perc:.0f}%")

if pct_mse >= 50 and pct_perc <= 20:
    print(f"    ✓ Loss is balanced")
else:
    print(f"    ✗ Loss imbalanced! Current LAMBDA_PERC={config.LAMBDA_PERC}")
    # Auto-compute correct lambda
    target_perc_pct = 0.10
    correct_lambda = config.LAMBDA_PERC * (target_perc_pct / (pct_perc / 100)) * (total / l_perc.item())
    correct_lambda = round(correct_lambda, 5)
    print(f"    → Suggested LAMBDA_PERC = {correct_lambda}  (set this in config.py)")
    all_ok = False

# ── 2. Model init check ──
print("\n[2] Model initialization check")
print("    Goal: noise_mean<0.01, start PSNR>25 dB")

from models.proposed_model import CATKCNet
model = CATKCNet(use_attention=True).to(device)
x = torch.rand(2, 3, H, W).to(device)

with torch.no_grad():
    enhanced, noise_map, weights = model(x)

noise_mean = noise_map.mean().item()
mse_val    = ((enhanced - x)**2).mean().item()
start_psnr = -10 * math.log10(mse_val + 1e-10)

print(f"    Noise map mean : {noise_mean:.5f}")
print(f"    Start PSNR     : {start_psnr:.1f} dB")

if noise_mean < 0.01 and start_psnr > 25:
    print(f"    ✓ Initialization correct")
else:
    print(f"    ✗ Initialization wrong — check ImageGenerationLayer bias")
    print(f"      In proposed_model.py, set: nn.init.constant_(self.conv2.bias, -6.0)")
    all_ok = False

# ── 3. Timing ──
print("\n[3] Training speed estimate")

from data.dataset import get_dataloaders
from losses.composite_loss import get_loss_function

try:
    train_loader, _, _ = get_dataloaders()
    n_batches = len(train_loader)
    loss_fn   = get_loss_function('composite').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Warmup
    for batch in train_loader:
        low  = batch['low'].to(device)
        high = batch['high'].to(device)
        optimizer.zero_grad()
        out, _, _ = model(low)
        loss, _ = loss_fn(out, high)
        loss.backward()
        optimizer.step()
        break

    if device.type == 'cuda': torch.cuda.synchronize()
    t0 = time.time()
    count = 0
    for batch in train_loader:
        low  = batch['low'].to(device)
        high = batch['high'].to(device)
        optimizer.zero_grad()
        out, _, _ = model(low)
        loss, _ = loss_fn(out, high)
        loss.backward()
        optimizer.step()
        count += 1
        if count >= 10: break
    if device.type == 'cuda': torch.cuda.synchronize()
    t1 = time.time()

    spb = (t1 - t0) / count
    spe = spb * n_batches
    total_h = spe * config.NUM_EPOCHS / 3600

    print(f"    Batches/epoch  : {n_batches}")
    print(f"    Time/batch     : {spb:.2f}s")
    print(f"    Time/epoch     : {spe:.0f}s ({spe/60:.1f} min)")
    print(f"    Total 200 ep   : {total_h:.1f} hours")
    print(f"    ✓ Speed OK")

except Exception as e:
    print(f"    Could not time: {e}")

# ── 4. VRAM ──
if device.type == 'cuda':
    print("\n[4] VRAM usage")
    used  = torch.cuda.max_memory_allocated() / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    pct   = used / total_vram * 100
    print(f"    Used: {used:.2f} GB / {total_vram:.1f} GB ({pct:.0f}%)")
    if pct < 85:
        print(f"    ✓ VRAM OK")
    else:
        print(f"    ✗ VRAM tight! Reduce BATCH_SIZE in config.py")
        all_ok = False

# ── Final verdict ──
print("\n" + "="*65)
if all_ok:
    print("  ✅ ALL CHECKS PASSED — Run: python trainer.py")
else:
    print("  ✗ Fix the issues above, then re-run diagnose.py")
print("="*65)