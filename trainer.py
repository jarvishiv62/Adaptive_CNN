# trainer.py — v5 (AMP mixed precision + LR warmup)
#
# New features:
#   - torch.autocast() fp16: ~1.5x faster, ~40% less VRAM
#   - GradScaler: prevents fp16 underflow
#   - LR warmup: smooth 1e-4 → 1e-3 ramp over first 5 epochs
#   - Overfitting monitor: prints train/val PSNR gap each epoch

import os, time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from utils import (
    compute_psnr, compute_ssim,
    AverageMeter, EarlyStopping,
    save_comparison_image, create_dirs
)


class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        train_loader,
        val_loader,
        experiment_name = 'experiment',
        checkpoint_dir  = config.CHECKPOINT_DIR,
        log_dir         = config.LOG_DIR,
        results_dir     = config.RESULTS_DIR,
    ):
        self.model           = model.to(config.DEVICE)
        self.loss_fn         = loss_fn.to(config.DEVICE)
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.experiment_name = experiment_name

        self.checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        self.log_dir        = os.path.join(log_dir,        experiment_name)
        self.results_dir    = os.path.join(results_dir,    experiment_name)
        create_dirs(self.checkpoint_dir, self.log_dir, self.results_dir)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Cosine annealing (best for image restoration)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.NUM_EPOCHS, eta_min=config.LR_MIN
        )

        # Mixed precision scaler
        self.use_amp = getattr(config, 'USE_AMP', True) and config.DEVICE == 'cuda'
        self.scaler  = GradScaler() if self.use_amp else None

        # LR warmup
        self.warmup_epochs = getattr(config, 'LR_WARMUP_EPOCHS', 5)
        self.base_lr       = config.LEARNING_RATE
        self.warmup_start_lr = self.base_lr / 10.0  # Start at 1/10 of target LR

        self.early_stopper = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode='max')
        self.writer        = SummaryWriter(log_dir=self.log_dir)
        self.best_psnr     = 0.0
        self.start_epoch   = 1

        print(f"\n{'='*65}")
        print(f"  Experiment : {experiment_name}")
        print(f"  Device     : {config.DEVICE}")
        print(f"  Resolution : {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
        print(f"  Epochs     : {config.NUM_EPOCHS}  Batch: {config.BATCH_SIZE}  LR: {config.LEARNING_RATE}")
        print(f"  AMP (fp16) : {'✓ enabled' if self.use_amp else '✗ disabled'}")
        print(f"  LR warmup  : {self.warmup_epochs} epochs ({self.warmup_start_lr:.1e} → {self.base_lr:.1e})")
        print(f"  Workers    : {config.NUM_WORKERS}")
        print(f"  Loss       : MSE(λ={config.LAMBDA_MSE}) SSIM(λ={config.LAMBDA_SSIM}) Perc(λ={config.LAMBDA_PERC})")
        print(f"{'='*65}\n")

    def _get_warmup_lr(self, epoch):
        """Linear warmup: epoch 1→warmup_epochs ramps LR from warmup_start_lr to base_lr."""
        if epoch <= self.warmup_epochs:
            return self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (epoch - 1) / max(self.warmup_epochs - 1, 1)
        return None  # Use scheduler after warmup

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _forward(self, low):
        out = self.model(low)
        return (out[0], out[2] if len(out) > 2 else None) if isinstance(out, tuple) else (out, None)

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter  = AverageMeter('Loss')
        psnr_meter  = AverageMeter('PSNR')
        comp_sums   = {}
        bright_sum  = 0.0

        pbar = tqdm(self.train_loader, desc=f"Ep {epoch:3d} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            low  = batch['low'].to(config.DEVICE,  non_blocking=True)
            high = batch['high'].to(config.DEVICE, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

            # ── Mixed precision forward ──
            if self.use_amp:
                with autocast():
                    enhanced, _ = self._forward(low)
                    loss_out    = self.loss_fn(enhanced, high)
                total_loss, loss_dict = loss_out if isinstance(loss_out, tuple) else (loss_out, {'total': loss_out.item()})
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                enhanced, _ = self._forward(low)
                loss_out    = self.loss_fn(enhanced, high)
                total_loss, loss_dict = loss_out if isinstance(loss_out, tuple) else (loss_out, {'total': loss_out.item()})
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_NORM)
                self.optimizer.step()

            with torch.no_grad():
                psnr_val    = compute_psnr(enhanced.detach(), high)
                bright_sum += (enhanced.mean() - low.mean()).item()

            loss_meter.update(total_loss.item(), low.size(0))
            psnr_meter.update(psnr_val, low.size(0))
            for k, v in loss_dict.items():
                comp_sums[k] = comp_sums.get(k, 0) + v

            pbar.set_postfix({'Loss': f"{loss_meter.avg:.4f}", 'PSNR': f"{psnr_meter.avg:.2f}"})

            step = (epoch - 1) * len(self.train_loader) + batch_idx
            if batch_idx % config.LOG_INTERVAL == 0:
                self.writer.add_scalar('Train/Loss', loss_meter.val, step)
                self.writer.add_scalar('Train/PSNR', psnr_meter.val, step)

        n          = len(self.train_loader)
        comp_avgs  = {k: v / n for k, v in comp_sums.items()}
        bright_avg = bright_sum / n
        return loss_meter.avg, psnr_meter.avg, comp_avgs, bright_avg

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        psnr_meter = AverageMeter('Val PSNR')
        ssim_meter = AverageMeter('Val SSIM')

        for batch in tqdm(self.val_loader, desc=f"Ep {epoch:3d} [Val]  ", leave=False):
            low  = batch['low'].to(config.DEVICE,  non_blocking=True)
            high = batch['high'].to(config.DEVICE, non_blocking=True)

            if self.use_amp:
                with autocast():
                    enhanced, _ = self._forward(low)
            else:
                enhanced, _ = self._forward(low)

            psnr_meter.update(compute_psnr(enhanced, high))
            ssim_meter.update(compute_ssim(enhanced, high))

        return psnr_meter.avg, ssim_meter.avg

    def save_checkpoint(self, epoch, psnr, is_best=False):
        state = {
            'epoch':       epoch,
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'best_psnr':   self.best_psnr,
            'psnr':        psnr,
            'image_size':  config.IMAGE_SIZE,
        }
        if self.scaler:
            state['scaler'] = self.scaler.state_dict()
        torch.save(state, os.path.join(self.checkpoint_dir, config.LAST_MODEL_NAME))
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, config.BEST_MODEL_NAME))
            print(f"    ★ Best — PSNR: {psnr:.4f} dB")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=config.DEVICE)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        if self.scaler and 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])
        self.best_psnr   = ckpt['best_psnr']
        self.start_epoch = ckpt['epoch'] + 1
        res = ckpt.get('image_size', '?')
        print(f"Resumed from epoch {ckpt['epoch']} (best PSNR: {self.best_psnr:.4f} dB, trained at {res}×{res})")

    def save_val_samples(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 3: break
                low  = batch['low'].to(config.DEVICE)
                high = batch['high'].to(config.DEVICE)
                enhanced, _ = self._forward(low)
                save_comparison_image(
                    low[0], enhanced[0], high[0],
                    os.path.join(self.results_dir, f"ep{epoch:03d}_{batch['filename'][0]}"),
                    title=f"Ep {epoch} | PSNR: {compute_psnr(enhanced, high):.2f} dB"
                )

    def train(self, resume_path=None):
        if resume_path:
            self.load_checkpoint(resume_path)

        print(f"Training from epoch {self.start_epoch}...\n")

        for epoch in range(self.start_epoch, config.NUM_EPOCHS + 1):
            t0 = time.time()

            # ── LR warmup ──
            warmup_lr = self._get_warmup_lr(epoch)
            if warmup_lr is not None:
                self._set_lr(warmup_lr)
            elif epoch == self.warmup_epochs + 1:
                # Sync scheduler to current epoch after warmup ends
                for _ in range(self.warmup_epochs):
                    self.scheduler.step()

            train_loss, train_psnr, comp_avgs, bright_avg = self.train_epoch(epoch)
            val_psnr, val_ssim = self.validate(epoch)

            # Step scheduler only after warmup
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed    = time.time() - t0
            eta_s      = elapsed * (config.NUM_EPOCHS - epoch)
            eta_h, eta_m = int(eta_s // 3600), int((eta_s % 3600) // 60)

            # Overfitting monitor: gap between train and val PSNR
            overfit_gap = train_psnr - val_psnr
            overfit_str = f"  ⚠ overfit gap={overfit_gap:.1f}" if overfit_gap > 3.0 else ""

            comp_str = " | ".join(f"{k}={v:.4f}" for k, v in comp_avgs.items() if k != 'total')

            print(
                f"[Ep {epoch:3d}/{config.NUM_EPOCHS}] "
                f"Loss:{train_loss:.4f} ({comp_str}) | "
                f"Tr:{train_psnr:.2f} Val:{val_psnr:.2f}dB | "
                f"SSIM:{val_ssim:.4f} | "
                f"BrightΔ:{bright_avg:+.3f} | "
                f"LR:{current_lr:.2e} | "
                f"{elapsed:.0f}s ETA:{eta_h}h{eta_m:02d}m"
                f"{overfit_str}"
            )

            # TensorBoard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_PSNR', train_psnr, epoch)
            self.writer.add_scalar('Epoch/Val_PSNR',   val_psnr,   epoch)
            self.writer.add_scalar('Epoch/Val_SSIM',   val_ssim,   epoch)
            self.writer.add_scalar('Epoch/LR',         current_lr, epoch)
            self.writer.add_scalar('Epoch/Overfit_Gap', overfit_gap, epoch)
            self.writer.add_scalar('Epoch/BrightDelta', bright_avg, epoch)

            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
            self.save_checkpoint(epoch, val_psnr, is_best)

            if epoch % 10 == 0:
                self.save_val_samples(epoch)

            if self.early_stopper(val_psnr):
                print(f"\nEarly stopping at epoch {epoch}. Best: {self.best_psnr:.4f} dB")
                break

        self.writer.close()
        print(f"\nTraining complete! Best Val PSNR: {self.best_psnr:.4f} dB")
        print(f"Tensorboard: tensorboard --logdir={config.LOG_DIR}")
        return self.best_psnr


if __name__ == "__main__":
    from data.dataset import get_dataloaders
    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function

    train_loader, val_loader, _ = get_dataloaders()
    model   = CATKCNet(use_attention=True)
    loss_fn = get_loss_function('composite')

    trainer = Trainer(model, loss_fn, train_loader, val_loader, experiment_name='A4_full_model')
    trainer.train()