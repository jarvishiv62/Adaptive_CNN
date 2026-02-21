# trainer.py
# Training loop for CATKC-Net and ablation models.
# Handles training, validation, checkpointing, and TensorBoard logging.

import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from utils import (
    compute_psnr, compute_ssim,
    AverageMeter, EarlyStopping,
    save_comparison_image, create_dirs
)


class Trainer:
    """
    General-purpose trainer for all model variants (A1, A2, A3, A4).

    Supports:
        - Configurable model and loss function
        - CosineAnnealingLR or ReduceLROnPlateau scheduler
        - Gradient clipping
        - Early stopping
        - TensorBoard logging
        - Best checkpoint saving
        - Per-epoch validation with PSNR/SSIM
    """

    def __init__(
        self,
        model,
        loss_fn,
        train_loader,
        val_loader,
        experiment_name='experiment',
        checkpoint_dir=config.CHECKPOINT_DIR,
        log_dir=config.LOG_DIR,
        results_dir=config.RESULTS_DIR,
    ):
        self.model           = model.to(config.DEVICE)
        self.loss_fn         = loss_fn.to(config.DEVICE)
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.experiment_name = experiment_name

        # Directories
        self.checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        self.log_dir        = os.path.join(log_dir, experiment_name)
        self.results_dir    = os.path.join(results_dir, experiment_name)
        create_dirs(self.checkpoint_dir, self.log_dir, self.results_dir)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # LR Scheduler
        if config.LR_SCHEDULER == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS,
                eta_min=config.LR_MIN
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=config.LR_PATIENCE,
                factor=config.LR_FACTOR
            )

        # Early stopping (maximize PSNR)
        self.early_stopper = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode='max')

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # State
        self.best_psnr  = 0.0
        self.start_epoch = 1

        print(f"\n{'='*60}")
        print(f"  Experiment : {experiment_name}")
        print(f"  Device     : {config.DEVICE}")
        print(f"  Epochs     : {config.NUM_EPOCHS}")
        print(f"  Batch size : {config.BATCH_SIZE}")
        print(f"  LR         : {config.LEARNING_RATE}")
        print(f"{'='*60}\n")

    def train_epoch(self, epoch):
        """Run one training epoch. Returns average loss and PSNR."""
        self.model.train()

        loss_meter = AverageMeter('Loss')
        psnr_meter = AverageMeter('PSNR')

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            low    = batch['low'].to(config.DEVICE)     # (B, 3, H, W)
            high   = batch['high'].to(config.DEVICE)    # (B, 3, H, W)

            # Forward pass
            self.optimizer.zero_grad()

            output = self.model(low)
            # Handle models returning (enhanced, noise_map) or (enhanced, noise_map, weights)
            if isinstance(output, tuple):
                enhanced = output[0]
            else:
                enhanced = output

            # Loss
            loss_output = self.loss_fn(enhanced, high)
            if isinstance(loss_output, tuple):
                total_loss, loss_dict = loss_output
            else:
                total_loss = loss_output
                loss_dict  = {'total': total_loss.item()}

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_NORM)
            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                psnr_val = compute_psnr(enhanced.detach(), high)

            loss_meter.update(total_loss.item(), low.size(0))
            psnr_meter.update(psnr_val, low.size(0))

            pbar.set_postfix({
                'Loss': f"{loss_meter.avg:.4f}",
                'PSNR': f"{psnr_meter.avg:.2f}dB"
            })

            # TensorBoard batch logging
            step = (epoch - 1) * len(self.train_loader) + batch_idx
            if batch_idx % config.LOG_INTERVAL == 0:
                self.writer.add_scalar('Train/Loss', loss_meter.val, step)
                self.writer.add_scalar('Train/PSNR', psnr_meter.val, step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Train/Loss_{k}', v, step)

        return loss_meter.avg, psnr_meter.avg

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation. Returns average PSNR and SSIM."""
        self.model.eval()

        psnr_meter = AverageMeter('Val PSNR')
        ssim_meter = AverageMeter('Val SSIM')

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Val]", leave=False)

        for batch in pbar:
            low  = batch['low'].to(config.DEVICE)
            high = batch['high'].to(config.DEVICE)

            output = self.model(low)
            if isinstance(output, tuple):
                enhanced = output[0]
            else:
                enhanced = output

            psnr_val = compute_psnr(enhanced, high)
            ssim_val = compute_ssim(enhanced, high)

            psnr_meter.update(psnr_val)
            ssim_meter.update(ssim_val)

            pbar.set_postfix({
                'PSNR': f"{psnr_meter.avg:.2f}dB",
                'SSIM': f"{ssim_meter.avg:.4f}"
            })

        return psnr_meter.avg, ssim_meter.avg

    def save_checkpoint(self, epoch, psnr, is_best=False):
        """Save model checkpoint."""
        state = {
            'epoch'     : epoch,
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'best_psnr' : self.best_psnr,
            'psnr'      : psnr,
        }

        last_path = os.path.join(self.checkpoint_dir, config.LAST_MODEL_NAME)
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, config.BEST_MODEL_NAME)
            torch.save(state, best_path)
            print(f"  ★ New best model saved! PSNR: {psnr:.4f} dB")

    def load_checkpoint(self, path):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(path, map_location=config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optim_state'])
        self.best_psnr  = checkpoint['best_psnr']
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']} (best PSNR: {self.best_psnr:.4f})")

    def save_val_samples(self, epoch):
        """Save a few validation image comparisons."""
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 3:  # Save 3 samples
                    break
                low  = batch['low'].to(config.DEVICE)
                high = batch['high'].to(config.DEVICE)
                fname = batch['filename'][0]

                output = self.model(low)
                enhanced = output[0] if isinstance(output, tuple) else output

                save_path = os.path.join(self.results_dir, f"epoch{epoch:03d}_{fname}")
                save_comparison_image(
                    low[0], enhanced[0], high[0],
                    save_path,
                    title=f"Epoch {epoch} — {fname}"
                )

    def train(self, resume_path=None):
        """
        Main training loop.

        Args:
            resume_path: Optional path to checkpoint to resume from
        """
        if resume_path:
            self.load_checkpoint(resume_path)

        print(f"Starting training from epoch {self.start_epoch}...\n")

        for epoch in range(self.start_epoch, config.NUM_EPOCHS + 1):
            t0 = time.time()

            # Train
            train_loss, train_psnr = self.train_epoch(epoch)

            # Validate
            val_psnr, val_ssim = self.validate(epoch)

            # LR scheduler step
            if config.LR_SCHEDULER == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_psnr)

            # Current LR
            current_lr = self.optimizer.param_groups[0]['lr']

            # TensorBoard epoch logging
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_PSNR', train_psnr, epoch)
            self.writer.add_scalar('Epoch/Val_PSNR',   val_psnr,   epoch)
            self.writer.add_scalar('Epoch/Val_SSIM',   val_ssim,   epoch)
            self.writer.add_scalar('Epoch/LR',         current_lr, epoch)

            # Print epoch summary
            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch:3d}/{config.NUM_EPOCHS}] "
                f"Loss: {train_loss:.4f} | "
                f"Train PSNR: {train_psnr:.2f} dB | "
                f"Val PSNR: {val_psnr:.2f} dB | "
                f"Val SSIM: {val_ssim:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
            self.save_checkpoint(epoch, val_psnr, is_best)

            # Save sample images every 10 epochs
            if epoch % 10 == 0:
                self.save_val_samples(epoch)

            # Early stopping
            if self.early_stopper(val_psnr):
                print(f"\nEarly stopping triggered at epoch {epoch}!")
                print(f"Best Val PSNR: {self.best_psnr:.4f} dB")
                break

        self.writer.close()
        print(f"\nTraining complete! Best Val PSNR: {self.best_psnr:.4f} dB")
        print(f"Best model saved at: {os.path.join(self.checkpoint_dir, config.BEST_MODEL_NAME)}")
        print(f"TensorBoard logs at: {self.log_dir}")
        print(f"Run: tensorboard --logdir={config.LOG_DIR}")

        return self.best_psnr


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Training CATKC-Net (Full Model - A4)...")
    print("Make sure data/LOL/ exists with train and test folders.\n")

    from data.dataset import get_dataloaders
    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Build model and loss
    model   = CATKCNet(use_attention=True)
    loss_fn = get_loss_function('composite')

    # Train
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name='A4_full_model'
    )
    trainer.train()