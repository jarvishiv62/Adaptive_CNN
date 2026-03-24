# trainer.py — v8
# Changes from v7:
#   ✓ Uses model.get_param_groups() if available → gate gets 10x LR
#   ✓ Logs gate value each epoch so you can watch it open up
#   ✓ set_epoch() call retained for loss warmup scheduling

import os, time
import torch
import torch.amp as amp
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
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

        # Use param groups if model supports it (gate gets 10x LR)
        use_adamw = getattr(config, 'USE_ADAMW', True)
        optimizer_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam

        if hasattr(model, 'get_param_groups'):
            param_groups = model.get_param_groups(config.LEARNING_RATE)
        else:
            param_groups = model.parameters()

        self.optimizer = optimizer_cls(
            param_groups,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        warmup_epochs = getattr(config, 'LR_WARMUP_EPOCHS', 5)
        cosine_epochs = config.NUM_EPOCHS - warmup_epochs

        warmup_sched = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=config.LR_MIN
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs]
        )

        self.use_amp = getattr(config, 'USE_AMP', True) and config.DEVICE == 'cuda'
        self.scaler  = amp.GradScaler('cuda') if self.use_amp else None

        self.early_stopper = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode='max')
        self.writer        = SummaryWriter(log_dir=self.log_dir)
        self.best_psnr     = 0.0
        self.start_epoch   = 1

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'='*70}")
        print(f"  Experiment  : {experiment_name}")
        print(f"  Device      : {config.DEVICE}")
        print(f"  Resolution  : {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
        print(f"  Parameters  : {n_params:,}")
        print(f"  Epochs      : {config.NUM_EPOCHS}  Batch: {config.BATCH_SIZE}")
        print(f"  Optimizer   : {'AdamW' if use_adamw else 'Adam'}  LR: {config.LEARNING_RATE}")
        print(f"  Gate LR     : {config.LEARNING_RATE * 10:.1e} (10x base)")
        print(f"  AMP (fp16)  : {'✓ enabled' if self.use_amp else '✗ disabled'}")
        print(f"  Loss warmup : Charbonnier-only for {getattr(config, 'LOSS_WARMUP_EPOCHS', 5)} epochs")
        print(f"  Early stop  : patience={config.EARLY_STOP_PATIENCE}")
        print(f"{'='*70}\n")

    def _forward(self, low):
        out = self.model(low)
        return (out[0], out[2] if len(out) > 2 else None) if isinstance(out, tuple) else (out, None)

    def _get_gate_val(self):
        if hasattr(self.model, 'residual_gate'):
            return torch.sigmoid(self.model.residual_gate).item()
        return None

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter('Loss')
        psnr_meter = AverageMeter('PSNR')
        comp_sums  = {}
        bright_sum = 0.0

        pbar = tqdm(self.train_loader, desc=f"Ep {epoch:3d} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            low  = batch['low'].to(config.DEVICE,  non_blocking=True)
            high = batch['high'].to(config.DEVICE, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with amp.autocast('cuda'):
                    enhanced, _ = self._forward(low)
                    loss_out    = self.loss_fn(enhanced, high)
                total_loss, loss_dict = loss_out if isinstance(loss_out, tuple) else (loss_out, {})
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                enhanced, _ = self._forward(low)
                loss_out    = self.loss_fn(enhanced, high)
                total_loss, loss_dict = loss_out if isinstance(loss_out, tuple) else (loss_out, {})
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
                with amp.autocast('cuda'):
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
            'sched_state': self.scheduler.state_dict(),
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
        if 'sched_state' in ckpt:
            self.scheduler.load_state_dict(ckpt['sched_state'])
        if self.scaler and 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])
        self.best_psnr   = ckpt['best_psnr']
        self.start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (best PSNR: {self.best_psnr:.4f} dB)")

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

            # Tell loss function current epoch (warmup scheduling)
            if hasattr(self.loss_fn, 'set_epoch'):
                self.loss_fn.set_epoch(epoch)

            train_loss, train_psnr, comp_avgs, bright_avg = self.train_epoch(epoch)
            val_psnr, val_ssim = self.validate(epoch)

            self.scheduler.step()

            current_lr  = self.optimizer.param_groups[0]['lr']
            gate_val    = self._get_gate_val()
            elapsed     = time.time() - t0
            eta_s       = elapsed * (config.NUM_EPOCHS - epoch)
            eta_h, eta_m = int(eta_s // 3600), int((eta_s % 3600) // 60)

            overfit_gap = train_psnr - val_psnr
            overfit_str = f"  ⚠ overfit gap={overfit_gap:.1f}" if overfit_gap > 3.0 else ""
            gate_str    = f" | Gate:{gate_val:.3f}" if gate_val is not None else ""
            comp_str    = " | ".join(f"{k}={v:.4f}" for k, v in comp_avgs.items() if k != 'total')

            print(
                f"[Ep {epoch:3d}/{config.NUM_EPOCHS}] "
                f"Loss:{train_loss:.4f} ({comp_str}) | "
                f"Tr:{train_psnr:.2f} Val:{val_psnr:.2f}dB | "
                f"SSIM:{val_ssim:.4f}{gate_str} | "
                f"BrightΔ:{bright_avg:+.3f} | "
                f"LR:{current_lr:.2e} | "
                f"{elapsed:.0f}s ETA:{eta_h}h{eta_m:02d}m"
                f"{overfit_str}"
            )

            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_PSNR', train_psnr, epoch)
            self.writer.add_scalar('Epoch/Val_PSNR',   val_psnr,   epoch)
            self.writer.add_scalar('Epoch/Val_SSIM',   val_ssim,   epoch)
            self.writer.add_scalar('Epoch/LR',         current_lr, epoch)
            self.writer.add_scalar('Epoch/BrightDelta', bright_avg, epoch)
            if gate_val is not None:
                self.writer.add_scalar('Epoch/Gate', gate_val, epoch)

            for k, v in comp_avgs.items():
                if k != 'total':
                    self.writer.add_scalar(f'Loss/{k}', v, epoch)

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
        return self.best_psnr


if __name__ == "__main__":
    from data.dataset import get_dataloaders
    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function

    train_loader, val_loader, _ = get_dataloaders()
    model   = CATKCNet(use_attention=True)
    loss_fn = get_loss_function('composite')

    trainer = Trainer(model, loss_fn, train_loader, val_loader, experiment_name='A4_full_model')

    print('\nRunning baseline validation on untrained model...')
    base_psnr, base_ssim = trainer.validate(0)
    print(f"Baseline Val PSNR: {base_psnr:.4f} dB | SSIM: {base_ssim:.4f}\n")

    trainer.train()