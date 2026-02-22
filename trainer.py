# trainer.py — v4
# Updated for residual-output model (returns enhanced, residual, weights).
# Also adds a quick sanity print at epoch 1 showing mean brightness change
# so you can confirm the model is actually brightening images.

import os
import time
import torch
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

        if config.LR_SCHEDULER == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.NUM_EPOCHS, eta_min=config.LR_MIN
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=config.LR_PATIENCE, factor=config.LR_FACTOR
            )

        self.early_stopper = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode='max')
        self.writer        = SummaryWriter(log_dir=self.log_dir)
        self.best_psnr     = 0.0
        self.start_epoch   = 1

        print(f"\n{'='*65}")
        print(f"  Experiment : {experiment_name}")
        print(f"  Device     : {config.DEVICE}")
        print(f"  Image size : {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
        print(f"  Epochs     : {config.NUM_EPOCHS}  Batch: {config.BATCH_SIZE}  LR: {config.LEARNING_RATE}")
        print(f"  Loss       : MSE(λ={config.LAMBDA_MSE}) SSIM(λ={config.LAMBDA_SSIM}) Perc(λ={config.LAMBDA_PERC})")
        print(f"{'='*65}\n")

    def _forward(self, low):
        """Run model forward, handle both 2-tuple and 3-tuple outputs."""
        out = self.model(low)
        if isinstance(out, tuple):
            enhanced = out[0]
            weights  = out[2] if len(out) > 2 else None
        else:
            enhanced = out
            weights  = None
        return enhanced, weights

    def train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter('Loss')
        psnr_meter = AverageMeter('PSNR')
        comp_sums  = {}
        bright_delta = 0.0  # Track mean brightness increase

        pbar = tqdm(self.train_loader, desc=f"Ep {epoch:3d} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            low  = batch['low'].to(config.DEVICE)
            high = batch['high'].to(config.DEVICE)

            self.optimizer.zero_grad()
            enhanced, _ = self._forward(low)

            loss_out = self.loss_fn(enhanced, high)
            if isinstance(loss_out, tuple):
                total_loss, loss_dict = loss_out
            else:
                total_loss = loss_out
                loss_dict  = {'total': total_loss.item()}

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_NORM)
            self.optimizer.step()

            with torch.no_grad():
                psnr_val = compute_psnr(enhanced.detach(), high)
                bright_delta += (enhanced.mean() - low.mean()).item()

            loss_meter.update(total_loss.item(), low.size(0))
            psnr_meter.update(psnr_val, low.size(0))

            for k, v in loss_dict.items():
                comp_sums[k] = comp_sums.get(k, 0) + v

            pbar.set_postfix({'Loss': f"{loss_meter.avg:.4f}", 'PSNR': f"{psnr_meter.avg:.2f}"})

            step = (epoch - 1) * len(self.train_loader) + batch_idx
            if batch_idx % config.LOG_INTERVAL == 0:
                self.writer.add_scalar('Train/Loss', loss_meter.val, step)
                self.writer.add_scalar('Train/PSNR', psnr_meter.val, step)

        n = len(self.train_loader)
        comp_avgs   = {k: v / n for k, v in comp_sums.items()}
        bright_avg  = bright_delta / n

        return loss_meter.avg, psnr_meter.avg, comp_avgs, bright_avg

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        psnr_meter = AverageMeter('Val PSNR')
        ssim_meter = AverageMeter('Val SSIM')

        for batch in tqdm(self.val_loader, desc=f"Ep {epoch:3d} [Val]  ", leave=False):
            low  = batch['low'].to(config.DEVICE)
            high = batch['high'].to(config.DEVICE)
            enhanced, _ = self._forward(low)
            psnr_meter.update(compute_psnr(enhanced, high))
            ssim_meter.update(compute_ssim(enhanced, high))

        return psnr_meter.avg, ssim_meter.avg

    def save_checkpoint(self, epoch, psnr, is_best=False):
        state = {
            'epoch':        epoch,
            'model_state':  self.model.state_dict(),
            'optim_state':  self.optimizer.state_dict(),
            'best_psnr':    self.best_psnr,
            'psnr':         psnr,
        }
        torch.save(state, os.path.join(self.checkpoint_dir, config.LAST_MODEL_NAME))
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, config.BEST_MODEL_NAME))
            print(f"    ★ Best — PSNR: {psnr:.4f} dB")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=config.DEVICE)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.best_psnr   = ckpt['best_psnr']
        self.start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (best PSNR: {self.best_psnr:.4f})")

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
                    title=f"Ep {epoch}"
                )

    def train(self, resume_path=None):
        if resume_path:
            self.load_checkpoint(resume_path)

        print(f"Training from epoch {self.start_epoch}...\n")
        print("WATCH: 'BrightΔ' should become positive and grow — that means the model is learning to brighten images.\n")

        for epoch in range(self.start_epoch, config.NUM_EPOCHS + 1):
            t0 = time.time()

            train_loss, train_psnr, comp_avgs, bright_avg = self.train_epoch(epoch)
            val_psnr, val_ssim = self.validate(epoch)

            if config.LR_SCHEDULER == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_psnr)

            lr      = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0
            eta_h   = int(elapsed * (config.NUM_EPOCHS - epoch) // 3600)
            eta_m   = int((elapsed * (config.NUM_EPOCHS - epoch) % 3600) // 60)

            comp_str = " | ".join(f"{k}={v:.4f}" for k, v in comp_avgs.items() if k != 'total')

            print(
                f"[Ep {epoch:3d}/{config.NUM_EPOCHS}] "
                f"Loss:{train_loss:.4f} ({comp_str}) | "
                f"TrPSNR:{train_psnr:.2f} | "
                f"ValPSNR:{val_psnr:.2f} | "
                f"SSIM:{val_ssim:.4f} | "
                f"BrightΔ:{bright_avg:+.4f} | "
                f"LR:{lr:.2e} | "
                f"{elapsed:.0f}s ETA:{eta_h}h{eta_m:02d}m"
            )

            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_PSNR', train_psnr, epoch)
            self.writer.add_scalar('Epoch/Val_PSNR',   val_psnr,   epoch)
            self.writer.add_scalar('Epoch/Val_SSIM',   val_ssim,   epoch)
            self.writer.add_scalar('Epoch/BrightDelta', bright_avg, epoch)
            self.writer.add_scalar('Epoch/LR',          lr,         epoch)

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
        print(f"\nDone! Best Val PSNR: {self.best_psnr:.4f} dB")
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