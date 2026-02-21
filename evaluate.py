# evaluate.py
# Test set evaluation for CATKC-Net.
# Computes PSNR, SSIM, LPIPS on all test images and produces a results table.

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import lpips
import json

import config
from utils import (
    compute_psnr, compute_ssim,
    save_comparison_image, create_dirs,
    visualize_attention_weights, plot_metrics_bar
)


class Evaluator:
    """
    Evaluates a trained model on the test set.

    Reports:
        - PSNR (per image + mean ± std)
        - SSIM (per image + mean ± std)
        - LPIPS (per image + mean)
        - Inference time (ms/image)
        - Attention weights (if applicable)
    """

    def __init__(self, model, test_loader, experiment_name, results_dir=config.RESULTS_DIR):
        self.model           = model.to(config.DEVICE)
        self.test_loader     = test_loader
        self.experiment_name = experiment_name
        self.results_dir     = os.path.join(results_dir, experiment_name, 'test')
        create_dirs(self.results_dir)

        # LPIPS metric (Alex network — fastest and correlates well with human perception)
        print("Loading LPIPS model...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(config.DEVICE)

        self.results = []

    def load_checkpoint(self, checkpoint_path):
        """Load trained model weights."""
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  (Trained for {checkpoint['epoch']} epochs, best Val PSNR: {checkpoint['best_psnr']:.4f} dB)")

    @torch.no_grad()
    def evaluate(self, save_images=True, visualize_attention=True):
        """
        Run full evaluation on test set.

        Args:
            save_images         : Save side-by-side comparison images
            visualize_attention : Save attention weight plots (if model supports it)

        Returns:
            summary: dict with mean/std of each metric
        """
        self.model.eval()
        self.results = []
        attention_weights_dict = {}

        import time

        pbar = tqdm(self.test_loader, desc=f"Evaluating [{self.experiment_name}]")

        for batch in pbar:
            low    = batch['low'].to(config.DEVICE)
            high   = batch['high'].to(config.DEVICE)
            fname  = batch['filename'][0]

            # Inference timing
            if config.DEVICE == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            output = self.model(low)

            if config.DEVICE == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            inference_ms = (t1 - t0) * 1000

            # Parse output
            if isinstance(output, tuple):
                enhanced = output[0]
                weights  = output[2] if len(output) > 2 else None
            else:
                enhanced = output
                weights  = None

            # Metrics
            psnr_val  = compute_psnr(enhanced, high)
            ssim_val  = compute_ssim(enhanced, high)

            # LPIPS expects images in [-1, 1]
            lpips_pred   = enhanced * 2.0 - 1.0
            lpips_target = high     * 2.0 - 1.0
            lpips_val = self.lpips_fn(lpips_pred, lpips_target).item()

            self.results.append({
                'filename'    : fname,
                'PSNR'        : psnr_val,
                'SSIM'        : ssim_val,
                'LPIPS'       : lpips_val,
                'Inference_ms': inference_ms,
            })

            pbar.set_postfix({'PSNR': f"{psnr_val:.2f}", 'SSIM': f"{ssim_val:.4f}"})

            # Collect attention weights
            if weights is not None and visualize_attention:
                attention_weights_dict[fname] = weights[0].cpu().numpy()

            # Save comparison images
            if save_images:
                save_path = os.path.join(self.results_dir, f"result_{fname}")
                save_comparison_image(
                    low[0], enhanced[0], high[0],
                    save_path,
                    title=f"{self.experiment_name} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}"
                )

        # Compute summary statistics
        summary = self._compute_summary()
        self._print_summary(summary)
        self._save_results_csv(summary)

        # Visualize attention weights
        if attention_weights_dict and visualize_attention:
            weights_save_path = os.path.join(self.results_dir, 'attention_weights.png')
            visualize_attention_weights(attention_weights_dict, weights_save_path)

        return summary

    def _compute_summary(self):
        """Compute mean, std of all metrics."""
        metrics = ['PSNR', 'SSIM', 'LPIPS', 'Inference_ms']
        summary = {}
        for m in metrics:
            vals = [r[m] for r in self.results]
            summary[m] = {
                'mean'  : np.mean(vals),
                'std'   : np.std(vals),
                'min'   : np.min(vals),
                'max'   : np.max(vals),
                'values': vals
            }
        summary['experiment'] = self.experiment_name
        summary['n_images']   = len(self.results)
        return summary

    def _print_summary(self, summary):
        """Print formatted results table."""
        print(f"\n{'='*65}")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Test images: {summary['n_images']}")
        print(f"{'='*65}")
        print(f"  {'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*60}")
        for m in ['PSNR', 'SSIM', 'LPIPS', 'Inference_ms']:
            s = summary[m]
            print(f"  {m:<20} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")
        print(f"{'='*65}\n")

    def _save_results_csv(self, summary):
        """Save per-image results and summary to CSV/JSON."""
        # Per-image results
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.results_dir, 'per_image_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Per-image results saved to: {csv_path}")

        # Summary JSON
        summary_to_save = {
            k: {kk: vv for kk, vv in v.items() if kk != 'values'}
            for k, v in summary.items()
            if isinstance(v, dict)
        }
        summary_to_save['experiment'] = summary['experiment']
        summary_to_save['n_images']   = summary['n_images']

        json_path = os.path.join(self.results_dir, 'summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary_to_save, f, indent=2)
        print(f"Summary saved to: {json_path}")


def evaluate_all_ablations(test_loader):
    """
    Evaluate all ablation models (A1, A2, A3, A4) and produce comparison table.
    Assumes best checkpoints exist in checkpoints/{experiment_name}/best_model.pth
    """
    from models.base_model import BaselineModel
    from models.proposed_model import CATKCNet
    from losses.composite_loss import get_loss_function

    ablation_configs = [
        {'name': 'A1_baseline',     'model': BaselineModel(),              'checkpoint': 'checkpoints/A1_baseline/best_model.pth'},
        {'name': 'A2_parallel_only','model': CATKCNet(use_attention=False), 'checkpoint': 'checkpoints/A2_parallel_only/best_model.pth'},
        {'name': 'A3_cam_mse',      'model': CATKCNet(use_attention=True),  'checkpoint': 'checkpoints/A3_cam_mse/best_model.pth'},
        {'name': 'A4_full',         'model': CATKCNet(use_attention=True),  'checkpoint': 'checkpoints/A4_full_model/best_model.pth'},
    ]

    all_summaries = {}

    for cfg in ablation_configs:
        if not os.path.exists(cfg['checkpoint']):
            print(f"Checkpoint not found: {cfg['checkpoint']} — skipping {cfg['name']}")
            continue

        evaluator = Evaluator(cfg['model'], test_loader, cfg['name'])
        evaluator.load_checkpoint(cfg['checkpoint'])
        summary = evaluator.evaluate(save_images=True, visualize_attention=True)
        all_summaries[cfg['name']] = summary

    # Print combined comparison table
    if all_summaries:
        print("\n" + "="*70)
        print("  ABLATION STUDY — FINAL COMPARISON")
        print("="*70)
        print(f"  {'Method':<25} {'PSNR (dB)':>12} {'SSIM':>10} {'LPIPS':>10} {'ms/img':>10}")
        print(f"  {'-'*65}")
        for name, s in all_summaries.items():
            print(
                f"  {name:<25} "
                f"{s['PSNR']['mean']:>12.4f} "
                f"{s['SSIM']['mean']:>10.4f} "
                f"{s['LPIPS']['mean']:>10.4f} "
                f"{s['Inference_ms']['mean']:>10.2f}"
            )
        print("="*70)

        # Plot comparison bar charts
        psnr_dict = {k: v['PSNR']['mean'] for k, v in all_summaries.items()}
        ssim_dict = {k: v['SSIM']['mean'] for k, v in all_summaries.items()}
        plot_metrics_bar(psnr_dict, metric='PSNR', save_path='results/ablation_psnr.png')
        plot_metrics_bar(ssim_dict, metric='SSIM', save_path='results/ablation_ssim.png')

    return all_summaries


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CATKC-Net")
    parser.add_argument('--model',      type=str, default='proposed',   help='Model: baseline / proposed')
    parser.add_argument('--checkpoint', type=str, required=True,        help='Path to checkpoint .pth file')
    parser.add_argument('--experiment', type=str, default='A4_full',    help='Experiment name')
    parser.add_argument('--ablation',   action='store_true',             help='Evaluate all ablation models')
    args = parser.parse_args()

    from data.dataset import get_dataloaders
    _, _, test_loader = get_dataloaders()

    if args.ablation:
        evaluate_all_ablations(test_loader)
    else:
        if args.model == 'baseline':
            from models.base_model import BaselineModel
            model = BaselineModel()
        else:
            from models.proposed_model import CATKCNet
            model = CATKCNet(use_attention=True)

        evaluator = Evaluator(model, test_loader, args.experiment)
        evaluator.load_checkpoint(args.checkpoint)
        evaluator.evaluate()