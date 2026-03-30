#!/usr/bin/env python3
"""
Simple log viewer - reads training logs from checkpoints instead of TensorBoard
"""

import os
import glob
import json

def read_checkpoint_info(checkpoint_path):
    """Read info from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        import torch
        state = torch.load(checkpoint_path, map_location='cpu')
        return {
            'epoch': state.get('epoch', 0),
            'best_psnr': state.get('best_psnr', 0),
            'psnr': state.get('psnr', 0)
        }
    except Exception as e:
        return {'error': str(e)}

def check_experiments():
    """Check all experiment checkpoints."""
    experiments = ['A1_baseline', 'A2_parallel_only', 'A3_cam_mse', 'A4_full_model']
    
    print("Training Progress Summary")
    print("="*50)
    
    for exp in experiments:
        print(f"\n{exp}:")
        
        # Check best model
        best_path = f"checkpoints/{exp}/best_model.pth"
        best_info = read_checkpoint_info(best_path)
        
        # Check last model
        last_path = f"checkpoints/{exp}/last_model.pth"
        last_info = read_checkpoint_info(last_path)
        
        if best_info:
            if 'error' in best_info:
                print(f"  Best model: Error - {best_info['error']}")
            else:
                print(f"  Best model: Epoch {best_info['epoch']}, PSNR = {best_info['best_psnr']:.2f} dB")
        
        if last_info:
            if 'error' in last_info:
                print(f"  Last model: Error - {last_info['error']}")
            else:
                print(f"  Last model: Epoch {last_info['epoch']}, PSNR = {last_info['psnr']:.2f} dB")
        
        if not best_info and not last_info:
            print(f"  No checkpoints found")
        
        # Check if results exist
        results_dir = f"results/{exp}"
        if os.path.exists(results_dir):
            result_files = os.listdir(results_dir)
            image_files = [f for f in result_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  Result images: {len(image_files)} files")
        else:
            print(f"  No results directory")

if __name__ == "__main__":
    check_experiments()
