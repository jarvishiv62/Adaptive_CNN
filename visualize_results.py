import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('results/A3_cam_mse/test/per_image_results.csv')

# Calculate average values
averages = {
    'PSNR': df['PSNR'].mean(),
    'SSIM': df['SSIM'].mean(),
    'LPIPS': df['LPIPS'].mean(),
    'Inference_ms': df['Inference_ms'].mean()
}

print("Average Values:")
print("-" * 30)
for metric, avg in averages.items():
    print(f"{metric}: {avg:.4f}")

# Create bar chart
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Performance Metrics per Image', fontsize=16, fontweight='bold')

# PSNR bar chart
axes[0, 0].bar(range(len(df)), df['PSNR'], color='skyblue', alpha=0.7)
axes[0, 0].axhline(y=averages['PSNR'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["PSNR"]:.2f}')
axes[0, 0].set_title('PSNR Values per Image')
axes[0, 0].set_xlabel('Image Index')
axes[0, 0].set_ylabel('PSNR (dB)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# SSIM bar chart
axes[0, 1].bar(range(len(df)), df['SSIM'], color='lightgreen', alpha=0.7)
axes[0, 1].axhline(y=averages['SSIM'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["SSIM"]:.3f}')
axes[0, 1].set_title('SSIM Values per Image')
axes[0, 1].set_xlabel('Image Index')
axes[0, 1].set_ylabel('SSIM')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# LPIPS bar chart
axes[1, 0].bar(range(len(df)), df['LPIPS'], color='salmon', alpha=0.7)
axes[1, 0].axhline(y=averages['LPIPS'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["LPIPS"]:.3f}')
axes[1, 0].set_title('LPIPS Values per Image')
axes[1, 0].set_xlabel('Image Index')
axes[1, 0].set_ylabel('LPIPS')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Inference time bar chart
axes[1, 1].bar(range(len(df)), df['Inference_ms'], color='gold', alpha=0.7)
axes[1, 1].axhline(y=averages['Inference_ms'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["Inference_ms"]:.2f} ms')
axes[1, 1].set_title('Inference Time per Image')
axes[1, 1].set_xlabel('Image Index')
axes[1, 1].set_ylabel('Inference Time (ms)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('results/A3_cam_mse/test/per_image_metrics_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nChart saved as: results/A3_cam_mse/test/per_image_metrics_chart.png")
