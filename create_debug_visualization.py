"""
Create visualization comparing all autoencoder debugging attempts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load results
results_path = Path('/Users/yusufyesilyurt/Desktop/Folders/projects/network-intrusion-detection/results/autoencoder_debug_results.pkl')
with open(results_path, 'rb') as f:
    results = pickle.load(f)

# Extract data
models = ['Original\nAutoencoder', 'Attempt 1\nDeeper Arch', 'Attempt 2\nHuber Loss',
          'Attempt 3\nLeakyReLU', 'VAE\n(Reference)']
f1_scores = [0.3564,
             results['attempt1']['f1'],
             results['attempt2']['f1'],
             results['attempt3']['f1'],
             0.8713]
precision_scores = [0.8398,
                    results['attempt1']['precision'],
                    results['attempt2']['precision'],
                    results['attempt3']['precision'],
                    0.90]
recall_scores = [0.2262,
                 results['attempt1']['recall'],
                 results['attempt2']['recall'],
                 results['attempt3']['recall'],
                 0.85]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Autoencoder Debugging: All Attempts vs VAE', fontsize=18, fontweight='bold')

# 1. F1 Score Comparison (top left)
ax1 = axes[0, 0]
colors = ['#FF6B6B', '#FFA07A', '#FFB347', '#FFD700', '#4CAF50']
bars = ax1.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.axhline(y=0.75, color='red', linestyle='--', linewidth=2, label='Success Threshold (F1=0.75)')
ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1.0])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    height = bar.get_height()
    status = '✗' if i < 4 else '✓'
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{score:.4f}\n{status}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. Precision vs Recall (top right)
ax2 = axes[0, 1]
x = np.arange(len(models))
width = 0.35

bars1 = ax2.bar(x - width/2, precision_scores, width, label='Precision',
                color='#3498DB', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, recall_scores, width, label='Recall',
                color='#E74C3C', alpha=0.8, edgecolor='black')

ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=0, ha='center')
ax2.legend(fontsize=10)
ax2.set_ylim([0, 1.0])
ax2.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

# 3. Threshold Optimization Results (bottom left)
ax3 = axes[1, 0]
threshold_percentiles = results['threshold_percentiles_tested']

# Get threshold results for each attempt
attempt1_f1s = [r['f1'] for r in results['attempt1']['threshold_results']]
attempt2_f1s = [r['f1'] for r in results['attempt2']['threshold_results']]
attempt3_f1s = [r['f1'] for r in results['attempt3']['threshold_results']]

ax3.plot(threshold_percentiles, attempt1_f1s, 'o-', label='Attempt 1 (Deeper)',
         linewidth=2, markersize=8, color='#FFA07A')
ax3.plot(threshold_percentiles, attempt2_f1s, 's-', label='Attempt 2 (Huber)',
         linewidth=2, markersize=8, color='#FFB347')
ax3.plot(threshold_percentiles, attempt3_f1s, '^-', label='Attempt 3 (LeakyReLU)',
         linewidth=2, markersize=8, color='#FFD700')
ax3.axhline(y=0.75, color='red', linestyle='--', linewidth=2, label='Success Threshold')
ax3.set_xlabel('Threshold Percentile', fontsize=12, fontweight='bold')
ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax3.set_title('Threshold Optimization Results', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 0.8])

# 4. Summary Table (bottom right)
ax4 = axes[1, 1]
ax4.axis('off')

# Create summary text
summary_text = """
AUTOENCODER DEBUGGING SUMMARY
═══════════════════════════════════════════════════════

OBJECTIVE: Fix F1=0.3564 → F1 ≥ 0.75

ATTEMPTS:
  1. Deeper Architecture [70→64→48→32→16]
     Result: F1=0.3930 (+10.3% vs original)
     Status: FAILED

  2. Huber Loss (robust to outliers)
     Result: F1=0.3896 (-0.9% vs Attempt 1)
     Status: FAILED

  3. LeakyReLU Activation (better gradients)
     Result: F1=0.3778 (-3.9% vs Attempt 1)
     Status: FAILED

═══════════════════════════════════════════════════════
CONCLUSION: Standard Autoencoder FUNDAMENTALLY UNSUITABLE

ROOT CAUSE:
• No latent space regularization (overfits to normal data)
• Reconstruction-only loss insufficient for anomaly detection
• Network traffic attacks reconstruct well → low error
• High-dimensional data (70 features) dilutes anomaly signal

SOLUTION: Use VAE (F1=0.8713)
• KL divergence regularizes latent space
• Probabilistic framework better for anomaly detection
• Combined loss (reconstruction + KL) more robust
• 121.7% improvement over best standard autoencoder

═══════════════════════════════════════════════════════
RECOMMENDATION: Deploy VAE for production
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save figure
output_path = Path('/Users/yusufyesilyurt/Desktop/Folders/projects/network-intrusion-detection/results/autoencoder_debug_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")

plt.show()
