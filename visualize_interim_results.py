#!/usr/bin/env python3
"""
Quick visualization of interim grid search results
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Interim results (configs 1-14)
results = [
    {'nu': 0.001, 'kernel': 'rbf', 'gamma': 'scale', 'f1': 0.8013, 'precision': 0.9684, 'recall': 0.6834},
    {'nu': 0.001, 'kernel': 'rbf', 'gamma': 'auto', 'f1': 0.8013, 'precision': 0.9684, 'recall': 0.6834},
    {'nu': 0.001, 'kernel': 'linear', 'gamma': None, 'f1': 0.2112, 'precision': 0.1742, 'recall': 0.2683},
    {'nu': 0.001, 'kernel': 'poly', 'gamma': None, 'f1': 0.1747, 'precision': 0.8971, 'recall': 0.0968},
    {'nu': 0.005, 'kernel': 'rbf', 'gamma': 'scale', 'f1': 0.8526, 'precision': 0.9629, 'recall': 0.7650},
    {'nu': 0.005, 'kernel': 'rbf', 'gamma': 'auto', 'f1': 0.8526, 'precision': 0.9629, 'recall': 0.7650},
    {'nu': 0.005, 'kernel': 'linear', 'gamma': None, 'f1': 0.1031, 'precision': 0.0963, 'recall': 0.1110},
    {'nu': 0.005, 'kernel': 'poly', 'gamma': None, 'f1': 0.1557, 'precision': 0.7806, 'recall': 0.0865},
    {'nu': 0.010, 'kernel': 'rbf', 'gamma': 'scale', 'f1': 0.8528, 'precision': 0.9467, 'recall': 0.7759},
    {'nu': 0.010, 'kernel': 'rbf', 'gamma': 'auto', 'f1': 0.8528, 'precision': 0.9467, 'recall': 0.7759},
    {'nu': 0.010, 'kernel': 'linear', 'gamma': None, 'f1': 0.0115, 'precision': 0.0187, 'recall': 0.0083},
    {'nu': 0.010, 'kernel': 'poly', 'gamma': None, 'f1': 0.1456, 'precision': 0.6977, 'recall': 0.0813},
    {'nu': 0.020, 'kernel': 'rbf', 'gamma': 'scale', 'f1': 0.8540, 'precision': 0.9236, 'recall': 0.7942},
    {'nu': 0.020, 'kernel': 'rbf', 'gamma': 'auto', 'f1': 0.8540, 'precision': 0.9236, 'recall': 0.7942},
]

df = pd.DataFrame(results)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('OCSVM Hyperparameter Tuning Results (200K Training Samples)\nInterim Results: Configurations 1-14 of 20',
             fontsize=16, fontweight='bold')

# 1. F1 Score by Kernel and Nu
ax1 = fig.add_subplot(gs[0, :2])
for kernel in ['rbf', 'linear', 'poly']:
    kernel_data = df[df['kernel'] == kernel].groupby('nu')['f1'].max()
    ax1.plot(kernel_data.index, kernel_data.values, marker='o', label=kernel, linewidth=2)
ax1.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='Quality Gate (0.80)')
ax1.axhline(y=0.5984, color='orange', linestyle='--', alpha=0.5, label='Baseline (0.5984)')
ax1.set_xlabel('Nu Parameter', fontsize=12)
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_title('F1 Score vs Nu (by Kernel)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# 2. RBF Kernel: Nu vs Metrics
ax2 = fig.add_subplot(gs[0, 2])
rbf_data = df[df['kernel'] == 'rbf'].groupby('nu').first()
ax2.plot(rbf_data.index, rbf_data['precision'], marker='o', label='Precision', color='blue')
ax2.plot(rbf_data.index, rbf_data['recall'], marker='s', label='Recall', color='green')
ax2.plot(rbf_data.index, rbf_data['f1'], marker='^', label='F1', color='red', linewidth=2)
ax2.set_xlabel('Nu', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('RBF Kernel: Metrics vs Nu', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# 3. Bar chart: Best F1 by Kernel
ax3 = fig.add_subplot(gs[1, 0])
best_by_kernel = df.groupby('kernel')['f1'].max().sort_values(ascending=False)
colors = ['#2ecc71' if x > 0.8 else '#e74c3c' for x in best_by_kernel.values]
best_by_kernel.plot(kind='bar', ax=ax3, color=colors)
ax3.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='Quality Gate')
ax3.set_title('Best F1 by Kernel', fontsize=14, fontweight='bold')
ax3.set_xlabel('Kernel')
ax3.set_ylabel('F1 Score')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 1)

# 4. Scatter: Precision vs Recall (RBF only)
ax4 = fig.add_subplot(gs[1, 1])
rbf_only = df[df['kernel'] == 'rbf']
scatter = ax4.scatter(rbf_only['recall'], rbf_only['precision'],
                     c=rbf_only['f1'], s=200, cmap='RdYlGn', vmin=0.5, vmax=1.0, alpha=0.7)
for idx, row in rbf_only.iterrows():
    ax4.annotate(f"nu={row['nu']}", (row['recall'], row['precision']),
                fontsize=8, ha='center')
ax4.set_xlabel('Recall', fontsize=12)
ax4.set_ylabel('Precision', fontsize=12)
ax4.set_title('Precision vs Recall (RBF Kernel)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('F1 Score')

# 5. Comparison: Baseline vs Best
ax5 = fig.add_subplot(gs[1, 2])
comparison = pd.DataFrame({
    'F1': [0.5984, 0.8540],
    'Precision': [0.8044, 0.9236],
    'Recall': [0.4664, 0.7942]
}, index=['Baseline\n(nu=0.01)', 'Optimized\n(nu=0.02)'])

x = np.arange(len(comparison.columns))
width = 0.35
ax5.bar(x - width/2, comparison.iloc[0], width, label='Baseline', color='orange', alpha=0.7)
ax5.bar(x + width/2, comparison.iloc[1], width, label='Optimized', color='green', alpha=0.7)
ax5.set_ylabel('Score', fontsize=12)
ax5.set_title('Baseline vs Optimized', fontsize=14, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticks(x)
ax5.set_xticklabels(comparison.columns)
ax5.legend()
ax5.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim(0, 1)

# 6. Top 5 Configurations
ax6 = fig.add_subplot(gs[2, :])
top_5 = df.nlargest(5, 'f1')[['nu', 'kernel', 'gamma', 'f1', 'precision', 'recall']]
ax6.axis('tight')
ax6.axis('off')

table_data = []
for idx, row in top_5.iterrows():
    gamma_str = row['gamma'] if pd.notna(row['gamma']) else 'N/A'
    table_data.append([
        f"{row['nu']:.3f}",
        row['kernel'],
        gamma_str,
        f"{row['f1']:.4f}",
        f"{row['precision']:.4f}",
        f"{row['recall']:.4f}"
    ])

table = ax6.table(cellText=table_data,
                 colLabels=['Nu', 'Kernel', 'Gamma', 'F1', 'Precision', 'Recall'],
                 cellLoc='center',
                 loc='center',
                 colColours=['lightblue']*6)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax6.set_title('Top 5 Configurations', fontsize=14, fontweight='bold', pad=20)

# Add summary text
summary_text = f"""
KEY FINDINGS:
• Best Configuration: nu=0.02, kernel=rbf, gamma=scale
• Best F1 Score: 0.8540 (vs baseline 0.5984, +42.7% improvement)
• Quality Gate: ✓ PASSED (F1 > 0.80)
• Conclusion: OCSVM scales well to 200K with proper tuning
• RBF kernel essential (linear/poly perform poorly)
• Gamma parameter has no effect (scale vs auto identical)
"""

plt.figtext(0.02, 0.02, summary_text, fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.savefig('results/ocsvm_interim_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved to results/ocsvm_interim_visualization.png")
plt.show()
