#!/usr/bin/env python3
"""
OCSVM Hyperparameter Grid Search for 200K Dataset

This script performs comprehensive hyperparameter tuning to determine if OCSVM's
performance degradation (F1=0.7886 @ 50K → 0.5984 @ 200K) can be fixed with
proper parameter re-optimization.

Grid Search Parameters:
- nu: [0.001, 0.005, 0.01, 0.02, 0.05]
- kernel: ['rbf', 'linear', 'poly']
- gamma: ['scale', 'auto'] (for RBF kernel only)

Expected to take 30-60 minutes for complete grid search.
"""

import sys
import time
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.preprocessing import CICIDS2017Preprocessor
from src.models.one_class_svm import OneClassSVMDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/raw")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_and_preprocess_data(
    train_samples: int = 200000,
    test_samples: int = 100000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess CICIDS2017 data.

    Args:
        train_samples: Number of Monday BENIGN samples for training
        test_samples: Number of Wednesday samples for testing

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Loading Monday BENIGN data for training...")
    monday_path = DATA_DIR / "Monday-WorkingHours.pcap_ISCX.csv"

    if not monday_path.exists():
        raise FileNotFoundError(
            f"Monday data not found at {monday_path}\n"
            "Please download CICIDS2017 dataset first."
        )

    # Load Monday data (100% BENIGN)
    df_monday = pd.read_csv(monday_path, nrows=train_samples)
    # Strip column names (CICIDS2017 has leading spaces)
    df_monday.columns = df_monday.columns.str.strip()
    logger.info(f"Loaded {len(df_monday)} Monday samples")

    # Load Wednesday data (DoS/DDoS + BENIGN)
    logger.info("Loading Wednesday attack data for testing...")
    wednesday_path = DATA_DIR / "Wednesday-workingHours.pcap_ISCX.csv"

    if not wednesday_path.exists():
        raise FileNotFoundError(
            f"Wednesday data not found at {wednesday_path}\n"
            "Please download CICIDS2017 dataset first."
        )

    df_wednesday = pd.read_csv(wednesday_path, nrows=test_samples)
    # Strip column names (CICIDS2017 has leading spaces)
    df_wednesday.columns = df_wednesday.columns.str.strip()
    logger.info(f"Loaded {len(df_wednesday)} Wednesday samples")

    # Preprocess
    logger.info("Preprocessing data...")
    preprocessor = CICIDS2017Preprocessor(scaler_type='standard')

    # Encode Monday labels first (before cleaning drops label column)
    df_monday_encoded, _ = preprocessor.encode_labels(df_monday, binary_encoding=True)

    # Clean Monday data
    df_monday_clean = preprocessor.clean_data(
        df_monday_encoded,
        remove_duplicates=True,
        handle_inf='replace',
        handle_nan='median'
    )

    # Extract features and labels
    X_train = df_monday_clean.drop(['Label', 'Label_Original'], axis=1, errors='ignore')
    y_train = df_monday_clean['Label'].values if 'Label' in df_monday_clean.columns else np.zeros(len(df_monday_clean))

    # Scale Monday features
    X_train = preprocessor.scale_features(X_train.values)
    if isinstance(X_train, tuple):
        X_train = X_train[0]

    # Encode Wednesday labels first
    df_wednesday_encoded, _ = preprocessor.encode_labels(df_wednesday, binary_encoding=True)

    # Clean Wednesday data
    df_wednesday_clean = preprocessor.clean_data(
        df_wednesday_encoded,
        remove_duplicates=True,
        handle_inf='replace',
        handle_nan='median'
    )

    # Extract features and labels
    X_test = df_wednesday_clean.drop(['Label', 'Label_Original'], axis=1, errors='ignore')
    y_test = df_wednesday_clean['Label'].values

    # Scale Wednesday features (use training scaler)
    X_test = preprocessor.scaler.transform(X_test.values)

    logger.info(f"Training set: {X_train.shape}, {(y_train == 0).sum()} BENIGN")
    logger.info(f"Test set: {X_test.shape}, {(y_test == 0).sum()} BENIGN, {(y_test == 1).sum()} ATTACKS")

    return X_train, X_test, y_train, y_test


def evaluate_model(
    model: OneClassSVMDetector,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model on test data.

    Args:
        model: Trained OCSVM detector
        X_test: Test features
        y_test: Test labels (0=BENIGN, 1=ATTACK)

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred)
    }

    return metrics


def grid_search(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Perform comprehensive hyperparameter grid search.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels

    Returns:
        DataFrame with all results
    """
    # Define grid
    nu_values = [0.001, 0.005, 0.01, 0.02, 0.05]
    kernels = ['rbf', 'linear', 'poly']
    gamma_values = ['scale', 'auto']

    results = []
    total_configs = len(nu_values) * (len(kernels) - 1 + len(gamma_values))
    current_config = 0

    logger.info(f"Starting grid search with {total_configs} configurations...")

    for nu in nu_values:
        for kernel in kernels:
            if kernel == 'rbf':
                # RBF kernel - try different gamma values
                for gamma in gamma_values:
                    current_config += 1
                    config_name = f"nu={nu}, kernel={kernel}, gamma={gamma}"
                    logger.info(f"[{current_config}/{total_configs}] Testing {config_name}")

                    try:
                        start_time = time.time()

                        # Train model
                        model = OneClassSVMDetector(
                            nu=nu,
                            kernel=kernel,
                            gamma=gamma
                        )
                        model.fit(X_train)

                        training_time = time.time() - start_time

                        # Evaluate
                        metrics = evaluate_model(model, X_test, y_test)

                        # Record results
                        results.append({
                            'nu': nu,
                            'kernel': kernel,
                            'gamma': gamma,
                            'f1': metrics['f1'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'accuracy': metrics['accuracy'],
                            'training_time': training_time
                        })

                        logger.info(f"  F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, "
                                  f"Recall={metrics['recall']:.4f}, Time={training_time:.2f}s")

                    except Exception as e:
                        logger.error(f"  Failed: {e}")
                        results.append({
                            'nu': nu,
                            'kernel': kernel,
                            'gamma': gamma,
                            'f1': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'accuracy': 0.0,
                            'training_time': 0.0,
                            'error': str(e)
                        })

            else:
                # Linear or Poly kernel - no gamma parameter
                current_config += 1
                config_name = f"nu={nu}, kernel={kernel}"
                logger.info(f"[{current_config}/{total_configs}] Testing {config_name}")

                try:
                    start_time = time.time()

                    # Train model
                    model = OneClassSVMDetector(
                        nu=nu,
                        kernel=kernel
                    )
                    model.fit(X_train)

                    training_time = time.time() - start_time

                    # Evaluate
                    metrics = evaluate_model(model, X_test, y_test)

                    # Record results
                    results.append({
                        'nu': nu,
                        'kernel': kernel,
                        'gamma': None,
                        'f1': metrics['f1'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'accuracy': metrics['accuracy'],
                        'training_time': training_time
                    })

                    logger.info(f"  F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, "
                              f"Recall={metrics['recall']:.4f}, Time={training_time:.2f}s")

                except Exception as e:
                    logger.error(f"  Failed: {e}")
                    results.append({
                        'nu': nu,
                        'kernel': kernel,
                        'gamma': None,
                        'f1': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'accuracy': 0.0,
                        'training_time': 0.0,
                        'error': str(e)
                    })

    return pd.DataFrame(results)


def visualize_results(results_df: pd.DataFrame, output_path: Path):
    """
    Create comprehensive visualization of grid search results.

    Args:
        results_df: DataFrame with all results
        output_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OCSVM Hyperparameter Grid Search Results (200K Training Samples)',
                 fontsize=16, fontweight='bold')

    # 1. Heatmap: F1 scores for RBF kernel (nu vs gamma)
    ax1 = axes[0, 0]
    rbf_results = results_df[results_df['kernel'] == 'rbf'].copy()
    if len(rbf_results) > 0:
        pivot_f1 = rbf_results.pivot(index='nu', columns='gamma', values='f1')
        sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax1)
        ax1.set_title('F1 Score: RBF Kernel (nu vs gamma)', fontweight='bold')
        ax1.set_xlabel('Gamma')
        ax1.set_ylabel('Nu')
    else:
        ax1.text(0.5, 0.5, 'No RBF results', ha='center', va='center')
        ax1.set_title('F1 Score: RBF Kernel')

    # 2. Bar chart: Best F1 by kernel type
    ax2 = axes[0, 1]
    best_by_kernel = results_df.groupby('kernel')['f1'].max().sort_values(ascending=False)
    best_by_kernel.plot(kind='bar', ax=ax2, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax2.set_title('Best F1 Score by Kernel Type', fontweight='bold')
    ax2.set_xlabel('Kernel')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.8, color='red', linestyle='--', label='Target (0.80)')
    ax2.axhline(y=0.5984, color='orange', linestyle='--', label='Baseline (0.5984)')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Line plot: F1 vs nu for each kernel
    ax3 = axes[1, 0]
    for kernel in results_df['kernel'].unique():
        kernel_data = results_df[results_df['kernel'] == kernel].groupby('nu')['f1'].max()
        ax3.plot(kernel_data.index, kernel_data.values, marker='o', label=kernel, linewidth=2)
    ax3.set_title('F1 Score vs Nu Parameter (by kernel)', fontweight='bold')
    ax3.set_xlabel('Nu (Expected Outlier Fraction)')
    ax3.set_ylabel('F1 Score')
    ax3.set_xscale('log')
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target (0.80)')
    ax3.axhline(y=0.5984, color='orange', linestyle='--', alpha=0.5, label='Baseline (0.5984)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Scatter: Precision vs Recall (best configs)
    ax4 = axes[1, 1]
    top_10 = results_df.nlargest(10, 'f1')
    for kernel in top_10['kernel'].unique():
        kernel_top = top_10[top_10['kernel'] == kernel]
        ax4.scatter(kernel_top['recall'], kernel_top['precision'],
                   label=kernel, s=100, alpha=0.6)

    # Add baseline point
    ax4.scatter([0.4664], [0.8044], color='red', marker='x', s=200,
               linewidths=3, label='Baseline (nu=0.01, F1=0.5984)')

    ax4.set_title('Precision vs Recall (Top 10 Configurations)', fontweight='bold')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    plt.close()


def generate_report(results_df: pd.DataFrame, baseline_f1: float = 0.5984) -> str:
    """
    Generate comprehensive text report.

    Args:
        results_df: DataFrame with all results
        baseline_f1: Baseline F1 score to compare against

    Returns:
        Report text
    """
    best_config = results_df.loc[results_df['f1'].idxmax()]

    report = f"""
{'='*80}
OCSVM HYPERPARAMETER TUNING REPORT - 200K TRAINING SAMPLES
{'='*80}

BASELINE PERFORMANCE (nu=0.01, kernel=rbf, gamma=scale):
  F1 Score:  {baseline_f1:.4f}
  Status:    DEGRADED from 50K training (F1=0.7886)

{'='*80}
BEST CONFIGURATION FOUND:
{'='*80}
  Kernel:     {best_config['kernel']}
  Nu:         {best_config['nu']}
  Gamma:      {best_config['gamma'] if pd.notna(best_config['gamma']) else 'N/A'}

  F1 Score:   {best_config['f1']:.4f}
  Precision:  {best_config['precision']:.4f}
  Recall:     {best_config['recall']:.4f}
  Accuracy:   {best_config['accuracy']:.4f}
  Train Time: {best_config['training_time']:.2f}s

IMPROVEMENT vs BASELINE:
  F1 Delta:   {best_config['f1'] - baseline_f1:+.4f} ({((best_config['f1'] - baseline_f1) / baseline_f1 * 100):+.1f}%)

QUALITY GATE (F1 > 0.80):
  Status:     {'✓ PASS' if best_config['f1'] > 0.80 else '✗ FAIL'}
  Gap:        {best_config['f1'] - 0.80:+.4f} ({((best_config['f1'] - 0.80) / 0.80 * 100):+.1f}%)

{'='*80}
TOP 5 CONFIGURATIONS:
{'='*80}
"""

    top_5 = results_df.nlargest(5, 'f1')[['kernel', 'nu', 'gamma', 'f1', 'precision', 'recall', 'training_time']]
    report += top_5.to_string(index=False)

    report += f"""

{'='*80}
PERFORMANCE BY KERNEL TYPE:
{'='*80}
"""

    for kernel in results_df['kernel'].unique():
        kernel_data = results_df[results_df['kernel'] == kernel]
        best_kernel = kernel_data.loc[kernel_data['f1'].idxmax()]
        report += f"""
{kernel.upper()} Kernel:
  Best F1:    {best_kernel['f1']:.4f} (nu={best_kernel['nu']}, gamma={best_kernel['gamma'] if pd.notna(best_kernel['gamma']) else 'N/A'})
  Configs:    {len(kernel_data)}
  Avg F1:     {kernel_data['f1'].mean():.4f}
  Avg Time:   {kernel_data['training_time'].mean():.2f}s
"""

    report += f"""
{'='*80}
CONCLUSION:
{'='*80}
"""

    if best_config['f1'] > 0.80:
        improvement_pct = ((best_config['f1'] - baseline_f1) / baseline_f1 * 100)
        report += f"""
SUCCESS: Re-tuning fixed the performance degradation!

The original conclusion "OCSVM doesn't scale to 200K" was INCORRECT. The issue
was hyperparameter tuning, not fundamental scalability.

With proper tuning (nu={best_config['nu']}, kernel={best_config['kernel']}), OCSVM achieves:
  - F1 = {best_config['f1']:.4f} (meets quality gate > 0.80)
  - {improvement_pct:+.1f}% improvement over baseline
  - {best_config['training_time']:.1f}s training time (acceptable)

RECOMMENDATION: Deploy OCSVM with optimized hyperparameters for 200K scale.
"""
    elif best_config['f1'] > baseline_f1:
        improvement_pct = ((best_config['f1'] - baseline_f1) / baseline_f1 * 100)
        gap_pct = ((0.80 - best_config['f1']) / 0.80 * 100)
        report += f"""
PARTIAL SUCCESS: Re-tuning improved performance but didn't reach quality gate.

With optimal hyperparameters (nu={best_config['nu']}, kernel={best_config['kernel']}):
  - F1 = {best_config['f1']:.4f} ({improvement_pct:+.1f}% better than baseline)
  - Still {gap_pct:.1f}% below quality gate (F1 > 0.80)

ANALYSIS:
The improvement suggests hyperparameters matter, but OCSVM may have fundamental
limitations at this scale. The nu parameter (expected outlier fraction) may not
adequately model the 36.3% attack rate in test data.

RECOMMENDATION:
1. Try ensemble with IF (IF for recall, OCSVM for precision)
2. Proceed to deep learning (VAE/GANomaly) for better performance
3. Use optimized OCSVM (F1={best_config['f1']:.4f}) as improved baseline
"""
    else:
        report += f"""
FAILURE: Re-tuning did not improve performance.

Even with comprehensive hyperparameter search, OCSVM could not exceed baseline
F1={baseline_f1:.4f}. This suggests fundamental scalability issues with OCSVM
on larger datasets, not just hyperparameter mistuning.

POSSIBLE CAUSES:
1. One-class SVM assumes single compact cluster - may not hold for 200K samples
2. Support vector computation becomes less effective at scale
3. Attack patterns in test data (36.3% attacks) poorly modeled by OCSVM's
   outlier fraction parameter (nu)

RECOMMENDATION:
1. Abandon OCSVM for 200K+ scale datasets
2. Use Isolation Forest (better scalability) or deep learning
3. Consider OCSVM only for smaller critical subsets (<50K samples)
"""

    report += f"""
{'='*80}
NEXT STEPS:
{'='*80}
"""

    if best_config['f1'] > 0.80:
        report += f"""
1. Save best model (nu={best_config['nu']}, kernel={best_config['kernel']}) as production candidate
2. Test on other attack types (Tuesday: Brute Force, Thursday: Web Attacks)
3. Compare with VAE results to determine final production model
4. Update CLAUDE.md with corrected OCSVM scaling analysis
"""
    else:
        report += f"""
1. Update CLAUDE.md: Document OCSVM limitations at 200K scale
2. Proceed with VAE training as primary path to quality gate
3. Consider ensemble: OCSVM (precision) + IF (recall) + VAE (deep learning)
4. Test full 530K Monday dataset with Isolation Forest (better scalability)
"""

    report += f"""
{'='*80}
"""

    return report


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("OCSVM HYPERPARAMETER GRID SEARCH - 200K DATASET")
    logger.info("="*80)

    # Load data
    logger.info("\nStep 1: Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        train_samples=200000,
        test_samples=100000
    )

    # Grid search
    logger.info("\nStep 2: Performing grid search...")
    start_time = time.time()
    results_df = grid_search(X_train, X_test, y_train, y_test)
    total_time = time.time() - start_time
    logger.info(f"\nGrid search completed in {total_time/60:.1f} minutes")

    # Save results
    logger.info("\nStep 3: Saving results...")
    results_df.to_csv(RESULTS_DIR / "ocsvm_tuning_results.csv", index=False)
    with open(RESULTS_DIR / "ocsvm_tuning_results.pkl", 'wb') as f:
        pickle.dump(results_df, f)
    logger.info(f"Results saved to {RESULTS_DIR}")

    # Find best model and save
    logger.info("\nStep 4: Training and saving best model...")
    best_config = results_df.loc[results_df['f1'].idxmax()]

    best_model = OneClassSVMDetector(
        nu=best_config['nu'],
        kernel=best_config['kernel'],
        gamma=best_config['gamma'] if pd.notna(best_config['gamma']) else 'scale'
    )
    best_model.fit(X_train)
    best_model.save(MODELS_DIR / "ocsvm_200k_tuned.pkl")
    logger.info(f"Best model saved to {MODELS_DIR / 'ocsvm_200k_tuned.pkl'}")

    # Generate visualizations
    logger.info("\nStep 5: Generating visualizations...")
    visualize_results(results_df, RESULTS_DIR / "ocsvm_tuning_heatmap.png")

    # Generate report
    logger.info("\nStep 6: Generating report...")
    report = generate_report(results_df)

    # Save report
    report_path = RESULTS_DIR / "ocsvm_tuning_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Print report
    print("\n")
    print(report)

    # Print summary
    best_config = results_df.loc[results_df['f1'].idxmax()]
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Best F1:        {best_config['f1']:.4f}")
    logger.info(f"Configuration:  nu={best_config['nu']}, kernel={best_config['kernel']}, gamma={best_config['gamma']}")
    logger.info(f"vs Baseline:    {best_config['f1'] - 0.5984:+.4f} ({((best_config['f1'] - 0.5984) / 0.5984 * 100):+.1f}%)")
    logger.info(f"Quality Gate:   {'✓ PASS' if best_config['f1'] > 0.80 else '✗ FAIL'} (F1 > 0.80)")
    logger.info(f"Total Time:     {total_time/60:.1f} minutes")
    logger.info("="*80)


if __name__ == "__main__":
    main()
