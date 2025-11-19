"""
Full CICIDS2017 Dataset Training Script

PHASE 1: Train OCSVM and VAE on FULL Monday dataset (530K samples)
PHASE 2: Test on multiple attack types (Tuesday-Friday)

Usage:
    # Run both phases (full training + multi-day testing)
    python train_full_dataset.py

    # Run only Phase 1 (full training)
    python train_full_dataset.py --phase1-only

    # Run only Phase 2 (multi-day testing on existing models)
    python train_full_dataset.py --phase2-only

Output:
    - Full dataset models: models/ocsvm_full.pkl, models/vae_full.h5
    - Comparison report: results/full_dataset_comparison.pkl
    - Multi-day results: results/multi_day_performance.csv
    - Visualizations: results/full_*.png
"""

import sys
import os
import argparse
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple

# Add src to path
sys.path.insert(0, '/Users/yusufyesilyurt/Desktop/Folders/projects/network-intrusion-detection')

from src.data.preprocessing import CICIDS2017Preprocessor
from src.models.one_class_svm import OneClassSVMDetector
from src.models.vae import VAEDetector
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Set random seeds
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Paths
PROJECT_ROOT = Path('/Users/yusufyesilyurt/Desktop/Folders/projects/network-intrusion-detection')
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset files mapping
DATASET_FILES = {
    'Monday': 'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday': 'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday': 'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-WebAttacks': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-Infiltration': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-Morning': 'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-PortScan': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-DDoS': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
}

ATTACK_TYPES = {
    'Monday': 'BENIGN',
    'Tuesday': 'Brute Force',
    'Wednesday': 'DoS/DDoS',
    'Thursday-WebAttacks': 'Web Attacks',
    'Thursday-Infiltration': 'Infiltration',
    'Friday-Morning': 'PortScan',
    'Friday-PortScan': 'PortScan',
    'Friday-DDoS': 'DDoS'
}


def load_and_preprocess_data(file_path: Path, sample_size: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess CICIDS2017 data."""
    print(f"\nLoading data from: {file_path.name}")
    print(f"File size: {file_path.stat().st_size / (1024**2):.1f} MB")

    # Load data
    start_time = time.time()
    df = pd.read_csv(file_path)
    load_time = time.time() - start_time
    print(f"Loaded {len(df):,} rows in {load_time:.2f}s")

    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size:,} rows from {len(df):,}")
        df = df.sample(n=sample_size, random_state=42)

    # Preprocess
    start_time = time.time()

    # Drop metadata columns
    metadata_cols = [' Destination Port', 'Flow ID', ' Source IP', ' Source Port', ' Destination IP',
                     ' Timestamp', 'Flow Bytes/s', ' Flow Packets/s']
    metadata_cols = [col for col in metadata_cols if col in df.columns]
    if metadata_cols:
        df = df.drop(columns=metadata_cols)

    # Separate features and labels first
    X = df.drop(columns=[' Label'])
    y = df[' Label']

    # Clean data
    preprocessor = CICIDS2017Preprocessor()
    X = preprocessor.clean_data(X, remove_duplicates=True, handle_inf='replace', handle_nan='median')

    # Binary encoding
    y_binary = (y != 'BENIGN').astype(int)

    preprocess_time = time.time() - start_time
    print(f"Preprocessed in {preprocess_time:.2f}s")
    print(f"Features: {X.shape[1]}, Samples: {len(X):,}")
    print(f"Label distribution:")
    print(f"  BENIGN: {(y_binary == 0).sum():,} ({(y_binary == 0).sum()/len(y_binary)*100:.1f}%)")
    print(f"  ATTACK: {(y_binary == 1).sum():,} ({(y_binary == 1).sum()/len(y_binary)*100:.1f}%)")

    return X, y_binary


def train_ocsvm_full(X_train: pd.DataFrame) -> OneClassSVMDetector:
    """Train One-Class SVM on full Monday dataset."""
    print("\n" + "="*80)
    print("TRAINING ONE-CLASS SVM ON FULL MONDAY DATASET")
    print("="*80)

    print(f"\nTraining samples: {len(X_train):,}")
    print("Configuration: nu=0.01, gamma='scale', kernel='rbf'")

    # Initialize detector
    detector = OneClassSVMDetector(nu=0.01, gamma='scale', kernel='rbf')

    # Train
    start_time = time.time()
    detector.fit(X_train)
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.2f}s")
    print(f"Support vectors: {detector.n_support_vectors_:,}")

    # Save model
    model_path = MODELS_DIR / 'ocsvm_full.pkl'
    detector.save_model(model_path)
    print(f"Model saved to: {model_path}")

    return detector


def train_vae_full(X_train: pd.DataFrame) -> VAEDetector:
    """Train VAE on full Monday dataset."""
    print("\n" + "="*80)
    print("TRAINING VAE ON FULL MONDAY DATASET")
    print("="*80)

    print(f"\nTraining samples: {len(X_train):,}")
    print("Configuration: latent_dim=20, encoder_dims=[50,30], kl_weight=0.001")

    # Initialize detector
    detector = VAEDetector(
        input_dim=X_train.shape[1],
        latent_dim=20,
        encoder_dims=[50, 30],
        decoder_dims=[30, 50],
        kl_weight=0.001,
        dropout_rate=0.2
    )

    # Train
    print("\nStarting VAE training...")
    start_time = time.time()
    history = detector.fit(
        X_train.values,
        epochs=100,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time/60:.2f} minutes")

    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Total loss
    axes[0, 0].plot(history['total_loss'], label='Train')
    axes[0, 0].plot(history['val_total_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('VAE Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Reconstruction loss
    axes[0, 1].plot(history['reconstruction_loss'], label='Train')
    axes[0, 1].plot(history['val_reconstruction_loss'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('VAE Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # KL divergence
    axes[1, 0].plot(history['kl_divergence'], label='Train')
    axes[1, 0].plot(history['val_kl_divergence'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('VAE KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'vae_full_training_history.png', dpi=300, bbox_inches='tight')
    print(f"Training history saved to: {RESULTS_DIR / 'vae_full_training_history.png'}")
    plt.close()

    # Save model
    model_path = MODELS_DIR / 'vae_full.h5'
    detector.save_model(str(model_path))
    print(f"Model saved to: {model_path}")

    return detector


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """Evaluate model on test data."""
    print(f"\nEvaluating {model_name}...")

    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test.values if isinstance(X_test, pd.DataFrame) else X_test)
    pred_time = time.time() - start_time

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    results = {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'fp_rate': fp_rate,
        'pred_time': pred_time,
        'samples_per_sec': len(X_test) / pred_time
    }

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  FP Rate:   {fp_rate:.4f}")
    print(f"  Inference: {pred_time:.2f}s ({results['samples_per_sec']:.0f} samples/sec)")

    return results


def phase1_full_training():
    """PHASE 1: Train on full Monday dataset."""
    print("\n" + "="*80)
    print("PHASE 1: FULL DATASET TRAINING")
    print("="*80)

    # Load FULL Monday data (530K samples)
    X_train, _ = load_and_preprocess_data(DATA_DIR / DATASET_FILES['Monday'])

    # Load Wednesday test data (100K samples for consistency with previous experiments)
    X_test, y_test = load_and_preprocess_data(
        DATA_DIR / DATASET_FILES['Wednesday'],
        sample_size=100000
    )

    # Train OCSVM
    ocsvm_detector = train_ocsvm_full(X_train)

    # Train VAE
    vae_detector = train_vae_full(X_train)

    # Evaluate both models
    print("\n" + "="*80)
    print("EVALUATION ON WEDNESDAY DOS/DDOS DATA")
    print("="*80)

    ocsvm_results = evaluate_model(ocsvm_detector, X_test, y_test, "OCSVM (Full)")
    vae_results = evaluate_model(vae_detector, X_test, y_test, "VAE (Full)")

    # Compare with previous 50K results
    print("\n" + "="*80)
    print("COMPARISON: 50K vs 530K TRAINING")
    print("="*80)

    comparison = pd.DataFrame([
        {'Model': 'OCSVM (50K)', 'Precision': 0.9339, 'Recall': 0.6824, 'F1': 0.7886, 'FP Rate': 0.0276},
        {'Model': 'OCSVM (530K)', 'Precision': ocsvm_results['precision'],
         'Recall': ocsvm_results['recall'], 'F1': ocsvm_results['f1'], 'FP Rate': ocsvm_results['fp_rate']},
        {'Model': 'VAE (50K)', 'Precision': 0.9320, 'Recall': 0.6795, 'F1': 0.7873, 'FP Rate': 0.0284},
        {'Model': 'VAE (530K)', 'Precision': vae_results['precision'],
         'Recall': vae_results['recall'], 'F1': vae_results['f1'], 'FP Rate': vae_results['fp_rate']}
    ])

    print("\n" + comparison.to_string(index=False))

    # Calculate improvements
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)

    ocsvm_f1_improvement = (ocsvm_results['f1'] - 0.7886) / 0.7886 * 100
    vae_f1_improvement = (vae_results['f1'] - 0.7873) / 0.7873 * 100

    print(f"\nOCSVM F1 improvement: {ocsvm_results['f1']:.4f} vs 0.7886 ({ocsvm_f1_improvement:+.2f}%)")
    print(f"VAE F1 improvement:   {vae_results['f1']:.4f} vs 0.7873 ({vae_f1_improvement:+.2f}%)")

    # Quality gate check
    print("\n" + "="*80)
    print("QUALITY GATE CHECK (F1 >= 0.85)")
    print("="*80)

    ocsvm_passes = ocsvm_results['f1'] >= 0.85
    vae_passes = vae_results['f1'] >= 0.85

    print(f"\nOCSVM (530K): F1 = {ocsvm_results['f1']:.4f} {'✓ PASS' if ocsvm_passes else '✗ FAIL'}")
    print(f"VAE (530K):   F1 = {vae_results['f1']:.4f} {'✓ PASS' if vae_passes else '✗ FAIL'}")

    if ocsvm_passes or vae_passes:
        print("\n✓ QUALITY GATE MET!")
        if ocsvm_passes and vae_passes:
            print("  Both models meet F1 >= 0.85 threshold")
        elif ocsvm_passes:
            print("  OCSVM meets quality gate - recommend for production")
        else:
            print("  VAE meets quality gate - recommend for production")
    else:
        print("\n✗ QUALITY GATE NOT MET")
        print("  Need ensemble or advanced architectures")

    # Save results
    results_summary = {
        'ocsvm': ocsvm_results,
        'vae': vae_results,
        'comparison': comparison.to_dict(),
        'quality_gate_passed': ocsvm_passes or vae_passes
    }

    with open(RESULTS_DIR / 'full_dataset_comparison.pkl', 'wb') as f:
        pickle.dump(results_summary, f)

    comparison.to_csv(RESULTS_DIR / 'full_dataset_comparison.csv', index=False)
    print(f"\nResults saved to: {RESULTS_DIR / 'full_dataset_comparison.pkl'}")

    return ocsvm_detector, vae_detector, results_summary


def phase2_multi_day_testing(ocsvm_detector=None, vae_detector=None):
    """PHASE 2: Test on multiple attack types."""
    print("\n" + "="*80)
    print("PHASE 2: MULTI-DAY ATTACK TESTING")
    print("="*80)

    # Load models if not provided
    if ocsvm_detector is None:
        print("\nLoading OCSVM model...")
        ocsvm_detector = OneClassSVMDetector()
        ocsvm_detector.load_model(MODELS_DIR / 'ocsvm_full.pkl')

    if vae_detector is None:
        print("Loading VAE model...")
        vae_detector = VAEDetector()
        vae_detector.load_model(str(MODELS_DIR / 'vae_full.h5'))

    # Test days (exclude Monday - training data)
    test_days = ['Tuesday', 'Wednesday', 'Thursday-WebAttacks',
                 'Thursday-Infiltration', 'Friday-Morning', 'Friday-PortScan', 'Friday-DDoS']

    results_list = []

    for day in test_days:
        print(f"\n{'='*80}")
        print(f"Testing on {day} ({ATTACK_TYPES[day]})")
        print('='*80)

        # Load and preprocess data (sample 100K for consistency)
        X_test, y_test = load_and_preprocess_data(
            DATA_DIR / DATASET_FILES[day],
            sample_size=min(100000, None)  # Use all data if < 100K
        )

        # Evaluate OCSVM
        ocsvm_results = evaluate_model(ocsvm_detector, X_test, y_test, "OCSVM")
        ocsvm_results['day'] = day
        ocsvm_results['attack_type'] = ATTACK_TYPES[day]
        results_list.append(ocsvm_results)

        # Evaluate VAE
        vae_results = evaluate_model(vae_detector, X_test, y_test, "VAE")
        vae_results['day'] = day
        vae_results['attack_type'] = ATTACK_TYPES[day]
        results_list.append(vae_results)

    # Create results dataframe
    results_df = pd.DataFrame(results_list)

    # Print summary
    print("\n" + "="*80)
    print("MULTI-DAY PERFORMANCE SUMMARY")
    print("="*80)

    summary = results_df.pivot_table(
        index='attack_type',
        columns='model',
        values=['f1', 'precision', 'recall', 'fp_rate']
    )

    print("\n" + summary.to_string())

    # Save results
    results_df.to_csv(RESULTS_DIR / 'multi_day_performance.csv', index=False)
    print(f"\nResults saved to: {RESULTS_DIR / 'multi_day_performance.csv'}")

    # Visualize multi-day performance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # F1 scores by attack type
    pivot_f1 = results_df.pivot(index='attack_type', columns='model', values='f1')
    pivot_f1.plot(kind='bar', ax=axes[0, 0], rot=45)
    axes[0, 0].set_title('F1 Score by Attack Type', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].axhline(y=0.85, color='r', linestyle='--', label='Quality Gate (0.85)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Precision by attack type
    pivot_precision = results_df.pivot(index='attack_type', columns='model', values='precision')
    pivot_precision.plot(kind='bar', ax=axes[0, 1], rot=45)
    axes[0, 1].set_title('Precision by Attack Type', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Recall by attack type
    pivot_recall = results_df.pivot(index='attack_type', columns='model', values='recall')
    pivot_recall.plot(kind='bar', ax=axes[1, 0], rot=45)
    axes[1, 0].set_title('Recall by Attack Type', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # FP Rate by attack type
    pivot_fp = results_df.pivot(index='attack_type', columns='model', values='fp_rate')
    pivot_fp.plot(kind='bar', ax=axes[1, 1], rot=45)
    axes[1, 1].set_title('False Positive Rate by Attack Type', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('FP Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'multi_day_performance.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {RESULTS_DIR / 'multi_day_performance.png'}")
    plt.close()

    return results_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Full CICIDS2017 Dataset Training')
    parser.add_argument('--phase1-only', action='store_true', help='Run only Phase 1 (full training)')
    parser.add_argument('--phase2-only', action='store_true', help='Run only Phase 2 (multi-day testing)')
    args = parser.parse_args()

    start_time = time.time()

    if args.phase2_only:
        # Run only Phase 2
        phase2_multi_day_testing()
    elif args.phase1_only:
        # Run only Phase 1
        phase1_full_training()
    else:
        # Run both phases
        ocsvm_detector, vae_detector, phase1_results = phase1_full_training()
        phase2_multi_day_testing(ocsvm_detector, vae_detector)

    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    print("\nFiles created:")
    print("  - models/ocsvm_full.pkl")
    print("  - models/vae_full.h5")
    print("  - results/full_dataset_comparison.pkl")
    print("  - results/full_dataset_comparison.csv")
    print("  - results/multi_day_performance.csv")
    print("  - results/vae_full_training_history.png")
    print("  - results/multi_day_performance.png")


if __name__ == '__main__':
    main()
