#!/usr/bin/env python3
"""
CRITICAL INVESTIGATION: Verify SHAP Test Performance

This script investigates the suspicious 2.82% detection rate reported
during SHAP analysis. Expected ~87% based on previous testing.
"""

import sys
import pickle
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

# Add src to path
sys.path.append(str(Path.cwd() / 'src'))

from src.models.vae import VAEDetector

# Directories
DATA_DIR = Path('data/raw')
MODELS_DIR = Path('models')

print("=" * 80)
print("CRITICAL INVESTIGATION: VAE PERFORMANCE ON SHAP TEST SET")
print("=" * 80)
print("\nIssue: SHAP reported only 2.82% detection rate on DoS traffic")
print("Expected: ~87% detection rate based on previous F1=0.8713 testing\n")

# ============================================================================
# Step 1: Load VAE Model and Check Threshold
# ============================================================================
print("Step 1: Loading VAE model and checking threshold...")
vae = VAEDetector.load(MODELS_DIR / 'vae_200k.h5')
print(f"✓ Model loaded")
print(f"  Threshold: {vae.threshold:.6f}")
print(f"  Input features: {vae.n_features}")
print(f"  Latent dimensions: {vae.latent_dim}\n")

# ============================================================================
# Step 2: Load Test Data (Same as SHAP Script)
# ============================================================================
print("Step 2: Loading Wednesday DoS test data (matching SHAP script)...")

def load_and_preprocess_for_verification(filepath, max_samples=10000):
    """
    Load and preprocess data EXACTLY like SHAP script.
    """
    # Load CSV
    df = pd.read_csv(filepath)
    print(f"  Total rows in file: {len(df):,}")

    # Separate BENIGN and DoS samples
    benign_df = df[df[' Label'] == 'BENIGN'].copy()
    dos_df = df[df[' Label'].str.contains('DoS', na=False)].copy()

    print(f"  Available BENIGN rows: {len(benign_df):,}")
    print(f"  Available DoS rows: {len(dos_df):,}")

    def preprocess_subset(subset_df, n_samples, label):
        """Preprocess a subset of data."""
        if len(subset_df) == 0:
            return None, None

        # Sample
        actual_samples = min(n_samples, len(subset_df))
        sampled_df = subset_df.sample(n=actual_samples, random_state=42)

        # Create labels BEFORE preprocessing
        labels = np.ones(len(sampled_df)) if 'DoS' in label else np.zeros(len(sampled_df))

        # Drop metadata columns
        metadata_cols = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp']
        for col in metadata_cols:
            if col in sampled_df.columns:
                sampled_df = sampled_df.drop(col, axis=1)

        # Drop label column
        if ' Label' in sampled_df.columns:
            sampled_df = sampled_df.drop(' Label', axis=1)

        # Handle infinite values
        sampled_df = sampled_df.replace([np.inf, -np.inf], np.finfo(np.float64).max)

        # Handle NaN values
        for col in sampled_df.columns:
            if sampled_df[col].isna().any():
                median_val = sampled_df[col].median()
                if np.isnan(median_val):
                    median_val = 0
                sampled_df[col] = sampled_df[col].fillna(median_val)

        # Convert to numpy
        X = sampled_df.values

        # Models expect 66 features
        if X.shape[1] > 66:
            X = X[:, :66]

        # Final validation
        assert not np.any(np.isnan(X)), "Data contains NaN after preprocessing!"
        assert not np.any(np.isinf(X)), "Data contains inf after preprocessing!"

        return X, labels

    # Preprocess both subsets (5000 each)
    X_benign, y_benign = preprocess_subset(benign_df, max_samples // 2, 'BENIGN')
    X_dos, y_dos = preprocess_subset(dos_df, max_samples // 2, 'DoS')

    # Combine
    X_test = np.vstack([X_benign, X_dos])
    y_test = np.concatenate([y_benign, y_dos])

    return X_test, y_test

# Load data
wednesday_file = DATA_DIR / 'Wednesday-workingHours.pcap_ISCX.csv'
X_test, y_test = load_and_preprocess_for_verification(wednesday_file, max_samples=10000)

print(f"\n  Loaded test set: {len(X_test):,} samples")
print(f"  Features: {X_test.shape[1]}\n")

# ============================================================================
# ACTION 1: Verify Test Data Composition
# ============================================================================
print("=" * 80)
print("ACTION 1: VERIFY TEST DATA COMPOSITION")
print("=" * 80)
print(f"Total samples: {len(y_test):,}")
print(f"Benign (0): {(y_test == 0).sum():,}")
print(f"DoS (1): {(y_test == 1).sum():,}")
print(f"Benign %: {(y_test == 0).sum() / len(y_test) * 100:.1f}%")
print(f"DoS %: {(y_test == 1).sum() / len(y_test) * 100:.1f}%")

if (y_test == 0).sum() != 5000 or (y_test == 1).sum() != 5000:
    print("⚠️  WARNING: Test set is NOT balanced 50/50!")
else:
    print("✓ Test set is balanced 50/50 as expected")

# ============================================================================
# ACTION 2: Verify Prediction Distribution
# ============================================================================
print(f"\n{'=' * 80}")
print("ACTION 2: VERIFY PREDICTION DISTRIBUTION")
print("=" * 80)

# Get predictions from model
predictions = vae.predict(X_test)
anomaly_scores = vae.decision_function(X_test)

print(f"Total predictions: {len(predictions):,}")
print(f"Predicted benign (0): {(predictions == 0).sum():,}")
print(f"Predicted attack (1): {(predictions == 1).sum():,}")
print(f"Attack %: {(predictions == 1).sum() / len(predictions) * 100:.2f}%")

print(f"\nAnomaly scores statistics:")
print(f"  Min: {np.nanmin(anomaly_scores):.6f}")
print(f"  Max: {np.nanmax(anomaly_scores):.6f}")
print(f"  Mean: {np.nanmean(anomaly_scores):.6f}")
print(f"  Median: {np.nanmedian(anomaly_scores):.6f}")
print(f"  NaN count: {np.isnan(anomaly_scores).sum()}")
print(f"  Inf count: {np.isinf(anomaly_scores).sum()}")
print(f"  Model threshold: {vae.threshold:.6f}")

if (predictions == 1).sum() / len(predictions) < 0.10:
    print("⚠️  CRITICAL: Only {:.2f}% flagged as attack - this is VERY LOW!".format(
        (predictions == 1).sum() / len(predictions) * 100))
else:
    print("✓ Attack percentage seems reasonable")

# ============================================================================
# ACTION 3: Calculate Proper Metrics
# ============================================================================
print(f"\n{'=' * 80}")
print("ACTION 3: CALCULATE PROPER CONFUSION MATRIX AND METRICS")
print("=" * 80)

cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives (TN): {cm[0,0]:,} (benign correctly classified)")
print(f"False Positives (FP): {cm[0,1]:,} (benign wrongly flagged)")
print(f"False Negatives (FN): {cm[1,0]:,} (DoS missed)")
print(f"True Positives (TP): {cm[1,1]:,} (DoS correctly detected)")

# Calculate metrics
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
recall = tp / (fn + tp) if (fn + tp) > 0 else 0
precision = tp / (fp + tp) if (fp + tp) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'=' * 80}")
print("PERFORMANCE METRICS")
print("=" * 80)
print(f"False Positive Rate: {fp_rate*100:.2f}% (FP / Total Benign)")
print(f"Recall (Detection Rate): {recall*100:.2f}% (TP / Total DoS)")
print(f"Precision: {precision*100:.2f}% (TP / Total Flagged)")
print(f"F1 Score: {f1:.4f}")

print(f"\n{'=' * 80}")
print("CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test, predictions,
                          target_names=['BENIGN', 'DoS'],
                          digits=4))

# ============================================================================
# COMPARISON WITH EXPECTED VALUES
# ============================================================================
print("=" * 80)
print("COMPARISON WITH EXPECTED PERFORMANCE")
print("=" * 80)

expected_f1 = 0.8713
expected_recall = 0.866
expected_fp_rate = 0.0212

print(f"\nMetric                 | Expected  | Actual    | Difference")
print(f"-" * 80)
print(f"F1 Score               | {expected_f1:.4f}    | {f1:.4f}    | {(f1 - expected_f1):+.4f}")
print(f"Recall (Detection)     | {expected_recall:.2%}     | {recall:.2%}     | {(recall - expected_recall):+.2%}")
print(f"False Positive Rate    | {expected_fp_rate:.2%}      | {fp_rate:.2%}      | {(fp_rate - expected_fp_rate):+.2%}")

# Determine issue
print(f"\n{'=' * 80}")
print("DIAGNOSIS")
print("=" * 80)

if abs(f1 - expected_f1) < 0.05:
    print("✓ Performance matches expected values - model is working correctly!")
    print("✓ The 2.82% was likely a calculation/reporting error in SHAP script")
elif recall < 0.10:
    print("✗ CRITICAL ISSUE: Recall is extremely low ({:.2%})".format(recall))
    print("✗ Model is barely detecting any attacks!")
    print("\nPossible causes:")
    print("  1. Wrong threshold being used")
    print("  2. Model not loaded correctly")
    print("  3. Preprocessing mismatch with training")
    print("  4. Model was trained incorrectly")
elif fp_rate > 0.50:
    print("✗ CRITICAL ISSUE: False positive rate is extremely high ({:.2%})".format(fp_rate))
    print("✗ Model is flagging most benign traffic as attacks!")
else:
    print("⚠️  Performance differs from expected but not catastrophically")
    print(f"   F1 difference: {(f1 - expected_f1):+.4f}")
    print(f"   Recall difference: {(recall - expected_recall):+.2%}")

# ============================================================================
# INVESTIGATION: Check if threshold is the issue
# ============================================================================
print(f"\n{'=' * 80}")
print("INVESTIGATION: THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 80)

print("\nTesting different threshold percentiles:")
for percentile in [90, 95, 99]:
    threshold_test = np.nanpercentile(anomaly_scores[y_test == 0], percentile)
    preds_test = (anomaly_scores > threshold_test).astype(int)

    cm_test = confusion_matrix(y_test, preds_test)
    tp_test = cm_test[1, 1]
    fp_test = cm_test[0, 1]
    fn_test = cm_test[1, 0]

    recall_test = tp_test / (fn_test + tp_test)
    precision_test = tp_test / (fp_test + tp_test) if (fp_test + tp_test) > 0 else 0
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0
    fp_rate_test = fp_test / 5000

    marker = " ← MODEL THRESHOLD" if abs(threshold_test - vae.threshold) < 0.01 else ""
    print(f"  P{percentile}: threshold={threshold_test:.6f}, F1={f1_test:.4f}, Recall={recall_test:.2%}, FP={fp_rate_test:.2%}{marker}")

print(f"\n{'=' * 80}")
print("END OF INVESTIGATION")
print("=" * 80)
