#!/usr/bin/env python3
"""
Test TRUE False Positive Rate on Multi-Day BENIGN Traffic

This script tests the models on BENIGN traffic from different days (Wednesday, Thursday, Friday)
to measure the actual false positive rate (normal traffic incorrectly flagged as attacks).

Previous testing measured "how different is attack traffic" not "how many false alarms on normal traffic".
"""

import sys
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path.cwd() / 'src'))

from data.preprocessing import CICIDS2017Preprocessor

# Directories
DATA_DIR = Path('data/raw')
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')

print("="*80)
print("TRUE FALSE POSITIVE RATE TEST")
print("="*80)
print("\nTesting models on BENIGN traffic from multiple days")
print("Training: Monday BENIGN")
print("Testing: Wednesday, Thursday, Friday BENIGN only\n")

# Load models
print("Loading models...")
from src.models.vae import VAEDetector
from src.models.one_class_svm import OneClassSVMDetector

vae = VAEDetector.load(MODELS_DIR / 'vae_200k.h5')
ocsvm = OneClassSVMDetector.load(MODELS_DIR / 'ocsvm_200k.pkl')
print("✓ Models loaded\n")

# Preprocessor
preprocessor = CICIDS2017Preprocessor()

# Test datasets
test_files = {
    'Wednesday': 'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday-Morning': 'Thursday-Morning-WebAttacks.pcap_ISCX.csv',
    'Friday-PortScan': 'Friday-Afternoon-PortScan.pcap_ISCX.csv'
}

results = []

for day_name, filename in test_files.items():
    print(f"\n{'='*80}")
    print(f"Testing on {day_name} BENIGN Traffic")
    print(f"{'='*80}")

    # Load data
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"⚠️  File not found: {filepath}")
        continue

    df = pd.read_csv(filepath)
    print(f"Total rows: {len(df):,}")

    # Filter BENIGN only
    benign_df = df[df[' Label'] == 'BENIGN'].copy()
    print(f"BENIGN rows: {len(benign_df):,}")

    if len(benign_df) == 0:
        print("⚠️  No BENIGN traffic found")
        continue

    # Sample for testing (use all if < 50K, otherwise sample 50K)
    sample_size = min(50000, len(benign_df))
    test_df = benign_df.sample(n=sample_size, random_state=42)
    print(f"Testing on: {len(test_df):,} samples\n")

    # Preprocess (simpler - models have their own scalers)
    # 1. Encode labels
    df_encoded, _ = preprocessor.encode_labels(test_df, binary_encoding=True)

    # 2. Clean data
    df_clean = preprocessor.clean_data(
        df_encoded,
        remove_duplicates=True,
        handle_inf='replace',
        handle_nan='median'
    )

    # 3. Separate features (no scaling - models will do it)
    X_test = df_clean.drop(['Label', 'Label_Original'], axis=1, errors='ignore').values

    # VAE predictions
    print("Testing VAE...")
    vae_preds = vae.predict(X_test)
    vae_fp_rate = (vae_preds == 1).mean()
    vae_fp_count = (vae_preds == 1).sum()

    print(f"  VAE False Positives: {vae_fp_count:,} / {len(vae_preds):,}")
    print(f"  VAE FP Rate: {vae_fp_rate:.2%}")

    # OCSVM predictions
    print("\nTesting OCSVM...")
    ocsvm_preds = ocsvm.predict(X_test)
    ocsvm_fp_rate = (ocsvm_preds == 1).mean()
    ocsvm_fp_count = (ocsvm_preds == 1).sum()

    print(f"  OCSVM False Positives: {ocsvm_fp_count:,} / {len(ocsvm_preds):,}")
    print(f"  OCSVM FP Rate: {ocsvm_fp_rate:.2%}")

    results.append({
        'day': day_name,
        'samples_tested': len(test_df),
        'vae_fp_count': vae_fp_count,
        'vae_fp_rate': vae_fp_rate,
        'ocsvm_fp_count': ocsvm_fp_count,
        'ocsvm_fp_rate': ocsvm_fp_rate
    })

# Summary
print(f"\n\n{'='*80}")
print("SUMMARY - TRUE FALSE POSITIVE RATES ON BENIGN TRAFFIC")
print(f"{'='*80}\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Average FP rates
print(f"\n{'='*80}")
print("AVERAGE FALSE POSITIVE RATES ACROSS DAYS")
print(f"{'='*80}")
print(f"VAE Average FP Rate:   {results_df['vae_fp_rate'].mean():.2%}")
print(f"OCSVM Average FP Rate: {results_df['ocsvm_fp_rate'].mean():.2%}")

# Quality assessment
print(f"\n{'='*80}")
print("PRODUCTION READINESS ASSESSMENT")
print(f"{'='*80}")

vae_avg = results_df['vae_fp_rate'].mean()
ocsvm_avg = results_df['ocsvm_fp_rate'].mean()

print(f"\nTarget: FP Rate < 10% on multi-day benign traffic")
print(f"\nVAE:   {vae_avg:.2%} {'✓ PASS' if vae_avg < 0.10 else '✗ FAIL'}")
print(f"OCSVM: {ocsvm_avg:.2%} {'✓ PASS' if ocsvm_avg < 0.10 else '✗ FAIL'}")

# Save results
results_df.to_csv(RESULTS_DIR / 'true_fp_rate_results.csv', index=False)
print(f"\n✓ Results saved to {RESULTS_DIR / 'true_fp_rate_results.csv'}")

print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")
print("""
These FP rates represent the percentage of NORMAL traffic from different days
that gets incorrectly flagged as attacks. This is the TRUE false positive rate
that matters for production deployment.

Previous multi-attack testing measured "how different is attack traffic" not
"how many false alarms on normal traffic".

If FP rates are < 10%, the models are production-ready for volumetric attack
detection within their operational scope (flow-based anomaly detection).
""")
