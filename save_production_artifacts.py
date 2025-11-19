"""
Create production artifacts bundle for deployment
Includes: fitted scaler + SHAP background data

This enables full API functionality without requiring 470MB training data.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("CREATING PRODUCTION ARTIFACTS FOR DEPLOYMENT")
print("=" * 80)
print()

# Step 1: Load Monday data
print("Step 1: Loading Monday BENIGN training data...")
monday_path = 'data/raw/Monday-WorkingHours.pcap_ISCX.csv'

if not os.path.exists(monday_path):
    print(f"ERROR: {monday_path} not found!")
    print("This script must be run locally where training data exists.")
    exit(1)

df_monday = pd.read_csv(monday_path)
print(f"  Loaded {len(df_monday):,} rows")

# Filter BENIGN only
df_monday = df_monday[df_monday[' Label'] == 'BENIGN'].copy()
print(f"  Filtered to {len(df_monday):,} BENIGN samples")

# Step 2: Preprocess (match training pipeline)
print()
print("Step 2: Preprocessing data...")

# Drop metadata columns
metadata_cols = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp', ' Label']
for col in metadata_cols:
    if col in df_monday.columns:
        df_monday = df_monday.drop(col, axis=1)

print(f"  Features after dropping metadata: {df_monday.shape[1]}")

# Handle inf/nan (same as training)
df_monday = df_monday.replace([np.inf, -np.inf], np.finfo(np.float64).max)
for col in df_monday.columns:
    if df_monday[col].isna().any():
        median_val = df_monday[col].median()
        df_monday[col] = df_monday[col].fillna(median_val)

print(f"  ✓ Cleaned inf/nan values")

# Get feature names and data (first 66 features to match model)
feature_names = df_monday.columns.tolist()[:66]
X_train = df_monday[feature_names].values[:200000]  # Use first 200K

print(f"  Training data shape: {X_train.shape}")
print(f"  Feature count: {len(feature_names)}")

# Step 3: Fit scaler
print()
print("Step 3: Creating and fitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_train)

print(f"  ✓ Scaler fitted on {len(X_train):,} samples")
print(f"  Mean shape: {scaler.mean_.shape}")
print(f"  Scale shape: {scaler.scale_.shape}")

# Step 4: Get SHAP background data (small sample)
print()
print("Step 4: Creating SHAP background data...")

# Transform and clean background data
X_scaled = scaler.transform(X_train[:100])
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

print(f"  Background samples: {X_scaled.shape[0]}")
print(f"  Background shape: {X_scaled.shape}")
print(f"  NaN count: {np.isnan(X_scaled).sum()}")
print(f"  Inf count: {np.isinf(X_scaled).sum()}")

# Step 5: Package and save
print()
print("Step 5: Saving production artifacts...")

production_data = {
    'scaler': scaler,
    'background_data': X_scaled,
    'feature_names': feature_names
}

output_path = 'models/production_artifacts.pkl'
joblib.dump(production_data, output_path)

file_size_kb = os.path.getsize(output_path) / 1024
print(f"  ✓ Saved to: {output_path}")
print(f"  File size: {file_size_kb:.1f} KB")

# Step 6: Verify by loading
print()
print("Step 6: Verifying saved artifacts...")
loaded = joblib.load(output_path)

assert 'scaler' in loaded, "Missing scaler!"
assert 'background_data' in loaded, "Missing background_data!"
assert 'feature_names' in loaded, "Missing feature_names!"

print(f"  ✓ Scaler: {type(loaded['scaler']).__name__}")
print(f"  ✓ Background: {loaded['background_data'].shape}")
print(f"  ✓ Features: {len(loaded['feature_names'])} features")
print(f"  ✓ First 5 features: {loaded['feature_names'][:5]}")

print()
print("=" * 80)
print("✓ PRODUCTION ARTIFACTS CREATED SUCCESSFULLY")
print("=" * 80)
print()
print("Next steps:")
print("  1. Add to git: git add -f models/production_artifacts.pkl")
print("  2. Update src/api/app.py to load this file")
print("  3. Test API locally")
print("  4. Push to deploy")
print()
print(f"File ready: {output_path} ({file_size_kb:.1f} KB)")
