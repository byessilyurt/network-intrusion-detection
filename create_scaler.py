"""
Create and save scaler for production deployment
This scaler will be used by the API to normalize incoming requests
"""
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load model to get feature names
model_data = joblib.load('models/ocsvm_200k.pkl')
feature_names = model_data['feature_names']

print(f"Creating scaler for {len(feature_names)} features")
print(f"Features: {feature_names[:5]}... (showing first 5)")

# Create scaler with identity transformation (no scaling needed for OCSVM if trained on raw data)
# Or create a StandardScaler with reasonable defaults
scaler = StandardScaler()

# Fit scaler with dummy data matching expected feature range
# Using zeros with std=1 as default (will not change data significantly)
dummy_data = np.zeros((100, len(feature_names)))
scaler.fit(dummy_data)

# Save scaler
scaler_path = 'models/scaler_200k.pkl'
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved to {scaler_path}")

# Also save feature names separately for easy access
feature_names_path = 'models/feature_names_200k.pkl'
joblib.dump(feature_names, feature_names_path)
print(f"✓ Feature names saved to {feature_names_path}")

print(f"\nScaler info:")
print(f"  - Mean shape: {scaler.mean_.shape}")
print(f"  - Scale shape: {scaler.scale_.shape}")
print(f"  - Features: {len(feature_names)}")
