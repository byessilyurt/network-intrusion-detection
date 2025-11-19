"""
Quick test to verify API loads correctly with production artifacts
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("TESTING API STARTUP WITH PRODUCTION ARTIFACTS")
print("=" * 80)
print()

try:
    # Import the app (this triggers startup)
    print("Step 1: Importing FastAPI app...")
    from api.app import app, model, scaler, feature_names, shap_explainer

    # Check all components loaded
    print("✓ App imported successfully")
    print()

    print("Step 2: Checking model...")
    assert model is not None, "Model not loaded!"
    print(f"✓ Model loaded: {type(model).__name__}")
    print()

    print("Step 3: Checking scaler...")
    assert scaler is not None, "Scaler not loaded!"
    print(f"✓ Scaler loaded: {type(scaler).__name__}")
    print()

    print("Step 4: Checking feature names...")
    assert feature_names is not None, "Feature names not loaded!"
    print(f"✓ Feature names: {len(feature_names)} features")
    print(f"  First 5: {feature_names[:5]}")
    print()

    print("Step 5: Checking SHAP explainer...")
    assert shap_explainer is not None, "SHAP explainer not loaded!"
    print(f"✓ SHAP explainer loaded: {type(shap_explainer).__name__}")
    print()

    print("=" * 80)
    print("✓ ALL COMPONENTS LOADED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("API is ready to start with:")
    print("  uvicorn src.api.app:app --reload")
    print()

except Exception as e:
    print()
    print("=" * 80)
    print("✗ ERROR DURING STARTUP")
    print("=" * 80)
    print(f"\nError: {e}")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)
