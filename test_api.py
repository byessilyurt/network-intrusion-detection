"""
Test script for FastAPI Network Intrusion Detection API
"""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
API_URL = "http://localhost:8000"
DATA_DIR = Path("data/raw")
WEDNESDAY_FILE = DATA_DIR / "Wednesday-workingHours.pcap_ISCX.csv"

# ============================================================================
# Test Functions
# ============================================================================

def test_root():
    """Test root endpoint"""
    print("\n" + "=" * 80)
    print("TEST 1: Root Endpoint (GET /)")
    print("=" * 80)

    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    print("✓ Root endpoint working")


def test_health():
    """Test health check endpoint"""
    print("\n" + "=" * 80)
    print("TEST 2: Health Check (GET /health)")
    print("=" * 80)

    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] == True
    assert data["scaler_loaded"] == True
    assert data["shap_loaded"] == True
    print("✓ Health check passed - all components loaded")


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "=" * 80)
    print("TEST 3: Model Info (GET /model/info)")
    print("=" * 80)

    response = requests.get(f"{API_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    data = response.json()
    assert data["test_f1_score"] == 0.8540
    assert data["production_status"] == "DEPLOYED"
    print("✓ Model info retrieved successfully")


def test_features():
    """Test features endpoint"""
    print("\n" + "=" * 80)
    print("TEST 4: Feature Names (GET /features)")
    print("=" * 80)

    response = requests.get(f"{API_URL}/features")
    print(f"Status Code: {response.status_code}")

    data = response.json()
    print(f"Feature Count: {data['count']}")
    print(f"First 10 Features: {data['features'][:10]}")

    assert response.status_code == 200
    assert data["count"] == 66
    print("✓ Feature list retrieved successfully")

    return data["features"]


def test_predict_benign(feature_names):
    """Test prediction on a benign sample"""
    print("\n" + "=" * 80)
    print("TEST 5: Predict Benign Traffic (POST /predict)")
    print("=" * 80)

    # Load a real benign sample from Wednesday data
    df = pd.read_csv(WEDNESDAY_FILE)
    benign_df = df[df[' Label'] == 'BENIGN'].copy()

    # Drop metadata columns
    metadata_cols = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp', ' Label']
    for col in metadata_cols:
        if col in benign_df.columns:
            benign_df = benign_df.drop(col, axis=1)

    # Take first sample
    sample = benign_df.iloc[0]

    # Create feature dict
    features_dict = {fname: float(sample.iloc[i]) for i, fname in enumerate(feature_names)}

    # Handle NaN/inf
    for key, value in features_dict.items():
        if np.isnan(value) or np.isinf(value):
            features_dict[key] = 0.0

    # Make request
    payload = {"features": features_dict}
    response = requests.post(f"{API_URL}/predict", json=payload)

    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Prediction: {data['prediction_label']}")
    print(f"Anomaly Score: {data['anomaly_score']:.6f}")
    print(f"Confidence: {data['confidence']:.2f}%")
    print(f"Explanation: {data['explanation']}")
    print(f"\nTop 5 Contributing Features:")
    for i, feat in enumerate(data['top_features'][:5], 1):
        print(f"  {i}. {feat['feature']}: SHAP={feat['shap_value']:.6f}, Value={feat['feature_value']:.2f}")

    assert response.status_code == 200
    print(f"\n✓ Benign prediction completed (predicted: {data['prediction_label']})")

    return data


def test_predict_attack(feature_names):
    """Test prediction on a DoS attack sample"""
    print("\n" + "=" * 80)
    print("TEST 6: Predict DoS Attack (POST /predict)")
    print("=" * 80)

    # Load a real DoS sample from Wednesday data
    df = pd.read_csv(WEDNESDAY_FILE)
    dos_df = df[df[' Label'].str.contains('DoS', na=False)].copy()

    # Drop metadata columns
    metadata_cols = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp', ' Label']
    for col in metadata_cols:
        if col in dos_df.columns:
            dos_df = dos_df.drop(col, axis=1)

    # Take first DoS sample
    sample = dos_df.iloc[0]

    # Create feature dict
    features_dict = {fname: float(sample.iloc[i]) for i, fname in enumerate(feature_names)}

    # Handle NaN/inf
    for key, value in features_dict.items():
        if np.isnan(value) or np.isinf(value):
            features_dict[key] = 0.0

    # Make request
    payload = {"features": features_dict}
    response = requests.post(f"{API_URL}/predict", json=payload)

    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Prediction: {data['prediction_label']}")
    print(f"Anomaly Score: {data['anomaly_score']:.6f}")
    print(f"Confidence: {data['confidence']:.2f}%")
    print(f"Explanation: {data['explanation']}")
    print(f"\nTop 5 Contributing Features:")
    for i, feat in enumerate(data['top_features'][:5], 1):
        print(f"  {i}. {feat['feature']}: SHAP={feat['shap_value']:.6f}, Value={feat['feature_value']:.2f}")

    assert response.status_code == 200
    print(f"\n✓ DoS attack prediction completed (predicted: {data['prediction_label']})")

    return data


def test_batch_prediction(feature_names):
    """Test batch prediction performance"""
    print("\n" + "=" * 80)
    print("TEST 7: Batch Prediction Performance")
    print("=" * 80)

    # Load samples
    df = pd.read_csv(WEDNESDAY_FILE)

    # Get 10 benign and 10 DoS samples
    benign_samples = df[df[' Label'] == 'BENIGN'].head(10).copy()
    dos_samples = df[df[' Label'].str.contains('DoS', na=False)].head(10).copy()

    # Drop metadata
    metadata_cols = ['Flow ID', ' Source IP', ' Destination IP', ' Timestamp', ' Label']
    for col in metadata_cols:
        if col in benign_samples.columns:
            benign_samples = benign_samples.drop(col, axis=1)
            dos_samples = dos_samples.drop(col, axis=1)

    # Test inference speed
    import time

    total_requests = 0
    total_time = 0.0

    print("\nTesting 20 predictions (10 benign + 10 attack)...")

    for idx, sample in enumerate(benign_samples.itertuples(index=False)):
        features_dict = {fname: float(getattr(sample, str(i))) for i, fname in enumerate(feature_names)}
        features_dict = {k: (0.0 if np.isnan(v) or np.isinf(v) else v) for k, v in features_dict.items()}

        start = time.time()
        response = requests.post(f"{API_URL}/predict", json={"features": features_dict})
        elapsed = time.time() - start

        total_requests += 1
        total_time += elapsed

        if response.status_code == 200:
            print(f"  ✓ Benign #{idx+1}: {response.json()['prediction_label']} ({elapsed*1000:.2f} ms)")

    for idx, sample in enumerate(dos_samples.itertuples(index=False)):
        features_dict = {fname: float(getattr(sample, str(i))) for i, fname in enumerate(feature_names)}
        features_dict = {k: (0.0 if np.isnan(v) or np.isinf(v) else v) for k, v in features_dict.items()}

        start = time.time()
        response = requests.post(f"{API_URL}/predict", json={"features": features_dict})
        elapsed = time.time() - start

        total_requests += 1
        total_time += elapsed

        if response.status_code == 200:
            print(f"  ✓ DoS #{idx+1}: {response.json()['prediction_label']} ({elapsed*1000:.2f} ms)")

    avg_latency = (total_time / total_requests) * 1000
    throughput = total_requests / total_time

    print(f"\n📊 Performance Metrics:")
    print(f"  Total Requests: {total_requests}")
    print(f"  Total Time: {total_time:.2f} seconds")
    print(f"  Average Latency: {avg_latency:.2f} ms/request")
    print(f"  Throughput: {throughput:.2f} requests/second")

    print("\n✓ Batch prediction test completed")


# ============================================================================
# Main Test Runner
# ============================================================================
def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 NETWORK INTRUSION DETECTION API - TEST SUITE")
    print("=" * 80)
    print(f"API URL: {API_URL}")
    print(f"Test Data: {WEDNESDAY_FILE}")

    try:
        # Basic endpoint tests
        test_root()
        test_health()
        test_model_info()
        feature_names = test_features()

        # Prediction tests
        test_predict_benign(feature_names)
        test_predict_attack(feature_names)
        test_batch_prediction(feature_names)

        # Summary
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("API is production-ready for deployment!")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Please start the API server first:")
        print("  cd src/api && python app.py")
        print("  or: uvicorn src.api.app:app --reload")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
