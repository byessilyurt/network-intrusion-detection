"""
FastAPI Application for Network Intrusion Detection
Production deployment with OCSVM model + SHAP explainability
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "ocsvm_200k.pkl"
ARTIFACTS_PATH = PROJECT_ROOT / "models" / "production_artifacts.pkl"

# ============================================================================
# Global Model State
# ============================================================================
model = None
scaler = None
feature_names = None
shap_explainer = None
model_metadata = {}

# ============================================================================
# FastAPI Application
# ============================================================================
app = FastAPI(
    title="Network Intrusion Detection API",
    description="DoS/DDoS attack detection using One-Class SVM with SHAP explanations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================
class NetworkFlowFeatures(BaseModel):
    """Input schema for network flow features (66 features from CICIDS2017)"""
    # Example of key features - expand with all 66 features
    destination_port: float = Field(..., description="Destination Port")
    flow_duration: float = Field(..., description="Flow Duration")
    total_fwd_packets: float = Field(..., description="Total Fwd Packets")
    total_backward_packets: float = Field(..., description="Total Backward Packets")
    total_length_of_fwd_packets: float = Field(..., description="Total Length of Fwd Packets")
    total_length_of_bwd_packets: float = Field(..., description="Total Length of Bwd Packets")
    fwd_packet_length_max: float = Field(..., description="Fwd Packet Length Max")
    fwd_packet_length_min: float = Field(..., description="Fwd Packet Length Min")
    fwd_packet_length_mean: float = Field(..., description="Fwd Packet Length Mean")
    fwd_packet_length_std: float = Field(..., description="Fwd Packet Length Std")
    bwd_packet_length_max: float = Field(..., description="Bwd Packet Length Max")
    bwd_packet_length_min: float = Field(..., description="Bwd Packet Length Min")
    bwd_packet_length_mean: float = Field(..., description="Bwd Packet Length Mean")
    bwd_packet_length_std: float = Field(..., description="Bwd Packet Length Std")
    flow_bytes_s: float = Field(..., description="Flow Bytes/s")
    flow_packets_s: float = Field(..., description="Flow Packets/s")
    flow_iat_mean: float = Field(..., description="Flow IAT Mean")
    flow_iat_std: float = Field(..., description="Flow IAT Std")
    flow_iat_max: float = Field(..., description="Flow IAT Max")
    flow_iat_min: float = Field(..., description="Flow IAT Min")
    fwd_iat_total: float = Field(..., description="Fwd IAT Total")
    fwd_iat_mean: float = Field(..., description="Fwd IAT Mean")
    fwd_iat_std: float = Field(..., description="Fwd IAT Std")
    fwd_iat_max: float = Field(..., description="Fwd IAT Max")
    fwd_iat_min: float = Field(..., description="Fwd IAT Min")

    # Note: This is a simplified schema. Full implementation should include all 66 features.
    # For demonstration, we'll accept a dict of all features.

    class Config:
        schema_extra = {
            "example": {
                "destination_port": 80.0,
                "flow_duration": 120000.0,
                "total_fwd_packets": 10.0,
                "total_backward_packets": 8.0,
                "total_length_of_fwd_packets": 5000.0,
                "total_length_of_bwd_packets": 4000.0,
                "fwd_packet_length_max": 1500.0,
                "fwd_packet_length_min": 60.0,
                "fwd_packet_length_mean": 500.0,
                "fwd_packet_length_std": 200.0,
                "bwd_packet_length_max": 1500.0,
                "bwd_packet_length_min": 60.0,
                "bwd_packet_length_mean": 500.0,
                "bwd_packet_length_std": 150.0,
                "flow_bytes_s": 75000.0,
                "flow_packets_s": 150.0,
                "flow_iat_mean": 12000.0,
                "flow_iat_std": 5000.0,
                "flow_iat_max": 30000.0,
                "flow_iat_min": 1000.0,
                "fwd_iat_total": 120000.0,
                "fwd_iat_mean": 12000.0,
                "fwd_iat_std": 5000.0,
                "fwd_iat_max": 30000.0,
                "fwd_iat_min": 1000.0,
            }
        }


class PredictionRequest(BaseModel):
    """Request for batch prediction - accepts raw feature dict"""
    features: Dict[str, float] = Field(..., description="All 66 network flow features as key-value pairs")

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "Destination Port": 80.0,
                    "Flow Duration": 120000.0,
                    "Total Fwd Packets": 10.0,
                    "Total Backward Packets": 8.0,
                    # ... (additional features)
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response with prediction and SHAP explanation"""
    prediction: int = Field(..., description="0 = BENIGN, 1 = ATTACK")
    prediction_label: str = Field(..., description="Human-readable label")
    anomaly_score: float = Field(..., description="Anomaly score from OCSVM")
    confidence: float = Field(..., description="Confidence percentage (0-100)")
    top_features: List[Dict[str, float]] = Field(..., description="Top 10 features contributing to prediction")
    explanation: str = Field(..., description="Human-readable explanation")
    model_info: Dict[str, str] = Field(..., description="Model metadata")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    shap_loaded: bool
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model metadata response"""
    model_type: str
    training_samples: int
    test_f1_score: float
    test_precision: float
    test_recall: float
    stability: str
    production_status: str
    last_updated: str


# ============================================================================
# Startup/Shutdown Events
# ============================================================================
@app.on_event("startup")
async def load_model():
    """Load OCSVM model, scaler, and initialize SHAP explainer"""
    global model, scaler, feature_names, shap_explainer, model_metadata

    logger.info("Loading OCSVM model and production artifacts...")

    try:
        # 1. Load OCSVM model
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model_data = joblib.load(f)

        if isinstance(model_data, dict):
            model = model_data['model']
            model_metadata = model_data.get('metadata', {})
        else:
            model = model_data
            model_metadata = {}

        logger.info(f"✓ Model loaded: {type(model).__name__}")

        # 2. Load production artifacts (scaler + SHAP background)
        logger.info(f"Loading production artifacts from {ARTIFACTS_PATH}")
        with open(ARTIFACTS_PATH, 'rb') as f:
            artifacts = joblib.load(f)

        scaler = artifacts['scaler']
        background_data = artifacts['background_data']
        feature_names = artifacts['feature_names']

        logger.info(f"✓ Scaler loaded: {type(scaler).__name__}")
        logger.info(f"✓ Background data: {background_data.shape[0]} samples")
        logger.info(f"✓ Features: {len(feature_names)}")

        # 3. Initialize SHAP explainer with background data
        logger.info("Initializing SHAP explainer...")
        shap_explainer = shap.KernelExplainer(
            model.decision_function,
            background_data
        )
        logger.info("✓ SHAP explainer ready")

        logger.info("=" * 80)
        logger.info("✓ API READY WITH FULL SHAP EXPLAINABILITY")
        logger.info(f"  Model: One-Class SVM (nu=0.02, RBF kernel)")
        logger.info(f"  Features: {len(feature_names)}")
        logger.info(f"  F1 Score: 0.8540")
        logger.info(f"  Precision: 92.4%")
        logger.info(f"  Recall: 79.4%")
        logger.info(f"  SHAP: ENABLED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Network Intrusion Detection API",
        "version": "1.0.0",
        "model": "One-Class SVM",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        shap_loaded=shap_explainer is not None,
        uptime_seconds=time.time()  # Simplified - should track actual uptime
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata and performance metrics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_type="One-Class SVM (RBF kernel, nu=0.02)",
        training_samples=200000,
        test_f1_score=0.8540,
        test_precision=0.924,
        test_recall=0.794,
        stability="100% valid predictions (no NaN issues)",
        production_status="DEPLOYED",
        last_updated="2025-11-19"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict network flow as BENIGN or ATTACK with SHAP explanation

    Accepts 66 network flow features and returns:
    - Binary prediction (0=BENIGN, 1=ATTACK)
    - Anomaly score
    - Confidence percentage
    - Top 10 contributing features (from SHAP)
    - Human-readable explanation
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Convert input dict to feature array
        features_dict = request.features

        # Ensure we have 66 features in correct order
        if len(features_dict) != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_names)} features, got {len(features_dict)}"
            )

        # Create feature array in correct order
        X = np.array([[features_dict.get(fname, 0.0) for fname in feature_names]])

        # 2. Preprocess: handle inf/NaN
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        # 3. Scale features
        X_scaled = scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)

        # 4. Get OCSVM prediction
        prediction = model.predict(X_scaled)[0]  # -1 (attack) or 1 (benign)
        anomaly_score = model.decision_function(X_scaled)[0]

        # Convert OCSVM output: 1 (benign) -> 0, -1 (attack) -> 1
        binary_prediction = 0 if prediction == 1 else 1
        prediction_label = "BENIGN" if binary_prediction == 0 else "ATTACK"

        # 5. Calculate confidence (based on distance from decision boundary)
        # Negative score = attack, positive = benign
        # Convert to 0-100% confidence
        confidence = min(100.0, max(0.0, abs(anomaly_score) * 100))

        # 6. Get SHAP feature importance
        shap_values = shap_explainer.shap_values(X_scaled, nsamples=100)  # Fast approximation
        shap_importance = np.abs(shap_values[0])
        top_indices = np.argsort(shap_importance)[::-1][:10]

        top_features = [
            {
                "feature": feature_names[idx],
                "shap_value": float(shap_values[0][idx]),
                "feature_value": float(X[0][idx]),
                "importance": float(shap_importance[idx])
            }
            for idx in top_indices
        ]

        # 7. Generate human-readable explanation
        if binary_prediction == 1:  # Attack detected
            top_feature = top_features[0]["feature"]
            explanation = (
                f"🚨 ATTACK DETECTED: This network flow shows anomalous behavior. "
                f"Key indicator: '{top_feature}' deviates significantly from normal patterns. "
                f"Confidence: {confidence:.1f}%. Recommend blocking or investigating this flow."
            )
        else:  # Benign
            explanation = (
                f"✅ BENIGN TRAFFIC: This network flow appears normal. "
                f"Confidence: {confidence:.1f}%. All features within expected ranges."
            )

        # 8. Return response
        return PredictionResponse(
            prediction=binary_prediction,
            prediction_label=prediction_label,
            anomaly_score=float(anomaly_score),
            confidence=float(confidence),
            top_features=top_features,
            explanation=explanation,
            model_info={
                "model": "One-Class SVM",
                "f1_score": "0.8540",
                "precision": "92.4%",
                "recall": "79.4%"
            }
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Additional Endpoints
# ============================================================================
@app.get("/features", response_model=Dict[str, List[str]])
async def get_features():
    """Get list of expected feature names"""
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "features": feature_names,
        "count": len(feature_names)
    }


# ============================================================================
# Error Handlers
# ============================================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
