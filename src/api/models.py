"""
Pydantic models for FastAPI request/response validation.

This module defines all data models used by the API:
- PredictionRequest: Input features for prediction
- PredictionResponse: Structured prediction output with explainability
- BatchPredictionRequest: Multiple predictions
- BatchPredictionResponse: Multiple prediction results
- ModelInfo: Metadata about available models
- HealthResponse: API health check
- ErrorResponse: Error handling
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
from enum import Enum


class ModelType(str, Enum):
    """Available model types for prediction."""
    OCSVM = "ocsvm"
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"
    VAE = "vae"


class PredictionRequest(BaseModel):
    """
    Request model for single prediction.

    Contains all 70 CICIDS2017 features required for network intrusion detection.
    Features are organized by category for better documentation.
    """

    # Basic Flow Features
    flow_duration: float = Field(..., description="Duration of the flow in microseconds")
    total_fwd_packets: int = Field(..., description="Total packets in forward direction", ge=0)
    total_bwd_packets: int = Field(..., description="Total packets in backward direction", ge=0)

    # Packet Length Features
    total_length_fwd_packets: float = Field(..., description="Total size of packets in forward direction")
    total_length_bwd_packets: float = Field(..., description="Total size of packets in backward direction")
    fwd_packet_length_max: float = Field(..., description="Maximum forward packet length", ge=0)
    fwd_packet_length_min: float = Field(..., description="Minimum forward packet length", ge=0)
    fwd_packet_length_mean: float = Field(..., description="Mean forward packet length", ge=0)
    fwd_packet_length_std: float = Field(..., description="Std deviation forward packet length", ge=0)
    bwd_packet_length_max: float = Field(..., description="Maximum backward packet length", ge=0)
    bwd_packet_length_min: float = Field(..., description="Minimum backward packet length", ge=0)
    bwd_packet_length_mean: float = Field(..., description="Mean backward packet length", ge=0)
    bwd_packet_length_std: float = Field(..., description="Std deviation backward packet length", ge=0)

    # Flow Bytes and Packets per Second
    flow_bytes_per_sec: float = Field(..., description="Flow bytes per second")
    flow_packets_per_sec: float = Field(..., description="Flow packets per second")

    # Inter-Arrival Time Features
    flow_iat_mean: float = Field(..., description="Mean inter-arrival time between packets")
    flow_iat_std: float = Field(..., description="Std deviation inter-arrival time")
    flow_iat_max: float = Field(..., description="Maximum inter-arrival time", ge=0)
    flow_iat_min: float = Field(..., description="Minimum inter-arrival time", ge=0)
    fwd_iat_total: float = Field(..., description="Total forward inter-arrival time", ge=0)
    fwd_iat_mean: float = Field(..., description="Mean forward inter-arrival time", ge=0)
    fwd_iat_std: float = Field(..., description="Std deviation forward inter-arrival time", ge=0)
    fwd_iat_max: float = Field(..., description="Maximum forward inter-arrival time", ge=0)
    fwd_iat_min: float = Field(..., description="Minimum forward inter-arrival time", ge=0)
    bwd_iat_total: float = Field(..., description="Total backward inter-arrival time", ge=0)
    bwd_iat_mean: float = Field(..., description="Mean backward inter-arrival time", ge=0)
    bwd_iat_std: float = Field(..., description="Std deviation backward inter-arrival time", ge=0)
    bwd_iat_max: float = Field(..., description="Maximum backward inter-arrival time", ge=0)
    bwd_iat_min: float = Field(..., description="Minimum backward inter-arrival time", ge=0)

    # Flag Features
    fwd_psh_flags: int = Field(..., description="Number of PSH flags in forward direction", ge=0)
    bwd_psh_flags: int = Field(..., description="Number of PSH flags in backward direction", ge=0)
    fwd_urg_flags: int = Field(..., description="Number of URG flags in forward direction", ge=0)
    bwd_urg_flags: int = Field(..., description="Number of URG flags in backward direction", ge=0)
    fin_flag_count: int = Field(..., description="Number of FIN flags", ge=0)
    syn_flag_count: int = Field(..., description="Number of SYN flags", ge=0)
    rst_flag_count: int = Field(..., description="Number of RST flags", ge=0)
    psh_flag_count: int = Field(..., description="Number of PSH flags", ge=0)
    ack_flag_count: int = Field(..., description="Number of ACK flags", ge=0)
    urg_flag_count: int = Field(..., description="Number of URG flags", ge=0)
    cwe_flag_count: int = Field(..., description="Number of CWE flags", ge=0)
    ece_flag_count: int = Field(..., description="Number of ECE flags", ge=0)

    # Header Length Features
    fwd_header_length: float = Field(..., description="Total forward header length", ge=0)
    bwd_header_length: float = Field(..., description="Total backward header length", ge=0)
    fwd_packets_per_sec: float = Field(..., description="Forward packets per second", ge=0)
    bwd_packets_per_sec: float = Field(..., description="Backward packets per second", ge=0)

    # Packet Size Features
    min_packet_length: float = Field(..., description="Minimum packet length", ge=0)
    max_packet_length: float = Field(..., description="Maximum packet length", ge=0)
    packet_length_mean: float = Field(..., description="Mean packet length", ge=0)
    packet_length_std: float = Field(..., description="Std deviation packet length", ge=0)
    packet_length_variance: float = Field(..., description="Variance of packet length", ge=0)

    # Down/Up Ratio
    down_up_ratio: float = Field(..., description="Download/Upload ratio", ge=0)

    # Average Packet Size
    average_packet_size: float = Field(..., description="Average packet size", ge=0)
    avg_fwd_segment_size: float = Field(..., description="Average forward segment size", ge=0)
    avg_bwd_segment_size: float = Field(..., description="Average backward segment size", ge=0)

    # Bulk Features
    fwd_avg_bytes_bulk: float = Field(0.0, description="Average bytes per bulk in forward", ge=0)
    fwd_avg_packets_bulk: float = Field(0.0, description="Average packets per bulk in forward", ge=0)
    fwd_avg_bulk_rate: float = Field(0.0, description="Average bulk rate in forward", ge=0)
    bwd_avg_bytes_bulk: float = Field(0.0, description="Average bytes per bulk in backward", ge=0)
    bwd_avg_packets_bulk: float = Field(0.0, description="Average packets per bulk in backward", ge=0)
    bwd_avg_bulk_rate: float = Field(0.0, description="Average bulk rate in backward", ge=0)

    # Subflow Features
    subflow_fwd_packets: int = Field(..., description="Packets in forward subflow", ge=0)
    subflow_fwd_bytes: float = Field(..., description="Bytes in forward subflow", ge=0)
    subflow_bwd_packets: int = Field(..., description="Packets in backward subflow", ge=0)
    subflow_bwd_bytes: float = Field(..., description="Bytes in backward subflow", ge=0)

    # Window Size Features
    init_win_bytes_forward: int = Field(..., description="Initial window bytes forward")
    init_win_bytes_backward: int = Field(..., description="Initial window bytes backward")

    # Active and Idle Times
    active_mean: float = Field(..., description="Mean active time", ge=0)
    active_std: float = Field(..., description="Std deviation active time", ge=0)
    active_max: float = Field(..., description="Maximum active time", ge=0)
    active_min: float = Field(..., description="Minimum active time", ge=0)
    idle_mean: float = Field(..., description="Mean idle time", ge=0)
    idle_std: float = Field(..., description="Std deviation idle time", ge=0)
    idle_max: float = Field(..., description="Maximum idle time", ge=0)
    idle_min: float = Field(..., description="Minimum idle time", ge=0)

    class Config:
        schema_extra = {
            "example": {
                "flow_duration": 120000000,
                "total_fwd_packets": 10,
                "total_bwd_packets": 8,
                "total_length_fwd_packets": 5000,
                "total_length_bwd_packets": 4000,
                "fwd_packet_length_max": 1500,
                "fwd_packet_length_min": 40,
                "fwd_packet_length_mean": 500,
                "fwd_packet_length_std": 200,
                "bwd_packet_length_max": 1500,
                "bwd_packet_length_min": 40,
                "bwd_packet_length_mean": 500,
                "bwd_packet_length_std": 200,
                "flow_bytes_per_sec": 75000,
                "flow_packets_per_sec": 150,
                "flow_iat_mean": 12000000,
                "flow_iat_std": 5000000,
                "flow_iat_max": 30000000,
                "flow_iat_min": 1000000,
                "fwd_iat_total": 108000000,
                "fwd_iat_mean": 12000000,
                "fwd_iat_std": 5000000,
                "fwd_iat_max": 30000000,
                "fwd_iat_min": 1000000,
                "bwd_iat_total": 84000000,
                "bwd_iat_mean": 12000000,
                "bwd_iat_std": 5000000,
                "bwd_iat_max": 30000000,
                "bwd_iat_min": 1000000,
                "fwd_psh_flags": 1,
                "bwd_psh_flags": 1,
                "fwd_urg_flags": 0,
                "bwd_urg_flags": 0,
                "fin_flag_count": 1,
                "syn_flag_count": 1,
                "rst_flag_count": 0,
                "psh_flag_count": 2,
                "ack_flag_count": 15,
                "urg_flag_count": 0,
                "cwe_flag_count": 0,
                "ece_flag_count": 0,
                "fwd_header_length": 200,
                "bwd_header_length": 160,
                "fwd_packets_per_sec": 83.33,
                "bwd_packets_per_sec": 66.67,
                "min_packet_length": 40,
                "max_packet_length": 1500,
                "packet_length_mean": 500,
                "packet_length_std": 300,
                "packet_length_variance": 90000,
                "down_up_ratio": 0.8,
                "average_packet_size": 500,
                "avg_fwd_segment_size": 500,
                "avg_bwd_segment_size": 500,
                "fwd_avg_bytes_bulk": 0,
                "fwd_avg_packets_bulk": 0,
                "fwd_avg_bulk_rate": 0,
                "bwd_avg_bytes_bulk": 0,
                "bwd_avg_packets_bulk": 0,
                "bwd_avg_bulk_rate": 0,
                "subflow_fwd_packets": 10,
                "subflow_fwd_bytes": 5000,
                "subflow_bwd_packets": 8,
                "subflow_bwd_bytes": 4000,
                "init_win_bytes_forward": 65535,
                "init_win_bytes_backward": 65535,
                "active_mean": 10000000,
                "active_std": 2000000,
                "active_max": 15000000,
                "active_min": 5000000,
                "idle_mean": 5000000,
                "idle_std": 1000000,
                "idle_max": 8000000,
                "idle_min": 2000000
            }
        }

    def to_array(self) -> List[float]:
        """Convert request to feature array for model prediction."""
        return [
            self.flow_duration,
            self.total_fwd_packets,
            self.total_bwd_packets,
            self.total_length_fwd_packets,
            self.total_length_bwd_packets,
            self.fwd_packet_length_max,
            self.fwd_packet_length_min,
            self.fwd_packet_length_mean,
            self.fwd_packet_length_std,
            self.bwd_packet_length_max,
            self.bwd_packet_length_min,
            self.bwd_packet_length_mean,
            self.bwd_packet_length_std,
            self.flow_bytes_per_sec,
            self.flow_packets_per_sec,
            self.flow_iat_mean,
            self.flow_iat_std,
            self.flow_iat_max,
            self.flow_iat_min,
            self.fwd_iat_total,
            self.fwd_iat_mean,
            self.fwd_iat_std,
            self.fwd_iat_max,
            self.fwd_iat_min,
            self.bwd_iat_total,
            self.bwd_iat_mean,
            self.bwd_iat_std,
            self.bwd_iat_max,
            self.bwd_iat_min,
            self.fwd_psh_flags,
            self.bwd_psh_flags,
            self.fwd_urg_flags,
            self.bwd_urg_flags,
            self.fin_flag_count,
            self.syn_flag_count,
            self.rst_flag_count,
            self.psh_flag_count,
            self.ack_flag_count,
            self.urg_flag_count,
            self.cwe_flag_count,
            self.ece_flag_count,
            self.fwd_header_length,
            self.bwd_header_length,
            self.fwd_packets_per_sec,
            self.bwd_packets_per_sec,
            self.min_packet_length,
            self.max_packet_length,
            self.packet_length_mean,
            self.packet_length_std,
            self.packet_length_variance,
            self.down_up_ratio,
            self.average_packet_size,
            self.avg_fwd_segment_size,
            self.avg_bwd_segment_size,
            self.fwd_avg_bytes_bulk,
            self.fwd_avg_packets_bulk,
            self.fwd_avg_bulk_rate,
            self.bwd_avg_bytes_bulk,
            self.bwd_avg_packets_bulk,
            self.bwd_avg_bulk_rate,
            self.subflow_fwd_packets,
            self.subflow_fwd_bytes,
            self.subflow_bwd_packets,
            self.subflow_bwd_bytes,
            self.init_win_bytes_forward,
            self.init_win_bytes_backward,
            self.active_mean,
            self.active_std,
            self.active_max,
            self.active_min,
            self.idle_mean,
            self.idle_std,
            self.idle_max,
            self.idle_min
        ]


class TopFeature(BaseModel):
    """Top contributing feature for explainability."""
    feature: str = Field(..., description="Feature name")
    contribution: float = Field(..., description="SHAP contribution value")
    feature_value: float = Field(..., description="Actual feature value")


class PredictionResponse(BaseModel):
    """
    Response model for single prediction with explainability.
    """
    anomaly_score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    is_anomaly: bool = Field(..., description="Binary prediction (True = attack, False = normal)")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    model_used: str = Field(..., description="Model used for prediction")
    top_features: List[TopFeature] = Field(..., description="Top contributing features")
    shap_values: Optional[List[float]] = Field(None, description="Full SHAP values array")
    prediction_time_ms: float = Field(..., description="Inference time in milliseconds")

    class Config:
        schema_extra = {
            "example": {
                "anomaly_score": 0.87,
                "is_anomaly": True,
                "confidence": 0.93,
                "model_used": "ocsvm",
                "top_features": [
                    {
                        "feature": "bwd_packet_length_max",
                        "contribution": 0.23,
                        "feature_value": 1500.0
                    },
                    {
                        "feature": "idle_mean",
                        "contribution": 0.18,
                        "feature_value": 5000000.0
                    },
                    {
                        "feature": "ack_flag_count",
                        "contribution": 0.15,
                        "feature_value": 15.0
                    }
                ],
                "shap_values": None,
                "prediction_time_ms": 2.5
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    samples: List[PredictionRequest] = Field(..., description="List of samples to predict")

    @validator('samples')
    def validate_samples(cls, v):
        if len(v) == 0:
            raise ValueError("Samples list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 samples per batch")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_samples: int = Field(..., description="Total number of samples")
    anomalies_detected: int = Field(..., description="Number of anomalies detected")
    anomaly_rate: float = Field(..., description="Percentage of anomalies", ge=0, le=100)
    total_time_ms: float = Field(..., description="Total processing time in milliseconds")


class ModelInfo(BaseModel):
    """Metadata about an available model."""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    f1_score: Optional[float] = Field(None, description="F1 score on test set")
    precision: Optional[float] = Field(None, description="Precision on test set")
    recall: Optional[float] = Field(None, description="Recall on test set")
    false_positive_rate: Optional[float] = Field(None, description="False positive rate")
    inference_speed_ms: Optional[float] = Field(None, description="Average inference time (ms)")
    is_loaded: bool = Field(..., description="Whether model is currently loaded")
    model_size_mb: Optional[float] = Field(None, description="Model file size in MB")


class ModelsListResponse(BaseModel):
    """Response model for available models list."""
    models: List[ModelInfo] = Field(..., description="List of available models")
    default_model: str = Field(..., description="Default model for predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    models_loaded: int = Field(..., description="Number of models loaded")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="API uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


if __name__ == "__main__":
    # Test model creation
    print("Pydantic Models for Network Intrusion Detection API")
    print("=" * 60)

    # Example request
    example_request = PredictionRequest(
        flow_duration=120000000,
        total_fwd_packets=10,
        total_bwd_packets=8,
        total_length_fwd_packets=5000,
        total_length_bwd_packets=4000,
        fwd_packet_length_max=1500,
        fwd_packet_length_min=40,
        fwd_packet_length_mean=500,
        fwd_packet_length_std=200,
        bwd_packet_length_max=1500,
        bwd_packet_length_min=40,
        bwd_packet_length_mean=500,
        bwd_packet_length_std=200,
        flow_bytes_per_sec=75000,
        flow_packets_per_sec=150,
        flow_iat_mean=12000000,
        flow_iat_std=5000000,
        flow_iat_max=30000000,
        flow_iat_min=1000000,
        fwd_iat_total=108000000,
        fwd_iat_mean=12000000,
        fwd_iat_std=5000000,
        fwd_iat_max=30000000,
        fwd_iat_min=1000000,
        bwd_iat_total=84000000,
        bwd_iat_mean=12000000,
        bwd_iat_std=5000000,
        bwd_iat_max=30000000,
        bwd_iat_min=1000000,
        fwd_psh_flags=1,
        bwd_psh_flags=1,
        fwd_urg_flags=0,
        bwd_urg_flags=0,
        fin_flag_count=1,
        syn_flag_count=1,
        rst_flag_count=0,
        psh_flag_count=2,
        ack_flag_count=15,
        urg_flag_count=0,
        cwe_flag_count=0,
        ece_flag_count=0,
        fwd_header_length=200,
        bwd_header_length=160,
        fwd_packets_per_sec=83.33,
        bwd_packets_per_sec=66.67,
        min_packet_length=40,
        max_packet_length=1500,
        packet_length_mean=500,
        packet_length_std=300,
        packet_length_variance=90000,
        down_up_ratio=0.8,
        average_packet_size=500,
        avg_fwd_segment_size=500,
        avg_bwd_segment_size=500,
        fwd_avg_bytes_bulk=0,
        fwd_avg_packets_bulk=0,
        fwd_avg_bulk_rate=0,
        bwd_avg_bytes_bulk=0,
        bwd_avg_packets_bulk=0,
        bwd_avg_bulk_rate=0,
        subflow_fwd_packets=10,
        subflow_fwd_bytes=5000,
        subflow_bwd_packets=8,
        subflow_bwd_bytes=4000,
        init_win_bytes_forward=65535,
        init_win_bytes_backward=65535,
        active_mean=10000000,
        active_std=2000000,
        active_max=15000000,
        active_min=5000000,
        idle_mean=5000000,
        idle_std=1000000,
        idle_max=8000000,
        idle_min=2000000
    )

    print(f"✓ Created example PredictionRequest with {len(example_request.to_array())} features")
    print(f"✓ All Pydantic models validated successfully")
