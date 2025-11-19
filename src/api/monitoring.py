"""
Monitoring and logging module for production ML systems.

Provides:
- Prediction logging with timestamps
- Feature statistics tracking
- Data drift detection (distribution shift)
- Performance metrics tracking
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class PredictionMonitor:
    """
    Monitor predictions and detect data drift.

    Tracks:
    - All predictions with features and results
    - Feature statistics over time
    - Distribution drift compared to training data
    - Performance metrics
    """

    def __init__(
        self,
        log_dir: Path,
        training_stats: Optional[Dict[str, Any]] = None,
        drift_threshold: float = 0.20,
        window_size: int = 1000
    ):
        """
        Initialize prediction monitor.

        Args:
            log_dir: Directory to save logs
            training_stats: Training data statistics (mean, std per feature)
            drift_threshold: Threshold for drift detection (default 20%)
            window_size: Sliding window size for recent predictions
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.training_stats = training_stats
        self.drift_threshold = drift_threshold
        self.window_size = window_size

        # Prediction history (sliding window)
        self.prediction_history = deque(maxlen=window_size)

        # Log files
        self.predictions_log = self.log_dir / 'predictions.log'
        self.drift_log = self.log_dir / 'drift.log'
        self.metrics_log = self.log_dir / 'metrics.log'

        # Performance tracking
        self.total_predictions = 0
        self.total_anomalies = 0
        self.total_inference_time = 0.0

        self.logger = logging.getLogger(__name__)

    def log_prediction(
        self,
        features: np.ndarray,
        prediction: int,
        anomaly_score: float,
        confidence: float,
        model_used: str,
        inference_time_ms: float,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a single prediction.

        Args:
            features: Input features
            prediction: Model prediction (0=normal, 1=anomaly)
            anomaly_score: Anomaly score
            confidence: Confidence score
            model_used: Model name
            inference_time_ms: Inference time
            feature_names: Feature names (optional)
            metadata: Additional metadata (optional)
        """
        timestamp = datetime.now().isoformat()

        # Build log entry
        log_entry = {
            'timestamp': timestamp,
            'prediction': int(prediction),
            'anomaly_score': float(anomaly_score),
            'confidence': float(confidence),
            'model_used': model_used,
            'inference_time_ms': float(inference_time_ms)
        }

        # Add features (optional - can be large)
        if feature_names:
            log_entry['features'] = {
                name: float(val) for name, val in zip(feature_names, features)
            }

        # Add metadata
        if metadata:
            log_entry['metadata'] = metadata

        # Write to log file
        with open(self.predictions_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Add to history
        self.prediction_history.append({
            'timestamp': timestamp,
            'features': features,
            'prediction': prediction,
            'anomaly_score': anomaly_score
        })

        # Update statistics
        self.total_predictions += 1
        if prediction == 1:
            self.total_anomalies += 1
        self.total_inference_time += inference_time_ms

        self.logger.info(
            f"Logged prediction: {prediction} (score={anomaly_score:.3f}, "
            f"conf={confidence:.3f}, model={model_used})"
        )

    def check_drift(
        self,
        current_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Check for data drift by comparing current features to training distribution.

        Args:
            current_features: Current batch of features (if None, uses recent history)

        Returns:
            Dictionary with drift information
        """
        if self.training_stats is None:
            return {
                'drift_detected': False,
                'message': 'No training statistics available'
            }

        # Get features to check
        if current_features is None:
            if len(self.prediction_history) == 0:
                return {
                    'drift_detected': False,
                    'message': 'No predictions logged yet'
                }
            # Use recent predictions
            current_features = np.array([
                p['features'] for p in self.prediction_history
            ])

        # Ensure 2D array
        if len(current_features.shape) == 1:
            current_features = current_features.reshape(1, -1)

        # Compute current statistics
        current_mean = np.mean(current_features, axis=0)
        current_std = np.std(current_features, axis=0)

        # Compare with training statistics
        training_mean = self.training_stats.get('mean', np.zeros_like(current_mean))
        training_std = self.training_stats.get('std', np.ones_like(current_std))

        # Compute relative difference in mean
        mean_diff = np.abs(current_mean - training_mean) / (training_mean + 1e-10)

        # Compute relative difference in std
        std_diff = np.abs(current_std - training_std) / (training_std + 1e-10)

        # Features with significant drift
        drift_features_mean = np.where(mean_diff > self.drift_threshold)[0]
        drift_features_std = np.where(std_diff > self.drift_threshold)[0]

        drift_detected = len(drift_features_mean) > 0 or len(drift_features_std) > 0

        drift_info = {
            'drift_detected': drift_detected,
            'timestamp': datetime.now().isoformat(),
            'n_samples_checked': len(current_features),
            'threshold': self.drift_threshold,
            'n_features_drift_mean': int(len(drift_features_mean)),
            'n_features_drift_std': int(len(drift_features_std)),
            'max_mean_drift': float(mean_diff.max()),
            'max_std_drift': float(std_diff.max()),
            'drift_features_mean': drift_features_mean.tolist(),
            'drift_features_std': drift_features_std.tolist()
        }

        if drift_detected:
            self.logger.warning(
                f"DRIFT DETECTED: {len(drift_features_mean)} features with mean drift, "
                f"{len(drift_features_std)} features with std drift"
            )

            # Log to drift log file
            with open(self.drift_log, 'a') as f:
                f.write(json.dumps(drift_info) + '\n')

        return drift_info

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current monitoring statistics.

        Returns:
            Dictionary with statistics
        """
        avg_inference_time = (
            self.total_inference_time / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )

        anomaly_rate = (
            self.total_anomalies / self.total_predictions * 100
            if self.total_predictions > 0
            else 0.0
        )

        # Recent anomaly rate (last N predictions)
        recent_anomalies = sum(
            1 for p in self.prediction_history if p['prediction'] == 1
        )
        recent_anomaly_rate = (
            recent_anomalies / len(self.prediction_history) * 100
            if len(self.prediction_history) > 0
            else 0.0
        )

        return {
            'total_predictions': self.total_predictions,
            'total_anomalies': self.total_anomalies,
            'anomaly_rate_percent': anomaly_rate,
            'recent_anomaly_rate_percent': recent_anomaly_rate,
            'avg_inference_time_ms': avg_inference_time,
            'total_inference_time_ms': self.total_inference_time,
            'history_window_size': len(self.prediction_history)
        }

    def export_logs(
        self,
        output_path: Path,
        format: str = 'csv'
    ):
        """
        Export prediction logs to file.

        Args:
            output_path: Output file path
            format: Export format ('csv' or 'json')
        """
        if not self.predictions_log.exists():
            self.logger.warning("No predictions logged yet")
            return

        # Read all log entries
        logs = []
        with open(self.predictions_log, 'r') as f:
            for line in f:
                logs.append(json.loads(line))

        if format == 'csv':
            # Convert to DataFrame and export
            df = pd.DataFrame(logs)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Exported {len(logs)} predictions to {output_path}")
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(logs, f, indent=2)
            self.logger.info(f"Exported {len(logs)} predictions to {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global monitor instance
_monitor: Optional[PredictionMonitor] = None


def initialize_monitor(
    log_dir: Path,
    training_stats: Optional[Dict] = None,
    drift_threshold: float = 0.20
) -> PredictionMonitor:
    """
    Initialize global prediction monitor.

    Args:
        log_dir: Directory to save logs
        training_stats: Training data statistics
        drift_threshold: Drift detection threshold

    Returns:
        PredictionMonitor instance
    """
    global _monitor
    _monitor = PredictionMonitor(
        log_dir=log_dir,
        training_stats=training_stats,
        drift_threshold=drift_threshold
    )
    return _monitor


def get_monitor() -> PredictionMonitor:
    """
    Get the global monitor instance.

    Returns:
        PredictionMonitor instance

    Raises:
        RuntimeError: If monitor not initialized
    """
    if _monitor is None:
        raise RuntimeError(
            "Monitor not initialized. Call initialize_monitor() first."
        )
    return _monitor


if __name__ == "__main__":
    # Test monitor
    print("Prediction Monitor Test")
    print("=" * 60)

    # Create test monitor
    log_dir = Path('./test_logs')
    monitor = PredictionMonitor(log_dir)

    # Log test predictions
    for i in range(10):
        features = np.random.randn(70)
        prediction = 1 if i % 3 == 0 else 0
        anomaly_score = 0.8 if prediction == 1 else 0.2

        monitor.log_prediction(
            features=features,
            prediction=prediction,
            anomaly_score=anomaly_score,
            confidence=0.9,
            model_used='ocsvm',
            inference_time_ms=2.5
        )

    # Get statistics
    stats = monitor.get_statistics()
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")

    print(f"\n✓ Logged {stats['total_predictions']} predictions")
    print(f"✓ Log file: {monitor.predictions_log}")
