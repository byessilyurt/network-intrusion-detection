"""
Model loading and management for FastAPI.

Handles loading, caching, and serving predictions from multiple trained models.
Models are loaded once at startup and cached in memory for fast inference.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import pickle
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.isolation_forest import IsolationForestDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.vae import VAEDetector


class ModelLoader:
    """
    Manages loading and caching of trained models.

    Models are loaded once at startup and kept in memory for fast inference.
    Supports multiple model types with automatic fallback if models not available.
    """

    def __init__(self, models_dir: Path):
        """
        Initialize model loader.

        Args:
            models_dir: Directory containing trained model files
        """
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.load_time = None

    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all available models from disk.

        Returns:
            Dictionary mapping model names to load status (True=success, False=fail)
        """
        start_time = time.time()
        load_status = {}

        # Model configurations
        model_configs = {
            'ocsvm': {
                'class': OneClassSVMDetector,
                'file': 'ocsvm_real.pkl',
                'type': 'ocsvm'
            },
            'isolation_forest': {
                'class': IsolationForestDetector,
                'file': 'isolation_forest_real.pkl',
                'type': 'isolation_forest'
            },
            'autoencoder': {
                'class': AutoencoderDetector,
                'file': 'autoencoder_real.pkl',
                'type': 'autoencoder'
            },
            'vae': {
                'class': VAEDetector,
                'file': 'vae_real.pkl',
                'type': 'vae'
            }
        }

        # Load each model
        for model_name, config in model_configs.items():
            try:
                model_path = self.models_dir / config['file']

                if not model_path.exists():
                    print(f"⚠ Model file not found: {model_path}")
                    load_status[model_name] = False
                    continue

                # Initialize detector
                detector = config['class']()

                # Load model
                detector.load(model_path)

                # Store in cache
                self.models[model_name] = detector
                self.model_metadata[model_name] = {
                    'type': config['type'],
                    'file': config['file'],
                    'size_mb': os.path.getsize(model_path) / (1024 * 1024)
                }

                load_status[model_name] = True
                print(f"✓ Loaded {model_name} from {config['file']}")

            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
                load_status[model_name] = False

        self.load_time = time.time() - start_time
        print(f"\n✓ Loaded {len(self.models)}/4 models in {self.load_time:.2f}s")

        return load_status

    def get_model(self, model_name: str) -> Optional[Any]:
        """
        Get a loaded model by name.

        Args:
            model_name: Name of model ('ocsvm', 'isolation_forest', 'autoencoder', 'vae')

        Returns:
            Model instance or None if not loaded
        """
        return self.models.get(model_name)

    def get_default_model(self) -> Optional[Any]:
        """
        Get the default model (OCSVM if available, else first loaded model).

        Returns:
            Default model instance or None if no models loaded
        """
        # Prefer OCSVM (best performer)
        if 'ocsvm' in self.models:
            return self.models['ocsvm']

        # Fallback to any loaded model
        if self.models:
            return list(self.models.values())[0]

        return None

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self.models

    def get_loaded_models(self) -> list:
        """Get list of loaded model names."""
        return list(self.models.keys())

    def get_model_info(self) -> Dict[str, Dict]:
        """
        Get information about all loaded models.

        Returns:
            Dictionary with model metadata
        """
        info = {}
        for model_name in self.models.keys():
            info[model_name] = {
                **self.model_metadata.get(model_name, {}),
                'is_loaded': True
            }
        return info

    def predict(
        self,
        model_name: str,
        X: np.ndarray
    ) -> Dict[str, Any]:
        """
        Make prediction using specified model.

        Args:
            model_name: Name of model to use
            X: Input features (single sample or batch)

        Returns:
            Dictionary with prediction results

        Raises:
            ValueError: If model not loaded
        """
        if model_name not in self.models:
            available = ', '.join(self.models.keys()) if self.models else 'none'
            raise ValueError(
                f"Model '{model_name}' not loaded. Available models: {available}"
            )

        model = self.models[model_name]

        # Ensure 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Measure inference time
        start_time = time.time()

        # Get prediction
        y_pred = model.predict(X)

        # Get anomaly score
        if hasattr(model, 'decision_function'):
            anomaly_score = model.decision_function(X)
        else:
            anomaly_score = model.predict_proba(X)[:, 1]

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Compute confidence (simple heuristic based on anomaly score)
        # For decision_function: higher absolute value = more confident
        # For predict_proba: value itself is confidence
        if hasattr(model, 'decision_function'):
            confidence = min(abs(anomaly_score[0]) / 2.0, 1.0)  # Normalize
        else:
            confidence = anomaly_score[0]

        return {
            'prediction': int(y_pred[0]),
            'anomaly_score': float(anomaly_score[0]),
            'confidence': float(confidence),
            'inference_time_ms': inference_time,
            'model_used': model_name
        }


# Global model loader instance (initialized by FastAPI on startup)
_model_loader: Optional[ModelLoader] = None


def initialize_models(models_dir: Path) -> ModelLoader:
    """
    Initialize global model loader.

    Args:
        models_dir: Directory containing trained models

    Returns:
        ModelLoader instance
    """
    global _model_loader
    _model_loader = ModelLoader(models_dir)
    _model_loader.load_all_models()
    return _model_loader


def get_model_loader() -> ModelLoader:
    """
    Get the global model loader instance.

    Returns:
        ModelLoader instance

    Raises:
        RuntimeError: If models not initialized
    """
    if _model_loader is None:
        raise RuntimeError(
            "Models not initialized. Call initialize_models() first."
        )
    return _model_loader


if __name__ == "__main__":
    # Test model loader
    print("Model Loader Test")
    print("=" * 60)

    models_dir = PROJECT_ROOT / 'models'
    loader = ModelLoader(models_dir)

    # Load models
    status = loader.load_all_models()
    print(f"\nLoad status: {status}")

    # Test prediction
    if loader.get_default_model():
        print("\nTesting prediction...")
        # Create dummy input (70 features)
        X_test = np.random.randn(1, 70)

        default_model_name = 'ocsvm' if 'ocsvm' in loader.models else loader.get_loaded_models()[0]
        result = loader.predict(default_model_name, X_test)
        print(f"Prediction result: {result}")
    else:
        print("\n⚠ No models loaded for testing")
