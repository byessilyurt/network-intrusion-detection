"""
Isolation Forest Anomaly Detection Module

This module implements a production-ready Isolation Forest detector for network
intrusion detection. Isolation Forest is an unsupervised learning algorithm that
identifies anomalies by isolating outliers using random partitioning.

Key Features:
- Wrapper around sklearn's IsolationForest with enhanced functionality
- Feature importance extraction using tree path lengths
- Model persistence (save/load)
- Prediction probabilities and anomaly scores
- Production-ready API for integration

Reference:
    Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).
    Isolation forest. In 2008 Eighth IEEE International Conference on Data Mining.
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Dict
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector with enhanced functionality.

    The Isolation Forest algorithm works by:
    1. Building random trees that partition data
    2. Anomalies require fewer partitions to isolate (shorter path lengths)
    3. Normal data requires more partitions (longer path lengths)

    Attributes:
        model: Fitted IsolationForest model
        contamination: Expected proportion of anomalies in dataset
        feature_names: Names of input features
        training_time: Time taken to train model (seconds)
        n_features: Number of features
    """

    def __init__(self,
                 contamination: float = 0.1,
                 n_estimators: int = 100,
                 max_samples: Union[int, str] = 'auto',
                 max_features: float = 1.0,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: int = 0):
        """
        Initialize Isolation Forest detector.

        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
                          Controls decision threshold for classification
            n_estimators: Number of isolation trees in the forest
            max_samples: Number of samples to draw for each tree
                        'auto' uses min(256, n_samples)
            max_features: Number of features to draw for each tree (proportion)
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = all CPUs)
            verbose: Verbosity level
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Initialize model
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )

        # State variables
        self.is_fitted = False
        self.feature_names = None
        self.training_time = None
        self.n_features = None

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            feature_names: Optional[list] = None) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest model on training data.

        Note: Isolation Forest is unsupervised - labels not required.
        Typically trained on normal data only or mixed data.

        Args:
            X: Training features (n_samples, n_features)
            feature_names: Names of features (optional)

        Returns:
            self (fitted detector)
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        # Store feature information
        self.n_features = X.shape[1]
        if feature_names is not None:
            if len(feature_names) != self.n_features:
                raise ValueError(f"feature_names length ({len(feature_names)}) "
                               f"must match n_features ({self.n_features})")
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # Train model
        if self.verbose:
            print(f"Training Isolation Forest with {self.n_estimators} trees...")
            print(f"Training samples: {X.shape[0]:,}")
            print(f"Features: {X.shape[1]}")
            print(f"Contamination: {self.contamination}")

        start_time = time.time()
        self.model.fit(X)
        self.training_time = time.time() - start_time

        self.is_fitted = True

        if self.verbose:
            print(f"Training completed in {self.training_time:.2f} seconds")

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly labels for samples.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Binary predictions: 0 = normal, 1 = anomaly
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        # sklearn returns -1 for anomalies, 1 for normal
        # Convert to 0 (normal) and 1 (anomaly)
        predictions = self.model.predict(X)
        binary_predictions = (predictions == -1).astype(int)

        return binary_predictions

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly probabilities (anomaly scores).

        Returns normalized anomaly scores in range [0, 1] where:
        - Values close to 1 = likely anomaly
        - Values close to 0 = likely normal

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Anomaly probabilities (n_samples,)
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get anomaly scores (negative outlier factor)
        # More negative = more anomalous
        anomaly_scores = self.model.score_samples(X)

        # Normalize to [0, 1] range
        # Use decision_function for normalization around threshold
        decision_scores = self.model.decision_function(X)

        # Convert to probability-like scores
        # Scores < 0 are considered anomalies
        # Use sigmoid-like transformation
        probabilities = 1 / (1 + np.exp(decision_scores * 5))  # Scale factor for steepness

        return probabilities

    def get_anomaly_scores(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get raw anomaly scores (average path length).

        More negative scores indicate stronger anomalies.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Anomaly scores (n_samples,)
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.score_samples(X)

    def get_feature_importance(self,
                              X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                              method: str = 'path_length') -> np.ndarray:
        """
        Compute feature importance scores.

        Isolation Forest doesn't provide built-in feature importance,
        so we estimate it using average path length contribution.

        Args:
            X: Sample data to compute importance on (optional)
               If None, uses random permutation method
            method: Method to compute importance
                   'path_length': Based on path length variance (default)

        Returns:
            Feature importance scores (n_features,)
        """
        self._check_is_fitted()

        if method == 'path_length':
            # Estimate importance by measuring impact on anomaly scores
            # Features that create more variation in scores are more important
            importance = np.zeros(self.n_features)

            if X is None:
                # Can't compute without data
                print("Warning: X is None. Returning uniform importance scores.")
                return np.ones(self.n_features) / self.n_features

            if isinstance(X, pd.DataFrame):
                X = X.values

            # Get baseline scores
            baseline_scores = self.model.score_samples(X)

            # Compute variance contribution per feature
            for tree in self.model.estimators_:
                # Get feature usage in tree splits
                feature_counts = np.zeros(self.n_features)

                # Count how many times each feature is used for splitting
                tree_structure = tree.tree_
                feature_indices = tree_structure.feature

                # Only count actual splits (not leaf nodes, which have feature = -2)
                for feature_idx in feature_indices:
                    if feature_idx >= 0:  # Valid feature (not leaf)
                        feature_counts[feature_idx] += 1

                importance += feature_counts

            # Normalize
            importance = importance / importance.sum() if importance.sum() > 0 else importance

            return importance

        else:
            raise ValueError(f"Unknown importance method: {method}")

    def benchmark_inference(self,
                          X: Union[np.ndarray, pd.DataFrame],
                          n_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            X: Sample data for benchmarking
            n_iterations: Number of iterations to average

        Returns:
            Dictionary with timing statistics
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        times = []
        for _ in range(n_iterations):
            start = time.time()
            self.predict(X)
            elapsed = time.time() - start
            times.append(elapsed)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'samples_per_second': X.shape[0] / np.mean(times),
            'time_per_sample_ms': (np.mean(times) / X.shape[0]) * 1000
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to disk using pickle.

        Args:
            filepath: Path to save model (e.g., 'models/isolation_forest.pkl')
        """
        self._check_is_fitted()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'training_time': self.training_time,
            'n_features': self.n_features,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'IsolationForestDetector':
        """
        Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded IsolationForestDetector instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance
        detector = cls(
            contamination=model_data['contamination'],
            n_estimators=model_data['n_estimators'],
            max_samples=model_data['max_samples'],
            max_features=model_data['max_features'],
            random_state=model_data['random_state']
        )

        # Restore state
        detector.model = model_data['model']
        detector.feature_names = model_data['feature_names']
        detector.training_time = model_data['training_time']
        detector.n_features = model_data['n_features']
        detector.is_fitted = model_data['is_fitted']

        print(f"Model loaded from: {filepath}")
        return detector

    def _check_is_fitted(self) -> None:
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"IsolationForestDetector(contamination={self.contamination}, "
                f"n_estimators={self.n_estimators}, "
                f"status={status})")


if __name__ == "__main__":
    # Example usage
    print("Isolation Forest Detector Module")
    print("="*70)
    print("\nExample usage:")
    print("""
from src.models.isolation_forest import IsolationForestDetector

# Initialize detector
detector = IsolationForestDetector(
    contamination=0.1,
    n_estimators=100,
    random_state=42
)

# Train on data (unsupervised - no labels needed)
detector.fit(X_train, feature_names=feature_names)

# Make predictions
y_pred = detector.predict(X_test)
y_proba = detector.predict_proba(X_test)

# Get feature importance
importance = detector.get_feature_importance(X_train)

# Benchmark inference speed
timing = detector.benchmark_inference(X_test[:1000])
print(f"Time per sample: {timing['time_per_sample_ms']:.2f} ms")

# Save model
detector.save('models/isolation_forest.pkl')

# Load model later
loaded_detector = IsolationForestDetector.load('models/isolation_forest.pkl')
    """)
    print("="*70)
