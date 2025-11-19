"""
One-Class SVM Anomaly Detection Module

This module implements a production-ready One-Class SVM detector for network
intrusion detection. One-Class SVM learns a boundary around normal data in
feature space and identifies points outside this boundary as anomalies.

Key Features:
- Wrapper around sklearn's OneClassSVM with enhanced functionality
- Support for RBF and linear kernels
- Model persistence (save/load)
- Prediction probabilities and decision scores
- Support vector analysis
- Production-ready API for integration

Reference:
    Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001).
    Estimating the support of a high-dimensional distribution. Neural computation, 13(7).
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from typing import Optional, Union, Dict
from sklearn.svm import OneClassSVM


class OneClassSVMDetector:
    """
    One-Class SVM based anomaly detector with enhanced functionality.

    The One-Class SVM algorithm works by:
    1. Learning a decision boundary around normal data in feature space
    2. Using kernel tricks to handle non-linear boundaries (RBF kernel)
    3. Points outside the boundary are classified as anomalies
    4. Support vectors define the boundary

    Attributes:
        model: Fitted OneClassSVM model
        nu: Upper bound on fraction of outliers (similar to contamination)
        kernel: Kernel type ('rbf' or 'linear')
        gamma: Kernel coefficient for 'rbf'
        feature_names: Names of input features
        training_time: Time taken to train model (seconds)
        n_features: Number of features
        n_support_vectors: Number of support vectors
    """

    def __init__(self,
                 nu: float = 0.1,
                 kernel: str = 'rbf',
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 tol: float = 1e-3,
                 shrinking: bool = True,
                 cache_size: int = 200,
                 max_iter: int = -1,
                 verbose: bool = False):
        """
        Initialize One-Class SVM detector.

        Args:
            nu: Upper bound on fraction of training errors and lower bound
                on fraction of support vectors (0 < nu <= 1).
                Similar to contamination in Isolation Forest.
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
                'rbf': Radial basis function (most common, handles non-linearity)
                'linear': Linear kernel (faster, simpler decision boundary)
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
                'scale': 1 / (n_features * X.var())
                'auto': 1 / n_features
                float: Custom gamma value
            degree: Degree for polynomial kernel (ignored for rbf/linear)
            coef0: Independent term in kernel function
            tol: Tolerance for stopping criterion
            shrinking: Whether to use shrinking heuristic
            cache_size: Kernel cache size in MB
            max_iter: Maximum iterations (-1 = no limit)
            verbose: Enable verbose output
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.verbose = verbose

        # Initialize model
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            verbose=verbose
        )

        # State variables
        self.is_fitted = False
        self.feature_names = None
        self.training_time = None
        self.n_features = None
        self.n_support_vectors = None
        self.support_vectors = None

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            feature_names: Optional[list] = None) -> 'OneClassSVMDetector':
        """
        Fit the One-Class SVM model on training data.

        Note: One-Class SVM is unsupervised - labels not required.
        Typically trained on normal data only.

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
            print(f"Training One-Class SVM with {self.kernel} kernel...")
            print(f"Training samples: {X.shape[0]:,}")
            print(f"Features: {X.shape[1]}")
            print(f"Nu parameter: {self.nu}")
            if self.kernel == 'rbf':
                print(f"Gamma: {self.gamma}")

        start_time = time.time()
        self.model.fit(X)
        self.training_time = time.time() - start_time

        # Store support vector information
        self.n_support_vectors = len(self.model.support_)
        self.support_vectors = self.model.support_vectors_

        self.is_fitted = True

        if self.verbose:
            print(f"Training completed in {self.training_time:.2f} seconds")
            print(f"Number of support vectors: {self.n_support_vectors:,} "
                  f"({self.n_support_vectors/X.shape[0]*100:.1f}% of training data)")

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
        Predict anomaly probabilities (decision scores).

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

        # Get decision function values
        # Positive = normal (inside boundary)
        # Negative = anomaly (outside boundary)
        decision_scores = self.model.decision_function(X)

        # Convert to probability-like scores [0, 1]
        # Use sigmoid transformation
        # Scale factor chosen to spread probabilities
        probabilities = 1 / (1 + np.exp(decision_scores * 2))

        return probabilities

    def get_decision_scores(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get raw decision function scores.

        Positive scores indicate normal points (inside boundary).
        Negative scores indicate anomalies (outside boundary).
        Distance from 0 indicates confidence.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Decision scores (n_samples,)
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.decision_function(X)

    def get_support_vector_info(self) -> Dict:
        """
        Get information about support vectors.

        Returns:
            Dictionary with support vector statistics
        """
        self._check_is_fitted()

        return {
            'n_support_vectors': self.n_support_vectors,
            'support_vector_indices': self.model.support_,
            'support_vectors': self.support_vectors,
            'dual_coef': self.model.dual_coef_,
            'intercept': self.model.intercept_[0]
        }

    def get_feature_importance(self,
                              X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                              method: str = 'weight') -> np.ndarray:
        """
        Compute feature importance scores.

        Note: Feature importance for One-Class SVM:
        - Linear kernel: Can extract feature weights directly
        - RBF kernel: Use permutation importance (requires data)

        Args:
            X: Sample data for permutation importance (required for RBF kernel)
            method: Method to compute importance
                   'weight': For linear kernel (coefficient magnitude)
                   'permutation': Permutation-based importance

        Returns:
            Feature importance scores (n_features,)
        """
        self._check_is_fitted()

        if self.kernel == 'linear' and method == 'weight':
            # Linear kernel: extract feature weights
            # w = sum(alpha_i * support_vector_i)
            weights = np.abs(self.model.coef_[0])
            # Normalize to [0, 1]
            importance = weights / weights.sum() if weights.sum() > 0 else weights
            return importance

        elif method == 'permutation':
            # Permutation importance (works for any kernel)
            if X is None:
                raise ValueError("X required for permutation importance")

            if isinstance(X, pd.DataFrame):
                X = X.values

            # Get baseline scores
            baseline_scores = self.model.decision_function(X)
            baseline_variance = np.var(baseline_scores)

            importance = np.zeros(self.n_features)

            # Permute each feature and measure impact
            for i in range(self.n_features):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_scores = self.model.decision_function(X_permuted)
                # Measure change in variance
                importance[i] = abs(baseline_variance - np.var(permuted_scores))

            # Normalize
            importance = importance / importance.sum() if importance.sum() > 0 else importance
            return importance

        else:
            raise ValueError(f"Method '{method}' not supported for kernel '{self.kernel}'")

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
            filepath: Path to save model (e.g., 'models/ocsvm.pkl')
        """
        self._check_is_fitted()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'nu': self.nu,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'feature_names': self.feature_names,
            'training_time': self.training_time,
            'n_features': self.n_features,
            'n_support_vectors': self.n_support_vectors,
            'support_vectors': self.support_vectors,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'OneClassSVMDetector':
        """
        Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded OneClassSVMDetector instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance
        detector = cls(
            nu=model_data['nu'],
            kernel=model_data['kernel'],
            gamma=model_data['gamma'],
            degree=model_data['degree'],
            coef0=model_data['coef0']
        )

        # Restore state
        detector.model = model_data['model']
        detector.feature_names = model_data['feature_names']
        detector.training_time = model_data['training_time']
        detector.n_features = model_data['n_features']
        detector.n_support_vectors = model_data['n_support_vectors']
        detector.support_vectors = model_data['support_vectors']
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
        return (f"OneClassSVMDetector(nu={self.nu}, "
                f"kernel={self.kernel}, "
                f"gamma={self.gamma}, "
                f"status={status})")


if __name__ == "__main__":
    # Example usage
    print("One-Class SVM Detector Module")
    print("="*70)
    print("\nExample usage:")
    print("""
from src.models.one_class_svm import OneClassSVMDetector

# Initialize detector with RBF kernel
detector = OneClassSVMDetector(
    nu=0.1,
    kernel='rbf',
    gamma='scale'
)

# Train on data (unsupervised - no labels needed)
detector.fit(X_train, feature_names=feature_names)

# Make predictions
y_pred = detector.predict(X_test)
y_proba = detector.predict_proba(X_test)

# Get decision scores
decision_scores = detector.get_decision_scores(X_test)

# Get support vector info
sv_info = detector.get_support_vector_info()
print(f"Support vectors: {sv_info['n_support_vectors']}")

# Get feature importance
importance = detector.get_feature_importance(X_train, method='permutation')

# Benchmark inference speed
timing = detector.benchmark_inference(X_test[:1000])
print(f"Time per sample: {timing['time_per_sample_ms']:.2f} ms")

# Save model
detector.save('models/ocsvm.pkl')

# Load model later
loaded_detector = OneClassSVMDetector.load('models/ocsvm.pkl')
    """)
    print("="*70)
