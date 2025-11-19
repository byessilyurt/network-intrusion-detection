"""
SHAP-based explainability module for network intrusion detection models.

This module provides unified explainability across different model types:
- TreeExplainer for Isolation Forest (native support, fast)
- KernelExplainer for One-Class SVM (model-agnostic, slower)
- GradientExplainer for Autoencoder/VAE (gradient-based, efficient)

Functions:
- explain_prediction(): Get SHAP values for single prediction
- get_top_features(): Extract top-N contributing features
- plot_force(): Generate SHAP force plot
- plot_summary(): Generate SHAP summary plot
- explain_batch(): Explain multiple predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Unified explainer interface for all model types.

    Automatically selects the appropriate SHAP explainer based on model type:
    - Isolation Forest: TreeExplainer (fast, exact)
    - One-Class SVM: KernelExplainer (accurate, slower)
    - Autoencoder/VAE: GradientExplainer (efficient for neural networks)
    """

    def __init__(
        self,
        model: Any,
        model_type: str,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize explainer for a specific model.

        Args:
            model: Trained model instance (IsolationForestDetector, OneClassSVMDetector, etc.)
            model_type: Type of model ('isolation_forest', 'ocsvm', 'autoencoder', 'vae')
            background_data: Background data for SHAP (required for kernel/gradient explainers)
            feature_names: List of feature names for interpretability
        """
        self.model = model
        self.model_type = model_type.lower()
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = None

        # Initialize appropriate explainer
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""

        if self.model_type == 'isolation_forest':
            # TreeExplainer for Isolation Forest (native support)
            try:
                # Access the underlying sklearn model
                sklearn_model = self.model.model if hasattr(self.model, 'model') else self.model
                self.explainer = shap.TreeExplainer(sklearn_model)
                print(f"✓ Initialized TreeExplainer for Isolation Forest")
            except Exception as e:
                print(f"⚠ TreeExplainer failed, falling back to KernelExplainer: {e}")
                self._initialize_kernel_explainer()

        elif self.model_type == 'ocsvm':
            # KernelExplainer for One-Class SVM
            self._initialize_kernel_explainer()

        elif self.model_type in ['autoencoder', 'vae']:
            # GradientExplainer for neural networks
            self._initialize_gradient_explainer()

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _initialize_kernel_explainer(self):
        """Initialize KernelExplainer (model-agnostic, works for any model)."""
        if self.background_data is None:
            raise ValueError("KernelExplainer requires background_data")

        # Sample subset for efficiency (100 samples)
        if len(self.background_data) > 100:
            indices = np.random.choice(len(self.background_data), 100, replace=False)
            background_sample = self.background_data[indices]
        else:
            background_sample = self.background_data

        # Create prediction function
        def predict_fn(X):
            """Wrapper for model prediction."""
            if hasattr(self.model, 'decision_function'):
                return self.model.decision_function(X)
            else:
                return self.model.predict_proba(X)[:, 1]

        self.explainer = shap.KernelExplainer(predict_fn, background_sample)
        print(f"✓ Initialized KernelExplainer with {len(background_sample)} background samples")

    def _initialize_gradient_explainer(self):
        """Initialize GradientExplainer for neural networks."""
        if self.background_data is None:
            raise ValueError("GradientExplainer requires background_data")

        # Sample subset for efficiency (100 samples)
        if len(self.background_data) > 100:
            indices = np.random.choice(len(self.background_data), 100, replace=False)
            background_sample = self.background_data[indices]
        else:
            background_sample = self.background_data

        try:
            # Access underlying Keras model
            keras_model = self.model.model if hasattr(self.model, 'model') else self.model

            # For autoencoders, we want to explain reconstruction error
            # Create custom model that outputs anomaly score
            import tensorflow as tf

            @tf.function
            def anomaly_score(X):
                """Compute anomaly score for gradient computation."""
                reconstructed = keras_model(X, training=False)
                reconstruction_error = tf.reduce_mean(tf.square(X - reconstructed), axis=1)
                return reconstruction_error

            self.explainer = shap.GradientExplainer(
                (keras_model.input, keras_model.output),
                background_sample
            )
            print(f"✓ Initialized GradientExplainer with {len(background_sample)} background samples")

        except Exception as e:
            print(f"⚠ GradientExplainer failed, falling back to KernelExplainer: {e}")
            self._initialize_kernel_explainer()

    def explain_prediction(
        self,
        X: np.ndarray,
        nsamples: int = 100
    ) -> np.ndarray:
        """
        Get SHAP values for a single prediction or batch.

        Args:
            X: Input features (single sample or batch)
            nsamples: Number of samples for KernelExplainer (ignored for TreeExplainer)

        Returns:
            shap_values: SHAP values explaining the prediction
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        try:
            if self.model_type == 'isolation_forest':
                # TreeExplainer returns exact values
                shap_values = self.explainer.shap_values(X)
            else:
                # KernelExplainer/GradientExplainer
                shap_values = self.explainer.shap_values(X, nsamples=nsamples)

            return shap_values

        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            raise

    def get_top_features(
        self,
        X: np.ndarray,
        n: int = 3,
        nsamples: int = 100
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Get top-N contributing features for a prediction.

        Args:
            X: Input features (single sample)
            n: Number of top features to return
            nsamples: Number of samples for KernelExplainer

        Returns:
            List of dicts with 'feature' and 'contribution' keys
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Get SHAP values
        shap_values = self.explain_prediction(X, nsamples=nsamples)

        # Get absolute contributions
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For multi-output models

        shap_values_single = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        abs_contributions = np.abs(shap_values_single)

        # Get top-N indices
        top_indices = np.argsort(abs_contributions)[-n:][::-1]

        # Build result
        top_features = []
        for idx in top_indices:
            feature_name = self.feature_names[idx] if self.feature_names else f"Feature_{idx}"
            contribution = float(shap_values_single[idx])

            top_features.append({
                'feature': feature_name,
                'contribution': contribution,
                'abs_contribution': float(abs_contributions[idx]),
                'feature_value': float(X[0, idx])
            })

        return top_features

    def plot_force(
        self,
        X: np.ndarray,
        nsamples: int = 100,
        matplotlib: bool = True,
        show: bool = True,
        save_path: Optional[Path] = None
    ):
        """
        Generate SHAP force plot for a single prediction.

        Args:
            X: Input features (single sample)
            nsamples: Number of samples for KernelExplainer
            matplotlib: Use matplotlib rendering (True) or HTML (False)
            show: Display the plot
            save_path: Path to save the plot
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Get SHAP values
        shap_values = self.explain_prediction(X, nsamples=nsamples)

        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            base_value = self.explainer.expected_value
        else:
            # Compute expected value from background data
            if hasattr(self.model, 'decision_function'):
                base_value = self.model.decision_function(self.background_data).mean()
            else:
                base_value = self.model.predict_proba(self.background_data)[:, 1].mean()

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]

        # Create force plot
        if matplotlib:
            shap.force_plot(
                base_value,
                shap_values[0],
                X[0],
                feature_names=self.feature_names,
                matplotlib=True,
                show=show
            )

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved force plot to {save_path}")
        else:
            # HTML rendering
            plot = shap.force_plot(
                base_value,
                shap_values[0],
                X[0],
                feature_names=self.feature_names
            )

            if save_path:
                shap.save_html(str(save_path), plot)
                print(f"✓ Saved force plot to {save_path}")

            return plot

    def plot_summary(
        self,
        X: np.ndarray,
        max_samples: int = 1000,
        nsamples: int = 100,
        plot_type: str = 'dot',
        max_display: int = 20,
        show: bool = True,
        save_path: Optional[Path] = None
    ):
        """
        Generate SHAP summary plot for multiple predictions.

        Args:
            X: Input features (multiple samples)
            max_samples: Maximum samples to explain (for performance)
            nsamples: Number of samples for KernelExplainer
            plot_type: 'dot', 'bar', or 'violin'
            max_display: Maximum features to display
            show: Display the plot
            save_path: Path to save the plot
        """
        # Sample if too many
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        print(f"Computing SHAP values for {len(X_sample)} samples...")

        # Get SHAP values
        shap_values = self.explain_prediction(X_sample, nsamples=nsamples)

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=show
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved summary plot to {save_path}")

    def explain_batch(
        self,
        X: np.ndarray,
        n_top_features: int = 3,
        nsamples: int = 100,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Explain multiple predictions and return as DataFrame.

        Args:
            X: Input features (multiple samples)
            n_top_features: Number of top features per prediction
            nsamples: Number of samples for KernelExplainer
            show_progress: Show progress bar

        Returns:
            DataFrame with predictions and top contributing features
        """
        results = []

        n_samples = len(X)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), desc="Explaining predictions")
        else:
            iterator = range(n_samples)

        for i in iterator:
            x = X[i:i+1]

            # Get prediction
            y_pred = self.model.predict(x)[0]

            # Get anomaly score
            if hasattr(self.model, 'decision_function'):
                score = self.model.decision_function(x)[0]
            else:
                score = self.model.predict_proba(x)[0, 1]

            # Get top features
            top_features = self.get_top_features(x, n=n_top_features, nsamples=nsamples)

            # Build result row
            result = {
                'sample_idx': i,
                'prediction': y_pred,
                'anomaly_score': score
            }

            # Add top features
            for rank, feature_info in enumerate(top_features, 1):
                result[f'top_{rank}_feature'] = feature_info['feature']
                result[f'top_{rank}_contribution'] = feature_info['contribution']
                result[f'top_{rank}_value'] = feature_info['feature_value']

            results.append(result)

        return pd.DataFrame(results)


def create_explainer(
    model: Any,
    model_type: str,
    background_data: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> ModelExplainer:
    """
    Factory function to create appropriate explainer for a model.

    Args:
        model: Trained model instance
        model_type: Type of model ('isolation_forest', 'ocsvm', 'autoencoder', 'vae')
        background_data: Background data for SHAP
        feature_names: List of feature names

    Returns:
        ModelExplainer instance
    """
    return ModelExplainer(
        model=model,
        model_type=model_type,
        background_data=background_data,
        feature_names=feature_names
    )


# Convenience functions for backward compatibility

def explain_prediction(
    model: Any,
    X: np.ndarray,
    model_type: str,
    background_data: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    nsamples: int = 100
) -> np.ndarray:
    """Get SHAP values for a prediction."""
    explainer = create_explainer(model, model_type, background_data, feature_names)
    return explainer.explain_prediction(X, nsamples=nsamples)


def get_top_features(
    model: Any,
    X: np.ndarray,
    model_type: str,
    n: int = 3,
    background_data: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    nsamples: int = 100
) -> List[Dict[str, Union[str, float]]]:
    """Get top-N contributing features."""
    explainer = create_explainer(model, model_type, background_data, feature_names)
    return explainer.get_top_features(X, n=n, nsamples=nsamples)


def plot_force(
    model: Any,
    X: np.ndarray,
    model_type: str,
    background_data: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    nsamples: int = 100,
    matplotlib: bool = True,
    show: bool = True,
    save_path: Optional[Path] = None
):
    """Generate SHAP force plot."""
    explainer = create_explainer(model, model_type, background_data, feature_names)
    return explainer.plot_force(X, nsamples=nsamples, matplotlib=matplotlib, show=show, save_path=save_path)


def plot_summary(
    model: Any,
    X: np.ndarray,
    model_type: str,
    background_data: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 1000,
    nsamples: int = 100,
    plot_type: str = 'dot',
    max_display: int = 20,
    show: bool = True,
    save_path: Optional[Path] = None
):
    """Generate SHAP summary plot."""
    explainer = create_explainer(model, model_type, background_data, feature_names)
    return explainer.plot_summary(
        X,
        max_samples=max_samples,
        nsamples=nsamples,
        plot_type=plot_type,
        max_display=max_display,
        show=show,
        save_path=save_path
    )


if __name__ == "__main__":
    print("SHAP Explainability Module")
    print("=" * 60)
    print("\nSupported model types:")
    print("  - isolation_forest: TreeExplainer (fast)")
    print("  - ocsvm: KernelExplainer (accurate)")
    print("  - autoencoder: GradientExplainer (efficient)")
    print("  - vae: GradientExplainer (efficient)")
    print("\nUsage:")
    print("  from src.evaluation.explainability import create_explainer")
    print("  explainer = create_explainer(model, 'ocsvm', background_data, feature_names)")
    print("  top_features = explainer.get_top_features(X_sample, n=3)")
