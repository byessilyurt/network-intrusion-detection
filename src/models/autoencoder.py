"""
Autoencoder Anomaly Detection Module

This module implements a production-ready deep learning Autoencoder detector for
network intrusion detection. Autoencoders learn to reconstruct normal traffic
patterns, and anomalies are detected by high reconstruction errors.

Key Features:
- Deep learning-based anomaly detection using TensorFlow/Keras
- Configurable encoder-decoder architecture with bottleneck
- Reconstruction error-based anomaly scoring
- Automatic threshold optimization using percentiles
- Model persistence (save/load with HDF5)
- Production-ready API matching classical ML interface
- GPU acceleration support

Reference:
    Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders
    with nonlinear dimensionality reduction. In Proceedings of the MLSDA.
"""

import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector with production-ready functionality.

    The Autoencoder learns to reconstruct normal network traffic patterns.
    Anomalies are detected by measuring reconstruction error - attacks have
    higher reconstruction errors than normal traffic.

    Architecture:
        Input → Encoder (compress) → Bottleneck (latent) → Decoder (reconstruct) → Output

    Attributes:
        model: Fitted Keras autoencoder model
        threshold: Reconstruction error threshold for anomaly detection
        encoder_dims: List of encoder layer dimensions
        decoder_dims: List of decoder layer dimensions (reverse of encoder)
        feature_names: Names of input features
        training_time: Time taken to train model (seconds)
        n_features: Number of input features
    """

    def __init__(self,
                 encoder_dims: List[int] = None,
                 activation: str = 'relu',
                 output_activation: str = 'linear',
                 dropout_rate: float = 0.2,
                 l2_reg: float = 1e-5,
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 100,
                 validation_split: float = 0.1,
                 early_stopping_patience: int = 10,
                 threshold_percentile: float = 95.0,
                 random_state: int = 42,
                 verbose: int = 1):
        """
        Initialize Autoencoder detector.

        Args:
            encoder_dims: List of encoder layer sizes (e.g., [50, 30, 20])
                         If None, uses default [50, 30, 20] for 70 input features
            activation: Activation function for hidden layers ('relu', 'tanh', 'elu')
            output_activation: Activation for output layer ('linear', 'sigmoid')
            dropout_rate: Dropout rate for regularization (0.0 = no dropout)
            l2_reg: L2 regularization strength
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs with no improvement before stopping
            threshold_percentile: Percentile of reconstruction errors to use as threshold
                                 (e.g., 95.0 = 95th percentile)
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        # Architecture parameters
        self.encoder_dims = encoder_dims if encoder_dims is not None else [50, 30, 20]
        self.decoder_dims = list(reversed(self.encoder_dims))  # Mirror encoder
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.verbose = verbose

        # State variables
        self.model = None
        self.encoder = None
        self.threshold = None
        self.is_fitted = False
        self.feature_names = None
        self.training_time = None
        self.n_features = None
        self.history = None
        self.scaler = StandardScaler()  # Internal scaler for stability

        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def _build_model(self, input_dim: int) -> Model:
        """
        Build autoencoder architecture.

        Args:
            input_dim: Number of input features

        Returns:
            Compiled Keras model
        """
        # Input layer
        input_layer = layers.Input(shape=(input_dim,), name='input')

        # Encoder
        x = input_layer
        for i, dim in enumerate(self.encoder_dims):
            x = layers.Dense(
                dim,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'encoder_{i+1}'
            )(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'dropout_enc_{i+1}')(x)

        # Bottleneck (latent representation)
        bottleneck_dim = self.encoder_dims[-1]
        bottleneck = x  # Already created in encoder loop

        # Decoder (mirror of encoder)
        for i, dim in enumerate(self.decoder_dims):
            x = layers.Dense(
                dim,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'decoder_{i+1}'
            )(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'dropout_dec_{i+1}')(x)

        # Output layer (reconstruction)
        output_layer = layers.Dense(
            input_dim,
            activation=self.output_activation,
            name='output'
        )(x)

        # Create autoencoder
        autoencoder = Model(inputs=input_layer, outputs=output_layer, name='autoencoder')

        # Compile model
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',  # Mean Squared Error for reconstruction
            metrics=['mae']  # Mean Absolute Error for monitoring
        )

        if self.verbose >= 2:
            autoencoder.summary()

        return autoencoder

    def _build_encoder(self, input_dim: int) -> Model:
        """
        Build encoder model for extracting latent representations.

        Args:
            input_dim: Number of input features

        Returns:
            Encoder model
        """
        input_layer = layers.Input(shape=(input_dim,), name='input')
        x = input_layer

        for i, dim in enumerate(self.encoder_dims):
            x = layers.Dense(
                dim,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'encoder_{i+1}'
            )(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'dropout_enc_{i+1}')(x)

        encoder = Model(inputs=input_layer, outputs=x, name='encoder')
        return encoder

    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            feature_names: Optional[List[str]] = None,
            validation_data: Optional[Tuple] = None) -> 'AutoencoderDetector':
        """
        Fit the Autoencoder model on training data.

        Note: Autoencoder is unsupervised - labels not required.
        Train ONLY on normal (benign) traffic for anomaly detection.

        Args:
            X: Training features (n_samples, n_features) - NORMAL DATA ONLY
            feature_names: Names of features (optional)
            validation_data: Optional validation data (X_val, y_val) for early stopping

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

        # Scale features for stable training
        X_scaled = self.scaler.fit_transform(X)

        if self.verbose:
            print("="*70)
            print("AUTOENCODER TRAINING")
            print("="*70)
            print(f"Training samples: {X.shape[0]:,}")
            print(f"Features: {X.shape[1]}")
            print(f"Architecture: {self.n_features} → {' → '.join(map(str, self.encoder_dims))} "
                  f"→ {' → '.join(map(str, self.decoder_dims))} → {self.n_features}")
            print(f"Activation: {self.activation}")
            print(f"Dropout: {self.dropout_rate}")
            print(f"L2 Regularization: {self.l2_reg}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Batch size: {self.batch_size}")
            print(f"Max epochs: {self.epochs}")
            print(f"Early stopping patience: {self.early_stopping_patience}")
            print("="*70)

        # Build model
        self.model = self._build_model(self.n_features)

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=self.verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=self.verbose
            )
        ]

        # Train model
        start_time = time.time()

        self.history = self.model.fit(
            X_scaled, X_scaled,  # Autoencoder: input = output
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split if validation_data is None else 0.0,
            validation_data=(self.scaler.transform(validation_data[0]),
                           self.scaler.transform(validation_data[0])) if validation_data is not None else None,
            callbacks=callbacks,
            verbose=self.verbose
        )

        self.training_time = time.time() - start_time

        # Compute reconstruction errors on training data
        train_reconstructed = self.model.predict(X_scaled, verbose=0)
        train_errors = np.mean(np.square(X_scaled - train_reconstructed), axis=1)

        # Set threshold at specified percentile of training errors
        self.threshold = np.percentile(train_errors, self.threshold_percentile)

        self.is_fitted = True

        if self.verbose:
            print("="*70)
            print(f"Training completed in {self.training_time:.2f} seconds")
            print(f"Final loss: {self.history.history['loss'][-1]:.6f}")
            print(f"Final val_loss: {self.history.history['val_loss'][-1]:.6f}")
            print(f"Reconstruction error threshold (P{self.threshold_percentile}): {self.threshold:.6f}")
            print(f"Epochs trained: {len(self.history.history['loss'])}")
            print("="*70)

        # Build encoder for latent space extraction
        self.encoder = self._build_encoder(self.n_features)
        # Copy encoder weights from trained autoencoder
        for i, layer in enumerate(self.encoder.layers):
            if 'encoder' in layer.name or 'dropout_enc' in layer.name:
                corresponding_layer = self.model.get_layer(layer.name)
                layer.set_weights(corresponding_layer.get_weights())

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly labels for samples.

        Samples with reconstruction error > threshold are classified as anomalies.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Binary predictions: 0 = normal, 1 = anomaly
        """
        self._check_is_fitted()

        reconstruction_errors = self.get_reconstruction_error(X)
        predictions = (reconstruction_errors > self.threshold).astype(int)

        return predictions

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly probabilities based on reconstruction error.

        Returns normalized anomaly scores in range [0, 1] where:
        - Values close to 1 = likely anomaly (high reconstruction error)
        - Values close to 0 = likely normal (low reconstruction error)

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Anomaly probabilities (n_samples,)
        """
        self._check_is_fitted()

        reconstruction_errors = self.get_reconstruction_error(X)

        # Normalize errors to [0, 1] using sigmoid-like transformation
        # Errors near threshold → 0.5, errors >> threshold → 1.0, errors << threshold → 0.0
        normalized_errors = reconstruction_errors / (self.threshold + 1e-10)
        probabilities = 1 / (1 + np.exp(-5 * (normalized_errors - 1)))

        return probabilities

    def get_reconstruction_error(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute reconstruction error for each sample.

        Reconstruction error = Mean Squared Error between input and reconstructed output.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Reconstruction errors (n_samples,)
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Reconstruct
        X_reconstructed = self.model.predict(X_scaled, verbose=0)

        # Compute MSE per sample
        reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)

        return reconstruction_errors

    def get_latent_representation(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get latent (bottleneck) representation of input data.

        Useful for visualization (t-SNE, PCA) and feature extraction.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Latent representations (n_samples, bottleneck_dim)
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Extract latent representation
        latent = self.encoder.predict(X_scaled, verbose=0)

        return latent

    def optimize_threshold(self,
                          X_val: Union[np.ndarray, pd.DataFrame],
                          y_val: np.ndarray,
                          percentiles: List[float] = None) -> Dict[str, float]:
        """
        Optimize reconstruction error threshold using validation data.

        Tests multiple percentile thresholds and selects the one with best F1 score.

        Args:
            X_val: Validation features
            y_val: Validation labels (0=normal, 1=anomaly)
            percentiles: List of percentiles to test (e.g., [90, 95, 99])
                        If None, uses [90, 92, 95, 97, 99]

        Returns:
            Dictionary with optimization results
        """
        self._check_is_fitted()

        if percentiles is None:
            percentiles = [90, 92, 95, 97, 99]

        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values

        # Compute reconstruction errors
        val_errors = self.get_reconstruction_error(X_val)

        # Test each percentile
        results = []
        for p in percentiles:
            threshold = np.percentile(val_errors, p)
            predictions = (val_errors > threshold).astype(int)

            # Compute metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_val, predictions, zero_division=0)
            recall = recall_score(y_val, predictions, zero_division=0)
            f1 = f1_score(y_val, predictions, zero_division=0)

            results.append({
                'percentile': p,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        # Find best F1
        best_result = max(results, key=lambda x: x['f1'])
        self.threshold = best_result['threshold']
        self.threshold_percentile = best_result['percentile']

        if self.verbose:
            print("\nThreshold Optimization Results:")
            print("-" * 70)
            for r in results:
                marker = " ← SELECTED" if r['percentile'] == best_result['percentile'] else ""
                print(f"P{r['percentile']:>3}: threshold={r['threshold']:.6f}, "
                      f"F1={r['f1']:.4f}, Precision={r['precision']:.4f}, "
                      f"Recall={r['recall']:.4f}{marker}")
            print("-" * 70)
            print(f"Best threshold: {self.threshold:.6f} (P{self.threshold_percentile})")
            print(f"Best F1: {best_result['f1']:.4f}")

        return {
            'results': results,
            'best': best_result,
            'best_threshold': self.threshold,
            'best_percentile': self.threshold_percentile
        }

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
        Save model to disk.

        Saves Keras model as .h5 and metadata as .pkl

        Args:
            filepath: Path to save model (e.g., 'models/autoencoder_real.h5')
        """
        self._check_is_fitted()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        model_path = filepath.with_suffix('.h5')
        self.model.save(model_path)

        # Save metadata
        metadata_path = filepath.with_suffix('.pkl')
        metadata = {
            'encoder_dims': self.encoder_dims,
            'decoder_dims': self.decoder_dims,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile,
            'feature_names': self.feature_names,
            'training_time': self.training_time,
            'n_features': self.n_features,
            'scaler': self.scaler,
            'history': self.history.history if self.history else None,
            'is_fitted': self.is_fitted
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        if self.verbose:
            print(f"Model saved to: {model_path}")
            print(f"Metadata saved to: {metadata_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'AutoencoderDetector':
        """
        Load model from disk.

        Args:
            filepath: Path to saved model (with or without extension)

        Returns:
            Loaded AutoencoderDetector instance
        """
        filepath = Path(filepath)

        # Load Keras model
        model_path = filepath.with_suffix('.h5')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load metadata
        metadata_path = filepath.with_suffix('.pkl')
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Create instance
        detector = cls(
            encoder_dims=metadata['encoder_dims'],
            activation=metadata['activation'],
            output_activation=metadata['output_activation'],
            dropout_rate=metadata['dropout_rate'],
            l2_reg=metadata['l2_reg'],
            learning_rate=metadata['learning_rate'],
            batch_size=metadata['batch_size'],
            epochs=metadata['epochs'],
            threshold_percentile=metadata['threshold_percentile'],
            verbose=0
        )

        # Load Keras model
        detector.model = keras.models.load_model(model_path)

        # Restore state
        detector.threshold = metadata['threshold']
        detector.feature_names = metadata['feature_names']
        detector.training_time = metadata['training_time']
        detector.n_features = metadata['n_features']
        detector.scaler = metadata['scaler']
        detector.is_fitted = metadata['is_fitted']

        # Rebuild encoder
        detector.encoder = detector._build_encoder(detector.n_features)
        for i, layer in enumerate(detector.encoder.layers):
            if 'encoder' in layer.name or 'dropout_enc' in layer.name:
                corresponding_layer = detector.model.get_layer(layer.name)
                layer.set_weights(corresponding_layer.get_weights())

        print(f"Model loaded from: {model_path}")
        print(f"Metadata loaded from: {metadata_path}")

        return detector

    def _check_is_fitted(self) -> None:
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        arch = f"{self.n_features} → {' → '.join(map(str, self.encoder_dims))} → {' → '.join(map(str, self.decoder_dims))} → {self.n_features}" if self.n_features else "architecture not built"
        return (f"AutoencoderDetector(architecture={arch}, "
                f"threshold={self.threshold:.6f if self.threshold else 'not set'}, "
                f"status={status})")


if __name__ == "__main__":
    # Example usage
    print("Autoencoder Detector Module")
    print("="*70)
    print("\nExample usage:")
    print("""
from src.models.autoencoder import AutoencoderDetector

# Initialize detector
detector = AutoencoderDetector(
    encoder_dims=[50, 30, 20],
    activation='relu',
    dropout_rate=0.2,
    learning_rate=0.001,
    batch_size=256,
    epochs=100,
    threshold_percentile=95.0,
    random_state=42
)

# Train on normal data ONLY (unsupervised)
detector.fit(X_train_normal, feature_names=feature_names)

# Optimize threshold on validation set
detector.optimize_threshold(X_val, y_val, percentiles=[90, 95, 99])

# Make predictions
y_pred = detector.predict(X_test)
y_proba = detector.predict_proba(X_test)
reconstruction_errors = detector.get_reconstruction_error(X_test)

# Get latent representations for visualization
latent = detector.get_latent_representation(X_test)

# Benchmark inference speed
timing = detector.benchmark_inference(X_test[:1000])
print(f"Time per sample: {timing['time_per_sample_ms']:.2f} ms")

# Save model
detector.save('models/autoencoder_real.h5')

# Load model later
loaded_detector = AutoencoderDetector.load('models/autoencoder_real')
    """)
    print("="*70)
