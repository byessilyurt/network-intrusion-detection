"""
Variational Autoencoder (VAE) Anomaly Detection Module

This module implements a production-ready Variational Autoencoder for network
intrusion detection. Unlike standard autoencoders, VAEs learn a probabilistic
latent representation, making them more robust for anomaly detection on tabular data.

Key Features:
- Probabilistic encoder with reparameterization trick
- KL divergence regularization to prevent overfitting
- Combined reconstruction + KL divergence anomaly scoring
- Latent space sampling and interpolation
- Model persistence (save/load with HDF5)
- Production-ready API matching other detectors
- GPU acceleration support

VAE Theory:
    VAE learns a generative model p(x|z) where z is a latent variable.
    The encoder approximates the posterior q(z|x) ≈ p(z|x).
    The decoder reconstructs x from sampled z.

    Loss = Reconstruction Loss + β * KL Divergence

    - Reconstruction: How well can we rebuild the input?
    - KL Divergence: How close is q(z|x) to prior p(z) = N(0,I)?

Reference:
    Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes.
    In International Conference on Learning Representations (ICLR).

    An, J., & Cho, S. (2015). Variational autoencoder based anomaly detection
    using reconstruction probability. SNU Data Mining Center.
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


class Sampling(layers.Layer):
    """
    Reparameterization trick layer for VAE.

    Instead of sampling z ~ N(μ, σ²) directly (non-differentiable),
    we sample ε ~ N(0, 1) and compute z = μ + σ * ε (differentiable).

    This allows backpropagation through the sampling operation.
    """

    def call(self, inputs):
        """
        Sample from latent distribution using reparameterization trick.

        Args:
            inputs: [z_mean, z_log_var] tensors from encoder

        Returns:
            z: Sampled latent vector
        """
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]

        # Sample epsilon from standard normal distribution
        epsilon = tf.random.normal(shape=(batch_size, latent_dim))

        # Reparameterization: z = μ + σ * ε
        # Use log(σ²) for numerical stability: σ = exp(0.5 * log(σ²))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEDetector:
    """
    Variational Autoencoder-based anomaly detector.

    VAE learns a probabilistic latent representation of normal network traffic.
    Anomalies are detected using a combined score of:
    1. Reconstruction error (MSE between input and output)
    2. KL divergence (deviation from learned prior distribution)

    Architecture:
        Input → Encoder → (μ, log_σ²) → Sampling → z → Decoder → Reconstructed Output

    Attributes:
        encoder: Keras encoder model (input → μ, log_σ²)
        decoder: Keras decoder model (z → reconstructed output)
        vae: Full VAE model (input → output)
        threshold: Anomaly score threshold for detection
        scaler: StandardScaler for feature normalization
        latent_dim: Dimensionality of latent space
        feature_names: Names of input features
        training_time: Training duration (seconds)
        n_features: Number of input features
        history: Training history
    """

    def __init__(self,
                 latent_dim: int = 20,
                 encoder_dims: List[int] = None,
                 activation: str = 'relu',
                 output_activation: str = 'linear',
                 dropout_rate: float = 0.2,
                 l2_reg: float = 1e-5,
                 kl_weight: float = 0.001,
                 learning_rate: float = 0.001,
                 batch_size: int = 256,
                 epochs: int = 100,
                 validation_split: float = 0.1,
                 early_stopping_patience: int = 10,
                 threshold_percentile: float = 95.0,
                 random_state: int = 42,
                 verbose: int = 1):
        """
        Initialize VAE detector.

        Args:
            latent_dim: Dimensionality of latent space (bottleneck)
                       Lower = more compression, higher = more capacity
            encoder_dims: List of encoder layer sizes (e.g., [50, 30])
                         If None, uses [50, 30] for 70 input features
            activation: Activation function for hidden layers
            output_activation: Activation for output layer (usually 'linear')
            dropout_rate: Dropout rate for regularization (0.0 = no dropout)
            l2_reg: L2 regularization strength
            kl_weight: Weight for KL divergence term in loss (β parameter)
                      Start small (0.001) to prevent posterior collapse
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
            epochs: Maximum training epochs
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs with no improvement before stopping
            threshold_percentile: Percentile of anomaly scores for threshold
            random_state: Random seed for reproducibility
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        # Architecture parameters
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims if encoder_dims is not None else [50, 30]
        self.decoder_dims = list(reversed(self.encoder_dims))  # Mirror encoder
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.kl_weight = kl_weight

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        self.verbose = verbose

        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        # Model components (initialized during fit)
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.feature_names = None
        self.n_features = None
        self.training_time = None
        self.history = None

    def _build_encoder(self, input_dim: int) -> Model:
        """
        Build encoder network: input → hidden layers → (μ, log_σ²).

        Args:
            input_dim: Number of input features

        Returns:
            Encoder model outputting [z_mean, z_log_var, z]
        """
        # Input layer
        inputs = keras.Input(shape=(input_dim,), name='encoder_input')
        x = inputs

        # Hidden layers
        for i, dim in enumerate(self.encoder_dims):
            x = layers.Dense(
                dim,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'encoder_dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'encoder_bn_{i+1}')(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'encoder_dropout_{i+1}')(x)

        # Latent parameters: μ and log(σ²)
        z_mean = layers.Dense(
            self.latent_dim,
            name='z_mean',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(x)

        z_log_var = layers.Dense(
            self.latent_dim,
            name='z_log_var',
            kernel_regularizer=keras.regularizers.l2(self.l2_reg)
        )(x)

        # Sampling layer (reparameterization trick)
        z = Sampling()([z_mean, z_log_var])

        # Encoder outputs both parameters and sampled z
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def _build_decoder(self, output_dim: int) -> Model:
        """
        Build decoder network: z → hidden layers → reconstructed output.

        Args:
            output_dim: Number of output features (same as input)

        Returns:
            Decoder model
        """
        # Latent input
        latent_inputs = keras.Input(shape=(self.latent_dim,), name='decoder_input')
        x = latent_inputs

        # Hidden layers (reverse of encoder)
        for i, dim in enumerate(self.decoder_dims):
            x = layers.Dense(
                dim,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                name=f'decoder_dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'decoder_bn_{i+1}')(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate, name=f'decoder_dropout_{i+1}')(x)

        # Output layer (reconstruct input)
        outputs = layers.Dense(
            output_dim,
            activation=self.output_activation,
            name='decoder_output'
        )(x)

        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder

    def _build_vae(self, input_dim: int) -> Model:
        """
        Build complete VAE model with custom loss.

        The VAE loss combines:
        1. Reconstruction loss: MSE(x, x_reconstructed)
        2. KL divergence: -0.5 * sum(1 + log_σ² - μ² - σ²)

        Args:
            input_dim: Number of input features

        Returns:
            Complete VAE model
        """
        # Build encoder and decoder
        self.encoder = self._build_encoder(input_dim)
        self.decoder = self._build_decoder(input_dim)

        # VAE: input → encoder → decoder → output
        inputs = keras.Input(shape=(input_dim,), name='vae_input')
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)

        vae = Model(inputs, outputs, name='vae')

        # Calculate VAE losses and add them to the model
        # This is the correct approach for TensorFlow 2.x/Keras 2.6

        # Reconstruction loss (MSE)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(inputs - outputs), axis=-1)
        )

        # KL divergence loss
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )

        # Total loss with KL weight
        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        # Add loss to model (this avoids symbolic tensor issues)
        vae.add_loss(total_loss)

        # Compile model without loss argument (already added via add_loss)
        vae.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )

        return vae

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Optional[np.ndarray] = None) -> 'VAEDetector':
        """
        Train VAE on normal data (unsupervised).

        VAE learns to reconstruct normal network traffic patterns.
        Labels (y) are ignored - this is unsupervised learning.

        Args:
            X: Training data (normal traffic only), shape (n_samples, n_features)
            y: Ignored (for sklearn compatibility)

        Returns:
            self: Fitted detector
        """
        start_time = time.time()

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        self.n_features = X.shape[1]

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Build VAE architecture
        if self.verbose > 0:
            print(f"Building VAE with architecture:")
            print(f"  Input: {self.n_features} features")
            print(f"  Encoder: {self.encoder_dims}")
            print(f"  Latent: {self.latent_dim} dimensions")
            print(f"  Decoder: {self.decoder_dims}")
            print(f"  KL weight: {self.kl_weight}")

        self.vae = self._build_vae(self.n_features)

        if self.verbose > 0:
            print(f"\nTotal parameters: {self.vae.count_params():,}")

        # Callbacks
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
                min_lr=1e-6,
                verbose=self.verbose
            )
        ]

        # Train VAE (unsupervised: input = output)
        if self.verbose > 0:
            print(f"\nTraining VAE for {self.epochs} epochs...")

        self.history = self.vae.fit(
            X_scaled, X_scaled,  # Autoencoder: reconstruct input
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose
        )

        self.training_time = time.time() - start_time

        # Optimize threshold using training data
        train_scores = self._compute_anomaly_scores(X_scaled)
        self.threshold = np.percentile(train_scores, self.threshold_percentile)

        if self.verbose > 0:
            print(f"\nTraining complete in {self.training_time:.2f}s")
            print(f"Anomaly score threshold (P{self.threshold_percentile}): {self.threshold:.6f}")
            print(f"Score range on training data: [{train_scores.min():.6f}, {train_scores.max():.6f}]")

        return self

    def _compute_anomaly_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores using reconstruction error + KL divergence.

        Args:
            X_scaled: Normalized input data

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        # Get encoder outputs
        z_mean, z_log_var, z = self.encoder.predict(X_scaled, verbose=0)

        # Get reconstructions
        reconstructions = self.decoder.predict(z, verbose=0)

        # Reconstruction error (MSE per sample)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)

        # KL divergence per sample
        # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_σ² - μ² - σ²)
        kl_divergences = -0.5 * np.sum(
            1 + z_log_var - np.square(z_mean) - np.exp(z_log_var),
            axis=1
        )

        # Combined anomaly score (weighted sum)
        # Scale KL by weight to match contribution to loss
        anomaly_scores = reconstruction_errors + self.kl_weight * kl_divergences

        return anomaly_scores

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly labels (0 = normal, 1 = anomaly).

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            predictions: Binary labels (0 = normal, 1 = anomaly)
        """
        if self.vae is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize and compute anomaly scores
        X_scaled = self.scaler.transform(X)
        scores = self._compute_anomaly_scores(X_scaled)

        # Threshold: score > threshold → anomaly (1), else normal (0)
        predictions = (scores > self.threshold).astype(int)

        return predictions

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly probabilities.

        Converts anomaly scores to probabilities using sigmoid normalization.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            probabilities: shape (n_samples, 2)
                          [:, 0] = P(normal), [:, 1] = P(anomaly)
        """
        if self.vae is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize and compute anomaly scores
        X_scaled = self.scaler.transform(X)
        scores = self._compute_anomaly_scores(X_scaled)

        # Normalize scores to [0, 1] using sigmoid around threshold
        # P(anomaly) = sigmoid((score - threshold) / scale)
        scale = self.threshold * 0.1  # Scale factor for smooth transition
        prob_anomaly = 1 / (1 + np.exp(-(scores - self.threshold) / scale))
        prob_normal = 1 - prob_anomaly

        # Return as [P(normal), P(anomaly)]
        probabilities = np.column_stack([prob_normal, prob_anomaly])

        return probabilities

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute anomaly scores (raw decision function).

        Higher scores indicate more anomalous samples.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            scores: Anomaly scores (reconstruction error + KL divergence)
        """
        if self.vae is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize and compute anomaly scores
        X_scaled = self.scaler.transform(X)
        scores = self._compute_anomaly_scores(X_scaled)

        return scores

    def get_reconstruction_error(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get reconstruction error (MSE) for each sample.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            errors: Reconstruction error per sample
        """
        if self.vae is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize
        X_scaled = self.scaler.transform(X)

        # Reconstruct
        reconstructions = self.vae.predict(X_scaled, verbose=0)

        # MSE per sample
        errors = np.mean(np.square(X_scaled - reconstructions), axis=1)

        return errors

    def get_latent(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latent space representation (μ, log_σ²).

        Useful for visualization and analysis of learned representations.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            z_mean: Latent means, shape (n_samples, latent_dim)
            z_log_var: Latent log variances, shape (n_samples, latent_dim)
        """
        if self.encoder is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize
        X_scaled = self.scaler.transform(X)

        # Get encoder outputs
        z_mean, z_log_var, _ = self.encoder.predict(X_scaled, verbose=0)

        return z_mean, z_log_var

    def reconstruct(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Reconstruct input data (denormalized).

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            reconstructions: Reconstructed data in original scale
        """
        if self.vae is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize
        X_scaled = self.scaler.transform(X)

        # Reconstruct (normalized)
        reconstructions_scaled = self.vae.predict(X_scaled, verbose=0)

        # Denormalize
        reconstructions = self.scaler.inverse_transform(reconstructions_scaled)

        return reconstructions

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model to disk.

        Saves:
        - VAE model architecture and weights (.h5)
        - Encoder and decoder separately (.h5)
        - Scaler and metadata (.pkl)

        Args:
            filepath: Base path for saving (without extension)
                     Creates: filepath.h5, filepath_encoder.h5,
                             filepath_decoder.h5, filepath.pkl
        """
        if self.vae is None:
            raise ValueError("Model not fitted. Call fit() first.")

        filepath = Path(filepath)

        # Save Keras models
        self.vae.save(f"{filepath}.h5")
        self.encoder.save(f"{filepath}_encoder.h5")
        self.decoder.save(f"{filepath}_decoder.h5")

        # Save metadata and scaler
        metadata = {
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'latent_dim': self.latent_dim,
            'encoder_dims': self.encoder_dims,
            'decoder_dims': self.decoder_dims,
            'kl_weight': self.kl_weight,
            'threshold_percentile': self.threshold_percentile,
            'training_time': self.training_time
        }

        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        if self.verbose > 0:
            print(f"Model saved to {filepath}.*")

    @classmethod
    def load(cls, filepath: Union[str, Path], verbose: int = 1) -> 'VAEDetector':
        """
        Load trained model from disk.

        Args:
            filepath: Base path to saved model (without extension)
            verbose: Verbosity level

        Returns:
            detector: Loaded VAE detector
        """
        filepath = Path(filepath)

        # Load metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create detector instance
        detector = cls(
            latent_dim=metadata['latent_dim'],
            encoder_dims=metadata['encoder_dims'],
            kl_weight=metadata['kl_weight'],
            threshold_percentile=metadata['threshold_percentile'],
            verbose=verbose
        )

        # Load Keras models
        detector.vae = keras.models.load_model(
            f"{filepath}.h5",
            custom_objects={'Sampling': Sampling},
            compile=False  # Don't need to compile for inference
        )
        detector.encoder = keras.models.load_model(
            f"{filepath}_encoder.h5",
            custom_objects={'Sampling': Sampling},
            compile=False
        )
        detector.decoder = keras.models.load_model(
            f"{filepath}_decoder.h5",
            compile=False
        )

        # Restore metadata
        detector.scaler = metadata['scaler']
        detector.threshold = metadata['threshold']
        detector.feature_names = metadata['feature_names']
        detector.n_features = metadata['n_features']
        detector.decoder_dims = metadata['decoder_dims']
        detector.training_time = metadata['training_time']

        if verbose > 0:
            print(f"Model loaded from {filepath}.*")
            print(f"Architecture: {detector.n_features} → {detector.encoder_dims} → "
                  f"{detector.latent_dim} → {detector.decoder_dims} → {detector.n_features}")

        return detector

    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.

        Returns:
            info: Dictionary with model details
        """
        if self.vae is None:
            return {'status': 'not_fitted'}

        return {
            'status': 'fitted',
            'architecture': {
                'input_dim': self.n_features,
                'encoder_dims': self.encoder_dims,
                'latent_dim': self.latent_dim,
                'decoder_dims': self.decoder_dims,
                'total_params': self.vae.count_params()
            },
            'training': {
                'training_time': self.training_time,
                'epochs_trained': len(self.history.history['loss']) if self.history else 0,
                'final_loss': self.history.history['loss'][-1] if self.history else None,
                'final_val_loss': self.history.history['val_loss'][-1] if self.history else None
            },
            'hyperparameters': {
                'kl_weight': self.kl_weight,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'dropout_rate': self.dropout_rate,
                'l2_reg': self.l2_reg
            },
            'detection': {
                'threshold': self.threshold,
                'threshold_percentile': self.threshold_percentile
            }
        }
