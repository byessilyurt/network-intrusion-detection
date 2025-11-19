#!/usr/bin/env python3
"""
Master Training Script - All 4 Network Intrusion Detection Models
Trains: Isolation Forest, One-Class SVM, Autoencoder, VAE
"""

import sys
import time
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.preprocessing import CICIDS2017Preprocessor
from models.isolation_forest import IsolationForestDetector
from models.one_class_svm import OneClassSVMDetector

# Try importing deep learning models
TF_AVAILABLE = False
try:
    from models.autoencoder import AutoencoderDetector
    from models.vae import VAEDetector
    TF_AVAILABLE = True
    print("✅ TensorFlow available - will train all 4 models")
except ImportError as e:
    print(f"⚠️  TensorFlow import failed: {e}")
    print("⚠️  Will train only classical models (IF, OCSVM)")
    TF_AVAILABLE = False

# Configuration
DATA_DIR = Path(__file__).parent / 'data' / 'raw'
MODELS_DIR = Path(__file__).parent / 'models'
CHECKPOINTS_DIR = Path(__file__).parent / 'checkpoints'
RESULTS_DIR = Path(__file__).parent / 'results'

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)
(CHECKPOINTS_DIR / 'autoencoder').mkdir(exist_ok=True)
(CHECKPOINTS_DIR / 'vae').mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_SAMPLES = 50000
TEST_SAMPLES = 100000
RANDOM_STATE = 42

print(f"\n{'='*80}")
print("NETWORK INTRUSION DETECTION - MASTER TRAINING SCRIPT")
print(f"{'='*80}")
print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Models directory: {MODELS_DIR}")
print(f"📁 Results directory: {RESULTS_DIR}")
print(f"📊 Training samples: {TRAIN_SAMPLES:,}")
print(f"📊 Test samples: {TEST_SAMPLES:,}")


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Check quality gate
    if f1 >= 0.85:
        print(f"✅ QUALITY GATE PASSED (F1 >= 0.85)")
    else:
        gap = 0.85 - f1
        print(f"❌ QUALITY GATE FAILED (F1 < 0.85, gap: {gap:.4f})")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fp_rate = fp / (fp + tn)

    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:,} | FP: {fp:,}")
    print(f"  FN: {fn:,} | TP: {tp:,}")
    print(f"\nFalse Positive Rate: {fp_rate:.2%}")
    print(f"{'='*60}")

    return {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp_rate': fp_rate,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }


def save_results(results, filename):
    """Save results to pickle file."""
    filepath = RESULTS_DIR / filename
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"💾 Results saved to {filepath}")


def main():
    """Main training pipeline."""

    print(f"\n{'='*80}")
    print("STEP 1: LOAD AND PREPROCESS DATA")
    print(f"{'='*80}")

    # Initialize preprocessor
    preprocessor = CICIDS2017Preprocessor(scaler_type='standard', random_state=RANDOM_STATE)

    # Load data
    print("Loading Monday BENIGN data (training)...")
    df_monday = preprocessor.load_data(str(DATA_DIR / 'Monday-WorkingHours.pcap_ISCX.csv'))

    print("\nLoading Wednesday DoS/DDoS data (testing)...")
    df_wednesday = preprocessor.load_data(str(DATA_DIR / 'Wednesday-workingHours.pcap_ISCX.csv'))

    # Sample
    print(f"\nSampling {TRAIN_SAMPLES} training samples...")
    df_train = df_monday.sample(n=min(TRAIN_SAMPLES, len(df_monday)), random_state=RANDOM_STATE)

    print(f"Sampling {TEST_SAMPLES} test samples...")
    df_test = df_wednesday.sample(n=min(TEST_SAMPLES, len(df_wednesday)), random_state=RANDOM_STATE)

    print(f"\n✅ Data loaded")
    print(f"   Training samples: {len(df_train):,}")
    print(f"   Test samples: {len(df_test):,}")

    # Preprocess
    print("\nPreprocessing data...")

    # Clean training data
    df_train_clean = preprocessor.clean_data(df_train, remove_duplicates=True,
                                             handle_inf='replace', handle_nan='median')
    df_train_encoded, _ = preprocessor.encode_labels(df_train_clean, binary_encoding=True)
    X_train = df_train_encoded.drop(['Label', 'Label_Original'], axis=1, errors='ignore')
    y_train = df_train_encoded['Label'].values if 'Label' in df_train_encoded.columns else np.zeros(len(df_train_encoded))

    # Scale training data
    X_train_scaled = preprocessor.scale_features(X_train.values)
    if isinstance(X_train_scaled, tuple):
        X_train_scaled = X_train_scaled[0]

    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Training labels (should be all BENIGN=0): {np.unique(y_train, return_counts=True)}")

    # Clean test data
    df_test_clean = preprocessor.clean_data(df_test, remove_duplicates=True,
                                            handle_inf='replace', handle_nan='median')
    df_test_encoded, _ = preprocessor.encode_labels(df_test_clean, binary_encoding=True)
    X_test = df_test_encoded.drop(['Label', 'Label_Original'], axis=1, errors='ignore')
    y_test = df_test_encoded['Label'].values

    # Scale test data
    X_test_scaled = preprocessor.scaler.transform(X_test.values)

    print(f"\nTest data shape: {X_test_scaled.shape}")
    attack_rate = y_test.mean()
    print(f"Attack rate in test set: {attack_rate:.1%}")
    print(f"✅ Preprocessing complete")

    # Use scaled data
    X_train = X_train_scaled
    X_test = X_test_scaled

    # Store results
    all_results = []

    # ========================================================================
    # MODEL 1: ISOLATION FOREST
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 2: TRAINING ISOLATION FOREST")
    print(f"{'='*80}")

    start_time = time.time()

    if_detector = IsolationForestDetector(
        contamination=0.1,
        n_estimators=100,
        random_state=RANDOM_STATE
    )

    print("Training Isolation Forest...")
    if_detector.fit(X_train)
    training_time = time.time() - start_time
    print(f"✅ Training complete in {training_time:.2f}s")

    print("Predicting on test set...")
    y_pred_if = if_detector.predict(X_test)

    if_results = evaluate_model(y_test, y_pred_if, "Isolation Forest")
    if_results['training_time'] = training_time
    all_results.append(if_results)

    model_path = MODELS_DIR / 'isolation_forest_final.pkl'
    if_detector.save(model_path)
    print(f"💾 Model saved to {model_path}")
    save_results(if_results, 'isolation_forest_results.pkl')

    # ========================================================================
    # MODEL 2: ONE-CLASS SVM
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: TRAINING ONE-CLASS SVM")
    print(f"{'='*80}")

    start_time = time.time()

    ocsvm_detector = OneClassSVMDetector(
        kernel='rbf',
        nu=0.01,
        gamma='scale'
    )

    # Subsample for OCSVM (faster training)
    X_train_ocsvm = X_train[:20000]
    print(f"Training One-Class SVM on {len(X_train_ocsvm):,} samples...")
    ocsvm_detector.fit(X_train_ocsvm)
    training_time = time.time() - start_time
    print(f"✅ Training complete in {training_time:.2f}s")

    print("Predicting on test set...")
    y_pred_ocsvm = ocsvm_detector.predict(X_test)

    ocsvm_results = evaluate_model(y_test, y_pred_ocsvm, "One-Class SVM")
    ocsvm_results['training_time'] = training_time
    all_results.append(ocsvm_results)

    model_path = MODELS_DIR / 'ocsvm_final.pkl'
    ocsvm_detector.save(model_path)
    print(f"💾 Model saved to {model_path}")
    save_results(ocsvm_results, 'ocsvm_results.pkl')

    # ========================================================================
    # MODEL 3 & 4: DEEP LEARNING (if TensorFlow available)
    # ========================================================================
    if TF_AVAILABLE:
        # AUTOENCODER
        print(f"\n{'='*80}")
        print("STEP 4: TRAINING AUTOENCODER")
        print(f"{'='*80}")

        start_time = time.time()

        ae_detector = AutoencoderDetector(
            encoding_dims=[40, 20],
            dropout_rate=0.2,
            l2_reg=1e-5
        )

        print("Training Autoencoder...")
        print("  - Early stopping enabled (patience=10)")
        print("  - This will take ~40 seconds...\n")

        try:
            from tensorflow.keras.callbacks import EarlyStopping

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )

            history = ae_detector.fit(
                X_train,
                epochs=100,
                batch_size=256,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=2
            )

            training_time = time.time() - start_time
            print(f"\n✅ Training complete in {training_time:.2f}s")

            print("Predicting on test set...")
            y_pred_ae = ae_detector.predict(X_test)

            ae_results = evaluate_model(y_test, y_pred_ae, "Autoencoder")
            ae_results['training_time'] = training_time
            all_results.append(ae_results)

            model_path = MODELS_DIR / 'autoencoder_final.h5'
            ae_detector.save(model_path)
            print(f"💾 Model saved to {model_path}")
            save_results(ae_results, 'autoencoder_results.pkl')

        except Exception as e:
            print(f"❌ Autoencoder training failed: {e}")

        # VAE
        print(f"\n{'='*80}")
        print("STEP 5: TRAINING VARIATIONAL AUTOENCODER (VAE)")
        print(f"{'='*80}")

        start_time = time.time()

        vae_detector = VAEDetector(
            latent_dim=20,
            encoder_dims=[50, 30],
            kl_weight=0.001,
            dropout_rate=0.2,
            l2_reg=1e-5
        )

        print("Training VAE...")
        print("  - Early stopping enabled (patience=10)")
        print("  - This will take ~5-10 minutes...\n")

        try:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )

            history = vae_detector.fit(
                X_train,
                epochs=100,
                batch_size=256,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=2
            )

            training_time = time.time() - start_time
            print(f"\n✅ Training complete in {training_time:.2f}s ({training_time/60:.1f} minutes)")

            print("Predicting on test set...")
            y_pred_vae = vae_detector.predict(X_test)

            vae_results = evaluate_model(y_test, y_pred_vae, "VAE")
            vae_results['training_time'] = training_time
            all_results.append(vae_results)

            model_path = MODELS_DIR / 'vae_final.h5'
            vae_detector.save(model_path)
            print(f"💾 Model saved to {model_path}")
            save_results(vae_results, 'vae_results.pkl')

        except Exception as e:
            print(f"❌ VAE training failed: {e}")

    else:
        print(f"\n{'='*80}")
        print("⚠️  SKIPPING DEEP LEARNING MODELS (TensorFlow not available)")
        print(f"{'='*80}")
        print("To train Autoencoder and VAE, fix TensorFlow installation.")
        print("See CLAUDE.md for details on Python 3.7 compatibility issue.")

    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_results)

    print("\nModel Performance Summary:")
    print(results_df[['model', 'f1', 'precision', 'recall', 'fp_rate', 'training_time']].to_string(index=False))

    # Find best model
    best_idx = results_df['f1'].idxmax()
    best_model = results_df.iloc[best_idx]

    print(f"\n🏆 BEST MODEL: {best_model['model']} (F1={best_model['f1']:.4f})")
    print(f"   - Precision: {best_model['precision']:.4f}")
    print(f"   - Recall: {best_model['recall']:.4f}")
    print(f"   - FP Rate: {best_model['fp_rate']:.2%}")
    print(f"   - Training Time: {best_model['training_time']:.2f}s")

    # Quality gate check
    print(f"\n{'='*60}")
    print("QUALITY GATE ASSESSMENT (F1 >= 0.85)")
    print(f"{'='*60}")

    if best_model['f1'] >= 0.85:
        print(f"✅ QUALITY GATE PASSED")
        print(f"   Ready for production deployment!")
    else:
        gap = 0.85 - best_model['f1']
        print(f"❌ QUALITY GATE NOT MET")
        print(f"   Gap to threshold: {gap:.4f} ({gap/0.85*100:.1f}%)")
        print(f"\nRecommendations:")
        print(f"  1. Try ensemble methods (combine OCSVM + best model)")
        print(f"  2. Hyperparameter tuning")
        print(f"  3. Train on full dataset (530K samples vs current 50K)")
        print(f"  4. Feature engineering (temporal patterns, statistical aggregations)")
        print(f"  5. Test alternative architectures (Transformer, GAN)")

    # Save comparison
    results_df.to_csv(RESULTS_DIR / 'final_comparison.csv', index=False)
    print(f"\n💾 Final comparison saved to {RESULTS_DIR / 'final_comparison.csv'}")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Models trained: {len(all_results)}")
    print(f"Total time: {sum(r['training_time'] for r in all_results):.1f}s")
    print(f"\nNext steps:")
    print(f"  1. Review results in {RESULTS_DIR}")
    print(f"  2. Analyze per-attack performance with notebooks/03_model_comparison.ipynb")
    print(f"  3. Update CLAUDE.md with final results")
    print(f"  4. Decide on production deployment strategy")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
