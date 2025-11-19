"""
Multi-Attack Type Testing Script

Tests VAE and OCSVM models across different attack categories to validate
generalization beyond DoS/DDoS attacks.

Attack types tested:
1. PortScan (Friday-Afternoon-PortScan.pcap_ISCX.csv)
2. Web Attacks (Thursday-Morning-WebAttacks.pcap_ISCX.csv)
3. Brute Force (Tuesday-WorkingHours.pcap_ISCX.csv)

Quality Gate: F1 > 0.80 on at least 2 of 3 attack types
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent))

from src.data.preprocessing import CICIDS2017Preprocessor
from src.models.vae import VAEDetector
from src.models.one_class_svm import OneClassSVMDetector
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report, roc_auc_score
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class MultiAttackTester:
    """Test models across multiple attack types."""

    def __init__(self, vae_path: str, ocsvm_path: str, data_dir: str):
        """
        Initialize tester with model paths.

        Args:
            vae_path: Path to VAE model (.h5 file)
            ocsvm_path: Path to OCSVM model (.pkl file)
            data_dir: Directory containing CICIDS2017 CSV files
        """
        self.data_dir = Path(data_dir)
        self.vae_path = Path(vae_path)
        self.ocsvm_path = Path(ocsvm_path)

        # Test scenarios
        self.test_scenarios = {
            'PortScan': {
                'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                'expected_attacks': ['PortScan'],
                'description': 'Network reconnaissance attacks'
            },
            'Web Attacks': {
                'file': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
                'expected_attacks': ['Web Attack – Brute Force', 'Web Attack – XSS',
                                    'Web Attack – Sql Injection'],
                'description': 'Application-layer web attacks'
            },
            'Brute Force': {
                'file': 'Tuesday-WorkingHours.pcap_ISCX.csv',
                'expected_attacks': ['FTP-Patator', 'SSH-Patator'],
                'description': 'Password brute-forcing attacks'
            }
        }

        self.results = {}

    def load_models(self):
        """Load trained VAE and OCSVM models."""
        print("\n" + "="*70)
        print("LOADING MODELS")
        print("="*70)

        # Load VAE (classmethod returns new instance)
        print(f"\nLoading VAE from: {self.vae_path}")
        self.vae = VAEDetector.load(str(self.vae_path))
        print(f"VAE loaded: {self.vae.latent_dim}D latent space, {self.vae.n_features} features")

        # Load OCSVM
        print(f"\nLoading OCSVM from: {self.ocsvm_path}")
        with open(self.ocsvm_path, 'rb') as f:
            ocsvm_data = pickle.load(f)

        self.ocsvm = ocsvm_data['model']
        self.ocsvm_scaler = ocsvm_data.get('scaler', None)
        print(f"OCSVM loaded: nu={self.ocsvm.nu}, kernel={self.ocsvm.kernel}")

        print("="*70 + "\n")

    def load_and_prepare_data(self, scenario: str, max_samples: int = 100000) -> Tuple:
        """
        Load and prepare data for testing.

        Args:
            scenario: Attack type scenario name
            max_samples: Maximum samples to load

        Returns:
            Tuple of (X, y, y_original, df_info)
        """
        scenario_info = self.test_scenarios[scenario]
        file_path = self.data_dir / scenario_info['file']

        print(f"\n{'='*70}")
        print(f"PREPARING DATA: {scenario}")
        print(f"{'='*70}")
        print(f"File: {scenario_info['file']}")
        print(f"Description: {scenario_info['description']}")

        # Load data using preprocessor
        preprocessor = CICIDS2017Preprocessor(random_state=42)
        df = preprocessor.load_data(str(file_path))

        # Clean data
        df_clean = preprocessor.clean_data(
            df,
            remove_duplicates=True,
            handle_inf='replace',
            handle_nan='median'
        )

        # Encode labels
        df_encoded, label_mapping = preprocessor.encode_labels(
            df_clean,
            binary_encoding=True
        )

        # Sample if needed
        if len(df_encoded) > max_samples:
            # Stratified sampling to preserve attack/benign ratio
            df_benign = df_encoded[df_encoded['Label'] == 0]
            df_attack = df_encoded[df_encoded['Label'] == 1]

            attack_ratio = len(df_attack) / len(df_encoded)
            n_attack = int(max_samples * attack_ratio)
            n_benign = max_samples - n_attack

            df_attack_sample = df_attack.sample(n=min(n_attack, len(df_attack)), random_state=42)
            df_benign_sample = df_benign.sample(n=min(n_benign, len(df_benign)), random_state=42)

            df_encoded = pd.concat([df_attack_sample, df_benign_sample]).sample(frac=1, random_state=42)
            print(f"\nSampled {len(df_encoded):,} records (attack ratio preserved: {attack_ratio:.1%})")

        # Separate features and labels
        y = df_encoded['Label'].values
        y_original = df_encoded['Label_Original'].values if 'Label_Original' in df_encoded.columns else None

        X = df_encoded.drop(['Label'], axis=1)
        if 'Label_Original' in X.columns:
            X = X.drop('Label_Original', axis=1)

        feature_names = X.columns.tolist()

        # Check feature count alignment
        expected_features = self.vae.n_features if hasattr(self.vae, 'n_features') and self.vae.n_features is not None else None
        current_features = len(feature_names)

        if expected_features is not None and current_features != expected_features:
            print(f"\nWARNING: Feature count mismatch!")
            print(f"  Expected: {expected_features} features (from training)")
            print(f"  Current: {current_features} features (from {scenario} data)")
            print(f"  Difference: {current_features - expected_features} features")

            # If current has MORE features, we need to align
            if current_features > expected_features:
                print(f"\n  Strategy: Using first {expected_features} features for compatibility")
                X = X.iloc[:, :expected_features]
                feature_names = feature_names[:expected_features]
            else:
                # If current has FEWER features, this is a problem
                raise ValueError(f"Dataset has {current_features} features but model expects {expected_features}")
        elif expected_features is None:
            # Try to infer from scaler
            if hasattr(self.vae, 'scaler') and hasattr(self.vae.scaler, 'mean_'):
                expected_features = len(self.vae.scaler.mean_)
                print(f"\nInferred expected features from scaler: {expected_features}")
                if current_features != expected_features:
                    print(f"  Current features: {current_features}")
                    print(f"  Trimming to {expected_features} features")
                    X = X.iloc[:, :expected_features]
                    feature_names = feature_names[:expected_features]

        X = X.values

        # Scale features using VAE's scaler (or OCSVM's if VAE doesn't have one)
        if hasattr(self.vae, 'scaler') and self.vae.scaler is not None:
            try:
                X_scaled = self.vae.scaler.transform(X)
            except Exception as e:
                print(f"ERROR scaling with VAE scaler: {e}")
                print("Using unscaled features")
                X_scaled = X
        elif self.ocsvm_scaler is not None:
            X_scaled = self.ocsvm_scaler.transform(X)
        else:
            print("WARNING: No scaler found, using raw features")
            X_scaled = X

        # Get attack distribution
        attack_counts = pd.Series(y_original).value_counts() if y_original is not None else None

        df_info = {
            'total_samples': len(X_scaled),
            'benign_samples': (y == 0).sum(),
            'attack_samples': (y == 1).sum(),
            'attack_ratio': (y == 1).sum() / len(y),
            'attack_counts': attack_counts,
            'feature_count': X_scaled.shape[1]
        }

        print(f"\nDataset Summary:")
        print(f"  Total samples: {df_info['total_samples']:,}")
        print(f"  BENIGN: {df_info['benign_samples']:,} ({(1-df_info['attack_ratio'])*100:.1f}%)")
        print(f"  ATTACKS: {df_info['attack_samples']:,} ({df_info['attack_ratio']*100:.1f}%)")
        print(f"  Features: {df_info['feature_count']}")

        if attack_counts is not None:
            print("\n  Attack Type Distribution:")
            for attack_type, count in attack_counts.items():
                if attack_type != 'BENIGN':
                    print(f"    {attack_type}: {count:,}")

        print("="*70 + "\n")

        return X_scaled, y, y_original, df_info

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str) -> Dict:
        """
        Calculate evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            model_name: Name of model

        Returns:
            Dictionary of metrics
        """
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # False positive rate on benign traffic
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Detection rate (recall)
        detection_rate = recall

        metrics = {
            'model': model_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'fp_rate': fp_rate,
            'detection_rate': detection_rate
        }

        return metrics

    def per_attack_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_original: np.ndarray) -> pd.DataFrame:
        """
        Analyze performance per attack type.

        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels
            y_original: Original attack type labels

        Returns:
            DataFrame with per-attack metrics
        """
        if y_original is None:
            return None

        results = []

        for attack_type in pd.Series(y_original).unique():
            if attack_type == 'BENIGN':
                continue

            # Get samples of this attack type
            mask = (y_original == attack_type)
            y_true_subset = y_true[mask]
            y_pred_subset = y_pred[mask]

            # Calculate metrics
            detected = (y_pred_subset == 1).sum()
            total = len(y_true_subset)
            detection_rate = detected / total if total > 0 else 0

            # F1 score (treating this attack vs rest)
            f1 = f1_score(y_true_subset, y_pred_subset, zero_division=0)
            precision = precision_score(y_true_subset, y_pred_subset, zero_division=0)
            recall = recall_score(y_true_subset, y_pred_subset, zero_division=0)

            results.append({
                'attack_type': attack_type,
                'samples': total,
                'detected': detected,
                'missed': total - detected,
                'detection_rate': detection_rate,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('samples', ascending=False)

        return df_results

    def test_scenario(self, scenario: str) -> Dict:
        """
        Test both models on a single attack scenario.

        Args:
            scenario: Attack type scenario name

        Returns:
            Dictionary of results
        """
        # Load and prepare data
        X, y, y_original, df_info = self.load_and_prepare_data(scenario)

        print(f"\n{'='*70}")
        print(f"TESTING MODELS: {scenario}")
        print(f"{'='*70}")

        # Test VAE
        print("\nTesting VAE...")
        start_time = time.time()
        y_pred_vae = self.vae.predict(X)
        vae_time = time.time() - start_time
        vae_metrics = self.evaluate_model(y, y_pred_vae, 'VAE')
        vae_metrics['inference_time'] = vae_time
        vae_per_attack = self.per_attack_analysis(y, y_pred_vae, y_original)

        print(f"  VAE F1: {vae_metrics['f1']:.4f}")
        print(f"  VAE Precision: {vae_metrics['precision']:.4f}")
        print(f"  VAE Recall: {vae_metrics['recall']:.4f}")
        print(f"  VAE FP Rate: {vae_metrics['fp_rate']*100:.2f}%")
        print(f"  VAE Inference Time: {vae_time:.2f}s")

        # Test OCSVM
        print("\nTesting OCSVM...")
        start_time = time.time()
        y_pred_ocsvm = self.ocsvm.predict(X)
        # OCSVM returns -1 for outliers, 1 for inliers - convert to 0/1
        y_pred_ocsvm = (y_pred_ocsvm == -1).astype(int)
        ocsvm_time = time.time() - start_time
        ocsvm_metrics = self.evaluate_model(y, y_pred_ocsvm, 'OCSVM')
        ocsvm_metrics['inference_time'] = ocsvm_time
        ocsvm_per_attack = self.per_attack_analysis(y, y_pred_ocsvm, y_original)

        print(f"  OCSVM F1: {ocsvm_metrics['f1']:.4f}")
        print(f"  OCSVM Precision: {ocsvm_metrics['precision']:.4f}")
        print(f"  OCSVM Recall: {ocsvm_metrics['recall']:.4f}")
        print(f"  OCSVM FP Rate: {ocsvm_metrics['fp_rate']*100:.2f}%")
        print(f"  OCSVM Inference Time: {ocsvm_time:.2f}s")

        print("="*70 + "\n")

        # Store results
        scenario_result = {
            'scenario': scenario,
            'description': self.test_scenarios[scenario]['description'],
            'dataset_info': df_info,
            'vae_metrics': vae_metrics,
            'ocsvm_metrics': ocsvm_metrics,
            'vae_per_attack': vae_per_attack,
            'ocsvm_per_attack': ocsvm_per_attack
        }

        return scenario_result

    def run_all_tests(self) -> Dict:
        """Run tests on all attack scenarios."""
        print("\n" + "="*70)
        print("MULTI-ATTACK TYPE TESTING")
        print("="*70)
        print("\nTesting VAE and OCSVM models across 3 attack categories:")
        print("1. PortScan (Network Reconnaissance)")
        print("2. Web Attacks (Application Layer)")
        print("3. Brute Force (Authentication)")
        print("="*70)

        # Load models
        self.load_models()

        # Test each scenario
        all_results = {}
        for scenario in self.test_scenarios.keys():
            result = self.test_scenario(scenario)
            all_results[scenario] = result
            self.results = all_results

        # Generate summary
        summary = self.generate_summary()
        all_results['summary'] = summary

        return all_results

    def generate_summary(self) -> Dict:
        """Generate summary of all tests."""
        print("\n" + "="*70)
        print("SUMMARY - MULTI-ATTACK TYPE PERFORMANCE")
        print("="*70)

        summary = {
            'vae_f1_scores': {},
            'ocsvm_f1_scores': {},
            'vae_avg_f1': 0,
            'ocsvm_avg_f1': 0,
            'quality_gate_status': {},
            'winner_per_scenario': {}
        }

        # Collect F1 scores
        for scenario, result in self.results.items():
            vae_f1 = result['vae_metrics']['f1']
            ocsvm_f1 = result['ocsvm_metrics']['f1']

            summary['vae_f1_scores'][scenario] = vae_f1
            summary['ocsvm_f1_scores'][scenario] = ocsvm_f1

            # Quality gate check (F1 > 0.80)
            summary['quality_gate_status'][scenario] = {
                'vae': 'PASS' if vae_f1 >= 0.80 else 'FAIL',
                'ocsvm': 'PASS' if ocsvm_f1 >= 0.80 else 'FAIL'
            }

            # Winner
            summary['winner_per_scenario'][scenario] = 'VAE' if vae_f1 > ocsvm_f1 else 'OCSVM'

        # Average F1
        summary['vae_avg_f1'] = np.mean(list(summary['vae_f1_scores'].values()))
        summary['ocsvm_avg_f1'] = np.mean(list(summary['ocsvm_f1_scores'].values()))

        # Overall quality gate
        vae_passes = sum(1 for status in summary['quality_gate_status'].values() if status['vae'] == 'PASS')
        ocsvm_passes = sum(1 for status in summary['quality_gate_status'].values() if status['ocsvm'] == 'PASS')

        summary['vae_quality_gate'] = 'PASS' if vae_passes >= 2 else 'FAIL'
        summary['ocsvm_quality_gate'] = 'PASS' if ocsvm_passes >= 2 else 'FAIL'

        # Print summary table
        print("\nF1 Score Comparison:")
        print(f"{'Scenario':<20} {'VAE F1':<12} {'OCSVM F1':<12} {'Winner':<10}")
        print("-" * 60)
        for scenario in self.test_scenarios.keys():
            vae_f1 = summary['vae_f1_scores'][scenario]
            ocsvm_f1 = summary['ocsvm_f1_scores'][scenario]
            winner = summary['winner_per_scenario'][scenario]
            print(f"{scenario:<20} {vae_f1:<12.4f} {ocsvm_f1:<12.4f} {winner:<10}")

        print("-" * 60)
        print(f"{'AVERAGE':<20} {summary['vae_avg_f1']:<12.4f} {summary['ocsvm_avg_f1']:<12.4f}")

        print("\nQuality Gate Status (F1 > 0.80 on 2+ scenarios):")
        print(f"  VAE: {summary['vae_quality_gate']} ({vae_passes}/3 scenarios passed)")
        print(f"  OCSVM: {summary['ocsvm_quality_gate']} ({ocsvm_passes}/3 scenarios passed)")

        print("="*70 + "\n")

        return summary

    def save_results(self, output_dir: str = 'results'):
        """Save all results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"\nSaving results to {output_dir}/")

        # Save complete results as pickle
        with open(output_dir / 'multi_attack_test_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print(f"  ✓ Saved: multi_attack_test_results.pkl")

        # Save summary as JSON
        if 'summary' in self.results:
            summary = self.results['summary'].copy()
            # Convert numpy types to Python types for JSON
            summary = json.loads(json.dumps(summary, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x))
            with open(output_dir / 'multi_attack_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  ✓ Saved: multi_attack_summary.json")

        # Save detailed results as CSV
        csv_rows = []
        for scenario, result in self.results.items():
            if scenario == 'summary':
                continue

            # VAE metrics
            csv_rows.append({
                'scenario': scenario,
                'model': 'VAE',
                'f1': result['vae_metrics']['f1'],
                'precision': result['vae_metrics']['precision'],
                'recall': result['vae_metrics']['recall'],
                'fp_rate': result['vae_metrics']['fp_rate'],
                'tp': result['vae_metrics']['tp'],
                'tn': result['vae_metrics']['tn'],
                'fp': result['vae_metrics']['fp'],
                'fn': result['vae_metrics']['fn'],
                'samples_tested': result['dataset_info']['total_samples'],
                'attack_proportion': result['dataset_info']['attack_ratio']
            })

            # OCSVM metrics
            csv_rows.append({
                'scenario': scenario,
                'model': 'OCSVM',
                'f1': result['ocsvm_metrics']['f1'],
                'precision': result['ocsvm_metrics']['precision'],
                'recall': result['ocsvm_metrics']['recall'],
                'fp_rate': result['ocsvm_metrics']['fp_rate'],
                'tp': result['ocsvm_metrics']['tp'],
                'tn': result['ocsvm_metrics']['tn'],
                'fp': result['ocsvm_metrics']['fp'],
                'fn': result['ocsvm_metrics']['fn'],
                'samples_tested': result['dataset_info']['total_samples'],
                'attack_proportion': result['dataset_info']['attack_ratio']
            })

        df_results = pd.DataFrame(csv_rows)
        df_results.to_csv(output_dir / 'multi_attack_test_results.csv', index=False)
        print(f"  ✓ Saved: multi_attack_test_results.csv")

        # Save per-attack breakdowns
        for scenario, result in self.results.items():
            if scenario == 'summary':
                continue

            # VAE per-attack
            if result['vae_per_attack'] is not None:
                filename = f"vae_{scenario.lower().replace(' ', '_')}_per_attack.csv"
                result['vae_per_attack'].to_csv(output_dir / filename, index=False)
                print(f"  ✓ Saved: {filename}")

            # OCSVM per-attack
            if result['ocsvm_per_attack'] is not None:
                filename = f"ocsvm_{scenario.lower().replace(' ', '_')}_per_attack.csv"
                result['ocsvm_per_attack'].to_csv(output_dir / filename, index=False)
                print(f"  ✓ Saved: {filename}")

        print("\nAll results saved successfully!\n")

    def plot_results(self, output_dir: str = 'results'):
        """Generate visualization comparing models across attack types."""
        output_dir = Path(output_dir)

        if 'summary' not in self.results:
            print("No summary found - run tests first")
            return

        summary = self.results['summary']

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. F1 Score Comparison (bar chart)
        ax1 = fig.add_subplot(gs[0, :])
        scenarios = list(self.test_scenarios.keys())
        vae_f1s = [summary['vae_f1_scores'][s] for s in scenarios]
        ocsvm_f1s = [summary['ocsvm_f1_scores'][s] for s in scenarios]

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax1.bar(x - width/2, vae_f1s, width, label='VAE', color='#2ecc71', alpha=0.8)
        bars2 = ax1.bar(x + width/2, ocsvm_f1s, width, label='OCSVM', color='#3498db', alpha=0.8)

        # Quality gate line
        ax1.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='Quality Gate (0.80)', alpha=0.7)

        ax1.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax1.set_title('F1 Score Comparison Across Attack Types', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=0)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 1.0)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

        # 2. Precision-Recall Comparison (scatter)
        ax2 = fig.add_subplot(gs[1, 0])
        for scenario in scenarios:
            vae_metrics = self.results[scenario]['vae_metrics']
            ocsvm_metrics = self.results[scenario]['ocsvm_metrics']

            ax2.scatter(vae_metrics['recall'], vae_metrics['precision'],
                       s=200, alpha=0.6, label=f'VAE - {scenario}', marker='o')
            ax2.scatter(ocsvm_metrics['recall'], ocsvm_metrics['precision'],
                       s=200, alpha=0.6, label=f'OCSVM - {scenario}', marker='s')

        ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax2.set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.05)

        # 3. False Positive Rate (bar chart)
        ax3 = fig.add_subplot(gs[1, 1])
        vae_fps = [self.results[s]['vae_metrics']['fp_rate'] * 100 for s in scenarios]
        ocsvm_fps = [self.results[s]['ocsvm_metrics']['fp_rate'] * 100 for s in scenarios]

        bars1 = ax3.bar(x - width/2, vae_fps, width, label='VAE', color='#e74c3c', alpha=0.8)
        bars2 = ax3.bar(x + width/2, ocsvm_fps, width, label='OCSVM', color='#f39c12', alpha=0.8)

        ax3.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
        ax3.set_ylabel('False Positive Rate (%)', fontsize=12, fontweight='bold')
        ax3.set_title('False Positive Rate (Lower is Better)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios, rotation=0)
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom', fontsize=9)

        # 4. Quality Gate Status (heatmap)
        ax4 = fig.add_subplot(gs[2, 0])
        gate_data = []
        for scenario in scenarios:
            status = summary['quality_gate_status'][scenario]
            gate_data.append([1 if status['vae'] == 'PASS' else 0,
                            1 if status['ocsvm'] == 'PASS' else 0])

        gate_data = np.array(gate_data).T
        sns.heatmap(gate_data, annot=True, fmt='d', cmap='RdYlGn',
                   xticklabels=scenarios, yticklabels=['VAE', 'OCSVM'],
                   cbar_kws={'label': '0=FAIL, 1=PASS'}, ax=ax4, vmin=0, vmax=1)
        ax4.set_title('Quality Gate Status (F1 > 0.80)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Attack Type', fontsize=12, fontweight='bold')

        # 5. Overall Metrics Summary (table)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        table_data = [
            ['Metric', 'VAE', 'OCSVM', 'Winner'],
            ['Avg F1', f"{summary['vae_avg_f1']:.4f}", f"{summary['ocsvm_avg_f1']:.4f}",
             'VAE' if summary['vae_avg_f1'] > summary['ocsvm_avg_f1'] else 'OCSVM'],
            ['Quality Gate', summary['vae_quality_gate'], summary['ocsvm_quality_gate'], '-'],
            ['Scenarios Passed',
             f"{sum(1 for s in summary['quality_gate_status'].values() if s['vae'] == 'PASS')}/3",
             f"{sum(1 for s in summary['quality_gate_status'].values() if s['ocsvm'] == 'PASS')}/3",
             '-']
        ]

        table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color quality gate cells
        table[(2, 1)].set_facecolor('#2ecc71' if summary['vae_quality_gate'] == 'PASS' else '#e74c3c')
        table[(2, 2)].set_facecolor('#2ecc71' if summary['ocsvm_quality_gate'] == 'PASS' else '#e74c3c')

        ax5.set_title('Overall Summary', fontsize=14, fontweight='bold', pad=20)

        plt.suptitle('Multi-Attack Type Testing: VAE vs OCSVM',
                    fontsize=16, fontweight='bold', y=0.98)

        # Save figure
        plt.savefig(output_dir / 'multi_attack_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {output_dir}/multi_attack_comparison.png\n")
        plt.close()


def main():
    """Main execution function."""
    # Paths
    project_dir = Path(__file__).parent
    vae_model_path = project_dir / 'models' / 'vae_200k.h5'
    ocsvm_model_path = project_dir / 'models' / 'ocsvm_200k.pkl'
    data_dir = project_dir / 'data' / 'raw'
    results_dir = project_dir / 'results'

    # Initialize tester
    tester = MultiAttackTester(
        vae_path=vae_model_path,
        ocsvm_path=ocsvm_model_path,
        data_dir=data_dir
    )

    # Run all tests
    results = tester.run_all_tests()

    # Save results
    tester.save_results(output_dir=results_dir)

    # Generate visualizations
    tester.plot_results(output_dir=results_dir)

    print("\n" + "="*70)
    print("MULTI-ATTACK TESTING COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}/")
    print("  - multi_attack_test_results.pkl (complete results)")
    print("  - multi_attack_test_results.csv (summary table)")
    print("  - multi_attack_summary.json (JSON summary)")
    print("  - multi_attack_comparison.png (visualization)")
    print("  - per-attack breakdown CSVs (detailed analysis)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
