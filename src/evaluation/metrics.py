"""
Evaluation Metrics for Anomaly Detection Models

This module provides reusable evaluation functions for network intrusion detection:
- Classification metrics (precision, recall, F1, accuracy)
- ROC curve and AUC calculation
- Confusion matrix visualization
- Performance summary reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_proba: Optional[np.ndarray] = None,
                                   pos_label: int = 1) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels (0=normal, 1=anomaly)
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC calculation)
        pos_label: Positive class label (default 1 for anomaly)

    Returns:
        Dictionary containing all metrics
    """
    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)

    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # AUC if probabilities provided
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)
        metrics['auc'] = auc(fpr, tpr)

    return metrics


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: Optional[list] = None,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix',
                         save_path: Optional[Union[str, Path]] = None,
                         figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix with visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes (default: ['Normal', 'Anomaly'])
        normalize: Whether to normalize by row (true class)
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    if class_names is None:
        class_names = ['Normal', 'Anomaly']

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    return fig


def plot_roc_curve(y_true: np.ndarray,
                  y_proba: np.ndarray,
                  title: str = 'ROC Curve',
                  save_path: Optional[Union[str, Path]] = None,
                  figsize: Tuple[int, int] = (8, 6),
                  pos_label: int = 1) -> plt.Figure:
    """
    Plot ROC curve with AUC.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        pos_label: Positive class label

    Returns:
        Matplotlib figure object
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
            label='Random Classifier')

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")

    return fig


def plot_feature_importance(feature_names: list,
                           importance_scores: np.ndarray,
                           top_n: int = 20,
                           title: str = 'Feature Importance',
                           save_path: Optional[Union[str, Path]] = None,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot feature importance scores.

    Args:
        feature_names: List of feature names
        importance_scores: Importance scores for each feature
        top_n: Number of top features to display
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Create DataFrame for easy sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })

    # Sort by importance and get top N
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar plot
    bars = ax.barh(range(len(importance_df)), importance_df['Importance'].values,
                   color='steelblue', edgecolor='black', linewidth=0.5)

    # Set y-tick labels
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'].values)

    # Formatting
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Highest importance at top
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")

    return fig


def generate_performance_summary(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                y_proba: Optional[np.ndarray] = None,
                                model_name: str = 'Model',
                                save_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Generate comprehensive performance summary report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        model_name: Name of the model
        save_path: Path to save CSV report (optional)

    Returns:
        DataFrame with performance metrics
    """
    # Compute metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    # Create summary DataFrame
    summary = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [f"{metrics['accuracy']:.4f}"],
        'Precision': [f"{metrics['precision']:.4f}"],
        'Recall': [f"{metrics['recall']:.4f}"],
        'F1 Score': [f"{metrics['f1']:.4f}"],
        'AUC': [f"{metrics.get('auc', 0):.4f}"],
        'True Positives': [metrics['true_positives']],
        'True Negatives': [metrics['true_negatives']],
        'False Positives': [metrics['false_positives']],
        'False Negatives': [metrics['false_negatives']],
        'Specificity': [f"{metrics['specificity']:.4f}"],
        'FPR': [f"{metrics['false_positive_rate']:.4f}"],
        'FNR': [f"{metrics['false_negative_rate']:.4f}"]
    })

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_path, index=False)
        print(f"Performance summary saved to: {save_path}")

    return summary


def print_classification_report(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               class_names: Optional[list] = None) -> None:
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes (default: ['Normal', 'Anomaly'])
    """
    if class_names is None:
        class_names = ['Normal', 'Anomaly']

    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("="*70 + "\n")


def evaluate_model(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  y_proba: Optional[np.ndarray] = None,
                  feature_names: Optional[list] = None,
                  feature_importance: Optional[np.ndarray] = None,
                  model_name: str = 'Model',
                  output_dir: Optional[Union[str, Path]] = None,
                  verbose: bool = True) -> Dict:
    """
    Complete model evaluation with all metrics and visualizations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        feature_names: Feature names (optional, for feature importance)
        feature_importance: Feature importance scores (optional)
        model_name: Name of the model
        output_dir: Directory to save all outputs (optional)
        verbose: Print detailed results

    Returns:
        Dictionary containing:
            - metrics: Dict of all metrics
            - summary: Performance summary DataFrame
            - figures: Dict of matplotlib figures
    """
    results = {}

    # Compute metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    results['metrics'] = metrics

    # Generate summary
    summary = generate_performance_summary(
        y_true, y_pred, y_proba,
        model_name=model_name,
        save_path=Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_summary.csv" if output_dir else None
    )
    results['summary'] = summary

    # Print report if verbose
    if verbose:
        print("\n" + "="*70)
        print(f"{model_name.upper()} - PERFORMANCE EVALUATION")
        print("="*70)
        print(f"\nAccuracy:   {metrics['accuracy']:.4f}")
        print(f"Precision:  {metrics['precision']:.4f}")
        print(f"Recall:     {metrics['recall']:.4f}")
        print(f"F1 Score:   {metrics['f1']:.4f}")
        if y_proba is not None:
            print(f"AUC:        {metrics.get('auc', 0):.4f}")
        print(f"\nTrue Positives:  {metrics['true_positives']:,}")
        print(f"True Negatives:  {metrics['true_negatives']:,}")
        print(f"False Positives: {metrics['false_positives']:,}")
        print(f"False Negatives: {metrics['false_negatives']:,}")
        print(f"\nSpecificity: {metrics['specificity']:.4f}")
        print(f"FPR:         {metrics['false_positive_rate']:.4f}")
        print("="*70 + "\n")

        print_classification_report(y_true, y_pred)

    # Create visualizations
    figures = {}

    # Confusion matrix
    cm_path = Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png" if output_dir else None
    figures['confusion_matrix'] = plot_confusion_matrix(
        y_true, y_pred,
        title=f'{model_name} - Confusion Matrix',
        save_path=cm_path
    )

    # ROC curve if probabilities available
    if y_proba is not None:
        roc_path = Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_roc_curve.png" if output_dir else None
        figures['roc_curve'] = plot_roc_curve(
            y_true, y_proba,
            title=f'{model_name} - ROC Curve',
            save_path=roc_path
        )

    # Feature importance if available
    if feature_names is not None and feature_importance is not None:
        fi_path = Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_feature_importance.png" if output_dir else None
        figures['feature_importance'] = plot_feature_importance(
            feature_names, feature_importance,
            title=f'{model_name} - Feature Importance',
            save_path=fi_path
        )

    results['figures'] = figures

    return results


if __name__ == "__main__":
    # Example usage
    print("Evaluation Metrics Module")
    print("="*70)
    print("\nExample usage:")
    print("""
from src.evaluation.metrics import evaluate_model

# Evaluate model with all metrics and visualizations
results = evaluate_model(
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    feature_names=feature_names,
    feature_importance=importance_scores,
    model_name='Isolation Forest',
    output_dir='results/',
    verbose=True
)

metrics = results['metrics']
summary = results['summary']
figures = results['figures']
    """)
    print("="*70)
