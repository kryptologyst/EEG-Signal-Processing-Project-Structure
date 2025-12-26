"""Evaluation metrics for EEG signal processing."""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, brier_score_loss
)
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics: List[str] = None
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric values
    """
    if metrics is None:
        metrics = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "accuracy"]
    
    results = {}
    
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    if "accuracy" in metrics:
        results["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    
    if "sensitivity" in metrics:
        results["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if "specificity" in metrics:
        results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    if "ppv" in metrics:
        results["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    if "npv" in metrics:
        results["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    if "f1" in metrics:
        precision = results.get("ppv", tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recall = results.get("sensitivity", tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        results["f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # ROC and PR metrics
    if "auroc" in metrics:
        try:
            results["auroc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            results["auroc"] = 0.5  # Random performance
    
    if "auprc" in metrics:
        try:
            results["auprc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            results["auprc"] = np.mean(y_true)  # Baseline performance
    
    return results


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Compute calibration metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary of calibration metrics
    """
    # Brier Score
    brier_score = brier_score_loss(y_true, y_prob)
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return {
        "brier_score": brier_score,
        "ece": ece
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ["Normal", "Epileptic"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {auprc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot calibration curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = (bin_lowers + bin_uppers) / 2
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(prop_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[len(bin_accuracies)])
            bin_counts.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bin_confidences, bin_accuracies, 'o-', label='Model Calibration')
    ax.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: List[str] = None
) -> Dict[str, float]:
    """Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run on
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in data_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute classification metrics
    classification_metrics = compute_classification_metrics(
        all_labels, all_preds, all_probs, metrics
    )
    
    # Compute calibration metrics
    calibration_metrics = compute_calibration_metrics(all_labels, all_probs)
    
    # Combine all metrics
    all_metrics = {**classification_metrics, **calibration_metrics}
    
    return all_metrics
