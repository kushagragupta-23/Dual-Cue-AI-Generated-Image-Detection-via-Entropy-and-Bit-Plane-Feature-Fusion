"""
Academic Metric Calculators for AI-Generated Image Detection.

Provides threshold-free and threshold-dependent binary classification metrics:
    - Accuracy, Precision, Recall, F1-Score
    - ROC-AUC (Area Under ROC Curve)
    - Average Precision (AP)
    - Confusion Matrix with per-domain breakdown
    - Optimal threshold selection via Youden's J statistic

All calculators accept numpy arrays and return serializable Python dicts.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger("metrics")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true: Ground truth binary labels, shape (N,).
        y_pred_prob: Predicted probabilities for class 1, shape (N,).
        threshold: Decision threshold for hard predictions (default 0.5).

    Returns:
        dict with keys: accuracy, precision, recall, f1_score, roc_auc,
        average_precision, threshold_used, num_samples.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold_used": float(threshold),
        "num_samples": int(len(y_true)),
    }

    # Threshold-free metrics (require sufficient class diversity)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_prob))
    except ValueError:
        metrics["roc_auc"] = 0.0
        logger.warning("ROC-AUC undefined (single class present in y_true).")

    try:
        metrics["average_precision"] = float(
            average_precision_score(y_true, y_pred_prob)
        )
    except ValueError:
        metrics["average_precision"] = 0.0

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Compute binary confusion matrix components.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred_prob: Predicted probabilities, shape (N,).
        threshold: Decision threshold (default 0.5).

    Returns:
        dict with keys: tn, fp, fn, tp.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the optimal classification threshold using Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1 = TPR - FPR

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred_prob: Predicted probabilities, shape (N,).

    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal_threshold).
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = float(thresholds[best_idx])
    except ValueError:
        optimal_threshold = 0.5

    metrics = compute_binary_metrics(y_true, y_pred_prob, threshold=optimal_threshold)
    return optimal_threshold, metrics


def compute_per_domain_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    domains: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by generator domain.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred_prob: Predicted probabilities, shape (N,).
        domains: Domain label for each sample, shape (N,).
        threshold: Decision threshold (default 0.5).

    Returns:
        dict mapping domain_name → metrics_dict.
    """
    unique_domains = sorted(set(domains))
    per_domain = {}

    for domain in unique_domains:
        mask = np.array([d == domain for d in domains])
        if mask.sum() < 2:
            continue

        domain_true = y_true[mask]
        domain_probs = y_pred_prob[mask]

        per_domain[domain] = compute_binary_metrics(
            domain_true, domain_probs, threshold=threshold
        )

    return per_domain


def format_metrics_table(
    metrics: Dict[str, float],
    title: str = "Evaluation Results",
) -> str:
    """
    Format metrics as a Markdown table string.

    Args:
        metrics: Dictionary of metric_name → value.
        title: Table title.

    Returns:
        str: Formatted Markdown table.
    """
    lines = [f"### {title}", "", "| Metric | Value |", "|--------|-------|"]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")

    return "\n".join(lines)


def format_per_domain_table(
    per_domain: Dict[str, Dict[str, float]],
    title: str = "Per-Domain Results",
) -> str:
    """
    Format per-domain metrics as a Markdown table.

    Args:
        per_domain: Dictionary of domain_name → metrics_dict.
        title: Table title.

    Returns:
        str: Formatted Markdown table.
    """
    lines = [
        f"### {title}",
        "",
        "| Domain | Accuracy | Precision | Recall | F1 | AUC | AP |",
        "|--------|----------|-----------|--------|-----|-----|-----|",
    ]
    for domain, m in sorted(per_domain.items()):
        lines.append(
            f"| {domain} "
            f"| {m.get('accuracy', 0):.4f} "
            f"| {m.get('precision', 0):.4f} "
            f"| {m.get('recall', 0):.4f} "
            f"| {m.get('f1_score', 0):.4f} "
            f"| {m.get('roc_auc', 0):.4f} "
            f"| {m.get('average_precision', 0):.4f} |"
        )

    return "\n".join(lines)


__all__ = [
    "compute_binary_metrics",
    "compute_confusion_matrix",
    "find_optimal_threshold",
    "compute_per_domain_metrics",
    "format_metrics_table",
    "format_per_domain_table",
]
