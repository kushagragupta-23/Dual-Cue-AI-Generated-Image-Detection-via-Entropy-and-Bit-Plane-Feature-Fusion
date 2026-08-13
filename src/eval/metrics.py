"""
Academic Metrics Calculator for HydraFusion-Net.

Provides standardized computation of:
  - Binary Accuracy
  - Average Precision (AP)
  - ROC-AUC (Area Under Receiver Operating Characteristic)
  - Precision, Recall, F1-Score
  - Confusion Matrix (normalized and raw)

All metrics follow sklearn conventions and are validated against
synthetic ground-truth arrays for exact reproducibility.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_curve,
        precision_recall_curve,
    )
except ImportError:
    raise ImportError(
        "scikit-learn is required for metrics computation. "
        "Install via: pip install scikit-learn"
    )


@dataclass
class MetricsResult:
    """Container for a complete set of binary classification metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    average_precision: float = 0.0
    confusion_matrix_raw: Optional[np.ndarray] = None
    confusion_matrix_normalized: Optional[np.ndarray] = None
    roc_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    pr_curve: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Serialize scalar metrics to a JSON-compatible dictionary."""
        return {
            "accuracy": round(self.accuracy * 100, 2),
            "precision": round(self.precision * 100, 2),
            "recall": round(self.recall * 100, 2),
            "f1_score": round(self.f1 * 100, 2),
            "roc_auc": round(self.roc_auc, 4),
            "average_precision": round(self.average_precision, 4),
            "num_samples": self.num_samples,
        }

    def summary_string(self) -> str:
        """Format a human-readable one-line summary."""
        return (
            f"Acc={self.accuracy*100:.2f}% | "
            f"P={self.precision*100:.2f}% | "
            f"R={self.recall*100:.2f}% | "
            f"F1={self.f1*100:.2f}% | "
            f"AUC={self.roc_auc:.4f} | "
            f"AP={self.average_precision:.4f}"
        )


class ForensicMetricsCalculator:
    """
    Computes a full suite of academic binary classification metrics.

    Designed for AI-Generated Image Detection evaluation where:
      - Label 0 = Real (Authentic)
      - Label 1 = Fake (AI-Generated)

    Usage:
        calc = ForensicMetricsCalculator()
        calc.update(predictions=[1, 0, 1], labels=[1, 1, 0], probabilities=[0.9, 0.3, 0.7])
        result = calc.compute()
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear accumulated predictions and labels."""
        self._labels: List[int] = []
        self._predictions: List[int] = []
        self._probabilities: List[float] = []

    def update(
        self,
        predictions: List[int],
        labels: List[int],
        probabilities: Optional[List[float]] = None,
    ) -> None:
        """
        Accumulate a batch of predictions and ground truth labels.

        Args:
            predictions: Binary predicted labels (0 or 1).
            labels: Ground truth binary labels (0 or 1).
            probabilities: Predicted probabilities for the positive class (Fake).
                          Required for ROC-AUC and Average Precision.
        """
        self._labels.extend(labels)
        self._predictions.extend(predictions)
        if probabilities is not None:
            self._probabilities.extend(probabilities)

    def compute(self) -> MetricsResult:
        """
        Compute all metrics from accumulated predictions and labels.

        Returns:
            MetricsResult containing accuracy, precision, recall, F1,
            ROC-AUC, AP, confusion matrices, and curve data.
        """
        if not self._labels:
            return MetricsResult()

        y_true = np.array(self._labels)
        y_pred = np.array(self._predictions)
        has_probs = len(self._probabilities) == len(self._labels)
        y_prob = np.array(self._probabilities) if has_probs else None

        result = MetricsResult(num_samples=len(y_true))

        # Core metrics
        result.accuracy = float(accuracy_score(y_true, y_pred))
        result.precision = float(precision_score(y_true, y_pred, zero_division=0))
        result.recall = float(recall_score(y_true, y_pred, zero_division=0))
        result.f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Confusion matrices
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        result.confusion_matrix_raw = cm
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_sum = np.where(cm_sum == 0, 1, cm_sum)  # Prevent division by zero
        result.confusion_matrix_normalized = cm.astype(float) / cm_sum

        # Probability-dependent metrics
        if y_prob is not None:
            y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
            # Guard against single-class test sets
            unique_labels = np.unique(y_true)
            if len(unique_labels) > 1:
                result.roc_auc = float(roc_auc_score(y_true, y_prob))
                result.average_precision = float(
                    average_precision_score(y_true, y_prob)
                )

                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
                result.roc_curve = (fpr, tpr, roc_thresholds)

                # Precision-Recall curve data
                pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
                    y_true, y_prob
                )
                result.pr_curve = (pr_precision, pr_recall, pr_thresholds)
            else:
                # Single-class edge case
                result.roc_auc = 0.0
                result.average_precision = 0.0

        return result


def compute_per_generator_metrics(
    generator_results: Dict[str, Tuple[List[int], List[int], List[float]]],
) -> Dict[str, MetricsResult]:
    """
    Compute metrics separately for each generator domain.

    Args:
        generator_results: Dict mapping generator name to
            (predictions, labels, probabilities) tuples.

    Returns:
        Dict mapping generator name to MetricsResult.
    """
    results = {}
    for gen_name, (preds, labels, probs) in generator_results.items():
        calc = ForensicMetricsCalculator()
        calc.update(predictions=preds, labels=labels, probabilities=probs)
        results[gen_name] = calc.compute()
    return results


def format_metrics_table(
    metrics_dict: Dict[str, MetricsResult],
    title: str = "Cross-Generator Zero-Shot Evaluation",
) -> str:
    """
    Format per-generator metrics as a Markdown table.

    Args:
        metrics_dict: Dict mapping generator name to MetricsResult.
        title: Title for the table.

    Returns:
        Formatted Markdown string.
    """
    lines = [
        f"### {title}\n",
        "| Generator | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC | AP | Samples |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    total_correct = 0
    total_samples = 0

    for gen_name, result in sorted(metrics_dict.items()):
        d = result.to_dict()
        lines.append(
            f"| {gen_name} | {d['accuracy']:.2f} | {d['precision']:.2f} | "
            f"{d['recall']:.2f} | {d['f1_score']:.2f} | {d['roc_auc']:.4f} | "
            f"{d['average_precision']:.4f} | {d['num_samples']} |"
        )
        total_correct += int(result.accuracy * result.num_samples)
        total_samples += result.num_samples

    # Average row
    if total_samples > 0:
        all_results = list(metrics_dict.values())
        avg_acc = np.mean([r.accuracy for r in all_results]) * 100
        avg_p = np.mean([r.precision for r in all_results]) * 100
        avg_r = np.mean([r.recall for r in all_results]) * 100
        avg_f1 = np.mean([r.f1 for r in all_results]) * 100
        avg_auc = np.mean([r.roc_auc for r in all_results])
        avg_ap = np.mean([r.average_precision for r in all_results])
        lines.append(
            f"| **Average** | **{avg_acc:.2f}** | **{avg_p:.2f}** | "
            f"**{avg_r:.2f}** | **{avg_f1:.2f}** | **{avg_auc:.4f}** | "
            f"**{avg_ap:.4f}** | **{total_samples}** |"
        )

    return "\n".join(lines)
