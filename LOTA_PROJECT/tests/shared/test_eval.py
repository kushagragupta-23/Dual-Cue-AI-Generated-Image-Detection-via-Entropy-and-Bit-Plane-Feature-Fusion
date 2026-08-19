"""
Unit Tests for Evaluation Suite: Metrics, Evaluator, and Formatting.

Verifies:
    1. Metric values match scikit-learn reference outputs
    2. Confusion matrix components sum correctly
    3. Optimal threshold selection via Youden's J
    4. Per-domain breakdown correctness
    5. Markdown table formatting
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.eval.metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    find_optimal_threshold,
    compute_per_domain_metrics,
    format_metrics_table,
    format_per_domain_table,
)


class TestBinaryMetrics:
    """Test suite for binary metric computation."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield accuracy=1.0, F1=1.0."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_probs = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        metrics = compute_binary_metrics(y_true, y_probs, threshold=0.5)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_worst_predictions(self):
        """All-wrong predictions should yield accuracy=0.0."""
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([1.0, 1.0, 0.0, 0.0])
        metrics = compute_binary_metrics(y_true, y_probs, threshold=0.5)
        assert metrics["accuracy"] == 0.0

    def test_matches_sklearn(self):
        """Verify our metrics match scikit-learn reference."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=100)
        y_probs = np.random.rand(100)
        y_pred = (y_probs >= 0.5).astype(int)

        metrics = compute_binary_metrics(y_true, y_probs, threshold=0.5)

        expected_acc = accuracy_score(y_true, y_pred)
        expected_f1 = f1_score(y_true, y_pred, zero_division=0)
        expected_auc = roc_auc_score(y_true, y_probs)

        assert abs(metrics["accuracy"] - expected_acc) < 1e-6
        assert abs(metrics["f1_score"] - expected_f1) < 1e-6
        assert abs(metrics["roc_auc"] - expected_auc) < 1e-6

    def test_num_samples(self):
        """Verify sample count is reported."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_probs = np.array([0.1, 0.9, 0.3, 0.8, 0.2])
        metrics = compute_binary_metrics(y_true, y_probs)
        assert metrics["num_samples"] == 5


class TestConfusionMatrix:
    """Test suite for confusion matrix computation."""

    def test_components_sum_to_total(self):
        """TP + TN + FP + FN should equal total samples."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_probs = np.array([0.1, 0.6, 0.8, 0.3, 0.2, 0.9, 0.7, 0.4])
        cm = compute_confusion_matrix(y_true, y_probs, threshold=0.5)
        total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
        assert total == len(y_true), f"CM components sum to {total}, expected {len(y_true)}"

    def test_perfect_cm(self):
        """Perfect predictions: FP=0, FN=0."""
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([0.0, 0.0, 1.0, 1.0])
        cm = compute_confusion_matrix(y_true, y_probs, threshold=0.5)
        assert cm["fp"] == 0
        assert cm["fn"] == 0
        assert cm["tn"] == 2
        assert cm["tp"] == 2


class TestOptimalThreshold:
    """Test suite for Youden's J threshold selection."""

    def test_perfect_threshold(self):
        """Well-separated distributions should find a good threshold."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.15, 0.05, 0.9, 0.85, 0.95, 0.8])
        threshold, metrics = find_optimal_threshold(y_true, y_probs)
        assert 0.2 < threshold < 0.9, f"Expected threshold between 0.2-0.9, got {threshold}"
        assert metrics["accuracy"] == 1.0

    def test_returns_metrics_dict(self):
        """Should return both threshold and full metrics dict."""
        y_true = np.array([0, 1, 0, 1])
        y_probs = np.array([0.3, 0.7, 0.4, 0.6])
        threshold, metrics = find_optimal_threshold(y_true, y_probs)
        assert isinstance(threshold, float)
        assert "accuracy" in metrics
        assert "roc_auc" in metrics


class TestPerDomainMetrics:
    """Test suite for per-domain breakdown."""

    def test_domain_breakdown(self):
        """Should compute metrics per domain."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        domains = ["ProGAN", "ProGAN", "StyleGAN", "StyleGAN", "SD", "SD"]

        per_domain = compute_per_domain_metrics(y_true, y_probs, domains)
        assert "ProGAN" in per_domain
        assert "StyleGAN" in per_domain
        assert "SD" in per_domain

    def test_each_domain_has_metrics(self):
        """Each domain should have complete metric set."""
        y_true = np.array([0, 1, 0, 1])
        y_probs = np.array([0.1, 0.9, 0.2, 0.8])
        domains = ["A", "A", "B", "B"]

        per_domain = compute_per_domain_metrics(y_true, y_probs, domains)
        for domain_metrics in per_domain.values():
            assert "accuracy" in domain_metrics
            assert "f1_score" in domain_metrics


class TestFormatting:
    """Test suite for Markdown table formatting."""

    def test_format_metrics_table(self):
        """Should produce valid Markdown table string."""
        metrics = {"accuracy": 0.95, "f1_score": 0.93}
        table = format_metrics_table(metrics, title="Test")
        assert "### Test" in table
        assert "0.9500" in table
        assert "|" in table

    def test_format_per_domain_table(self):
        """Should produce per-domain Markdown table."""
        per_domain = {
            "ProGAN": {"accuracy": 0.99, "precision": 0.98, "recall": 0.97, "f1_score": 0.975, "roc_auc": 0.99, "average_precision": 0.98},
        }
        table = format_per_domain_table(per_domain)
        assert "ProGAN" in table
        assert "0.9900" in table


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
