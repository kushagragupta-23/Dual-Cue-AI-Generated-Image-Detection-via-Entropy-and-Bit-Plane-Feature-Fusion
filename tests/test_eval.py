"""
Unit Tests for HydraFusion-Net Evaluation Metrics.

Validates ROC-AUC, Average Precision, Precision, Recall, F1, and
Confusion Matrix calculations against known synthetic ground-truth
classification arrays (cross-referenced with scikit-learn outputs).
"""

import pytest
import numpy as np
from src.eval.metrics import (
    ForensicMetricsCalculator,
    MetricsResult,
    compute_per_generator_metrics,
    format_metrics_table,
)


class TestForensicMetricsCalculator:
    """Tests for the main metrics calculator."""

    def test_perfect_predictions(self):
        """All predictions correct → accuracy=1.0, F1=1.0."""
        calc = ForensicMetricsCalculator()
        labels = [0, 0, 0, 1, 1, 1]
        preds = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.05, 0.95, 0.9, 0.85]

        calc.update(predictions=preds, labels=labels, probabilities=probs)
        result = calc.compute()

        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.roc_auc == 1.0
        assert result.average_precision == 1.0
        assert result.num_samples == 6

    def test_all_wrong_predictions(self):
        """All predictions wrong → accuracy=0.0."""
        calc = ForensicMetricsCalculator()
        labels = [0, 0, 1, 1]
        preds = [1, 1, 0, 0]
        probs = [0.9, 0.8, 0.2, 0.1]

        calc.update(predictions=preds, labels=labels, probabilities=probs)
        result = calc.compute()

        assert result.accuracy == 0.0
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.roc_auc == 0.0

    def test_mixed_predictions(self):
        """Verify metrics on a realistic mixed prediction set."""
        calc = ForensicMetricsCalculator()
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        preds = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
        probs = [0.1, 0.3, 0.6, 0.2, 0.9, 0.85, 0.4, 0.7, 0.95, 0.8]

        calc.update(predictions=preds, labels=labels, probabilities=probs)
        result = calc.compute()

        assert result.accuracy == pytest.approx(0.8, abs=0.01)
        assert 0.0 < result.precision <= 1.0
        assert 0.0 < result.recall <= 1.0
        assert 0.0 < result.f1 <= 1.0
        assert 0.5 < result.roc_auc <= 1.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 2x2."""
        calc = ForensicMetricsCalculator()
        calc.update([0, 1, 1], [0, 1, 0], [0.2, 0.8, 0.7])
        result = calc.compute()

        assert result.confusion_matrix_raw is not None
        assert result.confusion_matrix_raw.shape == (2, 2)
        assert result.confusion_matrix_normalized is not None
        assert result.confusion_matrix_normalized.shape == (2, 2)

    def test_confusion_matrix_normalized_rows_sum_to_one(self):
        """Each row of normalized confusion matrix should sum to ~1.0."""
        calc = ForensicMetricsCalculator()
        labels = [0, 0, 0, 1, 1, 1]
        preds = [0, 0, 1, 1, 1, 0]
        calc.update(predictions=preds, labels=labels)
        result = calc.compute()

        row_sums = result.confusion_matrix_normalized.sum(axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-6)

    def test_roc_curve_data(self):
        """ROC curve data should contain monotonically increasing TPR."""
        calc = ForensicMetricsCalculator()
        calc.update(
            predictions=[0, 0, 1, 1],
            labels=[0, 1, 0, 1],
            probabilities=[0.2, 0.4, 0.6, 0.8],
        )
        result = calc.compute()

        assert result.roc_curve is not None
        fpr, tpr, _ = result.roc_curve
        # TPR should be non-decreasing
        assert all(tpr[i] <= tpr[i + 1] for i in range(len(tpr) - 1))

    def test_empty_calculator(self):
        """Empty calculator should return zero metrics."""
        calc = ForensicMetricsCalculator()
        result = calc.compute()
        assert result.num_samples == 0
        assert result.accuracy == 0.0

    def test_reset(self):
        """Reset should clear accumulated data."""
        calc = ForensicMetricsCalculator()
        calc.update([1, 0], [1, 0])
        calc.reset()
        result = calc.compute()
        assert result.num_samples == 0

    def test_multi_batch_accumulation(self):
        """Multiple update calls should accumulate correctly."""
        calc = ForensicMetricsCalculator()
        calc.update([0, 1], [0, 1], [0.2, 0.9])
        calc.update([1, 0], [1, 0], [0.8, 0.1])
        result = calc.compute()
        assert result.num_samples == 4
        assert result.accuracy == 1.0

    def test_to_dict_format(self):
        """to_dict should return percentage-scaled values."""
        calc = ForensicMetricsCalculator()
        calc.update([1, 1, 0, 0], [1, 1, 0, 0], [0.9, 0.85, 0.1, 0.15])
        result = calc.compute()
        d = result.to_dict()

        assert d["accuracy"] == 100.0
        assert d["precision"] == 100.0
        assert d["recall"] == 100.0
        assert "num_samples" in d
        assert d["num_samples"] == 4


class TestFormatMetricsTable:
    """Tests for Markdown table formatting."""

    def test_format_produces_valid_markdown(self):
        """Output should contain proper Markdown table separators."""
        calc = ForensicMetricsCalculator()
        calc.update([0, 1], [0, 1], [0.1, 0.9])
        result = calc.compute()

        table = format_metrics_table({"TestGen": result})
        assert "| Generator |" in table
        assert "| TestGen |" in table
        assert "| **Average** |" in table

    def test_multiple_generators(self):
        """Table should have one row per generator plus average."""
        calc1 = ForensicMetricsCalculator()
        calc1.update([0, 1], [0, 1], [0.1, 0.9])
        calc2 = ForensicMetricsCalculator()
        calc2.update([1, 0], [1, 0], [0.8, 0.2])

        table = format_metrics_table({
            "ProGAN": calc1.compute(),
            "StyleGAN": calc2.compute(),
        })
        assert "ProGAN" in table
        assert "StyleGAN" in table
        assert "Average" in table
