"""
Automated Cross-Generator Zero-Shot Evaluator.

Provides:
    1. ModelEvaluator — Runs inference on evaluation datasets, collects predictions,
       and computes comprehensive metrics.
    2. ZeroShotBenchmark — Evaluates a model trained on one generator set against
       unseen generator domains without fine-tuning.
    3. Markdown report compiler for academic result tables.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    compute_per_domain_metrics,
    find_optimal_threshold,
    format_metrics_table,
    format_per_domain_table,
)
from src.utils.logger import get_logger

logger = get_logger("evaluator")


class ModelEvaluator:
    """
    Automated model evaluation engine.

    Runs inference on a DataLoader, collects predictions and ground truth,
    and computes comprehensive academic metrics.

    Args:
        model: Trained PyTorch model.
        device: Target compute device.
        use_sigmoid: Whether to apply sigmoid to logits (default True for BCEWithLogitsLoss).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_sigmoid: bool = True,
    ):
        self.model = model
        self.device = device
        self.use_sigmoid = use_sigmoid

    @torch.no_grad()
    def predict(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run inference and collect predictions.

        Args:
            dataloader: Evaluation DataLoader yielding (images, labels, meta).

        Returns:
            Tuple of (y_true, y_pred_prob, domains):
                - y_true: Ground truth labels, shape (N,)
                - y_pred_prob: Predicted probabilities, shape (N,)
                - domains: Domain string for each sample
        """
        self.model.eval()

        all_labels = []
        all_probs = []
        all_domains = []

        for batch in dataloader:
            images = batch[0].to(self.device)
            labels = batch[1]

            # Handle different model output signatures
            output = self.model(images)
            if isinstance(output, tuple):
                logits = output[0]  # class_logits from fusion model
            else:
                logits = output

            logits = logits.squeeze(-1)  # (B,)

            if self.use_sigmoid:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = logits.cpu().numpy()

            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

            # Extract domain metadata
            if len(batch) > 2:
                meta = batch[2]
                if isinstance(meta, dict) and "domain" in meta:
                    all_domains.extend(meta["domain"])
                elif isinstance(meta, list):
                    for m in meta:
                        all_domains.append(m.get("domain", "unknown") if isinstance(m, dict) else "unknown")
                else:
                    all_domains.extend(["unknown"] * len(labels))
            else:
                all_domains.extend(["unknown"] * len(labels))

        return (
            np.array(all_labels),
            np.array(all_probs),
            all_domains,
        )

    def evaluate(
        self,
        dataloader: DataLoader,
        threshold: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        Full evaluation pipeline: inference + metrics computation.

        Args:
            dataloader: Evaluation DataLoader.
            threshold: Classification threshold. If None, uses optimal threshold.

        Returns:
            dict with 'overall_metrics', 'confusion_matrix', 'per_domain_metrics',
            'optimal_threshold', 'inference_time_ms'.
        """
        t0 = time.time()
        y_true, y_pred_prob, domains = self.predict(dataloader)
        inference_time = (time.time() - t0) * 1000  # ms

        # Find optimal threshold if not specified
        if threshold is None:
            optimal_thresh, _ = find_optimal_threshold(y_true, y_pred_prob)
        else:
            optimal_thresh = threshold

        results = {
            "overall_metrics": compute_binary_metrics(
                y_true, y_pred_prob, threshold=optimal_thresh
            ),
            "confusion_matrix": compute_confusion_matrix(
                y_true, y_pred_prob, threshold=optimal_thresh
            ),
            "optimal_threshold": optimal_thresh,
            "inference_time_ms": inference_time,
            "num_samples": len(y_true),
        }

        # Per-domain breakdown if domain information available
        if any(d != "unknown" for d in domains):
            results["per_domain_metrics"] = compute_per_domain_metrics(
                y_true, y_pred_prob, domains, threshold=optimal_thresh
            )

        logger.info(
            f"Evaluation complete: {len(y_true)} samples in {inference_time:.1f}ms | "
            f"Acc={results['overall_metrics']['accuracy']:.4f} | "
            f"AUC={results['overall_metrics']['roc_auc']:.4f}"
        )

        return results


class ZeroShotBenchmark:
    """
    Cross-generator zero-shot evaluation benchmark.

    Evaluates a model across multiple dataset splits (each from a different
    generator domain) without any fine-tuning.

    Args:
        model: Trained model to evaluate.
        device: Compute device.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.evaluator = ModelEvaluator(model, device)

    def run_benchmark(
        self,
        dataloaders: Dict[str, DataLoader],
        threshold: float = 0.5,
    ) -> Dict[str, Dict]:
        """
        Run zero-shot evaluation across multiple generator domains.

        Args:
            dataloaders: Dict mapping generator_name → DataLoader.
            threshold: Classification threshold (default 0.5).

        Returns:
            dict mapping generator_name → evaluation_results.
        """
        all_results = {}

        for gen_name, loader in dataloaders.items():
            logger.info(f"Evaluating zero-shot on: {gen_name}")
            results = self.evaluator.evaluate(loader, threshold=threshold)
            all_results[gen_name] = results

        return all_results

    @staticmethod
    def compile_markdown_report(
        benchmark_results: Dict[str, Dict],
        title: str = "Zero-Shot Cross-Generator Benchmark",
    ) -> str:
        """
        Compile benchmark results into a Markdown report.

        Args:
            benchmark_results: Results from run_benchmark().
            title: Report title.

        Returns:
            str: Complete Markdown report.
        """
        lines = [f"# {title}", ""]

        # Summary table
        lines.extend([
            "## Summary",
            "",
            "| Generator | Accuracy | Precision | Recall | F1 | AUC | AP |",
            "|-----------|----------|-----------|--------|-----|-----|-----|",
        ])

        for gen_name, results in sorted(benchmark_results.items()):
            m = results.get("overall_metrics", {})
            lines.append(
                f"| {gen_name} "
                f"| {m.get('accuracy', 0):.4f} "
                f"| {m.get('precision', 0):.4f} "
                f"| {m.get('recall', 0):.4f} "
                f"| {m.get('f1_score', 0):.4f} "
                f"| {m.get('roc_auc', 0):.4f} "
                f"| {m.get('average_precision', 0):.4f} |"
            )

        # Compute averages
        all_accs = [r["overall_metrics"]["accuracy"] for r in benchmark_results.values()]
        all_aucs = [r["overall_metrics"]["roc_auc"] for r in benchmark_results.values()]
        lines.append(
            f"| **Average** "
            f"| **{np.mean(all_accs):.4f}** "
            f"| - | - | - "
            f"| **{np.mean(all_aucs):.4f}** | - |"
        )
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def save_results(
        results: Dict,
        save_path: Path,
    ) -> None:
        """Save benchmark results to JSON file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        def _convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = json.loads(
            json.dumps(results, default=_convert)
        )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4)
        logger.info(f"Saved benchmark results to {save_path}")


__all__ = ["ModelEvaluator", "ZeroShotBenchmark"]
