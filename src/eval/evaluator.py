"""
Automated Evaluation Loop & Markdown Table Compiler for HydraFusion-Net.

Provides:
  - Full model evaluation on arbitrary DataLoaders
  - Per-class (Real/Fake) breakdown
  - Cross-generator zero-shot evaluation orchestration
  - Markdown and JSON report export
  - Gating weight (alpha) distribution analysis
"""

import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from src.eval.metrics import (
    ForensicMetricsCalculator,
    MetricsResult,
    format_metrics_table,
)
from src.utils.logger import get_logger

logger = get_logger("evaluator")


class HydraFusionEvaluator:
    """
    Comprehensive evaluator for HydraFusion-Net.

    Handles:
      1. Standard test-set evaluation with full academic metrics
      2. Gating weight (alpha) distribution logging per sample
      3. Latency and throughput measurement
      4. Report generation in Markdown and JSON

    Usage:
        evaluator = HydraFusionEvaluator(model, device)
        result = evaluator.evaluate(test_loader)
        evaluator.export_report(result, output_dir="outputs/results")
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model = model
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = amp_dtype

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        desc: str = "Evaluating",
        collect_alphas: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on a DataLoader.

        Args:
            dataloader: PyTorch DataLoader yielding (images, labels) batches.
            desc: Progress bar description.
            collect_alphas: If True, collect gating weight distributions.

        Returns:
            Dict containing:
              - "metrics": MetricsResult with all academic metrics
              - "alphas": numpy array of gating weights (B, 4) if collected
              - "latency_ms": average per-image latency in milliseconds
              - "throughput_ips": images per second
        """
        self.model.eval()
        calc = ForensicMetricsCalculator()
        all_alphas: List[np.ndarray] = []

        total_time = 0.0
        total_images = 0
        total_correct = 0

        pbar = tqdm(dataloader, desc=desc, leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels_np = labels.numpy().tolist()
            batch_size = images.shape[0]

            # Time the forward pass
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    logits, _, gating_weights = self.model(images, stage=2)
            else:
                logits, _, gating_weights = self.model(images, stage=2)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_time += t1 - t0
            total_images += batch_size

            # Extract predictions and probabilities (float32 to avoid fp16 overflow)
            logits_f32 = logits.float()
            probs_tensor = torch.sigmoid(logits_f32).squeeze()
            # Clean up NaN / Inf and clamp to [0, 1]
            probs_tensor = torch.nan_to_num(probs_tensor, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            # Handle single-sample batch
            if probs_tensor.dim() == 0:
                probs_tensor = probs_tensor.unsqueeze(0)
            probs = probs_tensor.cpu().numpy().tolist()
            preds = [1 if p > 0.5 else 0 for p in probs]

            calc.update(predictions=preds, labels=labels_np, probabilities=probs)

            # Collect gating weights
            if collect_alphas and gating_weights is not None:
                all_alphas.append(gating_weights.cpu().numpy())

            # Update progress bar with simple running accuracy (avoid expensive compute())
            running_correct = sum(1 for p, l in zip(preds, labels_np) if p == l)
            total_correct += running_correct
            running_acc = total_correct / total_images * 100
            pbar.set_postfix(acc=f"{running_acc:.1f}%")

        # Compile results
        metrics = calc.compute()
        result: Dict[str, Any] = {"metrics": metrics}

        if all_alphas:
            result["alphas"] = np.concatenate(all_alphas, axis=0)
        else:
            result["alphas"] = None

        # Timing statistics
        if total_images > 0:
            result["latency_ms"] = (total_time / total_images) * 1000.0
            result["throughput_ips"] = total_images / total_time
        else:
            result["latency_ms"] = 0.0
            result["throughput_ips"] = 0.0

        return result

    def export_report(
        self,
        eval_result: Dict[str, Any],
        output_dir: str = "outputs/results",
        report_name: str = "evaluation_report",
    ) -> Path:
        """
        Export evaluation results as JSON and Markdown reports.

        Args:
            eval_result: Output from evaluate().
            output_dir: Directory to save reports.
            report_name: Base name for report files.

        Returns:
            Path to the generated Markdown report.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        metrics: MetricsResult = eval_result["metrics"]

        # --- JSON Report ---
        json_data = metrics.to_dict()
        json_data["latency_ms_per_image"] = round(eval_result["latency_ms"], 3)
        json_data["throughput_images_per_sec"] = round(
            eval_result["throughput_ips"], 1
        )

        # Alpha statistics
        if eval_result["alphas"] is not None:
            alphas = eval_result["alphas"]
            alpha_means = alphas.mean(axis=0).tolist()
            alpha_stds = alphas.std(axis=0).tolist()
            json_data["gating_weights"] = {
                "head_names": [
                    "SpatialAttn_MLEP→LOTA",
                    "SpatialAttn_LOTA→MLEP",
                    "ChannelSE",
                    "FreqCorrelation",
                ],
                "mean_weights": [round(a, 4) for a in alpha_means],
                "std_weights": [round(s, 4) for s in alpha_stds],
            }

        json_path = out_path / f"{report_name}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # --- Markdown Report ---
        md_lines = [
            f"# HydraFusion-Net Evaluation Report",
            f"",
            f"## Summary Metrics",
            f"",
            f"| Metric | Value |",
            f"| :--- | :---: |",
            f"| **Accuracy** | {metrics.accuracy*100:.2f}% |",
            f"| **Precision** | {metrics.precision*100:.2f}% |",
            f"| **Recall** | {metrics.recall*100:.2f}% |",
            f"| **F1 Score** | {metrics.f1*100:.2f}% |",
            f"| **ROC-AUC** | {metrics.roc_auc:.4f} |",
            f"| **Average Precision (AP)** | {metrics.average_precision:.4f} |",
            f"| **Test Samples** | {metrics.num_samples} |",
            f"| **Latency** | {eval_result['latency_ms']:.2f} ms/image |",
            f"| **Throughput** | {eval_result['throughput_ips']:.1f} images/sec |",
            f"",
        ]

        # Confusion matrix
        if metrics.confusion_matrix_raw is not None:
            cm = metrics.confusion_matrix_raw
            cm_norm = metrics.confusion_matrix_normalized
            md_lines.extend([
                f"## Confusion Matrix",
                f"",
                f"### Raw Counts",
                f"| | Predicted Real | Predicted Fake |",
                f"| :--- | :---: | :---: |",
                f"| **Actual Real** | {cm[0][0]} | {cm[0][1]} |",
                f"| **Actual Fake** | {cm[1][0]} | {cm[1][1]} |",
                f"",
                f"### Normalized (%)",
                f"| | Predicted Real | Predicted Fake |",
                f"| :--- | :---: | :---: |",
                f"| **Actual Real** | {cm_norm[0][0]*100:.1f}% | {cm_norm[0][1]*100:.1f}% |",
                f"| **Actual Fake** | {cm_norm[1][0]*100:.1f}% | {cm_norm[1][1]*100:.1f}% |",
                f"",
            ])

        # Gating weights
        if eval_result["alphas"] is not None:
            gw = json_data["gating_weights"]
            md_lines.extend([
                f"## Gating Weight Distribution",
                f"",
                f"| Fusion Head | Mean α | Std α |",
                f"| :--- | :---: | :---: |",
            ])
            for name, mean, std in zip(
                gw["head_names"], gw["mean_weights"], gw["std_weights"]
            ):
                md_lines.append(f"| {name} | {mean:.4f} | {std:.4f} |")
            md_lines.append("")

        md_path = out_path / f"{report_name}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Reports saved: {json_path} | {md_path}")
        return md_path

    @torch.no_grad()
    def evaluate_robustness(
        self,
        dataloader: torch.utils.data.DataLoader,
        jpeg_qualities: List[int] = [100, 90, 80, 70],
        blur_sigmas: List[float] = [0.0, 0.5, 1.0, 2.0],
    ) -> Dict[str, MetricsResult]:
        """
        Evaluate model robustness under JPEG compression and Gaussian blur.

        This applies degradations on-the-fly to the test set and reports
        accuracy under each condition.

        Args:
            dataloader: Test set DataLoader.
            jpeg_qualities: JPEG quality levels to test.
            blur_sigmas: Gaussian blur sigma values to test.

        Returns:
            Dict mapping condition name to MetricsResult.
        """
        from src.eval.robustness_suite import apply_jpeg_compression, apply_gaussian_blur

        results: Dict[str, MetricsResult] = {}

        # JPEG degradation
        for q in jpeg_qualities:
            calc = ForensicMetricsCalculator()
            for images, labels in tqdm(
                dataloader, desc=f"JPEG Q={q}", leave=False
            ):
                degraded = apply_jpeg_compression(images, quality=q)
                degraded = degraded.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        logits, _, _ = self.model(degraded, stage=2)
                else:
                    logits, _, _ = self.model(degraded, stage=2)

                probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy().tolist()
                preds = [1 if p > 0.5 else 0 for p in probs]
                calc.update(
                    predictions=preds,
                    labels=labels.numpy().tolist(),
                    probabilities=probs,
                )
            results[f"JPEG_Q{q}"] = calc.compute()
            logger.info(f"  JPEG Q={q}: {results[f'JPEG_Q{q}'].summary_string()}")

        # Gaussian blur degradation
        for sigma in blur_sigmas:
            calc = ForensicMetricsCalculator()
            for images, labels in tqdm(
                dataloader, desc=f"Blur σ={sigma}", leave=False
            ):
                degraded = apply_gaussian_blur(images, sigma=sigma)
                degraded = degraded.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        logits, _, _ = self.model(degraded, stage=2)
                else:
                    logits, _, _ = self.model(degraded, stage=2)

                probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy().tolist()
                preds = [1 if p > 0.5 else 0 for p in probs]
                calc.update(
                    predictions=preds,
                    labels=labels.numpy().tolist(),
                    probabilities=probs,
                )
            results[f"Blur_s{sigma}"] = calc.compute()
            logger.info(
                f"  Blur σ={sigma}: {results[f'Blur_s{sigma}'].summary_string()}"
            )

        return results
