"""
Automated Robustness Degradation Test Suite.

Evaluates classifier resilience against systematic image degradation:
    1. JPEG Compression: Quality levels Q ∈ {100, 90, 80, 70}
    2. Gaussian Blur: Sigma levels σ ∈ {0.0, 0.5, 1.0, 2.0}

Generates:
    - Quantitative degradation curves (accuracy/AUC vs. degradation level)
    - Comparison tables between baseline and enhanced architectures
    - Publication-ready Matplotlib figures
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from src.eval.metrics import compute_binary_metrics
from src.eval.evaluator import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger("robustness_suite")


class DegradedDatasetWrapper(Dataset):
    """
    Wraps an existing dataset to apply systematic image degradation.

    Args:
        base_dataset: Original dataset returning (tensor, label, meta).
        degradation_type: 'jpeg' or 'blur'.
        degradation_level: Quality factor for JPEG or sigma for blur.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        degradation_type: str = "jpeg",
        degradation_level: float = 70.0,
    ):
        self.base_dataset = base_dataset
        self.degradation_type = degradation_type
        self.degradation_level = degradation_level

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        tensor, label, meta = self.base_dataset[idx]

        if self.degradation_type == "jpeg":
            tensor = self._apply_jpeg(tensor, int(self.degradation_level))
        elif self.degradation_type == "blur":
            tensor = self._apply_blur(tensor, float(self.degradation_level))

        return tensor, label, meta

    @staticmethod
    def _apply_jpeg(tensor: torch.Tensor, quality: int) -> torch.Tensor:
        """Apply JPEG recompression to a tensor in [0, 255]."""
        # Convert to PIL
        pil_tensor = tensor / 255.0
        img = TF.to_pil_image(pil_tensor.clamp(0, 1))

        # JPEG round-trip
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img_recompressed = Image.open(buffer).convert("RGB")

        # Convert back to tensor in [0, 255]
        return TF.to_tensor(img_recompressed) * 255.0

    @staticmethod
    def _apply_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur to a tensor."""
        if sigma < 1e-6:
            return tensor
        return TF.gaussian_blur(tensor, kernel_size=5, sigma=sigma)

    @property
    def samples(self):
        """Proxy for base dataset's samples attribute."""
        return self.base_dataset.samples


class RobustnessSuite:
    """
    Automated robustness evaluation suite.

    Systematically tests a model against JPEG compression and Gaussian blur
    at multiple severity levels, generating degradation curves and reports.

    Args:
        model: Model to evaluate.
        device: Compute device.
        jpeg_levels: JPEG quality levels to test (default [100, 90, 80, 70]).
        blur_sigmas: Gaussian blur sigmas to test (default [0.0, 0.5, 1.0, 2.0]).
        batch_size: Evaluation batch size (default 32).
        num_workers: DataLoader workers (default 0).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        jpeg_levels: Optional[List[int]] = None,
        blur_sigmas: Optional[List[float]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        self.model = model
        self.device = device
        self.jpeg_levels = jpeg_levels or [100, 90, 80, 70]
        self.blur_sigmas = blur_sigmas or [0.0, 0.5, 1.0, 2.0]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.evaluator = ModelEvaluator(model, device)

    def run_jpeg_degradation(
        self, test_dataset: Dataset, threshold: float = 0.5
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model under JPEG compression at multiple quality levels.

        Args:
            test_dataset: Base test dataset.
            threshold: Classification threshold.

        Returns:
            dict mapping jpeg_quality → metrics_dict.
        """
        results = {}
        for quality in self.jpeg_levels:
            logger.info(f"Testing JPEG Q={quality}...")
            degraded_ds = DegradedDatasetWrapper(test_dataset, "jpeg", float(quality))
            loader = DataLoader(
                degraded_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            eval_results = self.evaluator.evaluate(loader, threshold=threshold)
            results[quality] = eval_results["overall_metrics"]
            logger.info(
                f"  JPEG Q={quality}: Acc={results[quality]['accuracy']:.4f}, "
                f"AUC={results[quality]['roc_auc']:.4f}"
            )
        return results

    def run_blur_degradation(
        self, test_dataset: Dataset, threshold: float = 0.5
    ) -> Dict[float, Dict[str, float]]:
        """
        Evaluate model under Gaussian blur at multiple sigma levels.

        Args:
            test_dataset: Base test dataset.
            threshold: Classification threshold.

        Returns:
            dict mapping blur_sigma → metrics_dict.
        """
        results = {}
        for sigma in self.blur_sigmas:
            logger.info(f"Testing Blur σ={sigma}...")
            degraded_ds = DegradedDatasetWrapper(test_dataset, "blur", sigma)
            loader = DataLoader(
                degraded_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            eval_results = self.evaluator.evaluate(loader, threshold=threshold)
            results[sigma] = eval_results["overall_metrics"]
            logger.info(
                f"  Blur σ={sigma}: Acc={results[sigma]['accuracy']:.4f}, "
                f"AUC={results[sigma]['roc_auc']:.4f}"
            )
        return results

    def run_full_suite(
        self, test_dataset: Dataset, threshold: float = 0.5
    ) -> Dict[str, Dict]:
        """
        Run complete robustness evaluation (JPEG + Blur).

        Args:
            test_dataset: Base test dataset.
            threshold: Classification threshold.

        Returns:
            dict with 'jpeg_results' and 'blur_results'.
        """
        return {
            "jpeg_results": self.run_jpeg_degradation(test_dataset, threshold),
            "blur_results": self.run_blur_degradation(test_dataset, threshold),
        }

    @staticmethod
    def plot_degradation_curves(
        results: Dict[str, Dict],
        save_path: Optional[Path] = None,
        title: str = "Robustness Degradation Analysis",
    ) -> plt.Figure:
        """
        Generate publication-ready degradation curve plots.

        Args:
            results: Output from run_full_suite().
            save_path: Optional path to save figure.
            title: Figure title.

        Returns:
            matplotlib.figure.Figure: Degradation curve figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # JPEG degradation
        if "jpeg_results" in results:
            jpeg_data = results["jpeg_results"]
            qualities = sorted(jpeg_data.keys(), reverse=True)
            accs = [jpeg_data[q]["accuracy"] for q in qualities]
            aucs = [jpeg_data[q]["roc_auc"] for q in qualities]

            ax1.plot(qualities, accs, "o-", color="#2196F3", linewidth=2, label="Accuracy")
            ax1.plot(qualities, aucs, "s--", color="#FF5722", linewidth=2, label="ROC-AUC")
            ax1.set_xlabel("JPEG Quality Factor", fontsize=12)
            ax1.set_ylabel("Score", fontsize=12)
            ax1.set_title("JPEG Compression Robustness", fontsize=13, fontweight="bold")
            ax1.legend(fontsize=10)
            ax1.set_ylim(0, 1.05)
            ax1.grid(True, alpha=0.3)
            ax1.invert_xaxis()

        # Blur degradation
        if "blur_results" in results:
            blur_data = results["blur_results"]
            sigmas = sorted(blur_data.keys())
            accs = [blur_data[s]["accuracy"] for s in sigmas]
            aucs = [blur_data[s]["roc_auc"] for s in sigmas]

            ax2.plot(sigmas, accs, "o-", color="#4CAF50", linewidth=2, label="Accuracy")
            ax2.plot(sigmas, aucs, "s--", color="#9C27B0", linewidth=2, label="ROC-AUC")
            ax2.set_xlabel("Gaussian Blur σ", fontsize=12)
            ax2.set_ylabel("Score", fontsize=12)
            ax2.set_title("Gaussian Blur Robustness", fontsize=13, fontweight="bold")
            ax2.legend(fontsize=10)
            ax2.set_ylim(0, 1.05)
            ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            logger.info(f"Saved degradation curve to {path}")
            plt.close(fig)

        return fig

    @staticmethod
    def compile_markdown_table(results: Dict[str, Dict]) -> str:
        """Compile robustness results as Markdown tables."""
        lines = ["## Robustness Evaluation Results", ""]

        if "jpeg_results" in results:
            lines.extend([
                "### JPEG Compression Robustness",
                "",
                "| Quality | Accuracy | AUC | F1 |",
                "|---------|----------|-----|-----|",
            ])
            for q in sorted(results["jpeg_results"].keys(), reverse=True):
                m = results["jpeg_results"][q]
                lines.append(
                    f"| Q={q} | {m['accuracy']:.4f} | {m['roc_auc']:.4f} | {m['f1_score']:.4f} |"
                )
            lines.append("")

        if "blur_results" in results:
            lines.extend([
                "### Gaussian Blur Robustness",
                "",
                "| Sigma | Accuracy | AUC | F1 |",
                "|-------|----------|-----|-----|",
            ])
            for s in sorted(results["blur_results"].keys()):
                m = results["blur_results"][s]
                lines.append(
                    f"| σ={s:.1f} | {m['accuracy']:.4f} | {m['roc_auc']:.4f} | {m['f1_score']:.4f} |"
                )

        return "\n".join(lines)


__all__ = ["DegradedDatasetWrapper", "RobustnessSuite"]
