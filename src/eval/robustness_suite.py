"""
Robustness Testing Suite for HydraFusion-Net.

Applies systematic degradations to test images and measures
how classification performance degrades:
  - JPEG Recompression at quality levels Q ∈ {100, 90, 80, 70}
  - Gaussian Blur at sigma levels σ ∈ {0.0, 0.5, 1.0, 2.0}
  - Combined degradation (Q=80 + σ=1.0)

These degradation functions operate on batched tensors in the range [0, 255]
to match HydraFusion's input convention.
"""

import io
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required for JPEG compression simulation.")


def apply_jpeg_compression(
    images: torch.Tensor, quality: int = 80
) -> torch.Tensor:
    """
    Apply JPEG recompression to a batch of images.

    Simulates real-world JPEG degradation by encoding each image to JPEG
    at the specified quality level and decoding it back.

    Args:
        images: Batch tensor (B, 3, H, W) in range [0, 255].
        quality: JPEG quality level (1-100). Lower = more compression artifacts.

    Returns:
        Degraded batch tensor (B, 3, H, W) in range [0, 255].
    """
    if quality >= 100:
        return images  # No degradation

    B, C, H, W = images.shape
    device = images.device
    compressed = []

    for i in range(B):
        # Convert to PIL
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode="RGB")

        # Compress and decompress via in-memory JPEG
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        pil_reloaded = Image.open(buffer).convert("RGB")

        # Back to tensor
        img_reloaded = np.array(pil_reloaded).astype(np.float32)
        tensor = torch.from_numpy(img_reloaded.transpose(2, 0, 1))
        compressed.append(tensor)

    return torch.stack(compressed, dim=0).to(device)


def apply_gaussian_blur(
    images: torch.Tensor, sigma: float = 1.0, kernel_size: int = 0
) -> torch.Tensor:
    """
    Apply Gaussian blur to a batch of images.

    Args:
        images: Batch tensor (B, 3, H, W) in range [0, 255].
        sigma: Standard deviation of the Gaussian kernel.
        kernel_size: Size of the kernel. If 0, computed as ceil(6*sigma) | 1.

    Returns:
        Blurred batch tensor (B, 3, H, W) in range [0, 255].
    """
    if sigma <= 0:
        return images

    if kernel_size == 0:
        kernel_size = int(np.ceil(6 * sigma)) | 1  # Ensure odd
        kernel_size = max(kernel_size, 3)

    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D separable kernel
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).to(images.device)

    # Apply depthwise convolution
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel_2d, padding=padding, groups=3)

    return blurred.clamp(0, 255)


def apply_combined_degradation(
    images: torch.Tensor,
    jpeg_quality: int = 80,
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Apply combined JPEG compression and Gaussian blur.

    Args:
        images: Batch tensor (B, 3, H, W) in range [0, 255].
        jpeg_quality: JPEG quality level.
        blur_sigma: Gaussian blur sigma.

    Returns:
        Degraded batch tensor.
    """
    images = apply_jpeg_compression(images, quality=jpeg_quality)
    images = apply_gaussian_blur(images, sigma=blur_sigma)
    return images


class RobustnessSuite:
    """
    Comprehensive robustness evaluation suite.

    Evaluates the model under a matrix of degradation conditions and
    compiles results into a summary table.

    Usage:
        suite = RobustnessSuite(model, device)
        results = suite.run(test_loader)
        print(suite.format_table(results))
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        use_amp: bool = True,
        jpeg_qualities: List[int] = [100, 90, 80, 70],
        blur_sigmas: List[float] = [0.0, 0.5, 1.0, 2.0],
    ) -> None:
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.jpeg_qualities = jpeg_qualities
        self.blur_sigmas = blur_sigmas

    @torch.no_grad()
    def run(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> dict:
        """
        Run the full robustness evaluation matrix.

        Returns:
            Dict mapping condition name to {accuracy, precision, recall, f1, roc_auc}.
        """
        from src.eval.metrics import ForensicMetricsCalculator
        from tqdm import tqdm

        self.model.eval()
        results = {}

        # JPEG sweep
        for q in self.jpeg_qualities:
            calc = ForensicMetricsCalculator()
            desc = f"JPEG Q={q}" if q < 100 else "No Degradation"
            for images, labels in tqdm(dataloader, desc=desc, leave=False):
                degraded = apply_jpeg_compression(images, quality=q)
                degraded = degraded.to(self.device, non_blocking=True)
                self._eval_batch(degraded, labels, calc)
            result = calc.compute()
            results[f"JPEG Q={q}"] = result.to_dict()

        # Blur sweep
        for sigma in self.blur_sigmas:
            if sigma <= 0:
                continue  # Already covered by JPEG Q=100
            calc = ForensicMetricsCalculator()
            for images, labels in tqdm(
                dataloader, desc=f"Blur σ={sigma}", leave=False
            ):
                degraded = apply_gaussian_blur(images, sigma=sigma)
                degraded = degraded.to(self.device, non_blocking=True)
                self._eval_batch(degraded, labels, calc)
            result = calc.compute()
            results[f"Blur σ={sigma}"] = result.to_dict()

        # Combined
        calc = ForensicMetricsCalculator()
        for images, labels in tqdm(
            dataloader, desc="Combined Q=80 σ=1.0", leave=False
        ):
            degraded = apply_combined_degradation(images, jpeg_quality=80, blur_sigma=1.0)
            degraded = degraded.to(self.device, non_blocking=True)
            self._eval_batch(degraded, labels, calc)
        result = calc.compute()
        results["Combined (Q=80, σ=1.0)"] = result.to_dict()

        return results

    def _eval_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        calc,
    ) -> None:
        """Evaluate a single batch and update the calculator."""
        if self.use_amp and self.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = self.model(images, stage=2)
        else:
            logits, _, _ = self.model(images, stage=2)

        logits_f32 = logits.float()
        probs_tensor = torch.sigmoid(logits_f32).squeeze()
        probs_tensor = probs_tensor.clamp(0.0, 1.0)
        if probs_tensor.dim() == 0:
            probs_tensor = probs_tensor.unsqueeze(0)
        probs = probs_tensor.cpu().numpy().tolist()
        preds = [1 if p > 0.5 else 0 for p in probs]
        calc.update(
            predictions=preds,
            labels=labels.numpy().tolist(),
            probabilities=probs,
        )

    @staticmethod
    def format_table(results: dict) -> str:
        """Format robustness results as a Markdown table."""
        lines = [
            "### Robustness Evaluation Results\n",
            "| Condition | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for condition, metrics in results.items():
            lines.append(
                f"| {condition} | {metrics['accuracy']:.2f} | "
                f"{metrics['precision']:.2f} | {metrics['recall']:.2f} | "
                f"{metrics['f1_score']:.2f} | {metrics['roc_auc']:.4f} |"
            )
        return "\n".join(lines)
