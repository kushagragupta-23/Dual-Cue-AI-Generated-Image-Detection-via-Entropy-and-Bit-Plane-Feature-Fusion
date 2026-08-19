"""
Online Robustness Augmentation Transforms for AIGID Training.

Simulates real-world image degradation to improve classifier robustness:
    1. JPEG Recompression: Random quality Q ∈ [70, 100] via PIL JPEG encoder
    2. Gaussian Blur: Random σ ∈ [0.5, 2.0] via torchvision GaussianBlur
    3. Combined pipeline with per-transform probability gates

These augmentations are applied BEFORE feature extraction (MLEP/LOTA) to train
the Learnable Frequency Pre-Filter to adaptively strip compression artifacts.
"""

import io
import random
from typing import Optional, Tuple

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.utils.logger import get_logger

logger = get_logger("augmentations")


class JPEGRecompression:
    """
    Simulate social media JPEG recompression by encoding and decoding via PIL.

    Applies random JPEG quality factor to introduce realistic DCT block artifacts.
    Operates on PIL Images or PyTorch tensors.

    Args:
        quality_min: Minimum JPEG quality factor (default 70).
        quality_max: Maximum JPEG quality factor (default 100).
        p: Probability of applying this augmentation (default 0.5).
    """

    def __init__(
        self, quality_min: int = 70, quality_max: int = 100, p: float = 0.5
    ):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply JPEG recompression to a PIL Image.

        Args:
            img: Input PIL Image in RGB mode.

        Returns:
            PIL.Image: Recompressed image (or original if probability gate fails).
        """
        if random.random() > self.p:
            return img

        quality = random.randint(self.quality_min, self.quality_max)

        # Encode to JPEG bytes and decode back
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img_recompressed = Image.open(buffer).convert("RGB")

        return img_recompressed


class GaussianBlurAugmentation:
    """
    Apply random Gaussian blur to simulate lens defocus or social media processing.

    Args:
        sigma_min: Minimum blur sigma (default 0.5).
        sigma_max: Maximum blur sigma (default 2.0).
        kernel_size: Blur kernel size (must be odd, default 5).
        p: Probability of applying this augmentation (default 0.5).
    """

    def __init__(
        self,
        sigma_min: float = 0.5,
        sigma_max: float = 2.0,
        kernel_size: int = 5,
        p: float = 0.5,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply Gaussian blur to a PIL Image.

        Args:
            img: Input PIL Image.

        Returns:
            PIL.Image: Blurred image (or original if probability gate fails).
        """
        if random.random() > self.p:
            return img

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        img_tensor = TF.to_tensor(img)
        blurred = TF.gaussian_blur(img_tensor, kernel_size=self.kernel_size, sigma=sigma)
        return TF.to_pil_image(blurred)


class TensorJPEGRecompression:
    """
    JPEG recompression operating directly on PyTorch tensors.

    Converts tensor → PIL → JPEG encode → decode → tensor round-trip.
    Suitable for use in training loops after initial tensor conversion.

    Args:
        quality_min: Minimum JPEG quality factor (default 70).
        quality_max: Maximum JPEG quality factor (default 100).
        p: Probability of applying this augmentation (default 0.5).
    """

    def __init__(
        self, quality_min: int = 70, quality_max: int = 100, p: float = 0.5
    ):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply JPEG recompression to a tensor.

        Args:
            tensor: Image tensor of shape (C, H, W) in [0, 255] or [0, 1].

        Returns:
            torch.Tensor: Recompressed tensor.
        """
        if random.random() > self.p:
            return tensor

        quality = random.randint(self.quality_min, self.quality_max)

        # Determine if tensor is in [0, 255] or [0, 1] range
        is_255_range = tensor.max() > 1.0
        if is_255_range:
            pil_tensor = tensor / 255.0
        else:
            pil_tensor = tensor

        # Convert to PIL, compress, and convert back
        img = TF.to_pil_image(pil_tensor.clamp(0, 1))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img_recompressed = Image.open(buffer).convert("RGB")
        result = TF.to_tensor(img_recompressed)

        if is_255_range:
            result = result * 255.0

        return result


class TensorGaussianBlur:
    """
    Gaussian blur operating directly on PyTorch tensors.

    Args:
        sigma_min: Minimum blur sigma (default 0.5).
        sigma_max: Maximum blur sigma (default 2.0).
        kernel_size: Blur kernel size (must be odd, default 5).
        p: Probability of applying this augmentation (default 0.5).
    """

    def __init__(
        self,
        sigma_min: float = 0.5,
        sigma_max: float = 2.0,
        kernel_size: int = 5,
        p: float = 0.5,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to a tensor.

        Args:
            tensor: Image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Blurred tensor.
        """
        if random.random() > self.p:
            return tensor

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return TF.gaussian_blur(tensor, kernel_size=self.kernel_size, sigma=sigma)


class RobustnessAugmentationPipeline:
    """
    Composable augmentation pipeline combining JPEG recompression and Gaussian blur.

    Applied to PIL Images before tensor conversion in the dataset __getitem__.

    Args:
        jpeg_quality_range: (min, max) JPEG quality range (default (70, 100)).
        blur_sigma_range: (min, max) blur sigma range (default (0.5, 2.0)).
        jpeg_p: Probability of JPEG recompression (default 0.5).
        blur_p: Probability of Gaussian blur (default 0.5).
    """

    def __init__(
        self,
        jpeg_quality_range: Tuple[int, int] = (70, 100),
        blur_sigma_range: Tuple[float, float] = (0.5, 2.0),
        jpeg_p: float = 0.5,
        blur_p: float = 0.5,
    ):
        self.jpeg = JPEGRecompression(
            quality_min=jpeg_quality_range[0],
            quality_max=jpeg_quality_range[1],
            p=jpeg_p,
        )
        self.blur = GaussianBlurAugmentation(
            sigma_min=blur_sigma_range[0],
            sigma_max=blur_sigma_range[1],
            p=blur_p,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply augmentation pipeline to a PIL Image."""
        img = self.jpeg(img)
        img = self.blur(img)
        return img


__all__ = [
    "JPEGRecompression",
    "GaussianBlurAugmentation",
    "TensorJPEGRecompression",
    "TensorGaussianBlur",
    "RobustnessAugmentationPipeline",
]
