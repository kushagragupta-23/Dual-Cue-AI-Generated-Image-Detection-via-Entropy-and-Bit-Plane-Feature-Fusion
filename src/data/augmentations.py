"""
Online Robustness Augmentation Transforms for HydraFusion-Net.

Applied during training to improve model resilience to real-world image
degradations. Includes:
  - JPEG recompression simulation (via torchvision or PIL)
  - Gaussian blur
  - Random noise injection
  - Color jitter variants

These transforms operate on tensors in the [0, 255] range to match
HydraFusion's input convention.
"""

import io
import random
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required for augmentations.")


class OnlineJPEGCompression:
    """
    Randomly applies JPEG recompression to simulate social media degradation.

    Args:
        quality_range: (min_quality, max_quality) for random quality selection.
        probability: Probability of applying the augmentation per image.
    """

    def __init__(
        self,
        quality_range: Tuple[int, int] = (70, 100),
        probability: float = 0.5,
    ) -> None:
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random JPEG compression to a single image tensor.

        Args:
            tensor: Image tensor (3, H, W) in range [0, 255].

        Returns:
            Augmented tensor (3, H, W) in range [0, 255].
        """
        if random.random() > self.probability:
            return tensor

        quality = random.randint(*self.quality_range)
        if quality >= 100:
            return tensor

        img_np = tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode="RGB")

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        pil_reloaded = Image.open(buffer).convert("RGB")

        img_reloaded = np.array(pil_reloaded).astype(np.float32)
        return torch.from_numpy(img_reloaded.transpose(2, 0, 1)).to(tensor.device)


class OnlineGaussianBlur:
    """
    Randomly applies Gaussian blur.

    Args:
        sigma_range: (min_sigma, max_sigma) for random sigma selection.
        probability: Probability of applying the augmentation per image.
    """

    def __init__(
        self,
        sigma_range: Tuple[float, float] = (0.5, 1.5),
        probability: float = 0.3,
    ) -> None:
        self.sigma_range = sigma_range
        self.probability = probability

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random Gaussian blur to a single image tensor.

        Args:
            tensor: Image tensor (3, H, W) in range [0, 255].

        Returns:
            Blurred tensor (3, H, W) in range [0, 255].
        """
        if random.random() > self.probability:
            return tensor

        sigma = random.uniform(*self.sigma_range)
        kernel_size = int(np.ceil(6 * sigma)) | 1
        kernel_size = max(kernel_size, 3)

        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).to(tensor.device)

        # Add batch dim for conv2d
        x = tensor.unsqueeze(0)
        padding = kernel_size // 2
        blurred = F.conv2d(x, kernel_2d, padding=padding, groups=3)
        return blurred.squeeze(0).clamp(0, 255)


class OnlineGaussianNoise:
    """
    Randomly adds Gaussian noise.

    Args:
        sigma_range: (min_sigma, max_sigma) for noise standard deviation.
        probability: Probability of applying the augmentation per image.
    """

    def __init__(
        self,
        sigma_range: Tuple[float, float] = (1.0, 5.0),
        probability: float = 0.2,
    ) -> None:
        self.sigma_range = sigma_range
        self.probability = probability

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.probability:
            return tensor

        sigma = random.uniform(*self.sigma_range)
        noise = torch.randn_like(tensor) * sigma
        return (tensor + noise).clamp(0, 255)


class ForensicAugmentationPipeline:
    """
    Combines all online augmentations into a single pipeline.

    Applied per-image during training to improve robustness to
    real-world degradations.

    Usage:
        pipeline = ForensicAugmentationPipeline()
        augmented_tensor = pipeline(tensor)  # (3, H, W) in [0, 255]
    """

    def __init__(
        self,
        jpeg_prob: float = 0.3,
        jpeg_quality_range: Tuple[int, int] = (70, 100),
        blur_prob: float = 0.2,
        blur_sigma_range: Tuple[float, float] = (0.5, 1.5),
        noise_prob: float = 0.15,
        noise_sigma_range: Tuple[float, float] = (1.0, 5.0),
    ) -> None:
        self.augmentations = [
            OnlineJPEGCompression(jpeg_quality_range, jpeg_prob),
            OnlineGaussianBlur(blur_sigma_range, blur_prob),
            OnlineGaussianNoise(noise_sigma_range, noise_prob),
        ]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            tensor = aug(tensor)
        return tensor
