"""
Transforms and Degradation Utilities for MLEP.
"""

import io
from typing import Any, Optional, Union
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms.functional as TF


class GaussianBlurDegradation:
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=self.sigma))


class JPEGRecompression:
    def __init__(self, quality: int = 75):
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class SharedImageTransform:
    """
    Standardized transform bridging PIL/Numpy inputs, Albumentations pipelines,
    and PyTorch [0, 255] float32 image tensors.
    """
    def __init__(
        self,
        image_size: int = 256,
        target_size: Optional[int] = None,
        img_size: Optional[int] = None,
        pipeline: Optional[Any] = None,
        is_training: bool = False,
    ):
        self.image_size = target_size or img_size or image_size
        self.pipeline = pipeline
        self.is_training = is_training

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
            np_img = np.array(img)
        elif isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                np_img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                np_img = img.cpu().numpy().astype(np.uint8)
        else:
            np_img = np.array(img, dtype=np.uint8)

        if self.pipeline is not None:
            augmented = self.pipeline(image=np_img)
            np_img = augmented["image"]
        else:
            pil_img = Image.fromarray(np_img).resize(
                (self.image_size, self.image_size), Image.LANCZOS
            )
            np_img = np.array(pil_img)

        # Convert to Tensor (3, H, W) scaled to [0, 255.0] float32
        tensor = torch.from_numpy(np_img).float().permute(2, 0, 1).contiguous()
        return tensor


class MLEPPreprocessingTransform(SharedImageTransform):
    pass


__all__ = [
    "GaussianBlurDegradation",
    "JPEGRecompression",
    "SharedImageTransform",
    "MLEPPreprocessingTransform",
]
