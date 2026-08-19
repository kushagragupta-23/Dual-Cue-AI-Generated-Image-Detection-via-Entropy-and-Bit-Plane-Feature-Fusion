"""
Transforms for LOTA Standalone Pipeline.
Provides image preprocessing transforms for the LOTA steganalysis feature extraction.
"""

import torch
from PIL import Image
import torchvision.transforms.functional as TF


class SharedImageTransform:
    """Resize and convert images to [0, 255] float tensors for LOTA pipeline."""

    def __init__(self, target_size: int = 256, image_size: int = None, crop_to_square: bool = False):
        self.target_size = image_size if image_size is not None else target_size
        self.crop_to_square = crop_to_square

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        if self.crop_to_square:
            # Center crop to square before resize
            w, h = img.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))
        img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
        tensor = TF.to_tensor(img) * 255.0  # Scale to [0, 255]
        return tensor


# Alias for backward compatibility
LOTAPreprocessingTransform = SharedImageTransform
LOTATrainTransform = SharedImageTransform

__all__ = ["SharedImageTransform", "LOTAPreprocessingTransform", "LOTATrainTransform"]
