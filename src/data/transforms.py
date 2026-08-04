"""
Image transformations, standardization, and online robustness augmentations for LOTA.
"""

from io import BytesIO
import random
from typing import Any, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms.functional as TF


class JPEGRecompression:
    """
    Simulate online JPEG compression degradation on PIL Images or Tensors.
    Used for testing the robustness of LSB bit-plane noise fingerprints.
    """
    def __init__(self, quality: int = 80):
        if not (1 <= quality <= 100):
            raise ValueError(f"JPEG quality must be between 1 and 100, got {quality}")
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply JPEG recompression to a PIL Image in memory."""
        if not isinstance(img, Image.Image):
            raise TypeError("Input to JPEGRecompression must be a PIL Image.")
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class GaussianBlurDegradation:
    """
    Apply Gaussian blur filtering to simulate sensor noise smoothing.
    """
    def __init__(self, sigma: float = 1.0):
        if sigma < 0.0:
            raise ValueError(f"Blur sigma must be non-negative, got {sigma}")
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply Gaussian blur to a PIL Image."""
        if not isinstance(img, Image.Image):
            raise TypeError("Input to GaussianBlurDegradation must be a PIL Image.")
        if self.sigma == 0.0:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=self.sigma))


class LOTAPreprocessingTransform:
    """
    Standard preprocessing transform for LOTA feature extraction.
    Converts image to RGB, applies optional center cropping and resizing to 256x256,
    executes optional robustness augmentations, and returns a PyTorch Tensor
    in range [0.0, 255.0] with shape (3, H, W).
    """
    def __init__(
        self,
        image_size: int = 256,
        crop_to_square: bool = True,
        enable_augmentations: bool = False,
        jpeg_quality_range: Tuple[int, int] = (70, 100),
        blur_sigma_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.image_size = image_size
        self.crop_to_square = crop_to_square
        self.enable_augmentations = enable_augmentations
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_sigma_range = blur_sigma_range

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        """Apply random JPEG recompression, Gaussian blur, ColorJitter, and Horizontal Flip."""
        if random.random() < 0.5:
            q = random.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1])
            img = JPEGRecompression(quality=q)(img)
        if random.random() < 0.5:
            sigma = random.uniform(self.blur_sigma_range[0], self.blur_sigma_range[1])
            img = GaussianBlurDegradation(sigma=sigma)(img)
        # Prevent overfitting with structural/spatial augmentations
        # Increased probabilities to force better generalization for the >90% validation goal
        if random.random() < 0.7:
            from torchvision.transforms import ColorJitter
            img = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)(img)
        if random.random() < 0.5:
            from torchvision.transforms.functional import hflip
            img = hflip(img)
        if random.random() < 0.5:
            from torchvision.transforms import RandomRotation
            # Slight random rotation to prevent spatial memorization
            img = RandomRotation(degrees=10)(img)
        if random.random() < 0.5:
            from torchvision.transforms import RandomResizedCrop
            # Random scale crop prevents the model from relying on fixed object positioning
            img = RandomResizedCrop(size=self.image_size, scale=(0.8, 1.0))(img)
        return img

    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Execute transformation pipeline.

        Args:
            img: Input PIL Image or PyTorch Tensor.

        Returns:
            torch.Tensor: Float32 tensor of shape (3, 256, 256) with values in [0.0, 255.0].
        """
        if isinstance(img, torch.Tensor):
            # Convert tensor back to PIL for standard formatting
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).astype("uint8")
                else:
                    img_np = img_np.astype("uint8")
                img = Image.fromarray(img_np)
            else:
                raise ValueError(f"Unsupported tensor shape for transform: {img.shape}")

        # Ensure RGB conversion
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Optional online augmentations
        if self.enable_augmentations:
            img = self._apply_augmentations(img)

        # Standard cropping and resizing
        if self.crop_to_square:
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))

        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), resample=Image.Resampling.BILINEAR)

        # Convert to Tensor without dividing by 255 (keep [0.0, 255.0] range for bit-plane slicing)
        tensor = TF.to_tensor(img) * 255.0
        return tensor.to(torch.float32)


class SharedImageTransform:
    """
    Standard PyTorch tensor bridge for shared dataset infrastructure.
    Converts input PIL Images or numpy arrays to RGB, executes optional Albumentations
    transform pipelines, and outputs float32 tensors of shape (3, H, W).
    """
    def __init__(
        self,
        image_size: int = 256,
        pipeline: Optional[Any] = None,
        normalize_to_01: bool = False,
    ):
        self.image_size = image_size
        self.pipeline = pipeline
        self.normalize_to_01 = normalize_to_01

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Execute transformation on PIL Image, numpy array, or PyTorch tensor.

        Args:
            img: Input image representation.

        Returns:
            torch.Tensor: Float32 tensor of shape (3, H, W).
        """
        if isinstance(img, torch.Tensor):
            if img.ndim == 3:
                img = img.permute(1, 2, 0).cpu().numpy()
                if img.max() <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

        if isinstance(img, Image.Image):
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = np.array(img)

        if not isinstance(img, np.ndarray):
            raise TypeError(f"Unsupported image input type: {type(img)}")

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]

        if self.pipeline is not None:
            res = self.pipeline(image=img)
            img = res["image"]
        else:
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                pil_tmp = Image.fromarray(img)
                pil_tmp = pil_tmp.resize((self.image_size, self.image_size), resample=Image.Resampling.BILINEAR)
                img = np.array(pil_tmp)

        tensor = torch.from_numpy(img.transpose(2, 0, 1)).to(torch.float32)
        if self.normalize_to_01:
            tensor = tensor / 255.0

        return tensor

