"""
Albumentations transform pipelines for shared MLEP & LOTA dataset infrastructure.
Provides standardized 256x256 resizing and configurable online robustness augmentations.
"""

from typing import Optional
import albumentations as A
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("dataset_augmentations")


def get_training_augmentations(
    image_size: int = 256,
    p_flip: float = 0.5,
    p_compression: float = 0.5,
    p_blur: float = 0.3,
) -> A.Compose:
    """
    Construct the online training augmentation pipeline using Albumentations.

    Args:
        image_size: Target spatial height and width for standardization (default 256).
        p_flip: Probability of horizontal flipping.
        p_compression: Probability of simulated social media JPEG compression.
        p_blur: Probability of Gaussian blur degradation.

    Returns:
        albumentations.Compose: Configured training transform pipeline.
    """
    pipeline = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=1), # cv2.INTER_LINEAR equivalent
        A.HorizontalFlip(p=p_flip),
        A.ImageCompression(quality_range=(70, 100), p=p_compression),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 2.0), p=p_blur),
    ])
    logger.debug(f"Created training augmentation pipeline (size={image_size}, flip={p_flip}, comp={p_compression}, blur={p_blur})")
    return pipeline


def get_validation_augmentations(image_size: int = 256) -> A.Compose:
    """
    Construct the deterministic evaluation and validation transform pipeline.
    Performs clean resizing without stochastic degradation.

    Args:
        image_size: Target spatial height and width (default 256).

    Returns:
        albumentations.Compose: Configured evaluation transform pipeline.
    """
    pipeline = A.Compose([
        A.Resize(height=image_size, width=image_size, interpolation=1),
    ])
    logger.debug(f"Created validation augmentation pipeline (size={image_size})")
    return pipeline


def apply_augmentation_pipeline(
    pipeline: A.Compose,
    image: np.ndarray,
) -> np.ndarray:
    """
    Execute an Albumentations transform pipeline on an HWC uint8 numpy array.

    Args:
        pipeline: The Albumentations Compose object.
        image: HWC RGB numpy array in uint8 format [0, 255].

    Returns:
        numpy.ndarray: Transformed HWC RGB numpy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    
    augmented = pipeline(image=image)
    return augmented["image"]
