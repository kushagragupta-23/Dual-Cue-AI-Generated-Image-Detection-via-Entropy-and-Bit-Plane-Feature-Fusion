"""
Data Processing Transforms for HydraFusion-Net.

Provides:
  - get_training_transforms: Standard resizing, flip, rotation, jitter
  - get_validation_transforms: Standard resizing and tensor conversion
"""

from torchvision import transforms


def get_training_transforms(img_size: int = 256):
    """Return default training data transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])


def get_validation_transforms(img_size: int = 256):
    """Return default validation/testing data transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
