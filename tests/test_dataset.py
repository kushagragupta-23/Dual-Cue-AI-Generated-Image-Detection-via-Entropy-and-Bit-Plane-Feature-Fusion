"""
Unit tests for data ingestion, transformations, dataset loading, and balanced sampling.
"""

import json
from pathlib import Path
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
from src.data.transforms import (
    GaussianBlurDegradation,
    JPEGRecompression,
    LOTAPreprocessingTransform,
)
from src.data.dataset import AIGIDDataset
from src.data.samplers import BalancedRealFakeSampler


@pytest.fixture
def dummy_dataset_dir(tmp_path: Path) -> Path:
    """Create synthetic dataset directory with valid real and ai image files."""
    root = tmp_path / "synthetic_data"
    real_dir = root / "0_real" / "imagenet"
    fake_dir = root / "1_fake" / "progan"
    
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 10 real images and 10 fake images
    for i in range(10):
        img_real = Image.new("RGB", (300, 300), color=(i * 20, 100, 150))
        img_real.save(real_dir / f"real_{i}.png")
        
        img_fake = Image.new("RGB", (200, 400), color=(150, i * 20, 100))
        img_fake.save(fake_dir / f"progan_{i}.jpg")
        
    return root


def test_transforms_resizing_and_range():
    """Verify LOTAPreprocessingTransform resizes to (3, 256, 256) and returns [0, 255]."""
    transform = LOTAPreprocessingTransform(image_size=256, crop_to_square=True)
    img = Image.new("RGB", (512, 384), color=(128, 64, 32))
    
    tensor = transform(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)
    assert tensor.dtype == torch.float32
    assert tensor.max() <= 255.0 and tensor.min() >= 0.0


def test_robustness_augmentations():
    """Verify JPEG compression and Gaussian blur execute without errors."""
    img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    
    jpeg_deg = JPEGRecompression(quality=75)
    img_jpeg = jpeg_deg(img)
    assert isinstance(img_jpeg, Image.Image)
    assert img_jpeg.size == (256, 256)
    
    blur_deg = GaussianBlurDegradation(sigma=1.5)
    img_blur = blur_deg(img)
    assert isinstance(img_blur, Image.Image)
    assert img_blur.size == (256, 256)


def test_dataset_scanning_and_splits(dummy_dataset_dir: Path):
    """Verify AIGIDDataset scans files, infers labels, and splits train/val/test."""
    dataset_train = AIGIDDataset(
        root_dir=dummy_dataset_dir,
        split="train",
        val_ratio=0.2,
        test_ratio=0.2,
        validate_images=True,
    )
    dataset_val = AIGIDDataset(
        root_dir=dummy_dataset_dir,
        split="val",
        val_ratio=0.2,
        test_ratio=0.2,
        validate_images=True,
    )
    dataset_test = AIGIDDataset(
        root_dir=dummy_dataset_dir,
        split="test",
        val_ratio=0.2,
        test_ratio=0.2,
        validate_images=True,
    )
    
    total_samples = len(dataset_train) + len(dataset_val) + len(dataset_test)
    assert total_samples == 20  # 10 real + 10 fake
    
    # Test __getitem__
    tensor, label, meta = dataset_train[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 256, 256)
    assert label in [0, 1]
    assert "path" in meta and "domain" in meta


def test_metadata_export(dummy_dataset_dir: Path, tmp_path: Path):
    """Verify dataset statistics summary JSON generation and export."""
    export_file = tmp_path / "metadata.json"
    dataset = AIGIDDataset(
        root_dir=dummy_dataset_dir,
        split="train",
        metadata_export_path=export_file,
    )
    
    assert export_file.exists()
    data = json.loads(export_file.read_text(encoding="utf-8"))
    assert data["total_samples"] == 20
    assert data["class_distribution"]["real"] == 10
    assert data["class_distribution"]["ai_generated"] == 10


def test_balanced_sampler(dummy_dataset_dir: Path):
    """Verify BalancedRealFakeSampler yields 50/50 Real/AI ratio per mini-batch."""
    dataset = AIGIDDataset(root_dir=dummy_dataset_dir, split="train", val_ratio=0.0, test_ratio=0.0)
    
    batch_size = 4
    sampler = BalancedRealFakeSampler(dataset=dataset, batch_size=batch_size, drop_last=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    for batch_tensors, batch_labels, _ in loader:
        assert len(batch_labels) == batch_size
        num_real = (batch_labels == 0).sum().item()
        num_fake = (batch_labels == 1).sum().item()
        assert num_real == 2 and num_fake == 2
