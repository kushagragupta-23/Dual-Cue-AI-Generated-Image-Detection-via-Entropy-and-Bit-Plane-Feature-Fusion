"""
Unit tests for the Shared Dataset Infrastructure (metadata, splits, augmentations, dataset, and dataloader).
"""

import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch

from src.data.metadata import (
    validate_image_file,
    scan_dataset_directory,
    generate_metadata_summary,
    export_metadata_summary,
)
from src.data.splits import (
    partition_dataset,
    save_split_manifests,
    load_split_manifest,
)
from src.data.augmentations import (
    get_training_augmentations,
    get_validation_augmentations,
    apply_augmentation_pipeline,
)
from src.data.transforms import SharedImageTransform
from src.data.dataset import SharedImageDataset
from src.data.dataloader import create_dataloader


@pytest.fixture
def mock_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset directory with real, AI, and corrupted image files."""
    real_dir = tmp_path / "0_real"
    ai_dir = tmp_path / "1_stylegan2"
    real_dir.mkdir()
    ai_dir.mkdir()

    # Create 10 real RGB images (varying sizes)
    for i in range(10):
        img = Image.new("RGB", (300, 200), color=(10 + i * 10, 50, 100))
        img.save(real_dir / f"real_{i}.png")

    # Create 10 AI RGB images
    for i in range(10):
        img = Image.new("RGB", (180, 180), color=(200, 20 + i * 10, 50))
        img.save(ai_dir / f"ai_{i}.png")

    # Create 1 corrupted unreadable image
    corrupted_path = ai_dir / "corrupted.png"
    with open(corrupted_path, "w") as f:
        f.write("Not an image header")

    return tmp_path


def test_metadata_integrity_and_scanning(mock_dataset_dir: Path, tmp_path: Path):
    """Verify corrupted image detection, metadata scanning, and JSON summary export."""
    # Check individual file validation
    good_img = mock_dataset_dir / "0_real" / "real_0.png"
    bad_img = mock_dataset_dir / "1_stylegan2" / "corrupted.png"
    assert validate_image_file(good_img) is True
    assert validate_image_file(bad_img) is False

    # Scan directory with integrity validation enabled
    samples = scan_dataset_directory(mock_dataset_dir, validate_integrity=True)
    assert len(samples) == 20  # Exactly 10 real + 10 valid AI (corrupted excluded)

    real_samples = [s for s in samples if s["label"] == 0]
    ai_samples = [s for s in samples if s["label"] == 1]
    assert len(real_samples) == 10
    assert len(ai_samples) == 10
    assert all(s["domain"] == "real" for s in real_samples)
    assert all(s["domain"] == "stylegan2" for s in ai_samples)

    # Generate and export summary
    summary = generate_metadata_summary(samples, root_dir=mock_dataset_dir)
    assert summary["total_samples"] == 20
    assert summary["class_distribution"]["real_count"] == 10
    assert summary["class_distribution"]["ai_generated_count"] == 10

    json_path = tmp_path / "summary.json"
    export_metadata_summary(summary, json_path)
    assert json_path.exists()
    with open(json_path, "r") as f:
        loaded_summary = json.load(f)
    assert loaded_summary["total_samples"] == 20


def test_stratified_dataset_partitioning(tmp_path: Path):
    """Verify stratified partitioning maintains exact 1:1 class ratios across Train/Val/Test splits."""
    samples = []
    # 20 real and 20 fake
    for i in range(20):
        samples.append({"path": f"real_{i}.png", "label": 0, "domain": "real"})
        samples.append({"path": f"fake_{i}.png", "label": 1, "domain": "progan"})

    train_set, val_set, test_set = partition_dataset(samples, val_ratio=0.2, test_ratio=0.2, seed=42)

    # 40 total -> 20% val = 8, 20% test = 8, 60% train = 24
    assert len(train_set) == 24
    assert len(val_set) == 8
    assert len(test_set) == 8

    # Verify exact class stratification in each split
    assert sum(1 for s in train_set if s["label"] == 0) == 12
    assert sum(1 for s in train_set if s["label"] == 1) == 12
    assert sum(1 for s in val_set if s["label"] == 0) == 4
    assert sum(1 for s in val_set if s["label"] == 1) == 4

    # Test saving and reloading manifest
    manifest_dir = tmp_path / "manifests"
    save_split_manifests(train_set, val_set, test_set, manifest_dir)
    loaded_val = load_split_manifest(manifest_dir / "val_split.json")
    assert len(loaded_val) == 8
    assert loaded_val[0]["path"] == val_set[0]["path"]


def test_albumentations_and_transforms():
    """Verify Albumentations training/val pipelines and SharedImageTransform tensor bridge."""
    train_pipe = get_training_augmentations(image_size=256, p_flip=1.0, p_compression=0.0, p_blur=0.0)
    val_pipe = get_validation_augmentations(image_size=256)

    # Test with random numpy HWC image (100x150)
    raw_img = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
    
    aug_train = apply_augmentation_pipeline(train_pipe, raw_img)
    aug_val = apply_augmentation_pipeline(val_pipe, raw_img)
    assert aug_train.shape == (256, 256, 3)
    assert aug_val.shape == (256, 256, 3)

    # Test SharedImageTransform with PIL, Numpy, and Tensor inputs
    tf = SharedImageTransform(image_size=256, pipeline=val_pipe)
    pil_input = Image.fromarray(raw_img)
    tensor_out = tf(pil_input)
    assert isinstance(tensor_out, torch.Tensor)
    assert tensor_out.shape == (3, 256, 256)
    assert tensor_out.dtype == torch.float32
    assert 0.0 <= tensor_out.max() <= 255.0


def test_shared_dataset_and_dataloader(mock_dataset_dir: Path, tmp_path: Path):
    """Verify SharedImageDataset and create_dataloader with balanced sampling."""
    manifest_dir = tmp_path / "splits_out"
    
    # Instantiate dataset for train split (this will scan and save manifests)
    train_ds = SharedImageDataset(
        root_dir=mock_dataset_dir,
        split="train",
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        validate_integrity=True,
        split_manifest_dir=manifest_dir,
    )
    assert len(train_ds) > 0
    img_tensor, label, meta = train_ds[0]
    assert img_tensor.shape == (3, 256, 256)
    assert label in [0, 1]
    assert "domain" in meta and meta["split"] == "train"
    assert (manifest_dir / "train_split.json").exists()

    # Create balanced dataloader
    loader = create_dataloader(
        dataset=train_ds,
        batch_size=4,
        num_workers=0,  # 0 for safe synchronous testing
        balanced_sampling=True,
        drop_last=False,
    )
    
    # Iterate over 1 batch and check balanced class distribution
    for batch_imgs, batch_labels, batch_metas in loader:
        assert batch_imgs.shape == (4, 3, 256, 256)
        assert batch_labels.shape == (4,)
        # In a balanced sampler with batch_size=4, we must get exactly 2 real (0) and 2 fake (1)
        real_count = (batch_labels == 0).sum().item()
        fake_count = (batch_labels == 1).sum().item()
        assert real_count == 2 and fake_count == 2
        break
