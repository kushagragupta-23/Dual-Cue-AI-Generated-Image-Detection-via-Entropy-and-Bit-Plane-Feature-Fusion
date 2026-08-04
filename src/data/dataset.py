"""
Dataset loading, train/val/test splitting, image validation, and metadata generation.
"""

import json
import logging
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.utils.logger import get_logger

logger = get_logger("dataset_loader")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class AIGIDDataset(Dataset):
    """
    PyTorch Dataset for AI-Generated Image Detection (ForenSynths & GenImage).
    Handles file scanning, integrity validation, deterministic splitting, and metadata generation.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        transform: Optional[Callable] = None,
        seed: int = 42,
        validate_images: bool = True,
        metadata_export_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the dataset loader and perform deterministic split indexing.

        Args:
            root_dir: Root directory of the dataset containing image folders.
            split: Target dataset split ('train', 'val', or 'test').
            val_ratio: Proportion of samples allocated to validation set.
            test_ratio: Proportion of samples allocated to test set.
            transform: Optional torchvision or custom transform applied to loaded images.
            seed: Random seed for deterministic data partitioning.
            validate_images: If True, checks file header integrity during scanning.
            metadata_export_path: If provided, exports calculated metadata statistics to this JSON path.
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split name '{split}'. Must be 'train', 'val', or 'test'.")
        if val_ratio + test_ratio >= 1.0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("Invalid validation/test ratios. Must be non-negative and sum to < 1.0.")

        self.root_dir = Path(root_dir)
        self.split = split
        if transform is None:
            from src.data.transforms import MLEPPreprocessingTransform
            self.transform = MLEPPreprocessingTransform(
                image_size=256, 
                crop_to_square=True,
                enable_augmentations=(split == "train")
            )
        else:
            self.transform = transform
        self.seed = seed
        self.validate_images = validate_images

        # Scan and index valid images
        all_samples = self._scan_and_validate()
        
        # Deterministic partition
        train_samples, val_samples, test_samples = self._partition_splits(
            all_samples, val_ratio, test_ratio, seed
        )

        if split == "train":
            self.samples = train_samples
        elif split == "val":
            self.samples = val_samples
        else:
            self.samples = test_samples

        logger.info(f"Initialized AIGIDDataset [{split.upper()}] with {len(self.samples)} valid samples.")

        # Optional metadata export
        if metadata_export_path is not None:
            self.export_metadata(metadata_export_path, all_samples, train_samples, val_samples, test_samples)

    def _infer_label_and_domain(self, file_path: Path) -> Tuple[int, str]:
        """
        Infer Real (0) vs AI-Generated (1) label and generator domain from relative file path.
        """
        parts_lower = [p.lower() for p in file_path.parts]
        
        # Check for standard real indicators
        if any(w in parts_lower for w in ["real", "0_real", "nature", "authentic", "original", "imagenet", "lsun"]):
            label = 0
            domain = "real"
        else:
            label = 1
            # Try to identify generator domain
            for gen in ["stylegan3", "stylegan2", "stylegan", "progan", "biggan", "cyclegan", "stargan", "gaugan", "sdv15", "sdv14", "sdxl", "midjourney", "flux", "adm", "vqdm", "wukong", "glide"]:
                if any(gen in part for part in parts_lower):
                    domain = gen
                    break
            else:
                domain = "ai_unknown"

        return label, domain

    def _is_valid_image(self, file_path: Path) -> bool:
        """Verify image header readability and RGB compatibility."""
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(f"Corrupted image file ignored: {file_path} ({e})")
            return False

    def _scan_and_validate(self) -> List[Dict[str, Any]]:
        """Scan directory tree for image files and build indexed metadata list."""
        if not self.root_dir.exists():
            logger.warning(f"Dataset root directory does not exist: {self.root_dir}")
            return []

        samples = []
        for path in sorted(self.root_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                if self.validate_images and not self._is_valid_image(path):
                    continue
                
                label, domain = self._infer_label_and_domain(path)
                samples.append({
                    "path": str(path.resolve()),
                    "label": label,
                    "domain": domain,
                })

        return samples

    def _partition_splits(
        self,
        samples: List[Dict[str, Any]],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Perform stratified deterministic data partitioning into Train/Val/Test."""
        if not samples:
            return [], [], []

        # Separate by class label to maintain class balance across splits
        real_samples = [s for s in samples if s["label"] == 0]
        fake_samples = [s for s in samples if s["label"] == 1]

        rng = random.Random(seed)
        rng.shuffle(real_samples)
        rng.shuffle(fake_samples)

        def split_list(lst: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
            n = len(lst)
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio)
            n_train = n - n_val - n_test
            return lst[:n_train], lst[n_train : n_train + n_val], lst[n_train + n_val :]

        real_train, real_val, real_test = split_list(real_samples)
        fake_train, fake_val, fake_test = split_list(fake_samples)

        train_set = real_train + fake_train
        val_set = real_val + fake_val
        test_set = real_test + fake_test

        rng.shuffle(train_set)
        rng.shuffle(val_set)
        rng.shuffle(test_set)

        return train_set, val_set, test_set

    def generate_metadata_summary(
        self,
        all_samples: Optional[List[Dict[str, Any]]] = None,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Calculate statistical distribution metadata for the dataset."""
        samples_to_analyze = all_samples if all_samples is not None else self.samples

        total = len(samples_to_analyze)
        real_count = sum(1 for s in samples_to_analyze if s["label"] == 0)
        fake_count = sum(1 for s in samples_to_analyze if s["label"] == 1)

        domain_counts: Dict[str, int] = {}
        for s in samples_to_analyze:
            dom = s["domain"]
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

        summary = {
            "root_directory": str(self.root_dir.resolve()),
            "total_samples": total,
            "class_distribution": {
                "real": real_count,
                "ai_generated": fake_count,
                "real_ratio": round(real_count / total, 4) if total > 0 else 0.0,
            },
            "generator_domain_counts": domain_counts,
            "splits": {
                "train_size": len(train_samples) if train_samples is not None else 0,
                "val_size": len(val_samples) if val_samples is not None else 0,
                "test_size": len(test_samples) if test_samples is not None else 0,
            },
        }
        return summary

    def export_metadata(
        self,
        export_path: Union[str, Path],
        all_samples: Optional[List[Dict[str, Any]]] = None,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Export calculated metadata summary to a formatted JSON file."""
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.generate_metadata_summary(all_samples, train_samples, val_samples, test_samples)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        logger.info(f"Dataset metadata summary exported to: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Fetch and transform image sample at index idx.

        Returns:
            tensor: Image tensor of shape (3, H, W).
            label: Integer class label (0 for Real, 1 for AI-Generated).
            metadata: Dictionary containing path and generator domain info.
        """
        sample_info = self.samples[idx]
        file_path = Path(sample_info["path"])
        label = sample_info["label"]

        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            logger.error(f"Error reading image {file_path}: {e}")
            # Fallback to zero tensor to prevent dataloader crashing
            img_rgb = Image.new("RGB", (256, 256), color=(0, 0, 0))

        if self.transform is not None:
            tensor = self.transform(img_rgb)
        else:
            # Default fallback transform to ensure 256x256 standardization
            from src.data.transforms import MLEPPreprocessingTransform
            default_tf = MLEPPreprocessingTransform(image_size=256, crop_to_square=True)
            tensor = default_tf(img_rgb)

        return tensor, label, {"path": str(file_path), "domain": sample_info["domain"]}


class SharedImageDataset(Dataset):
    """
    Unified PyTorch Dataset for Shared MLEP Infrastructure.
    Supports ForenSynths, GenImage, and custom directory structures.
    Provides clean RGB float32 tensors of shape (3, 256, 256) to downstream pipelines.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        transform: Optional[Any] = None,
        seed: int = 42,
        validate_integrity: bool = True,
        split_manifest_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize SharedImageDataset and load or compute stratified data splits.
        """
        if split not in ["train", "val", "validation", "test"]:
            raise ValueError(f"Invalid split name '{split}'. Must be 'train', 'val', or 'test'.")
        if val_ratio + test_ratio >= 1.0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("Invalid validation/test ratios. Must be non-negative and sum to < 1.0.")

        self.root_dir = Path(root_dir)
        self.split = "val" if split == "validation" else split
        self.seed = seed
        self.validate_integrity = validate_integrity

        if transform is None:
            from src.data.transforms import SharedImageTransform
            self.transform = SharedImageTransform(image_size=256)
        else:
            self.transform = transform

        loaded_from_manifest = False
        if split_manifest_dir is not None:
            manifest_dir = Path(split_manifest_dir)
            target_manifest = manifest_dir / f"{self.split}_split.json"
            if target_manifest.exists():
                from src.data.splits import load_split_manifest
                self.samples = load_split_manifest(target_manifest)
                loaded_from_manifest = True

        if not loaded_from_manifest:
            from src.data.metadata import scan_dataset_directory
            from src.data.splits import partition_dataset, save_split_manifests

            all_samples = scan_dataset_directory(self.root_dir, validate_integrity=self.validate_integrity)
            train_set, val_set, test_set = partition_dataset(all_samples, val_ratio, test_ratio, seed)

            if split_manifest_dir is not None:
                save_split_manifests(train_set, val_set, test_set, split_manifest_dir)

            if self.split == "train":
                self.samples = train_set
            elif self.split == "val":
                self.samples = val_set
            else:
                self.samples = test_set

        logger.info(f"Initialized SharedImageDataset [{self.split.upper()}] with {len(self.samples)} valid samples.")

    def get_labels(self) -> List[int]:
        """Return list of integer class labels for sampler initialization."""
        return [s["label"] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Fetch and transform RGB image sample at index idx.
        """
        sample_info = self.samples[idx]
        file_path = Path(sample_info["path"])
        label = sample_info["label"]

        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            logger.error(f"Error reading image {file_path}: {e}")
            img_rgb = Image.new("RGB", (256, 256), color=(0, 0, 0))

        if self.transform is not None:
            tensor = self.transform(img_rgb)
        else:
            from src.data.transforms import SharedImageTransform
            default_tf = SharedImageTransform(image_size=256)
            tensor = default_tf(img_rgb)

        return tensor, label, {"path": str(file_path), "domain": sample_info["domain"], "split": self.split}

