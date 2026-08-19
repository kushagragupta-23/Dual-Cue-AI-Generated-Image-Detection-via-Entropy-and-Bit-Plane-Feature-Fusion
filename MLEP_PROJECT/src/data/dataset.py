"""
Shared Image Dataset for MLEP Standalone Pipeline.
Provides metadata tracking, deterministic train/val/test splitting, and manifest persistence.
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.data.transforms import SharedImageTransform
from src.data.metadata import scan_dataset_directory, validate_image_file
from src.data.splits import partition_dataset, save_split_manifests, load_split_manifest
from src.utils.logger import get_logger

logger = get_logger("dataset")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class SharedImageDataset(Dataset):
    """
    Dataset that scans domain folders or physical train/val/test splits,
    performs stratified splitting, and persists split manifests.
    """
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        transform: Optional[Any] = None,
        image_size: int = 256,
        img_size: Optional[int] = None,
        validate_integrity: bool = False,
        validate_images: bool = False,
        split_manifest_dir: Optional[Union[str, Path]] = None,
        metadata_export_path: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.image_size = img_size or image_size
        self.validate_integrity = validate_integrity or validate_images
        self.seed = seed

        self.transform = transform or SharedImageTransform(image_size=self.image_size)

        # 1. Check physical split folders
        split_folders = {
            "train": ["train"],
            "val": ["validation", "val"],
            "test": ["test"],
        }
        target_splits = split_folders.get(split, [split])
        found_physical_split = False

        for sf in target_splits:
            s_path = self.root_dir / sf
            if s_path.exists() and s_path.is_dir():
                found_physical_split = True
                self.samples = []
                for label_str, label in [("real", 0), ("fake", 1)]:
                    c_path = s_path / label_str
                    if c_path.exists():
                        for img_f in sorted(c_path.glob("*.*")):
                            if img_f.suffix.lower() in VALID_EXTENSIONS:
                                if self.validate_integrity and not validate_image_file(img_f):
                                    continue
                                self.samples.append({
                                    "path": str(img_f),
                                    "label": label,
                                    "domain": "real" if label == 0 else "fake",
                                    "split": split,
                                })
                break

        # 2. Dynamic scanning & partitioning
        if not found_physical_split:
            all_samples = scan_dataset_directory(self.root_dir, validate_integrity=self.validate_integrity)
            train_s, val_s, test_s = partition_dataset(
                all_samples, val_ratio=self.val_ratio, test_ratio=self.test_ratio, seed=self.seed
            )

            if split_manifest_dir is not None:
                save_split_manifests(train_s, val_s, test_s, Path(split_manifest_dir))

            split_map = {"train": train_s, "val": val_s, "validation": val_s, "test": test_s}
            selected = split_map.get(split, train_s)
            self.samples = []
            for s in selected:
                s_copy = dict(s)
                s_copy["split"] = split
                self.samples.append(s_copy)

        if metadata_export_path:
            p = Path(metadata_export_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            export_source = all_samples if not found_physical_split else self.samples
            real_c = sum(1 for s in export_source if s["label"] == 0)
            fake_c = sum(1 for s in export_source if s["label"] == 1)
            meta_data = {
                "total_samples": len(export_source),
                "class_distribution": {"real": real_c, "ai_generated": fake_c},
                "split": self.split,
            }
            p.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

        logger.info(f"[{split}] Initialized SharedImageDataset with {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        sample = self.samples[idx]
        img_path = sample["path"]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, sample["label"], sample


AIGIDDataset = SharedImageDataset
ForensicsDataset = SharedImageDataset

__all__ = ["SharedImageDataset", "AIGIDDataset", "ForensicsDataset"]
