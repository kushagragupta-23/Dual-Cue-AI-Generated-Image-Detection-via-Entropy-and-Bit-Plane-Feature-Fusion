"""
Shared Image Dataset for LOTA Standalone Pipeline.
Provides stratified dataset splitting with domain-aware sampling for AI-generated image detection.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

import sys
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.logger import get_logger

logger = get_logger("dataset")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class SharedImageTransform:
    """Resize and convert images to [0, 255] float tensors for LOTA pipeline."""

    def __init__(self, target_size: int = 256):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
        tensor = TF.to_tensor(img) * 255.0  # Scale to [0, 255]
        return tensor


LOTAPreprocessingTransform = SharedImageTransform


class SharedImageDataset(Dataset):
    """
    Dataset that scans domain folders and performs stratified train/val/test splitting.

    Expected directory structure:
      root_dir/
        0_real/       -> label 0
        1_stylegan2/  -> label 1
        1_midjourney/ -> label 1
        ...
    OR:
      root_dir/
        train/ (or validation/ or test/)
          real/  -> label 0
          fake/  -> label 1
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
        target_size: int = 256,
        validate_integrity: bool = False,
        split_manifest_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = SharedImageTransform(target_size)

        # Try pre-split directory structure first (train/val/test folders)
        split_folder_map = {"train": "train", "val": "validation", "test": "test"}
        split_folder = self.root_dir / split_folder_map.get(split, split)

        if split_folder.exists() and any(split_folder.iterdir()):
            self.samples = self._load_presplit(split_folder)
        else:
            # Fall back to flat domain-folder structure with stratified splitting
            all_samples = self._scan_domain_folders(self.root_dir)
            train, val, test = self._stratified_split(all_samples, val_ratio, test_ratio, seed)
            split_data = {"train": train, "val": val, "test": test}
            self.samples = split_data.get(split, train)

        logger.info(f"[{split}] Loaded {len(self.samples)} images from {self.root_dir}")

    def _load_presplit(self, split_dir: Path) -> List[Tuple[Path, int, Dict]]:
        """Load from pre-split directory with real/ and fake/ subfolders."""
        samples = []
        for label_name, label_id in [("real", 0), ("fake", 1), ("0_real", 0)]:
            label_dir = split_dir / label_name
            if label_dir.exists():
                for f in sorted(label_dir.iterdir()):
                    if f.suffix.lower() in VALID_EXTENSIONS:
                        samples.append((f, label_id, {"domain": label_name, "filename": f.name}))

        # Also check for any 1_* folders (AI generators)
        for folder in sorted(split_dir.iterdir()):
            if folder.is_dir() and folder.name.startswith("1_"):
                for f in sorted(folder.iterdir()):
                    if f.suffix.lower() in VALID_EXTENSIONS:
                        samples.append((f, 1, {"domain": folder.name, "filename": f.name}))
        return samples

    def _scan_domain_folders(self, root: Path) -> List[Tuple[Path, int, Dict]]:
        """Scan flat domain folders (0_real, 1_stylegan2, etc.)."""
        samples = []
        for folder in sorted(root.iterdir()):
            if not folder.is_dir():
                continue
            if folder.name.startswith("0_"):
                label = 0
            elif folder.name.startswith("1_"):
                label = 1
            else:
                continue
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in VALID_EXTENSIONS:
                    samples.append((f, label, {"domain": folder.name, "filename": f.name}))
        return samples

    def _stratified_split(
        self,
        samples: List[Tuple[Path, int, Dict]],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> Tuple[list, list, list]:
        """Stratified split preserving class balance."""
        rng = random.Random(seed)
        class_0 = [s for s in samples if s[1] == 0]
        class_1 = [s for s in samples if s[1] == 1]
        rng.shuffle(class_0)
        rng.shuffle(class_1)

        def split_list(data):
            n = len(data)
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)
            return data[n_test + n_val:], data[n_test:n_test + n_val], data[:n_test]

        train_0, val_0, test_0 = split_list(class_0)
        train_1, val_1, test_1 = split_list(class_1)

        return train_0 + train_1, val_0 + val_1, test_0 + test_1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        img_path, label, meta = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                tensor = self.transform(img)
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            tensor = torch.zeros(3, 256, 256)
        return tensor, torch.tensor(label, dtype=torch.long), meta
