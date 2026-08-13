import os
from pathlib import Path
from typing import Tuple, Dict, Any, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils.logger import get_logger

logger = get_logger("dataset")


class ForensicsDataset(Dataset):
    """
    Dataset loader for HydraFusion.
    Expects structure:
      data_dir/
        train/  (or validation/ or test/)
          real/   -> label 0
          fake/   -> label 1
    """
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, data_dir: str, split: str = "train", img_size: int = 256, is_training: bool = False):
        super().__init__()
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            fallbacks = [
                Path("dataset10000"),
                Path(__file__).resolve().parent.parent.parent / "MLEP PROJECT" / "dataset10000",
                Path("d:/MAIN PROJECT CV AND DL/MLEP PROJECT/dataset10000"),
                Path("C:/Users/Eldoria/Music/project main cl dv/DL AND CV PROJECT (1)/dataset10000"),
            ]
            for fb in fallbacks:
                if fb.exists():
                    self.data_dir = fb
                    logger.info(f"Primary data_dir '{data_dir}' not found. Using fallback dataset path: '{fb}'")
                    break
        self.split = split
        self.img_size = img_size
        self.is_training = is_training

        self.samples = self._load_samples()
        logger.info(f"[{split}] Loaded {len(self.samples)} images from {self.data_dir / self._split_folder()}")

        # Training augmentations for regularisation
        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

    def _split_folder(self) -> str:
        """Map split name to actual folder name."""
        mapping = {"train": "train", "val": "validation", "test": "test"}
        return mapping.get(self.split, self.split)

    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        split_dir = self.data_dir / self._split_folder()

        real_dir = split_dir / "real"
        fake_dir = split_dir / "fake"

        if real_dir.exists():
            for f in sorted(real_dir.iterdir()):
                if f.suffix.lower() in self.VALID_EXTENSIONS:
                    samples.append((f, 0))

        if fake_dir.exists():
            for f in sorted(fake_dir.iterdir()):
                if f.suffix.lower() in self.VALID_EXTENSIONS:
                    samples.append((f, 1))

        if not samples:
            logger.warning(f"No images found in {split_dir}. Expected real/ and fake/ sub-directories.")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                tensor = self.transform(img)
                # Scale to [0.0, 255.0] to match LOTA/MLEP expected input ranges
                tensor = tensor * 255.0
                return tensor, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a zeroed tensor on failure so training doesn't crash
            return torch.zeros(3, self.img_size, self.img_size), 0


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test dataloaders with GPU-optimised settings."""
    data_dir = config["dataset"]["data_dir"]
    img_size = config["dataset"]["image_size"]
    batch_size = config["dataset"]["batch_size"]
    # Cap workers at 2 on Windows to prevent DataLoader deadlocks
    num_workers = min(config["dataset"].get("num_workers", 2), 2)

    train_ds = ForensicsDataset(data_dir, split="train", img_size=img_size, is_training=True)
    val_ds   = ForensicsDataset(data_dir, split="val",   img_size=img_size, is_training=False)
    test_ds  = ForensicsDataset(data_dir, split="test",  img_size=img_size, is_training=False)

    # pin_memory + non_blocking enables async CPU->GPU transfers
    # persistent_workers DISABLED on Windows to prevent deadlocks
    common = dict(pin_memory=True, persistent_workers=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, **common)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, **common)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, **common)

    return train_loader, val_loader, test_loader
