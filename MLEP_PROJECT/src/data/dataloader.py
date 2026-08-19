"""
Dataloader utility for MLEP standalone execution.
Re-uses centralized get_dataloaders from HydraFusion.
"""

import sys
import importlib.util
from pathlib import Path

hydrafusion_dataset_path = Path(__file__).resolve().parent.parent.parent.parent / "HydraFusion" / "src" / "data" / "dataset.py"

if not hydrafusion_dataset_path.exists():
    raise FileNotFoundError(f"Centralized dataset file not found at {hydrafusion_dataset_path}")

spec = importlib.util.spec_from_file_location("hydrafusion_dataset", hydrafusion_dataset_path)
hf_dataset_mod = importlib.util.module_from_spec(spec)
sys.modules["hydrafusion_dataset"] = hf_dataset_mod
spec.loader.exec_module(hf_dataset_mod)

get_dataloaders = hf_dataset_mod.get_dataloaders

def create_dataloader(data_dir, batch_size=16, is_training=True, img_size=256, num_workers=2):
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )
    return train_loader if is_training else val_loader

__all__ = ["create_dataloader", "get_dataloaders"]
