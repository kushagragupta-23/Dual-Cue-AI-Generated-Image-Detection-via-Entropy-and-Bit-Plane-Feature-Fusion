"""
Dataset Loader for MLEP.
Re-uses centralized ForensicsDataset from HydraFusion/src/data/dataset.py.
"""

import sys
import importlib.util
from pathlib import Path

# Load HydraFusion dataset module directly by path to prevent circular import
hydrafusion_dataset_path = Path(__file__).resolve().parent.parent.parent.parent / "HydraFusion" / "src" / "data" / "dataset.py"

if not hydrafusion_dataset_path.exists():
    raise FileNotFoundError(f"Centralized dataset file not found at {hydrafusion_dataset_path}")

spec = importlib.util.spec_from_file_location("hydrafusion_dataset", hydrafusion_dataset_path)
hf_dataset_mod = importlib.util.module_from_spec(spec)
sys.modules["hydrafusion_dataset"] = hf_dataset_mod
spec.loader.exec_module(hf_dataset_mod)

ForensicsDataset = hf_dataset_mod.ForensicsDataset
SharedImageDataset = hf_dataset_mod.ForensicsDataset
get_dataloaders = hf_dataset_mod.get_dataloaders

__all__ = ["ForensicsDataset", "SharedImageDataset", "get_dataloaders"]
