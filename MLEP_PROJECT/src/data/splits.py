"""
Stratified deterministic Train/Validation/Test splitting and index manifest persistence.
"""

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from src.utils.logger import get_logger

logger = get_logger("dataset_splits")


def partition_dataset(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partition sample dictionaries into stratified deterministic Train, Validation, and Test sets.

    Args:
        samples: List of sample dictionaries containing 'label' and 'path'.
        val_ratio: Proportion of dataset allocated to validation set (0.0 to 1.0).
        test_ratio: Proportion of dataset allocated to test set (0.0 to 1.0).
        seed: Random seed for deterministic shuffling.

    Returns:
        tuple: (train_samples, val_samples, test_samples)
    """
    if not (0.0 <= val_ratio + test_ratio < 1.0):
        raise ValueError(f"Sum of val_ratio ({val_ratio}) and test_ratio ({test_ratio}) must be < 1.0.")

    # Group by class label for stratified splitting
    real_samples = [s for s in samples if s.get("label") == 0]
    fake_samples = [s for s in samples if s.get("label") == 1]

    rng = random.Random(seed)
    rng.shuffle(real_samples)
    rng.shuffle(fake_samples)

    def _split_list(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        n = len(data)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = n - n_val - n_test
        
        train_sub = data[:n_train]
        val_sub = data[n_train : n_train + n_val]
        test_sub = data[n_train + n_val :]
        return train_sub, val_sub, test_sub

    real_train, real_val, real_test = _split_list(real_samples)
    fake_train, fake_val, fake_test = _split_list(fake_samples)

    train_samples = real_train + fake_train
    val_samples = real_val + fake_val
    test_samples = real_test + fake_test

    # Final deterministic shuffle of combined sets
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    logger.info(
        f"Partitioned {len(samples)} samples -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}"
    )
    return train_samples, val_samples, test_samples


def save_split_manifests(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    output_dir: Union[str, Path],
) -> None:
    """
    Save Train, Validation, and Test split sample lists as JSON manifests.

    Args:
        train_samples: Train split samples.
        val_samples: Validation split samples.
        test_samples: Test split samples.
        output_dir: Directory path where split JSON manifests will be saved.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "train_split.json": train_samples,
        "val_split.json": val_samples,
        "test_split.json": test_samples,
    }

    for fname, data in manifests.items():
        filepath = out_dir / fname
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved split manifest ({len(data)} items) to: {filepath}")


def load_split_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load a saved dataset split manifest from disk.

    Args:
        manifest_path: Path to the JSON manifest file.

    Returns:
        list of dicts: List of sample dictionaries.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(data)} samples from manifest: {path}")
    return data
