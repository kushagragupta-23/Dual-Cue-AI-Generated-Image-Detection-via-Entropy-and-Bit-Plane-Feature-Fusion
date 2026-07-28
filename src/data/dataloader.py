"""
DataLoader utilities and factory functions for shared dataset infrastructure.
Supports balanced 50/50 mini-batch sampling and reproducible multi-worker setup.
"""

import random
from typing import Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils.logger import get_logger

logger = get_logger("dataset_dataloader")


def _worker_init_fn(worker_id: int) -> None:
    """
    Seed random number generators in worker subprocesses for reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    balanced_sampling: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Factory function to construct a PyTorch DataLoader for MLEP and LOTA branches.

    Args:
        dataset: Target PyTorch Dataset instance (e.g., SharedImageDataset).
        batch_size: Number of images per mini-batch.
        shuffle: Whether to shuffle data sequentially (ignored if balanced_sampling=True).
        num_workers: Number of background subprocesses for image loading and augmentations.
        balanced_sampling: If True, uses BalancedRealFakeSampler for 50/50 class ratios.
        pin_memory: If True, pins memory for rapid CPU-to-GPU transfers.
        drop_last: If True, drops incomplete final batches.

    Returns:
        torch.utils.data.DataLoader: Configured dataloader instance.
    """
    sampler = None
    if balanced_sampling:
        from src.data.samplers import BalancedRealFakeSampler
        sampler = BalancedRealFakeSampler(dataset, batch_size=batch_size)
        shuffle = False  # Mutually exclusive with sampler in PyTorch DataLoader
        logger.info(f"Configuring DataLoader with BalancedRealFakeSampler (batch_size={batch_size})")

    # On Windows or limited environments, fallback gracefully if workers fail
    persistent_workers = num_workers > 0

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=persistent_workers,
    )
    logger.info(
        f"Created DataLoader -> batch_size: {batch_size}, shuffle: {shuffle}, "
        f"workers: {num_workers}, balanced: {balanced_sampling}, pin_memory: {pin_memory}"
    )
    return loader
