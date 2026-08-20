"""
DataLoader factory for MLEP Standalone Pipeline.
Supports balanced class sampling for AI-generated image detection.
"""

from typing import Optional
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SharedImageDataset
from src.data.samplers import BalancedRealFakeSampler


def create_dataloader(
    dataset: SharedImageDataset,
    batch_size: int = 8,
    num_workers: int = 0,
    balanced_sampling: bool = True,
    drop_last: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with optional 50/50 class-balanced sampling.
    """
    sampler = None
    if balanced_sampling and len(dataset) > 0 and (batch_size % 2 == 0):
        sampler = BalancedRealFakeSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=42,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


__all__ = ["create_dataloader"]
