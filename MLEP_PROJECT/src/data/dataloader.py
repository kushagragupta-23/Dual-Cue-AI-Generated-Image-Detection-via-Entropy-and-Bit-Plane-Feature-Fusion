"""
DataLoader factory for MLEP Standalone Pipeline.
Supports balanced class sampling for AI-generated image detection.
"""

from typing import Optional
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import SharedImageDataset


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
    if balanced_sampling and len(dataset) > 0:
        labels = [s["label"] if isinstance(s, dict) else s[1] for s in dataset.samples]
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        weights = [1.0 / class_counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


__all__ = ["create_dataloader"]
