"""
Balanced and Domain-Aware Samplers for AIGID Training.

Provides:
    1. BalancedBatchSampler — Ensures exact 50/50 real/fake class ratio per batch
       to prevent majority class drift during training.
    2. DomainAwareSampler — Ensures proportional representation across generator
       domains (e.g., ProGAN, StyleGAN, Midjourney, etc.) within each epoch.
"""

from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

from src.utils.logger import get_logger

logger = get_logger("samplers")


class BalancedBatchSampler(Sampler):
    """
    Balanced class sampler ensuring equal real/fake representation per epoch.

    Computes inverse-frequency weights for each sample and uses PyTorch's
    WeightedRandomSampler to draw balanced batches.

    Args:
        dataset: PyTorch Dataset with .samples attribute containing
                 (path, label, meta) tuples.
        num_samples: Total samples per epoch. If None, uses dataset length.
        replacement: Whether to sample with replacement (default True).
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        super().__init__(dataset)
        self.dataset = dataset

        # Extract labels from dataset samples
        if hasattr(dataset, "samples"):
            labels = [s[1] for s in dataset.samples]
        elif hasattr(dataset, "targets"):
            labels = dataset.targets
        else:
            raise AttributeError(
                "Dataset must have 'samples' or 'targets' attribute."
            )

        # Compute per-class weights
        class_counts: Dict[int, int] = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Inverse frequency weighting
        weights = [1.0 / class_counts[label] for label in labels]

        self.num_samples = num_samples or len(labels)
        self._sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=self.num_samples,
            replacement=replacement,
        )

        logger.info(
            f"BalancedBatchSampler: {len(labels)} samples, "
            f"class distribution: {dict(sorted(class_counts.items()))}, "
            f"sampling {self.num_samples}/epoch"
        )

    def __iter__(self) -> Iterator[int]:
        return iter(self._sampler)

    def __len__(self) -> int:
        return self.num_samples


class DomainAwareSampler(Sampler):
    """
    Domain-aware sampler ensuring proportional generator representation.

    Balances sampling across different generator domains (e.g., ProGAN, StyleGAN,
    SD v1.5, Midjourney, etc.) to prevent the model from overfitting to the
    most abundant generator in the training set.

    Args:
        dataset: PyTorch Dataset with .samples attribute containing
                 (path, label, meta) tuples where meta has 'domain' key.
        num_samples: Total samples per epoch. If None, uses dataset length.
        replacement: Whether to sample with replacement (default True).
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        super().__init__(dataset)
        self.dataset = dataset

        if not hasattr(dataset, "samples"):
            raise AttributeError("Dataset must have 'samples' attribute.")

        # Extract domain labels
        domain_counts: Dict[str, int] = {}
        for _, _, meta in dataset.samples:
            domain = meta.get("domain", "unknown")
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Inverse domain frequency weighting
        weights = []
        for _, _, meta in dataset.samples:
            domain = meta.get("domain", "unknown")
            weights.append(1.0 / domain_counts[domain])

        self.num_samples = num_samples or len(dataset.samples)
        self._sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=self.num_samples,
            replacement=replacement,
        )

        logger.info(
            f"DomainAwareSampler: {len(dataset.samples)} samples across "
            f"{len(domain_counts)} domains: {dict(sorted(domain_counts.items()))}"
        )

    def __iter__(self) -> Iterator[int]:
        return iter(self._sampler)

    def __len__(self) -> int:
        return self.num_samples


__all__ = ["BalancedBatchSampler", "DomainAwareSampler"]
