"""
Balanced Class Sampler for Forensics Dataset.

Prevents majority class imbalance drift during training by balancing
Real (0) and Fake (1) sample probabilities.
"""

import torch
from torch.utils.data import WeightedRandomSampler
from typing import List, Tuple


def get_balanced_sampler(dataset_samples: List[Tuple[str, int]]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that samples Real and Fake images with equal total weight.

    Args:
        dataset_samples: List of (image_path, label) tuples.

    Returns:
        WeightedRandomSampler instance.
    """
    labels = [sample[1] for sample in dataset_samples]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()

    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler
