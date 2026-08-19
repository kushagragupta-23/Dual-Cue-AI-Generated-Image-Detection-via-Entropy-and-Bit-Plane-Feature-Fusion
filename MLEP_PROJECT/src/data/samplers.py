"""
Balanced mini-batch samplers for equal Real vs AI representation.
"""

import math
import random
from typing import Iterator, List, Optional
import torch
from torch.utils.data import Dataset, Sampler
from src.utils.logger import get_logger

logger = get_logger("samplers")


class BalancedRealFakeSampler(Sampler[int]):
    """
    Sampler that guarantees a 1:1 equal ratio of Real (label 0) and AI-Generated (label 1)
    samples within every mini-batch. Essential for preventing majority class drift.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        drop_last: bool = False,
        seed: Optional[int] = 42,
    ):
        """
        Initialize balanced sampler by indexing real and fake sample positions.

        Args:
            dataset: Target dataset (must have `.samples` or return (tensor, label, ...)).
            batch_size: Target mini-batch size (must be an even number).
            drop_last: If True, drops trailing incomplete batches.
            seed: Random seed for shuffling index lists each epoch.
        """
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be an even integer for 50/50 balance, got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.real_indices: List[int] = []
        self.fake_indices: List[int] = []

        # Extract class indices
        if hasattr(dataset, "samples"):
            for idx, s in enumerate(getattr(dataset, "samples")):
                if s["label"] == 0:
                    self.real_indices.append(idx)
                else:
                    self.fake_indices.append(idx)
        else:
            for idx in range(len(dataset)):
                try:
                    _, label, _ = dataset[idx]  # type: ignore
                except ValueError:
                    _, label = dataset[idx]  # type: ignore
                if label == 0:
                    self.real_indices.append(idx)
                else:
                    self.fake_indices.append(idx)

        self.num_real = len(self.real_indices)
        self.num_fake = len(self.fake_indices)

        if self.num_real == 0 or self.num_fake == 0:
            logger.warning("Dataset is missing either Real or AI samples. Balanced sampling may fail.")

        # Determine total batches per epoch determined by the minority class
        self.half_batch = batch_size // 2
        min_class_count = min(self.num_real, self.num_fake)
        
        if self.drop_last:
            self.num_batches = min_class_count // self.half_batch
        else:
            self.num_batches = math.ceil(min_class_count / self.half_batch)

        self.total_size = self.num_batches * self.batch_size
        logger.info(
            f"Initialized BalancedRealFakeSampler: {self.num_real} Real, {self.num_fake} AI -> "
            f"{self.num_batches} batches/epoch ({self.total_size} total samples)."
        )

    def __iter__(self) -> Iterator[int]:
        """Yield balanced indices for mini-batch collation."""
        rng = random.Random(self.seed + self.epoch if self.seed is not None else None)

        real_pool = self.real_indices.copy()
        fake_pool = self.fake_indices.copy()

        rng.shuffle(real_pool)
        rng.shuffle(fake_pool)

        # If one class is smaller, cycle/oversample it to match the required batch count
        while len(real_pool) < self.num_batches * self.half_batch and self.num_real > 0:
            extra = self.real_indices.copy()
            rng.shuffle(extra)
            real_pool.extend(extra)

        while len(fake_pool) < self.num_batches * self.half_batch and self.num_fake > 0:
            extra = self.fake_indices.copy()
            rng.shuffle(extra)
            fake_pool.extend(extra)

        batch_indices: List[int] = []
        for i in range(self.num_batches):
            r_slice = real_pool[i * self.half_batch : (i + 1) * self.half_batch]
            f_slice = fake_pool[i * self.half_batch : (i + 1) * self.half_batch]
            
            combined = r_slice + f_slice
            rng.shuffle(combined)
            batch_indices.extend(combined)

        self.epoch += 1
        return iter(batch_indices[: self.total_size])

    def __len__(self) -> int:
        return self.total_size
