"""
Dynamic Device Selector & Seed Management for LOTA / MLEP Fusion Pipeline.
Resolves optimal compute backend (CUDA > MPS > CPU) and ensures strict
reproducibility via deterministic seed initialization across all RNG engines.
"""

import random
from typing import Optional

import numpy as np
import torch


def get_compute_device() -> torch.device:
    """
    Dynamically select the optimal available compute backend.

    Priority order:
        1. NVIDIA CUDA (RTX 4050 / datacenter GPUs)
        2. Apple Metal MPS (M1/M2/M4 Silicon)
        3. CPU fallback

    Returns:
        torch.device: The selected compute device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# Backward-compatible alias used by existing modules
get_device = get_compute_device


def set_global_seed(seed: int = 42) -> None:
    """
    Set seeds across all random number generators for strict reproducibility.

    Configures: Python stdlib random, NumPy, PyTorch CPU, PyTorch CUDA,
    and cuDNN deterministic mode.

    Args:
        seed: Integer seed value for all RNG engines.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_info() -> str:
    """
    Return a human-readable string describing the active compute device.

    Returns:
        str: Device name and capabilities summary.
    """
    device = get_compute_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        return f"CUDA: {name} ({mem:.1f} GB VRAM)"
    elif device.type == "mps":
        return "Apple Metal MPS (Unified Memory)"
    else:
        return "CPU (No GPU acceleration)"


__all__ = ["get_compute_device", "get_device", "set_global_seed", "get_device_info"]
