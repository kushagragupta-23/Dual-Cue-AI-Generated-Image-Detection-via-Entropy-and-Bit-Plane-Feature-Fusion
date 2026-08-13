"""
CUDA GPU Hardware Acceleration & Device Selector for HydraFusion-Net.
Optimized for NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace 6GB VRAM).
"""

import os
import random
import numpy as np
import torch

def get_compute_device() -> torch.device:
    """
    Dynamically select optimal CUDA backend.
    Enforces NVIDIA CUDA GPU execution and configures PyTorch hardware acceleration flags
    (TF32, cuDNN Benchmark, CUDNN Fast Algorithms, PyTorch Caching Allocator).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CRITICAL HARDWARE ERROR: NVIDIA CUDA GPU (e.g., RTX 4050) is required but not found. "
            "HydraFusion-Net is hardware-accelerated for Tensor Cores (TF32 and fp16 AMP). "
            "CPU execution is explicitly disabled to prevent silent performance degradation."
        )

    # Maximize RTX 4050 Tensor Core throughput
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Configure CUDA memory allocator for RTX 4050 6GB VRAM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

    return torch.device("cuda")

def get_device() -> torch.device:
    """Alias for get_compute_device."""
    return get_compute_device()

def set_global_seed(seed: int = 42) -> None:
    """Set seeds across all random number generators for strict reproducibility while maintaining cuDNN speed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

__all__ = ["get_compute_device", "get_device", "set_global_seed"]
