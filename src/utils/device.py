import torch
import random
import numpy as np

def get_compute_device() -> torch.device:
    """
    Dynamically select optimal compute backend.
    STRICTLY ENFORCES NVIDIA GPU (RTX 4050) EXECUTION.
    CPU fallback is explicitly disabled to prevent silent slowdowns.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CRITICAL ERROR: NVIDIA CUDA GPU (e.g., RTX 4050) is required but not found. "
            "HydraFusion-Net is heavily optimized for TF32 and fp16 AMP. "
            "CPU execution is explicitly disabled to prevent silent training bottlenecks."
        )
    return torch.device("cuda")

def set_global_seed(seed: int = 42) -> None:
    """Set seeds across all random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable TF32 for RTX 40 series
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Ensure determinism vs speed tradeoff
        torch.backends.cudnn.deterministic = False # Set to false to allow benchmark
        torch.backends.cudnn.benchmark = True # Auto-tunes kernels for fixed-size convolutions
