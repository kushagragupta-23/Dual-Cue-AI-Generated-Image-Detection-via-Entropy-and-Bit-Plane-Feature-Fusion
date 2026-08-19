"""
Architecture I Modules: Learnable Frequency Pre-Filter & Projection Head.

LearnableFrequencyPreFilter:
    Trainable Butterworth rFFT2 mask that strips JPEG block quantization
    artifacts while preserving generative decoder anomalies.

ProjectionHead:
    2-layer MLP mapping backbone features to a lower-dimensional embedding
    space on which SupCon loss is computed.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger("arch1_supcon.modules")


class LearnableFrequencyPreFilter(nn.Module):
    """
    Learnable Frequency-Domain Pre-Filter for MLEP Robustness against JPEG Compression.

    Applies a trainable 2D Butterworth-style frequency attenuation mask via Real FFT
    (rFFT2) to selectively suppress high-frequency JPEG 8×8 block quantization noise
    before computing local Shannon entropy.

    Mathematical formulation:
        H_θ(u, v) = 1 / (1 + (r(u,v) / ω_c)^(2σ))
    where r is the radial distance from DC, ω_c is the learnable cutoff frequency,
    and σ is the learnable roll-off slope.

    Args:
        height: Expected input image height (default 256).
        width: Expected input image width (default 256).
        init_cutoff: Initial normalized cutoff frequency (default 0.65).
        init_slope: Initial Butterworth filter order/slope (default 4.0).
    """

    def __init__(
        self,
        height: int = 256,
        width: int = 256,
        init_cutoff: float = 0.65,
        init_slope: float = 4.0,
    ):
        super().__init__()
        self.height = height
        self.width = width

        # In rfft2, frequency dimensions are (H, W // 2 + 1)
        freq_h, freq_w = height, width // 2 + 1

        # Use fftfreq-based normalized coordinates so DC (index 0) has radius 0
        y_freq = torch.fft.fftfreq(height)     # (H,) in [-0.5, 0.5)
        x_freq = torch.fft.rfftfreq(width)     # (W//2+1,) in [0, 0.5]
        yy, xx = torch.meshgrid(y_freq, x_freq, indexing="ij")
        radius = torch.sqrt(xx ** 2 + yy ** 2)  # Radial distance from DC

        # Register radius as a non-trainable buffer
        self.register_buffer("radius", radius)

        # Trainable parameters: cutoff frequency and filter roll-off slope
        self.cutoff = nn.Parameter(torch.tensor(init_cutoff, dtype=torch.float32))
        self.slope = nn.Parameter(torch.tensor(init_slope, dtype=torch.float32))

        logger.info(
            f"LearnableFrequencyPreFilter: {height}×{width}, "
            f"init_cutoff={init_cutoff}, init_slope={init_slope}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable frequency-domain filtering.

        Args:
            x: Input tensor of shape (B, C, H, W) in [0, 255] range.

        Returns:
            torch.Tensor: Filtered tensor of same shape, clamped to [0, 255].
        """
        B, C, H, W = x.shape

        # Handle MPS precision limitations: cast to float32 for FFT ops
        original_dtype = x.dtype
        if x.device.type == "mps" and x.dtype != torch.float32:
            x = x.float()

        # 1. Transform to frequency domain via 2D Real FFT
        x_fft = torch.fft.rfft2(x, norm="ortho")

        # 2. Construct dynamic Butterworth frequency attenuation mask
        cutoff_clamped = torch.clamp(self.cutoff, min=0.1, max=1.5)
        slope_clamped = torch.clamp(self.slope, min=1.0, max=10.0)

        # H_mask shape: (1, 1, H, W // 2 + 1)
        mask = 1.0 / (1.0 + (self.radius / cutoff_clamped) ** (2.0 * slope_clamped))
        mask = mask.unsqueeze(0).unsqueeze(0)

        # 3. Apply frequency gating
        x_fft_filtered = x_fft * mask

        # 4. Inverse Real FFT back to spatial domain
        x_filtered = torch.fft.irfft2(x_fft_filtered, s=(H, W), norm="ortho")

        # Restore original dtype if needed
        if x_filtered.dtype != original_dtype:
            x_filtered = x_filtered.to(original_dtype)

        return torch.clamp(x_filtered, min=0.0, max=255.0)


class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head for contrastive pre-training.

    Maps backbone features to a lower-dimensional embedding space
    on which SupCon loss is computed.

    Args:
        in_dim: Input feature dimension from backbone (e.g., 2048 for ResNet-50).
        hidden_dim: Hidden layer dimension (default 256).
        out_dim: Output embedding dimension (default 128).
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["LearnableFrequencyPreFilter", "ProjectionHead"]
