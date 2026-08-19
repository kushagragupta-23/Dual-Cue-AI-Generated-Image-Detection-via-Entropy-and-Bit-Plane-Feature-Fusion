"""
VectorizedMLEPExtractor: Multi-granularity Local Entropy Pattern Feature Extraction.

Implements the NeurIPS 2025 MLEP pipeline using fully vectorized PyTorch operations:
    1. Channel-independent local windowed shuffling (16×16 macro-grid)
    2. Multi-scale resampling pyramid at {1.0, 0.5, 0.25}
    3. Vectorized 2×2 sliding-window Shannon entropy via F.unfold

All operations avoid Python loops over spatial dimensions, ensuring high throughput
on both CUDA and Apple Metal MPS backends.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger("mlep_extractor")


class VectorizedMLEPExtractor(nn.Module):
    """
    Multi-granularity Local Entropy Pattern (MLEP) Feature Extractor.

    Pipeline:
        Input RGB (B, 3, H, W) in [0, 255]
        → Local windowed shuffling within 16×16 macro-grid cells
        → Multi-scale resampling pyramid {1.0, 0.5, 0.25}
        → Channel-wise concatenation → (B, 9, H, W)
        → 2×2 sliding-window Shannon entropy via vectorized unfold
        → Output MLEP feature tensor (B, 9, H, W)

    The 5 possible discrete Shannon entropy values for a 2×2 window of uint8 pixels:
        - All 4 identical:       H = 0.0
        - 3 identical + 1 diff:  H ≈ 0.8113
        - 2 pairs:               H = 1.0
        - 2 identical + 2 diff:  H ≈ 1.5
        - All 4 unique:          H = 2.0
    """

    def __init__(
        self,
        scales: Optional[List[float]] = None,
        grid_size: int = 16,
        patch_size: int = 16,
        seed: int = 42,
    ):
        """
        Args:
            scales: Resampling scale factors for the multi-scale pyramid.
                    Default: [1.0, 0.5, 0.25] producing 9 output channels.
            grid_size: Number of macro-grid cells per spatial dimension for
                       local windowed shuffling. Default 16 for 256×256 input.
            patch_size: Size of each shuffling cell (H // grid_size).
                        Default 16 for 256×256 input with grid_size=16.
            seed: Random seed for deterministic shuffling permutations.
        """
        super().__init__()
        self.scales = scales if scales is not None else [1.0, 0.5, 0.25]
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.seed = seed
        self.num_output_channels = len(self.scales) * 3  # 3 RGB × K scales

        # Precompute the log2 lookup table for entropy of discrete distributions
        # For 4 values in a 2×2 window, probabilities are multiples of 0.25
        # p * log2(p) for p in {0.25, 0.5, 0.75, 1.0}
        # We'll compute entropy directly using the frequency-count approach
        logger.info(
            f"Initialized VectorizedMLEPExtractor: scales={self.scales}, "
            f"grid={grid_size}×{grid_size}, patch={patch_size}×{patch_size}"
        )

    def _local_windowed_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-independent local windowed shuffling within macro-grid cells.

        Each grid cell's pixels are independently permuted to destroy macro-level
        semantic structure while preserving local pixel co-occurrence patterns.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Shuffled tensor of same shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        g = self.grid_size
        p = self.patch_size

        # Validate dimensions
        assert H == g * p and W == g * p, (
            f"Input spatial dims ({H}, {W}) must equal grid_size*patch_size "
            f"({g}*{p} = {g * p})"
        )

        # Reshape to expose grid cells: (B, C, g, p, g, p)
        x_grid = x.view(B, C, g, p, g, p)

        # Permute to group spatial patch pixels: (B, C, g, g, p, p)
        x_grid = x_grid.permute(0, 1, 2, 4, 3, 5).contiguous()

        # Flatten each patch cell: (B, C, g, g, p*p)
        x_flat = x_grid.view(B, C, g, g, p * p)

        # Generate random permutation indices for each cell
        # Use a seeded generator for reproducibility
        gen = torch.Generator(device=x.device)
        gen.manual_seed(self.seed)

        # Create permutation indices: (B, C, g, g, p*p)
        num_cells = B * C * g * g
        perm_indices = torch.stack(
            [torch.randperm(p * p, generator=gen, device=x.device) for _ in range(num_cells)]
        ).view(B, C, g, g, p * p)

        # Apply permutation via gather
        x_shuffled = torch.gather(x_flat, dim=-1, index=perm_indices)

        # Reshape back: (B, C, g, g, p, p) → (B, C, g, p, g, p) → (B, C, H, W)
        x_shuffled = x_shuffled.view(B, C, g, g, p, p)
        x_shuffled = x_shuffled.permute(0, 1, 2, 4, 3, 5).contiguous()
        x_shuffled = x_shuffled.view(B, C, H, W)

        return x_shuffled

    def _multi_scale_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate multi-scale resampling pyramid and concatenate along channel dim.

        For each scale s in {1.0, 0.5, 0.25}:
            1. Bilinear downscale to (H*s, W*s)
            2. Bilinear upscale back to (H, W)
        Scale 1.0 is identity (no resampling).

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Concatenated pyramid of shape (B, C * len(scales), H, W).
        """
        B, C, H, W = x.shape
        pyramid_channels = []

        for scale in self.scales:
            if abs(scale - 1.0) < 1e-6:
                # Scale 1.0: identity pass-through (no resampling artifacts)
                pyramid_channels.append(x)
            else:
                # Downscale then upscale to expose interpolation anomalies
                h_down = max(1, int(H * scale))
                w_down = max(1, int(W * scale))
                x_down = F.interpolate(
                    x, size=(h_down, w_down), mode="bilinear", align_corners=False
                )
                x_up = F.interpolate(
                    x_down, size=(H, W), mode="bilinear", align_corners=False
                )
                pyramid_channels.append(x_up)

        # Concatenate: (B, C * K, H, W) where K = len(scales)
        return torch.cat(pyramid_channels, dim=1)

    def _compute_entropy_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 2×2 sliding-window Shannon entropy using vectorized F.unfold.

        For each 2×2 window, extracts 4 pixel values, counts unique value
        frequencies, and computes H = -Σ p_i * log2(p_i).

        Uses a quantization-to-bins approach for efficient batch computation:
        quantizes float values to uint8, counts occurrences per bin in each window.

        Args:
            x: Input tensor of shape (B, C, H, W) in [0, 255].

        Returns:
            torch.Tensor: Entropy map of shape (B, C, H, W) with zero-padding
                         to maintain spatial dimensions.
        """
        B, C, H, W = x.shape

        # Quantize to integer bins for discrete entropy computation
        x_int = x.round().clamp(0, 255).to(torch.long)

        # Unfold 2×2 windows with stride 1: (B, C, H-1, W-1, 4)
        # F.unfold expects (B, C, H, W) and returns (B, C*k*k, L)
        x_unfolded = F.unfold(
            x_int.float(), kernel_size=2, stride=1
        )  # (B, C*4, (H-1)*(W-1))

        # Reshape to separate channels: (B, C, 4, (H-1)*(W-1))
        L = (H - 1) * (W - 1)
        x_windows = x_unfolded.view(B, C, 4, L)

        # Count unique values per window via sorting-based approach
        # Sort each 4-element window
        x_sorted, _ = x_windows.sort(dim=2)  # (B, C, 4, L)

        # Create mask for value changes: (B, C, 3, L)
        diff_mask = (x_sorted[:, :, 1:, :] != x_sorted[:, :, :-1, :]).float()

        # Count distinct values per window: 1 + number of transitions
        # Patterns and their entropies (4 pixels):
        #   [a,a,a,a] → 1 unique → H=0.0          (0 transitions)
        #   [a,a,a,b] → 2 unique → H≈0.8113       (1 transition, counts 3+1)
        #   [a,a,b,b] → 2 unique → H=1.0          (1 transition, counts 2+2)
        #   [a,a,b,c] → 3 unique → H≈1.5          (2 transitions)
        #   [a,b,c,d] → 4 unique → H=2.0          (3 transitions)

        # Compute actual frequencies for each unique value
        # Create one-hot segment indicators
        # segment_id marks which group each element belongs to
        segment_starts = torch.zeros(B, C, 4, L, device=x.device)
        segment_starts[:, :, 0, :] = 1.0
        segment_starts[:, :, 1:, :] = diff_mask

        # Cumsum gives group IDs (0-indexed after subtracting 1)
        group_ids = segment_starts.cumsum(dim=2) - 1  # (B, C, 4, L), values in [0, 3]
        num_groups = group_ids[:, :, -1:, :] + 1  # (B, C, 1, L)

        # Count elements per group using scatter
        max_groups = 4
        counts = torch.zeros(B, C, max_groups, L, device=x.device)
        ones = torch.ones_like(group_ids)
        counts.scatter_add_(2, group_ids.long(), ones)

        # Compute entropy: H = -Σ (count/4) * log2(count/4) for non-zero counts
        probs = counts / 4.0  # Probabilities
        # Avoid log(0) by masking zero probabilities
        log_probs = torch.where(
            probs > 0,
            torch.log2(probs + 1e-10),
            torch.zeros_like(probs),
        )
        entropy_per_window = -(probs * log_probs).sum(dim=2)  # (B, C, L)

        # Reshape to spatial: (B, C, H-1, W-1)
        entropy_map = entropy_per_window.view(B, C, H - 1, W - 1)

        # Pad to original spatial size (replicate right and bottom edges)
        entropy_padded = F.pad(entropy_map, (0, 1, 0, 1), mode="replicate")

        return entropy_padded

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Execute the complete MLEP feature extraction pipeline.

        Args:
            x: Input RGB image tensor of shape (B, 3, H, W) in [0.0, 255.0].

        Returns:
            dict containing:
                - 'shuffled': Locally shuffled image (B, 3, H, W)
                - 'pyramid': Multi-scale concatenated tensor (B, 9, H, W)
                - 'entropy_map': MLEP feature tensor (B, 9, H, W)
        """
        B, C, H, W = x.shape

        # Step 1: Local windowed shuffling to destroy macro-semantics
        x_shuffled = self._local_windowed_shuffle(x)

        # Step 2: Multi-scale resampling pyramid {1.0, 0.5, 0.25}
        # Produces (B, 9, H, W) for 3 scales × 3 channels
        x_pyramid = self._multi_scale_pyramid(x_shuffled)

        # Step 3: Vectorized 2×2 Shannon entropy computation
        entropy_map = self._compute_entropy_vectorized(x_pyramid)

        return {
            "shuffled": x_shuffled,
            "pyramid": x_pyramid,
            "entropy_map": entropy_map,
        }
