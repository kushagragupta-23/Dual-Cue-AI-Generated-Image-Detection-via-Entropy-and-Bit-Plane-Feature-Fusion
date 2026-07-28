"""
Multi-granularity Local Entropy Pattern (MLEP) Preprocessing and Feature Extraction Core.
Exclusively implements MLEP channel-independent patch shuffling, multi-scale resampling pyramid,
and 2×2 sliding window Shannon entropy without any LOTA or fusion modules.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logger import get_logger

logger = get_logger("mlep_extractor")


class MLEPExtractor(nn.Module):
    """
    Multi-granularity Local Entropy Pattern (MLEP) Extractor (NeurIPS 2025).

    Operations:
        1. Channel-Independent Patch Shuffling: Partitions each R, G, B channel into
           L×L micro-patches and applies a seeded pseudo-random spatial permutation π.
        2. Multi-Scale Resampling Pyramid: Bilinear downsampling at scales {1.0, 0.5, 0.25}
           followed by bilinear upsampling back to original resolution, concatenated along channels.
        3. 2×2 Sliding Window Shannon Entropy (LEP): Computes discrete Shannon entropy over
           every 4-pixel window, producing values in V = {0.0, 0.8113, 1.0, 1.5, 2.0}.
        4. MLEP Feature Map: Sparse, discrete tensor X̄ ∈ V^{(H-1)×(W-1)×(C·K)} ready for
           downstream CNN backbone ingestion.
    """

    def __init__(
        self,
        patch_size: int = 2,
        scales: Optional[List[float]] = None,
        window_size: int = 2,
        seed: int = 42,
    ):
        """
        Initialize the MLEP Extractor.

        Args:
            patch_size: Size of micro-patches for channel-independent shuffling (default L=2).
            scales: List of scaling factors for multi-scale pyramid (default [1.0, 0.5, 0.25]).
            window_size: Sliding window size for Shannon entropy computation (default 2).
            seed: Random seed for deterministic patch shuffling permutation.
        """
        super().__init__()
        self.patch_size = patch_size
        self.scales = scales if scales is not None else [1.0, 0.5, 0.25]
        self.window_size = window_size
        self.seed = seed

        if any(s <= 0.0 or s > 1.0 for s in self.scales):
            raise ValueError(f"All scales must be in (0.0, 1.0]. Got: {self.scales}")

        logger.info(
            f"Initialized MLEPExtractor: patch_size={patch_size}, "
            f"scales={self.scales}, window_size={window_size}, seed={seed}"
        )

    def shuffle_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-independent spatial patch shuffling to destroy macro-semantics.

        Each R, G, B channel is independently partitioned into L×L micro-patches
        and shuffled via a seeded pseudo-random permutation π.

        Args:
            x: Input tensor of shape (B, C, H, W) in [0.0, 255.0].

        Returns:
            torch.Tensor: Shuffled tensor of same shape (B, C, H, W) with
                          pixel values preserved but spatial locations permuted.
        """
        B, C, H, W = x.shape
        L = self.patch_size

        if H % L != 0 or W % L != 0:
            raise ValueError(
                f"Image dimensions ({H}, {W}) must be divisible by patch_size ({L})."
            )

        grid_h = H // L
        grid_w = W // L
        num_patches = grid_h * grid_w

        # Reshape into patches: (B, C, grid_h, L, grid_w, L) -> (B, C, num_patches, L, L)
        x_patches = x.view(B, C, grid_h, L, grid_w, L)
        x_patches = x_patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        x_patches = x_patches.view(B, C, num_patches, L, L)

        # Generate seeded permutation (same permutation applied to all batch items)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.seed)
        perm = torch.randperm(num_patches, generator=generator, device=x.device)

        # Apply permutation independently per channel
        x_shuffled = x_patches[:, :, perm, :, :]

        # Reshape back to spatial layout: (B, C, grid_h, grid_w, L, L) -> (B, C, H, W)
        x_shuffled = x_shuffled.view(B, C, grid_h, grid_w, L, L)
        x_shuffled = x_shuffled.permute(0, 1, 2, 4, 3, 5).contiguous()
        x_shuffled = x_shuffled.view(B, C, H, W)

        return x_shuffled

    def build_multiscale_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Construct multi-scale resampling pyramid via bilinear down-then-up interpolation.

        For each scaling factor s_k ∈ {1.0, 0.5, 0.25}:
            X_down = Bilinear_Down(X, s_k)
            X_up   = Bilinear_Up(X_down, H, W)

        All scales are concatenated along the channel dimension.

        Args:
            x: Shuffled tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Multi-scale concatenated tensor of shape (B, C * len(scales), H, W).
        """
        B, C, H, W = x.shape
        pyramid_channels = []

        for scale in self.scales:
            if scale == 1.0:
                # Identity scale: no interpolation needed
                pyramid_channels.append(x)
            else:
                h_down = max(1, int(H * scale))
                w_down = max(1, int(W * scale))

                # Bilinear downsampling
                x_down = F.interpolate(
                    x, size=(h_down, w_down), mode="bilinear", align_corners=False
                )
                # Bilinear upsampling back to original resolution
                x_up = F.interpolate(
                    x_down, size=(H, W), mode="bilinear", align_corners=False
                )
                pyramid_channels.append(x_up)

        # Concatenate along channel dimension: (B, C * K, H, W)
        return torch.cat(pyramid_channels, dim=1)

    def compute_shannon_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute discrete Shannon entropy over 2×2 sliding windows across all channels.

        For each 4-pixel window {x_1, x_2, x_3, x_4}, computes:
            LEP = -Σ p(x_i) · log2(p(x_i))

        where p(x_i) is the empirical frequency of pixel value x_i within the window.
        The output entropy values are bounded to V = {0.0, 0.8113, 1.0, 1.5, 2.0}.

        Implementation uses torch.nn.functional.unfold for fully vectorized computation
        without any Python loops over spatial dimensions.

        Args:
            x: Multi-scale pyramid tensor of shape (B, C_total, H, W).

        Returns:
            torch.Tensor: Entropy feature map of shape (B, C_total, H-1, W-1).
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # Unfold into sliding 2×2 windows: (B * C, 1, H, W) -> (B * C, ws*ws, num_windows)
        x_flat = x.view(B * C, 1, H, W)
        patches = F.unfold(x_flat, kernel_size=ws, stride=1)  # (B*C, 4, L)

        out_h = H - ws + 1
        out_w = W - ws + 1
        num_windows = out_h * out_w
        num_pixels = ws * ws  # 4

        # Compute discrete probability distribution within each 4-pixel window
        # Count occurrences of each unique value by pairwise equality comparison
        # patches shape: (B*C, 4, num_windows)
        # Compare each pixel to every other pixel in the window
        p1 = patches.unsqueeze(2)  # (B*C, 4, 1, num_windows)
        p2 = patches.unsqueeze(1)  # (B*C, 1, 4, num_windows)
        eq_matrix = (p1 == p2).float()  # (B*C, 4, 4, num_windows)

        # Count matches for each pixel position -> probability
        counts = eq_matrix.sum(dim=2)  # (B*C, 4, num_windows)
        probs = counts / float(num_pixels)  # (B*C, 4, num_windows)

        # Shannon entropy: -Σ p · log2(p), with 0·log2(0) = 0
        log_probs = torch.log2(probs.clamp(min=1e-10))
        pixel_entropy = -probs * log_probs  # (B*C, 4, num_windows)

        # Sum entropy contributions, but avoid double-counting identical values
        # For identical pixels, each contributes the same -p·log2(p), which is correct
        # However, we need unique value contributions only.
        # Trick: only count entropy contribution from the first occurrence of each value
        # A pixel at position i is "first" if no earlier position j < i has the same value
        first_mask = torch.ones_like(patches, dtype=torch.bool)  # (B*C, 4, num_windows)
        for i in range(1, num_pixels):
            for j in range(i):
                first_mask[:, i, :] &= (patches[:, i, :] != patches[:, j, :])

        # Only keep entropy from first occurrences
        unique_entropy = (pixel_entropy * first_mask.float()).sum(dim=1)  # (B*C, num_windows)

        # Reshape to spatial dimensions
        entropy_map = unique_entropy.view(B, C, out_h, out_w)

        return entropy_map

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Execute full MLEP Preprocessing and Feature Extraction pipeline.

        Pipeline:
            Input (B, 3, 256, 256) -> Shuffle -> Pyramid -> Entropy -> MLEP Features (B, 9, 255, 255)

        Args:
            x: Input image tensor of shape (B, C, H, W) in [0.0, 255.0].

        Returns:
            dict containing:
                - 'shuffled': Patch-shuffled image tensor (B, 3, 256, 256)
                - 'pyramid': Multi-scale resampled tensor (B, 9, 256, 256)
                - 'entropy_maps': Per-scale entropy maps list [3 tensors of (B, 3, 255, 255)]
                - 'mlep_features': Final MLEP feature map (B, 9, 255, 255)
        """
        B, C, H, W = x.shape

        # 1: Channel-independent patch shuffling (π)
        x_shuffled = self.shuffle_patches(x)

        # 2: Multi-scale resampling pyramid {1.0, 0.5, 0.25}
        x_pyramid = self.build_multiscale_pyramid(x_shuffled)

        # 3: 2×2 sliding window Shannon entropy (LEP)
        mlep_features = self.compute_shannon_entropy(x_pyramid)

        # 4: Split entropy maps per scale for diagnostic inspection
        num_scales = len(self.scales)
        entropy_maps = []
        for s in range(num_scales):
            c_start = s * C
            c_end = (s + 1) * C
            entropy_maps.append(mlep_features[:, c_start:c_end, :, :])

        return {
            "shuffled": x_shuffled,
            "pyramid": x_pyramid,
            "entropy_maps": entropy_maps,
            "mlep_features": mlep_features,
        }
