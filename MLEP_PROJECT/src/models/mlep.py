"""
Multi-granularity Local Entropy Pattern (MLEP) Preprocessing & Extractor Core.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger("mlep")


class MLEPExtractor(nn.Module):
    """
    MLEP Feature Extractor implementing multi-scale resampling,
    patch-level spatial shuffling, and sliding-window Shannon entropy mapping.
    """
    def __init__(
        self,
        patch_size: int = 2,
        scales: Optional[List[float]] = None,
        window_size: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.scales = scales if scales is not None else [1.0, 0.5, 0.25]
        self.window_size = window_size
        self.seed = seed

    def shuffle_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Channel-wise patch shuffling preserving pixel multiset distributions."""
        B, C, H, W = x.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            return x

        num_h, num_w = H // p, W // p
        num_patches = num_h * num_w

        # Reshape to patches: (B, C, num_h, p, num_w, p) -> (B, C, num_patches, p*p)
        x_patches = x.view(B, C, num_h, p, num_w, p).permute(0, 1, 2, 4, 3, 5).contiguous()
        x_flat = x_patches.view(B, C, num_patches, p * p)

        rng = torch.Generator(device=x.device)
        rng.manual_seed(self.seed)

        # Shuffle patch contents per channel
        perm = torch.randperm(num_patches, generator=rng, device=x.device)
        x_shuffled = x_flat[:, :, perm, :]

        # Restore original spatial dimensions
        x_restored = x_shuffled.view(B, C, num_h, num_w, p, p)
        x_out = x_restored.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
        return x_out

    def build_multiscale_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        """Build multi-scale resampling pyramid across defined scale factors."""
        B, C, H, W = x.shape
        pyramid_channels = []
        for s in self.scales:
            if abs(s - 1.0) < 1e-6:
                pyramid_channels.append(x)
            else:
                h_s, w_s = max(1, int(H * s)), max(1, int(W * s))
                down = F.interpolate(x, size=(h_s, w_s), mode="bilinear", align_corners=False)
                up = F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)
                pyramid_channels.append(up)
        return torch.cat(pyramid_channels, dim=1)

    def compute_shannon_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sliding-window Shannon entropy over local windows."""
        B, C, H, W = x.shape
        w = self.window_size
        k = w * w

        # Quantize to discrete integers for frequency estimation
        x_int = x.round().clamp(0, 255).to(torch.long)
        unfolded = F.unfold(x_int.float(), kernel_size=w, stride=1)
        L = (H - w + 1) * (W - w + 1)
        windows = unfolded.view(B, C, k, L)

        sorted_vals, _ = windows.sort(dim=2)
        diff_mask = (sorted_vals[:, :, 1:, :] != sorted_vals[:, :, :-1, :]).float()
        segment_starts = torch.zeros(B, C, k, L, device=x.device)
        segment_starts[:, :, 0, :] = 1.0
        segment_starts[:, :, 1:, :] = diff_mask
        group_ids = segment_starts.cumsum(dim=2) - 1

        counts = torch.zeros(B, C, k, L, device=x.device)
        ones = torch.ones_like(group_ids)
        counts.scatter_add_(2, group_ids.long(), ones)

        probs = counts / float(k)
        log_probs = torch.where(probs > 0, torch.log2(probs + 1e-10), torch.zeros_like(probs))
        entropy = -(probs * log_probs).sum(dim=2)

        H_out = H - w + 1
        W_out = W - w + 1
        return entropy.view(B, C, H_out, W_out)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Complete MLEP pipeline forward pass."""
        shuffled = self.shuffle_patches(x)
        pyramid = self.build_multiscale_pyramid(shuffled)

        entropy_maps = []
        B, total_c, H, W = pyramid.shape
        for i in range(len(self.scales)):
            scale_x = pyramid[:, i * 3 : (i + 1) * 3, :, :]
            emap = self.compute_shannon_entropy(scale_x)
            entropy_maps.append(emap)

        mlep_features = torch.cat(entropy_maps, dim=1)
        return {
            "shuffled": shuffled,
            "pyramid": pyramid,
            "mlep_features": mlep_features,
            "entropy_maps": entropy_maps,
        }


VectorizedMLEPExtractor = MLEPExtractor
VectorizedMLEP = MLEPExtractor

__all__ = ["MLEPExtractor", "VectorizedMLEPExtractor", "VectorizedMLEP"]
