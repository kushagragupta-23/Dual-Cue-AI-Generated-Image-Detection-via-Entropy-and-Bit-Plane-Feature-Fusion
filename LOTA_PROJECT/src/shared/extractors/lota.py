"""
LOw-biT pAtch (LOTA) Preprocessing Architecture and MGPS Extraction Core.
Exclusively implements LOTA bit-plane slicing, threshold normalization, 4-directional MGPS,
and single Maximum Gradient patch selection as per ICCV 2025.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logger import get_logger

logger = get_logger("lota_extractor")


class TopKLOTAExtractor(nn.Module):
    """
    LOw-biT pAtch (LOTA) Extractor and Multi-Grid Patch Scoring (MGPS) Module.
    
    Operations:
        1. Bit-Plane Extraction: Slices uint8 representation into individual bit-planes.
        2. LSB Composition: Composes k=0,1,2 least significant bits via z = 4*x2 + 2*x1 + x0.
        3. Binarized Thresholding: Maps positive LSB noise z > 0 to 255.0, 0 remains 0.0.
        4. Directional Gradient Scoring: Convolves z_norm against fixed 2x2 kernels (gx, gy, gxy, gyx)
           and calculates L1 divergence scores across an 8x8 spatial grid of 32x32 patches.
        5. Maximum Gradient Patch Selection: Extracts the single 32x32 patch with the highest
           gradient divergence score.
        6. Upscaling: Resizes the 32x32 patch back to 256x256 using nearest-neighbor interpolation.
    """
    def __init__(
        self,
        k_patches: int = 1,  # LOTA uses only 1 patch
        patch_size: int = 32,
        grid_size: int = 8,
        bit_planes: Optional[List[int]] = None,
        threshold_val: float = 255.0,
    ):
        """
        Initialize the LOTA Extractor and register fixed non-trainable gradient kernels.

        Args:
            k_patches: Number of top divergence patches to extract (default K=1 for LOTA).
            patch_size: Spatial height and width of each MGPS grid patch (default 32).
            grid_size: Number of patches along each spatial dimension (default 8 for 256x256).
            bit_planes: List of LSB bit-plane indices to compose (default [0, 1, 2]).
            threshold_val: Normalization value for positive LSB activations (default 255.0).
        """
        super().__init__()
        self.k_patches = 1  # Force K=1 as per paper
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.bit_planes = bit_planes if bit_planes is not None else [0, 1, 2]
        self.threshold_val = threshold_val

        if self.grid_size * self.patch_size != 256:
            logger.warning(
                f"grid_size ({grid_size}) * patch_size ({patch_size}) != 256. "
                "Ensure input images match grid dimensions."
            )

        # Register fixed 2x2 directional gradient kernels (gx, gy, gxy, gyx) as non-trainable buffers
        gx = torch.tensor([[-1.0, 1.0], [0.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gy = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gxy = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gyx = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Shape: (4, 1, 2, 2)
        self.register_buffer("kernels", torch.cat([gx, gy, gxy, gyx], dim=0))

    def extract_all_bit_planes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract all 8 bit-planes from input image tensor.

        Args:
            x: Float tensor of shape (B, C, H, W) in range [0.0, 255.0].

        Returns:
            torch.Tensor: Binary bit-plane tensor of shape (B, C, 8, H, W) with values in {0.0, 1.0}.
        """
        x_int = x.to(torch.uint8)
        planes = []
        for k in range(8):
            plane = (x_int & (1 << k)) >> k
            planes.append(plane.to(torch.float32))
        return torch.stack(planes, dim=2)

    def _extract_lsb_threshold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compose LSB bit-planes and apply binarized thresholding normalization.
        
        Formula:
            z^c = 4*x_2^c + 2*x_1^c + x_0^c
            z_norm = 255.0 if z > 0 else 0.0

        Args:
            x: Input tensor of shape (B, C, H, W) in [0.0, 255.0].

        Returns:
            torch.Tensor: Thresholded LSB noise map of shape (B, C, H, W).
        """
        x_int = x.to(torch.uint8)
        
        # We handle standard bit composition for k=0, 1, 2
        z = torch.zeros_like(x_int, dtype=torch.int32)
        for idx, k in enumerate(sorted(self.bit_planes)):
            weight = 1 << idx
            bit = ((x_int & (1 << k)) >> k) * weight
            z = z + bit.to(torch.int32)

        # LOTA-scl: Min-max scaling normalization to [0.0, 255.0]
        # We compute min and max across spatial dimensions for each image and channel.
        # This handles JPEG compression noise much better than raw thresholding!
        
        # Flatten spatial dimensions to find min/max
        B, C, H, W = z.shape
        z_float = z.to(torch.float32)
        z_flat = z_float.view(B, C, -1)
        
        z_min = z_flat.min(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        z_max = z_flat.max(dim=2, keepdim=True)[0].view(B, C, 1, 1)
        
        # Avoid division by zero
        range_val = torch.clamp(z_max - z_min, min=1e-5)
        
        z_norm = 255.0 * ((z_float - z_min) / range_val)
        return z_norm

    def _compute_mgps_scores(self, z_norm: torch.Tensor) -> torch.Tensor:
        """
        Compute Multi-Grid Patch Scoring (MGPS) gradient divergence across 8x8 patches.

        Args:
            z_norm: Thresholded LSB noise map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Spatial patch divergence score matrix of shape (B, grid_size * grid_size).
        """
        B, C, H, W = z_norm.shape
        
        # Pad bottom and right by 1 so 2x2 kernel convolution maintains exact (H, W) dimensions
        z_padded = F.pad(z_norm, (0, 1, 0, 1), mode="replicate")
        
        # Reshape to convolve each color channel independently against 4 directional gradient kernels
        z_flat = z_padded.view(B * C, 1, H + 1, W + 1)
        grads = F.conv2d(z_flat, self.kernels, padding=0)  # Shape: (B*C, 4, H, W)
        
        # Sum absolute gradient responses over directions and channels -> Shape: (B, H, W)
        grad_l1 = grads.abs().sum(dim=1).view(B, C, H, W).sum(dim=1)
        
        # Reshape into (B, grid_size, patch_size, grid_size, patch_size) to aggregate scores per patch
        grad_grid = grad_l1.view(B, self.grid_size, self.patch_size, self.grid_size, self.patch_size)
        
        # Sum over spatial patch dimensions (dim 2 and 4) -> Shape: (B, grid_size, grid_size)
        patch_scores = grad_grid.sum(dim=(2, 4))
        
        # Flatten grid to (B, grid_size * grid_size)
        scores = patch_scores.view(B, self.grid_size * self.grid_size)
        return scores

    def _select_max_gradient_patch(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Select the single patch index with the highest MGPS score (Eq 6 in paper).

        Args:
            scores: Flattened patch scores of shape (B, 64).

        Returns:
            torch.Tensor: Indices of selected Top-1 patch of shape (B, 1).
        """
        # argmax returns shape (B,) so we unsqueeze to (B, 1)
        return torch.argmax(scores, dim=1, keepdim=True)

    def extract_patches_from_indices(
        self, z_norm: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract spatial 32x32 patches from thresholded noise map based on grid indices.

        Args:
            z_norm: Thresholded LSB noise map of shape (B, C, H, W).
            indices: Selected patch grid indices of shape (B, 1).

        Returns:
            torch.Tensor: Extracted patches of shape (B, 1, C, patch_size, patch_size).
        """
        B, C, H, W = z_norm.shape
        K = indices.shape[1]
        
        # Reshape z_norm into 5D grid tensor: (B, C, grid_size, patch_size, grid_size, patch_size)
        # Permute to (B, grid_size, grid_size, C, patch_size, patch_size) then flatten grid
        z_grid = z_norm.view(B, C, self.grid_size, self.patch_size, self.grid_size, self.patch_size)
        z_grid = z_grid.permute(0, 2, 4, 1, 3, 5).contiguous()
        z_grid_flat = z_grid.view(B, self.grid_size * self.grid_size, C, self.patch_size, self.patch_size)

        # Gather patches along grid index dimension
        idx_expanded = indices.view(B, K, 1, 1, 1).expand(B, K, C, self.patch_size, self.patch_size)
        patches = torch.gather(z_grid_flat, dim=1, index=idx_expanded)
        return patches

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Execute full LOTA Preprocessing and MGPS pipeline.

        Args:
            x: Input image tensor of shape (B, C, H, W) in [0.0, 255.0].

        Returns:
            dict containing:
                - 'z_norm': Thresholded LSB noise map (B, 3, 256, 256)
                - 'mgps_scores': Spatial divergence score matrix (B, 64)
                - 'top1_index': Grid index of selected patch (B, 1)
                - 'top1_patch': Extracted 5D patch tensor (B, 1, 3, 32, 32)
                - 'noise_tensor': Final resized patch (B, 3, 256, 256)
        """
        # 1 & 2 & 3: LSB composition and thresholding normalization
        z_norm = self._extract_lsb_threshold(x)
        
        # 4 & 5: Calculate divergence scores and select top 1 patch
        scores = self._compute_mgps_scores(z_norm)
        top1_index = self._select_max_gradient_patch(scores)
        
        # 6: Extract the 32x32 patch
        top1_patch = self.extract_patches_from_indices(z_norm, top1_index)
        
        # Squeeze the K dimension out -> (B, 3, 32, 32)
        extracted_patch = top1_patch.squeeze(1)
        
        # 7: Upscale the 32x32 patch to 256x256 using nearest-neighbor interpolation.
        # This converts the binary noise into a continuous feature representation
        # suitable for ResNet-50's spatial convolutions (as specified in the paper).
        resized_patch = F.interpolate(extracted_patch, size=(256, 256), mode='nearest')

        return {
            "z_norm": z_norm,
            "mgps_scores": scores,
            "mgps_scores_grid": scores.view(-1, self.grid_size, self.grid_size),
            "top1_index": top1_index,
            "topk_indices": top1_index,
            "top1_patch": top1_patch,
            "noise_tensor": resized_patch,
        }
