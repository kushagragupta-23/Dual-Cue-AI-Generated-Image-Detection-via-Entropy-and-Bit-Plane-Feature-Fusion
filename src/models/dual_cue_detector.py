"""
Dual-Cue AI-Generated Image Detector.
Unifies the MLEP and LOTA feature extractors, backbones, and fusion head into
a single end-to-end differentiable module.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from src.models.mlep import MLEPExtractor
from src.models.lota import TopKLOTAExtractor
from src.models.backbones import SharedFeatureExtractor
from src.models.fusion_head import CrossModalGatingFusionHead
from src.utils.logger import get_logger

logger = get_logger("dual_cue_detector")


class DualCueDetector(nn.Module):
    """
    End-to-end unified Dual-Cue Architecture.
    Processes a raw input image through both the MLEP entropy pathway and the 
    LOTA quantization noise pathway, extracts features via ResNet-18 backbones,
    and fuses them using a cross-modal attention gating network.
    """

    def __init__(
        self,
        k_patches: int = 4,
        mlep_scales: tuple = (1.0, 0.5, 0.25),
        latent_dim: int = 256,
        pretrained_backbones: bool = True
    ):
        """
        Initialize the Dual-Cue Detector.

        Args:
            k_patches: Number of top divergence patches to extract from LOTA branch.
            mlep_scales: Image pyramid scaling factors for MLEP branch.
            latent_dim: Dimension of the shared latent space for alignment.
            pretrained_backbones: If True, uses ImageNet pre-trained weights for backbones.
        """
        super().__init__()
        
        # 1. Feature Extractors
        self.mlep_extractor = MLEPExtractor(scales=mlep_scales, window_size=2)
        self.lota_extractor = TopKLOTAExtractor(k_patches=k_patches, patch_size=32, grid_size=8)
        
        # Determine number of input channels to backbones
        # MLEP: 3 scales * 3 RGB channels = 9 channels
        mlep_channels = len(mlep_scales) * 3
        # LOTA: k_patches * 3 RGB channels = 12 channels (for K=4)
        lota_channels = k_patches * 3
        
        # 2. Backbones (Spatial encoders)
        self.mlep_backbone = SharedFeatureExtractor(in_channels=mlep_channels, pretrained=pretrained_backbones)
        self.lota_backbone = SharedFeatureExtractor(in_channels=lota_channels, pretrained=pretrained_backbones)
        
        # Both backbones output 512-D global average pooled feature vectors
        
        # 3. Fusion Head
        self.fusion_head = CrossModalGatingFusionHead(
            in_channels_mlep=512,
            in_channels_lota=512,
            latent_dim=latent_dim
        )
        
        logger.info(f"Initialized Dual-Cue Detector: MLEP branch (c={mlep_channels}) | LOTA branch (c={lota_channels})")

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Forward pass for the unified model.

        Args:
            x: Raw input image tensor of shape (B, 3, 256, 256) in range [0.0, 255.0].
            return_features: If True, returns a dictionary of intermediate features for explainability.

        Returns:
            Logit predictions of shape (B, 1) if return_features=False.
            Dict of intermediate activations and logits if return_features=True.
        """
        # --- PATHWAY A: MLEP (Multi-Level Entropy Patterns) ---
        # Normalize input from [0, 255] to [0, 1] for MLEP, or let extractor handle it.
        # MLEP requires standard float normalization [0, 1] usually, but let's assume it accepts raw or handles it.
        x_mlep = x / 255.0  # Normalize to [0, 1] as standard for image models
        mlep_output = self.mlep_extractor(x_mlep)
        entropy_map = mlep_output['mlep_features']  # (B, 9, H', W')
        feat_mlep = self.mlep_backbone(entropy_map)  # (B, 512)
        
        # --- PATHWAY B: LOTA (LOw-biT pAtch Noise) ---
        # LOTA explicitly requires raw [0.0, 255.0] range to extract bit-planes.
        lota_output = self.lota_extractor(x)
        noise_tensor = lota_output['noise_tensor']  # (B, 12, 32, 32)
        feat_lota = self.lota_backbone(noise_tensor)  # (B, 512)
        
        # --- FUSION ---
        logits = self.fusion_head(feat_mlep, feat_lota)
        
        if return_features:
            return {
                "logits": logits,
                "entropy_map": entropy_map,
                "noise_tensor": noise_tensor,
                "lota_topk_indices": lota_output['topk_indices'],
                "feat_mlep": feat_mlep,
                "feat_lota": feat_lota
            }
            
        return logits
