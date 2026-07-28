"""
Cross-Modal Gating and Attention Fusion Head.
Implements dynamic weighting of MLEP texture entropy versus LOTA LSB quantization noise.
"""

import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger("fusion_head")


class CrossModalGatingFusionHead(nn.Module):
    """
    Cross-Modal Attention Gating Network.
    Normalizes divergent feature scales, projects both cues into a shared latent space,
    and applies dynamic attention gating weights based on input degradation.
    """

    def __init__(self, in_channels_mlep: int = 512, in_channels_lota: int = 512, latent_dim: int = 256):
        """
        Initialize the Fusion Head.

        Args:
            in_channels_mlep: Size of the global feature vector from the MLEP backbone (default 512).
            in_channels_lota: Size of the global feature vector from the LOTA backbone (default 512).
            latent_dim: Dimension of the shared latent space for alignment (default 256).
        """
        super().__init__()
        
        # Layer Normalization to reconcile [0, 1] entropy maps with [0, 255] thresholded bit-planes
        self.norm_mlep = nn.LayerNorm(in_channels_mlep)
        self.norm_lota = nn.LayerNorm(in_channels_lota)
        
        # Projections to align both branches into a shared latent dimension
        self.proj_mlep = nn.Sequential(
            nn.Linear(in_channels_mlep, latent_dim), 
            nn.ReLU(), 
            nn.Dropout(0.3)
        )
        self.proj_lota = nn.Sequential(
            nn.Linear(in_channels_lota, latent_dim), 
            nn.ReLU(), 
            nn.Dropout(0.3)
        )
        
        # Attention gating network predicting scalar weights (alpha_mlep, alpha_lota)
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        
        # Final classification classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Logit output for Real (0) vs AI-Generated (1)
        )
        
        logger.info(f"Initialized CrossModalGatingFusionHead (latent_dim={latent_dim}).")

    def forward(self, feat_mlep: torch.Tensor, feat_lota: torch.Tensor) -> torch.Tensor:
        """
        Fuse MLEP and LOTA features using dynamically predicted attention weights.

        Args:
            feat_mlep: Global average pooled feature vector from MLEP branch of shape (B, in_channels_mlep).
            feat_lota: Global average pooled feature vector from LOTA branch of shape (B, in_channels_lota).

        Returns:
            torch.Tensor: Logit predictions of shape (B, 1). 
                          Positive values indicate AI-Generated, negative indicate Real.
        """
        # 1. Normalize
        feat_mlep = self.norm_mlep(feat_mlep)
        feat_lota = self.norm_lota(feat_lota)
        
        # 2. Project to shared latent space
        h_mlep = self.proj_mlep(feat_mlep)
        h_lota = self.proj_lota(feat_lota)
        
        # 3. Concatenate and compute attention gating weights
        combined = torch.cat([h_mlep, h_lota], dim=-1)
        weights = self.gating_network(combined)  # (B, 2)
        alpha_mlep = weights[:, 0:1]  # (B, 1)
        alpha_lota = weights[:, 1:2]  # (B, 1)
        
        # 4. Dynamic cross-modal fusion
        h_fused = alpha_mlep * h_mlep + alpha_lota * h_lota
        
        # 5. Final Classification
        logits = self.classifier(h_fused)
        
        return logits
