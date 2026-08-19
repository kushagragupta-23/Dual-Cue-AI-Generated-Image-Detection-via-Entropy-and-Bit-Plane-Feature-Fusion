"""
Fusion Modules: Cross-Modal Gating Fusion Head.

Dynamically predicts attention gating weights for MLEP and LOTA branches:
    α = Softmax(W · [f_MLEP; f_LOTA])
    z_fused = α_MLEP · f_MLEP + α_LOTA · f_LOTA
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalGatingFusionHead(nn.Module):
    """
    Dynamically predicts attention gating weights for MLEP and LOTA branches.

    The gating weights are input-dependent, allowing the model to dynamically
    emphasize whichever modality provides stronger forensic signal for each image.

    Args:
        feat_dim: Feature dimension of each stem output (default 2048).
    """

    def __init__(self, feat_dim: int = 2048):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, 2),  # 2 weights: α_MLEP, α_LOTA
        )

    def forward(
        self, f_mlep: torch.Tensor, f_lota: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_mlep: MLEP features of shape (B, feat_dim).
            f_lota: LOTA features of shape (B, feat_dim).

        Returns:
            Tuple of (z_fused, gate_weights).
        """
        concat = torch.cat([f_mlep, f_lota], dim=1)
        gate_logits = self.gate_net(concat)
        gate_weights = F.softmax(gate_logits, dim=-1)

        alpha_mlep = gate_weights[:, 0:1]
        alpha_lota = gate_weights[:, 1:2]

        z_fused = alpha_mlep * f_mlep + alpha_lota * f_lota
        return z_fused, gate_weights


__all__ = ["CrossModalGatingFusionHead"]
