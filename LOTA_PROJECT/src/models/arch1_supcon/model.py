"""
Architecture I: LearnableFreqSupConNet — Stage 1 Contrastive Pre-Training Network.

Integrates:
    1. LearnableFrequencyPreFilter → MLEP extractor → MLEP backbone → projection
    2. LOTA extractor → LOTA backbone → projection
    3. DualCueSupConLoss for cross-modal alignment

After pre-training, discard the projection heads and freeze the backbone stems
for Stage 2 gated classifier fine-tuning.

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.shared.backbones import ChannelAdaptedResNet
from src.shared.losses import DualCueSupConLoss
from src.models.arch1_supcon.modules import LearnableFrequencyPreFilter, ProjectionHead
from src.utils.logger import get_logger

logger = get_logger("arch1_supcon.model")


class LearnableFreqSupConNet(nn.Module):
    """
    Stage 1 Contrastive Pre-Training Network.

    Integrates:
        1. LearnableFrequencyPreFilter → MLEP extractor → MLEP backbone → projection
        2. LOTA extractor → LOTA backbone → projection
        3. DualCueSupConLoss for cross-modal alignment

    After pre-training, discard the projection heads and freeze the backbone stems
    for Stage 2 gated classifier fine-tuning.

    Args:
        mlep_extractor: VectorizedMLEPExtractor instance.
        lota_extractor: TopKLOTAExtractor instance.
        backbone_name: 'resnet18' or 'resnet50'.
        proj_dim: Projection head output dimension (default 128).
        temperature: SupCon temperature τ (default 0.07).
    """

    def __init__(
        self,
        mlep_extractor: nn.Module,
        lota_extractor: nn.Module,
        backbone_name: str = "resnet50",
        proj_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.freq_filter = LearnableFrequencyPreFilter()
        self.mlep_extractor = mlep_extractor
        self.lota_extractor = lota_extractor

        # Backbone stems with adapted input channels
        self.mlep_backbone = ChannelAdaptedResNet(
            in_channels=9, backbone_name=backbone_name, pretrained=True
        )
        self.lota_backbone = ChannelAdaptedResNet(
            in_channels=3, backbone_name=backbone_name, pretrained=True
        )

        # Determine feature dimension from backbone
        feat_dim = self.mlep_backbone.stage_channels["stage4"]

        # Projection heads (discarded after Stage 1)
        self.mlep_projector = ProjectionHead(feat_dim, 256, proj_dim)
        self.lota_projector = ProjectionHead(feat_dim, 256, proj_dim)

        # SupCon loss
        self.supcon_loss = DualCueSupConLoss(temperature=temperature)

        logger.info(
            f"LearnableFreqSupConNet: {backbone_name} stems → "
            f"{feat_dim}→256→{proj_dim} projections, τ={temperature}"
        )

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for contrastive pre-training.

        Args:
            x: Raw RGB image tensor (B, 3, H, W) in [0, 255].
            labels: Binary class labels (B,). Required during training for SupCon loss.

        Returns:
            Tuple of (mlep_embeddings, lota_embeddings, supcon_loss_or_None).
            Embeddings are L2-normalized on the unit hypersphere.
        """
        # MLEP branch: frequency filter → extractor → backbone → projection
        x_filtered = self.freq_filter(x)
        mlep_dict = self.mlep_extractor(x_filtered)
        mlep_features = self.mlep_backbone(mlep_dict["entropy_map"])  # (B, feat_dim)
        z_mlep = self.mlep_projector(mlep_features)  # (B, proj_dim)

        # LOTA branch: extractor → backbone → projection
        lota_dict = self.lota_extractor(x)
        lota_input = lota_dict["noise_tensor"]  # (B, 3, 256, 256)
        lota_features = self.lota_backbone(lota_input)  # (B, feat_dim)
        z_lota = self.lota_projector(lota_features)  # (B, proj_dim)

        # Compute SupCon loss if labels provided
        loss = None
        if labels is not None:
            # Stack: (B, 2, proj_dim)
            dual_embeddings = torch.stack([z_mlep, z_lota], dim=1)
            loss = self.supcon_loss(dual_embeddings, labels)

        return z_mlep, z_lota, loss


__all__ = ["LearnableFreqSupConNet"]
