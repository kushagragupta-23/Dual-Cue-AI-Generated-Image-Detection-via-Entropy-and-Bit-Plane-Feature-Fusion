"""
Architecture III Models:

1. DomainAdversarialMoEDetector — Takes pre-fused feature vector, routes through MoE,
   outputs classification + domain logits + aux loss.

2. MoEStandaloneDualCueDetector — [NEW] End-to-end standalone model with its own
   MLEP/LOTA extractors + backbone stems, enabling independent training.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.shared.backbones import ChannelAdaptedResNet
from src.models.arch3_moe.modules import (
    SparseMoEForensicModule,
    GradientReversalLayer,
    DomainDiscriminator,
)
from src.utils.logger import get_logger

logger = get_logger("arch3_moe.model")


class DomainAdversarialMoEDetector(nn.Module):
    """
    Unified Sparse MoE + Domain-Adversarial AIGID Architecture.

    Minimax objective:
        min_{θ,φ} max_{ψ} [ L_BCE(C_φ(z_MoE), y) - λ·L_Domain(D_ψ(R(z_MoE)), d) ]

    Args:
        in_dim: Concatenated feature dimension from both stems (default 1024).
        d_model: MoE internal dimension (default 512).
        num_experts: Number of MoE experts (default 4).
        top_k: Number of active experts per sample (default 2).
        num_domains: Number of generator domain classes (default 8).
        lambda_coeff: GRL reversal coefficient (default 0.5).
    """

    def __init__(
        self,
        in_dim: int = 1024,
        d_model: int = 512,
        num_experts: int = 4,
        top_k: int = 2,
        num_domains: int = 8,
        lambda_coeff: float = 0.5,
    ):
        super().__init__()

        # Input projection to MoE dimension
        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
        )

        # Sparse MoE with Top-K routing
        self.moe = SparseMoEForensicModule(
            d_model=d_model, num_experts=num_experts, top_k=top_k
        )

        # Binary Real/Fake Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # Domain-Adversarial Head (GRL + Discriminator)
        self.grl = GradientReversalLayer(lambda_coeff=lambda_coeff)
        self.domain_discriminator = DomainDiscriminator(
            in_dim=d_model, num_domains=num_domains
        )

        logger.info(
            f"DomainAdversarialMoEDetector: {in_dim}→{d_model} → "
            f"MoE({num_experts} experts, Top-{top_k}) → "
            f"Classifier(1) + DANN({num_domains} domains, λ={lambda_coeff})"
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with MoE routing and domain-adversarial branches.

        Args:
            x: Concatenated MLEP + LOTA feature vector of shape (B, in_dim).

        Returns:
            Tuple of (class_logits, domain_logits, aux_loss).
        """
        h = self.proj_in(x)
        z_moe, aux_loss = self.moe(h)
        class_logits = self.classifier(z_moe)
        z_reversed = self.grl(z_moe)
        domain_logits = self.domain_discriminator(z_reversed)

        return class_logits, domain_logits, aux_loss


class MoEStandaloneDualCueDetector(nn.Module):
    """
    End-to-End Standalone MoE + DANN Detector with built-in extractors.

    Complete pipeline:
        MLEP entropy → backbone → pool ─┐
                                          ├→ concat → MoE → classifier + DANN
        LOTA noise   → backbone → pool ─┘

    This enables independent training of Architecture III without depending
    on the Fusion model wrapper.

    Args:
        mlep_channels: Input channels for MLEP stem (default 9).
        lota_channels: Input channels for LOTA stem (default 3).
        backbone_name: 'resnet18' or 'resnet50'.
        d_model: MoE internal dimension (default 512).
        num_experts: Number of MoE experts (default 4).
        top_k: Top-K routing (default 2).
        num_domains: Number of generator domain classes (default 8).
        lambda_coeff: GRL reversal coefficient (default 0.5).
    """

    def __init__(
        self,
        mlep_channels: int = 9,
        lota_channels: int = 3,
        backbone_name: str = "resnet50",
        d_model: int = 512,
        num_experts: int = 4,
        top_k: int = 2,
        num_domains: int = 8,
        lambda_coeff: float = 0.5,
    ):
        super().__init__()

        # Dual ResNet stems
        self.mlep_stem = ChannelAdaptedResNet(
            in_channels=mlep_channels, backbone_name=backbone_name, pretrained=True
        )
        self.lota_stem = ChannelAdaptedResNet(
            in_channels=lota_channels, backbone_name=backbone_name, pretrained=True
        )

        feat_dim = self.mlep_stem.stage_channels["stage4"]  # 2048 for resnet50

        # MoE + DANN detector on concatenated features
        self.moe_detector = DomainAdversarialMoEDetector(
            in_dim=feat_dim * 2,  # concat of both stems
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            num_domains=num_domains,
            lambda_coeff=lambda_coeff,
        )

        logger.info(
            f"MoEStandaloneDualCueDetector: MLEP({mlep_channels}ch) + LOTA({lota_channels}ch) "
            f"→ {backbone_name} → concat({feat_dim * 2}) → MoE({num_experts})"
        )

    def forward(
        self, x_mlep: torch.Tensor, x_lota: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass.

        Args:
            x_mlep: MLEP entropy tensor (B, mlep_channels, H, W).
            x_lota: LOTA noise tensor (B, lota_channels, H, W).

        Returns:
            Tuple of (class_logits, domain_logits, aux_loss).
        """
        f_mlep = self.mlep_stem(x_mlep)  # (B, feat_dim)
        f_lota = self.lota_stem(x_lota)  # (B, feat_dim)
        concat = torch.cat([f_mlep, f_lota], dim=1)  # (B, feat_dim * 2)

        return self.moe_detector(concat)


__all__ = ["DomainAdversarialMoEDetector", "MoEStandaloneDualCueDetector"]
