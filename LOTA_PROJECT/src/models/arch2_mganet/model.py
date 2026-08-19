"""
Architecture II: MGANetDualCueDetector — End-to-End Cross-Attention Detector.

Ingests preprocessed MLEP entropy (9-ch) and LOTA noise (3/12-ch) tensors,
extracts spatial features via dual ResNet stems, performs pyramid cross-attention
fusion, and outputs binary Real/Fake classification predictions.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.shared.backbones import ChannelAdaptedResNet
from src.models.arch2_mganet.modules import PyramidCrossAttentionModule
from src.utils.logger import get_logger

logger = get_logger("arch2_mganet.model")


class MGANetDualCueDetector(nn.Module):
    """
    Complete End-to-End Multi-Granularity Cross-Attention Detector.

    Architecture flow:
        MLEP tensor → ResNet stem (up to layer3) → 1024-ch spatial map ─┐
                                                                         ├→ CrossAttention → Pool → Classifier
        LOTA tensor → ResNet stem (up to layer3) → 1024-ch spatial map ─┘

    Args:
        mlep_channels: Input channels for MLEP stem (default 9).
        lota_channels: Input channels for LOTA stem (default 3).
        backbone_name: 'resnet18' or 'resnet50'.
        d_model: Cross-attention latent dimension (default 256).
        num_heads: Number of attention heads (default 8).
        num_classes: Output classes (default 1 for binary).
        dropout: Dropout rate (default 0.3).
    """

    def __init__(
        self,
        mlep_channels: int = 9,
        lota_channels: int = 3,
        backbone_name: str = "resnet50",
        d_model: int = 256,
        num_heads: int = 8,
        num_classes: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Dual ResNet stems for spatial feature extraction
        self.mlep_stem = ChannelAdaptedResNet(
            in_channels=mlep_channels, backbone_name=backbone_name, pretrained=True
        )
        self.lota_stem = ChannelAdaptedResNet(
            in_channels=lota_channels, backbone_name=backbone_name, pretrained=True
        )

        # Get channel dims at stage3
        stem_channels = self.mlep_stem.stage_channels["stage3"]

        # Cross-Attention Module
        self.cross_attn = PyramidCrossAttentionModule(
            in_channels_mlep=stem_channels,
            in_channels_lota=stem_channels,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Global Spatial Pooling and Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        logger.info(
            f"MGANetDualCueDetector: MLEP({mlep_channels}ch) × LOTA({lota_channels}ch) "
            f"→ {backbone_name} stems → CrossAttn(d={d_model}) → {num_classes}-class output"
        )

    def forward(
        self, x_mlep: torch.Tensor, x_lota: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_mlep: Preprocessed MLEP entropy tensor (B, mlep_channels, H, W).
            x_lota: Preprocessed LOTA noise tensor (B, lota_channels, H, W).

        Returns:
            torch.Tensor: Logit predictions of shape (B, num_classes).
        """
        # Extract spatial feature maps at stage3: (B, 1024, H/16, W/16)
        mlep_features = self.mlep_stem.forward_features(x_mlep)["stage3"]
        lota_features = self.lota_stem.forward_features(x_lota)["stage3"]

        # Execute Multi-Granularity Cross-Attention
        fused_spatial = self.cross_attn(mlep_features, lota_features)

        # Pool and classify
        pooled = self.global_pool(fused_spatial).flatten(1)  # (B, d_model)
        logits = self.classifier(pooled)  # (B, num_classes)

        return logits


__all__ = ["MGANetDualCueDetector"]
