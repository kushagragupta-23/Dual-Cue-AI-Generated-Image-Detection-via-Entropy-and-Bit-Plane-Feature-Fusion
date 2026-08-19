"""
Dual-Stem ResNet Backbone Adapters for MLEP & LOTA Feature Extraction.

Provides modified ResNet-18/50 stems with channel adapters for:
    - MLEP branch: 9-channel input (3 scales × 3 RGB entropy maps)
    - LOTA branch: 3-channel (single Top-1 patch) or 12-channel (Top-4 patches × 3 RGB)

Supports feature extraction at multiple ResNet stages for cross-attention (MGA-Net).
Uses Kaiming initialization on modified conv1 layers and transfers ImageNet pretrained
weights for all shared layers (bn1, layer1-4).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
)

from src.utils.logger import get_logger

logger = get_logger("backbones")


class ChannelAdaptedResNet(nn.Module):
    """
    ResNet backbone with modified first convolution to accept arbitrary input channels.

    Provides multi-stage feature extraction for downstream cross-attention modules.
    Feature maps are available at four stages:
        - stage1: After layer1 (64/256 channels, H/4 × W/4)
        - stage2: After layer2 (128/512 channels, H/8 × W/8)
        - stage3: After layer3 (256/1024 channels, H/16 × W/16)
        - stage4: After layer4 (512/2048 channels, H/32 × W/32)

    Args:
        in_channels: Number of input channels (e.g., 9 for MLEP, 3/12 for LOTA).
        backbone_name: 'resnet18' or 'resnet50'.
        pretrained: Whether to load ImageNet pretrained weights for shared layers.
        freeze_early: If True, freeze conv1, bn1, layer1, and layer2 parameters.
    """

    # Channel dimensions at each stage for ResNet variants
    STAGE_CHANNELS = {
        "resnet18": {"stage1": 64, "stage2": 128, "stage3": 256, "stage4": 512},
        "resnet50": {"stage1": 256, "stage2": 512, "stage3": 1024, "stage4": 2048},
    }

    def __init__(
        self,
        in_channels: int = 3,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_early: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.backbone_name = backbone_name

        # Load base ResNet with optional pretrained weights
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            base = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Use 'resnet18' or 'resnet50'.")

        # Replace conv1 to accept custom input channels
        original_conv1 = base.conv1
        self.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
        )

        # Transfer pretrained conv1 weights where possible
        if pretrained and in_channels != 3:
            with torch.no_grad():
                # Repeat the 3-channel weights to fill new channels
                weight_3ch = original_conv1.weight.data  # (64, 3, 7, 7)
                repeats = (in_channels + 2) // 3  # Ceiling division
                expanded = weight_3ch.repeat(1, repeats, 1, 1)[:, :in_channels, :, :]
                # Scale to preserve activation magnitude
                self.conv1.weight.copy_(expanded * (3.0 / in_channels))
        elif pretrained and in_channels == 3:
            self.conv1.weight.data.copy_(original_conv1.weight.data)
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Copy remaining layers from pretrained model
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # Optionally freeze early layers to prevent low-level feature memorization
        if freeze_early:
            self._freeze_early_layers()

        self.stage_channels = self.STAGE_CHANNELS[backbone_name]

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ChannelAdaptedResNet({backbone_name}, in_ch={in_channels}) "
            f"| Trainable: {trainable:,} / Total: {total:,} "
            f"| Frozen early: {freeze_early}"
        )

    def _freeze_early_layers(self) -> None:
        """Freeze conv1, bn1, layer1, and layer2 to prevent low-level memorization."""
        for module in [self.conv1, self.bn1, self.layer1, self.layer2]:
            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract feature maps at all four ResNet stages.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            dict with keys 'stage1' through 'stage4', each containing
            the spatial feature map at that depth.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        s1 = self.layer1(x)    # (B, C1, H/4, W/4)
        s2 = self.layer2(s1)   # (B, C2, H/8, W/8)
        s3 = self.layer3(s2)   # (B, C3, H/16, W/16)
        s4 = self.layer4(s3)   # (B, C4, H/32, W/32)

        return {"stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass: returns global average pooled feature vector.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Feature vector of shape (B, stage4_channels).
        """
        features = self.forward_features(x)
        pooled = self.avgpool(features["stage4"])  # (B, C4, 1, 1)
        return pooled.flatten(1)  # (B, C4)


class DualStemBackbone(nn.Module):
    """
    Dual-branch backbone providing separate ResNet stems for MLEP and LOTA branches.

    The two stems share architecture but have independent weights, allowing each
    to specialize for its respective input modality (entropy maps vs. LSB noise).

    Args:
        mlep_channels: Input channels for MLEP stem (default 9: 3 scales × 3 RGB).
        lota_channels: Input channels for LOTA stem (default 3 for Top-1, 12 for Top-4).
        backbone_name: 'resnet18' or 'resnet50'.
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_early: Whether to freeze conv1, bn1, layer1, layer2.
    """

    def __init__(
        self,
        mlep_channels: int = 9,
        lota_channels: int = 3,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_early: bool = False,
    ):
        super().__init__()

        self.mlep_stem = ChannelAdaptedResNet(
            in_channels=mlep_channels,
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_early=freeze_early,
        )
        self.lota_stem = ChannelAdaptedResNet(
            in_channels=lota_channels,
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_early=freeze_early,
        )

        self.out_dim = self.mlep_stem.stage_channels["stage4"]
        logger.info(
            f"DualStemBackbone initialized: MLEP({mlep_channels}ch) + LOTA({lota_channels}ch) "
            f"→ {self.out_dim}-dim per stem"
        )

    def forward(
        self, x_mlep: torch.Tensor, x_lota: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract global feature vectors from both stems.

        Args:
            x_mlep: MLEP entropy tensor of shape (B, mlep_channels, H, W).
            x_lota: LOTA noise tensor of shape (B, lota_channels, H, W).

        Returns:
            Tuple of (mlep_features, lota_features), each (B, out_dim).
        """
        f_mlep = self.mlep_stem(x_mlep)
        f_lota = self.lota_stem(x_lota)
        return f_mlep, f_lota

    def forward_spatial(
        self, x_mlep: torch.Tensor, x_lota: torch.Tensor, stage: str = "stage3"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract spatial feature maps at a specified stage from both stems.

        Used by MGA-Net cross-attention which requires spatial dimensions.

        Args:
            x_mlep: MLEP entropy tensor of shape (B, mlep_channels, H, W).
            x_lota: LOTA noise tensor of shape (B, lota_channels, H, W).
            stage: Which ResNet stage to extract ('stage1' through 'stage4').

        Returns:
            Tuple of spatial feature maps from both stems at the requested stage.
        """
        mlep_features = self.mlep_stem.forward_features(x_mlep)
        lota_features = self.lota_stem.forward_features(x_lota)
        return mlep_features[stage], lota_features[stage]


__all__ = ["ChannelAdaptedResNet", "DualStemBackbone"]
