"""
Backbone architectures for the MLEP Detector.
Implements a ResNet-50 feature extractor for the MLEP branch.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from src.utils.logger import get_logger

logger = get_logger("backbones")


class SharedFeatureExtractor(nn.Module):
    """
    High-capacity ResNet-50 feature extractor.
    Strips the final classification FC layer and acts as a generic 
    spatial encoder producing a 2048-D global average pooled feature vector.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        """
        Initialize the ResNet-50 backbone.

        Args:
            in_channels: Number of input channels (e.g., 9 for MLEP pyramid).
            pretrained: If True, uses ImageNet pre-trained weights (where possible).
        """
        super().__init__()
        
        # Load standard ResNet-50
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Adjust the first convolution layer if input channels != 3
        if in_channels != 3:
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            if pretrained and in_channels % 3 == 0:
                # Mathematically copy the 3-channel pretrained weights across the new channels
                # and divide by the replication factor to preserve the original variance.
                repeats = in_channels // 3
                with torch.no_grad():
                    self.backbone.conv1.weight.data = original_conv.weight.data.repeat(1, repeats, 1, 1) / repeats
                logger.info(f"Successfully tiled ImageNet weights across {in_channels} channels.")
            else:
                # Fallback to standard Kaiming initialization if not pretrained or not a multiple of 3
                nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
            logger.info(f"Adapted ResNet-50 conv1 for {in_channels} input channels.")

        # Remove the fully connected layer
        self.backbone.fc = nn.Identity()
        
        logger.info(f"Initialized ResNet-50 feature extractor (pretrained={pretrained}).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global feature vector from input tensor.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Global average pooled feature vector of shape (B, 2048).
        """
        # resnet50 forward pass output after AdaptiveAvgPool2d + Flatten
        # Since self.backbone.fc is Identity, it returns the flattened vector
        return self.backbone(x)
