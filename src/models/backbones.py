"""
Backbone architectures for the Dual-Cue Fusion Model.
Implements shared ResNet-18 feature extractors for both MLEP and LOTA branches.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from src.utils.logger import get_logger

logger = get_logger("backbones")


class SharedFeatureExtractor(nn.Module):
    """
    Lightweight ResNet-18 feature extractor.
    Strips the final classification FC layer and acts as a generic 
    spatial encoder producing a 512-D global average pooled feature vector.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        """
        Initialize the ResNet-18 backbone.

        Args:
            in_channels: Number of input channels (e.g., 3 for LOTA patches, 9 for MLEP pyramid).
            pretrained: If True, uses ImageNet pre-trained weights (where possible).
        """
        super().__init__()
        
        # Load standard ResNet-18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        
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
            # Standard Kaiming initialization for the new channels
            nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
            logger.info(f"Adapted ResNet-18 conv1 for {in_channels} input channels.")

        # Remove the fully connected layer
        self.backbone.fc = nn.Identity()
        
        logger.info(f"Initialized ResNet-18 feature extractor (pretrained={pretrained}).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract global feature vector from input tensor.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Global average pooled feature vector of shape (B, 512).
        """
        # resnet18 forward pass output after AdaptiveAvgPool2d + Flatten
        # Since self.backbone.fc is Identity, it returns the flattened vector
        return self.backbone(x)
