"""
MLEP (Multi-granularity Local Entropy Patterns) Detector.
A unified end-to-end differentiable module focusing strictly on entropy features.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from src.models.mlep import MLEPExtractor
from src.models.backbones import SharedFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger("mlep_detector")


class MLEPDetector(nn.Module):
    """
    End-to-end MLEP Architecture.
    Processes a raw input image through the MLEP entropy pathway, 
    extracts features via a ResNet-18 backbone, and classifies it.
    """

    def __init__(
        self,
        mlep_scales: tuple = (1.0, 0.5, 0.25),
        pretrained_backbones: bool = True
    ):
        """
        Initialize the MLEP Detector.

        Args:
            mlep_scales: Image pyramid scaling factors for MLEP branch.
            pretrained_backbones: If True, uses ImageNet pre-trained weights for backbones.
        """
        super().__init__()
        
        # 1. Feature Extractor
        self.mlep_extractor = MLEPExtractor(scales=mlep_scales, window_size=2)
        
        # Determine number of input channels to backbone
        # MLEP: 3 scales * 3 RGB channels = 9 channels
        mlep_channels = len(mlep_scales) * 3
        
        # Add Batch Normalization to properly scale entropy features for the backbone
        self.bn = nn.BatchNorm2d(mlep_channels)
        
        # 2. Backbone (Spatial encoder)
        self.mlep_backbone = SharedFeatureExtractor(in_channels=mlep_channels, pretrained=pretrained_backbones)
        
        # 3. Classifier Head (ResNet-50 output is 2048)
        # Upgraded to an MLP with Dropout to prevent overfitting and improve generalization
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )
        
        logger.info(f"Initialized MLEP Detector (channels={mlep_channels})")

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Forward pass for the MLEP model.

        Args:
            x: Raw input image tensor of shape (B, 3, 256, 256) in range [0.0, 255.0].
            return_features: If True, returns a dictionary of intermediate features for explainability.

        Returns:
            Logit predictions of shape (B, 1) if return_features=False.
            Dict of intermediate activations and logits if return_features=True.
        """
        # MLEP requires [0, 1] input for entropy computation
        x_mlep = x / 255.0
        mlep_output = self.mlep_extractor(x_mlep)
        entropy_map = mlep_output['mlep_features']  # (B, 9, H', W') values in [0.0, 2.0]
        
        # CRITICAL FIX: Normalize entropy maps properly using BatchNorm2d 
        # so the ResNet backbone receives inputs at the zero-mean, unit-variance magnitude it was trained on.
        entropy_map = self.bn(entropy_map)
        
        feat_mlep = self.mlep_backbone(entropy_map)  # (B, 2048)
        
        logits = self.classifier(feat_mlep)
        
        if return_features:
            return {
                "logits": logits,
                "entropy_map": entropy_map,
                "feat_mlep": feat_mlep
            }
            
        return logits
