"""
Dual-Cue Feature Classifiers (LOTA and MLEP)
Updated to match ICCV 2025: LOTA uses pre-trained ResNet-50.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from src.shared.extractors.lota import TopKLOTAExtractor
from src.utils.logger import get_logger

logger = get_logger("classifiers")


class LOTAClassifier(nn.Module):
    """
    Pure LOTA Classifier (Noise-Based Classifier).
    
    1. Accepts raw RGB image tensor (B, 3, 256, 256).
    2. Extracts LOTA LSB noise map / maximum gradient patch via TopKLOTAExtractor.
    3. Feeds noise map into ImageNet pre-trained ResNet backbone.
    4. Outputs binary classification logits.
    """
    def __init__(
        self,
        k_patches: int = 1,
        patch_size: int = 32,
        grid_size: int = 8,
        backbone_name: str = "resnet50",
        use_full_lsb: bool = False,
    ):
        super().__init__()
        self.use_full_lsb = use_full_lsb
        
        # 1. LOTA Extractor
        self.lota_extractor = TopKLOTAExtractor(
            k_patches=k_patches,
            patch_size=patch_size,
            grid_size=grid_size,
        )
        
        # 2. ResNet backbone
        if backbone_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input RGB tensor of shape (B, 3, 256, 256) in range [0.0, 255.0].
        """
        # If input is already an extracted LOTA noise map, bypass extractor
        if hasattr(self, "_is_noise_map") and self._is_noise_map:
            noise_patch = x
        else:
            lota_dict = self.lota_extractor(x)
            if self.use_full_lsb:
                noise_patch = lota_dict["z_norm"]
            else:
                noise_patch = lota_dict["noise_tensor"]
                
        # Scale to [0.0, 1.0]
        noise_patch = noise_patch / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=noise_patch.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=noise_patch.device).view(1, 3, 1, 1)
        
        noise_patch = (noise_patch - mean) / std
        
        logits = self.backbone(noise_patch)
        return logits

