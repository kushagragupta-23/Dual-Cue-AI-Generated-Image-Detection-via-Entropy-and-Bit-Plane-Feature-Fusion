"""
Dual-Cue Feature Fusion Classifier (RGB/Entropy + LOTA LSB Bit-Plane Noise)
Combines global semantic spatial features with fine-grained LSB noise maps.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from src.models.lota import TopKLOTAExtractor
from src.utils.logger import get_logger

logger = get_logger("dual_cue_model")


class DualCueClassifier(nn.Module):
    """
    Dual-Cue Dual-Stream Feature Fusion Architecture.
    
    - Stream 1 (RGB Stream): Extracts global spatial semantics & color artifacts (2048 dims).
    - Stream 2 (LOTA LSB Stream): Extracts LSB noise anomalies via TopKLOTAExtractor (2048 dims).
    - Fusion Head: Concatenates (4096 dims) -> BatchNorm -> Dropout(0.5) -> FC(512) -> ReLU -> Dropout(0.5) -> Logits(1).
    """
    def __init__(
        self,
        k_patches: int = 1,
        patch_size: int = 32,
        grid_size: int = 8,
        backbone_name: str = "resnet50",
        dropout_rate: float = 0.6,
        freeze_early_layers: bool = True,
    ):
        super().__init__()
        
        # 1. LOTA LSB Extractor
        self.lota_extractor = TopKLOTAExtractor(
            k_patches=k_patches,
            patch_size=patch_size,
            grid_size=grid_size,
        )
        
        # 2. RGB Stream Backbone
        self.rgb_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features_rgb = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity()  # Output 2048-dim feature vector
        
        # 3. LOTA LSB Stream Backbone
        self.lota_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features_lota = self.lota_backbone.fc.in_features
        self.lota_backbone.fc = nn.Identity()  # Output 2048-dim feature vector
        
        # 4. Freeze Stem + layer1 + layer2 to prevent low-level feature memorization
        if freeze_early_layers:
            self._freeze_backbone_stem(self.rgb_backbone)
            self._freeze_backbone_stem(self.lota_backbone)
        
        total_in_features = in_features_rgb + in_features_lota  # 4096
        
        # 5. Feature Fusion Classification Head with Strong Regularization
        self.fusion_head = nn.Sequential(
            nn.BatchNorm1d(total_in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(total_in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1)
        )
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized DualCueClassifier (Trainable Params: {trainable_params:,} / Total: {total_params:,} | Frozen Early Layers: {freeze_early_layers})")

    def _freeze_backbone_stem(self, backbone: nn.Module) -> None:
        """Freeze conv1, bn1, layer1, and layer2 of a ResNet-50 model."""
        for param in backbone.conv1.parameters():
            param.requires_grad = False
        for param in backbone.bn1.parameters():
            param.requires_grad = False
        for param in backbone.layer1.parameters():
            param.requires_grad = False
        for param in backbone.layer2.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input RGB tensor of shape (B, 3, H, W) in range [0.0, 255.0].
        """
        # --- Stream 1: RGB Image Features ---
        x_norm = x / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        rgb_norm = (x_norm - mean) / std
        
        rgb_feats = self.rgb_backbone(rgb_norm)  # (B, 2048)
        
        # --- Stream 2: LOTA LSB Noise Features ---
        lota_dict = self.lota_extractor(x)
        noise_patch = lota_dict["noise_tensor"] / 255.0  # (B, 3, 256, 256)
        noise_norm = (noise_patch - mean) / std
        
        lota_feats = self.lota_backbone(noise_norm)  # (B, 2048)
        
        # --- Stream Fusion ---
        fused_feats = torch.cat([rgb_feats, lota_feats], dim=1)  # (B, 4096)
        logits = self.fusion_head(fused_feats)  # (B, 1)
        
        return logits
