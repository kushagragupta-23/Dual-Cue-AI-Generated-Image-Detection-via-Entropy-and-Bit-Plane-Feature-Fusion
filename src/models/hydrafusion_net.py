import torch
import torch.nn as nn
from typing import Tuple

from .freq_prefilter import LearnableFrequencyPreFilter
from .mlep_extractor import MLEPExtractor
from .lota_extractor import TopKLOTAExtractor
from .backbones import ResNet50SpatialStem
from .fusion_heads import MultiHeadFusionModule
from .gating_router import AdaptiveGatingRouter
from .domain_adversarial import DomainAdversarialHead

class HydraFusionNet(nn.Module):
    """
    HydraFusion-Net: Adaptive Multi-Head Forensic Fusion Architecture.
    Combines MLEP (NeurIPS 2025) and LOTA (ICCV 2025) streams.
    
    v3 Changes:
      - Backbones frozen by default (pretrained feature extractors)
      - Only fusion layers, router, classifier are trainable in Stage 2
      - Domain adversarial head kept but optional (disabled by default)
    """
    def __init__(
        self,
        num_domains: int = 8,
        latent_dim: int = 512,
        use_freq_filter: bool = True,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        
        # --- PRE-FUSION ---
        self.use_freq_filter = use_freq_filter
        if self.use_freq_filter:
            self.freq_filter = LearnableFrequencyPreFilter()
            
        self.mlep_extractor = MLEPExtractor()
        self.lota_extractor = TopKLOTAExtractor(k_patches=4)
        
        # --- SPATIAL BACKBONES ---
        # MLEP outputs 9 channels (3 scales * 3 colors)
        self.mlep_stem = ResNet50SpatialStem(in_channels=9, return_layer='layer3')
        # LOTA outputs 3 channels (differentiable soft-mask noise map)
        self.lota_stem = ResNet50SpatialStem(in_channels=3, return_layer='layer3')
        
        # In ResNet-50 layer3, output channels = 1024
        
        # Freeze backbones: use as pretrained feature extractors only
        # This prevents overfitting 47M params on 6k images
        if freeze_backbones:
            self._freeze_backbones()
        
        # --- MULTI-HEAD FUSION ---
        self.fusion_module = MultiHeadFusionModule(channels_mlep=1024, channels_lota=1024, dim=latent_dim)
        self.router = AdaptiveGatingRouter(in_channels_mlep=1024, in_channels_lota=1024, num_heads=4)
        
        # --- STAGE 1 (SUPCON) PROJECTIONS ---
        self.mlep_proj = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128))
        self.lota_proj = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128))
        
        # --- STAGE 2 CLASSIFICATION ---
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Binary: 0=Real, 1=Fake
        )
        
        # Domain adversarial (kept for optional use)
        self.domain_head = DomainAdversarialHead(in_features=latent_dim, num_domains=num_domains)

    def _freeze_backbones(self):
        """Freeze pretrained ResNet-50 backbones — only trainable layers learn."""
        for module in [self.mlep_stem, self.lota_stem]:
            for param in module.parameters():
                param.requires_grad = False

    def update_grl_lambda(self, lambda_val: float):
        """Update GRL reverse scale based on training epoch progression."""
        self.domain_head.grl.lambda_val = lambda_val

    def forward(self, x: torch.Tensor, stage: int = 2) -> Tuple[torch.Tensor, ...]:
        """
        stage=1: Returns L2-normalized projections for SupCon
        stage=2: Returns Binary Logits, Domain Logits, and Gating Weights
        """
        # Pre-Filter MLEP path
        x_mlep = self.freq_filter(x) if self.use_freq_filter else x
        
        # Extract features (both now differentiable)
        feat_mlep = self.mlep_extractor(x_mlep)  # (B, 9, H', W')
        feat_lota = self.lota_extractor(x)        # (B, 3, 256, 256)
        
        # Spatial Stems (frozen — act as feature extractors)
        spatial_mlep = self.mlep_stem(feat_mlep)  # (B, 1024, 16, 16)
        spatial_lota = self.lota_stem(feat_lota)  # (B, 1024, 16, 16)
        
        if stage == 1:
            # Global Average Pool and Project to unit hypersphere
            z_mlep = torch.nn.functional.adaptive_avg_pool2d(spatial_mlep, 1).flatten(1)
            z_lota = torch.nn.functional.adaptive_avg_pool2d(spatial_lota, 1).flatten(1)
            
            p_mlep = torch.nn.functional.normalize(self.mlep_proj(z_mlep), dim=1)
            p_lota = torch.nn.functional.normalize(self.lota_proj(z_lota), dim=1)
            return p_mlep, p_lota
            
        elif stage == 2:
            # Parallel Multi-Head Fusion
            head_outputs = self.fusion_module(spatial_mlep, spatial_lota)  # (B, 4, dim)
            
            # Adaptive Routing
            fused, gating_weights = self.router(spatial_mlep, spatial_lota, head_outputs)
            
            # Classification
            logits = self.classifier(fused)
            
            # Domain Adversarial
            domain_logits = self.domain_head(fused)
            
            return logits, domain_logits, gating_weights
