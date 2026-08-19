"""
DualCueAIGIDModel: Master End-to-End Assembly for Dual-Cue AI-Generated Image Detection.

Integrates all pipeline components into a single configurable PyTorch module:
    1. MLEP Extractor → Frequency Pre-Filter → MLEP Backbone Stem
    2. LOTA Extractor → LOTA Backbone Stem
    3. Cross-Modal Gating Fusion Head (α_MLEP + α_LOTA = 1.0)
    4. Optional: MGA-Net Cross-Attention (Architecture II)
    5. Optional: MoE Routing + Domain Discriminator (Architecture III)

Supports 2-stage training:
    Stage 1: SupCon contrastive pre-training (via arch1_supcon)
    Stage 2: Gated classifier fine-tuning (this module)
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.extractors import VectorizedMLEPExtractor, TopKLOTAExtractor
from src.shared.backbones import ChannelAdaptedResNet
from src.models.arch1_supcon.modules import LearnableFrequencyPreFilter
from src.models.arch2_mganet.modules import PyramidCrossAttentionModule
from src.models.arch3_moe.model import DomainAdversarialMoEDetector
from src.models.fusion.modules import CrossModalGatingFusionHead
from src.utils.logger import get_logger

logger = get_logger("fusion.model")


class DualCueAIGIDModel(nn.Module):
    """
    Master Dual-Cue AI-Generated Image Detection Model.

    End-to-end configurable architecture integrating MLEP, LOTA, frequency
    pre-filtering, backbone stems, cross-attention fusion, MoE routing,
    and domain-adversarial training into a single unified module.

    Args:
        backbone_name: 'resnet18' or 'resnet50' (default 'resnet50').
        pretrained: Whether to use pretrained backbone weights.
        use_frequency_filter: Whether to apply learnable frequency pre-filter.
        use_cross_attention: Whether to use MGA-Net spatial cross-attention.
        use_moe: Whether to use Sparse MoE routing.
        use_dann: Whether to use Domain-Adversarial training (requires use_moe=True).
        num_domains: Number of generator domains for DANN (default 8).
        num_experts: Number of MoE experts (default 4).
        top_k: Top-K routing for MoE (default 2).
        d_model: Cross-attention / MoE hidden dimension (default 256).
        num_heads: Cross-attention heads (default 8).
        moe_lambda: GRL reversal coefficient for DANN (default 0.5).
        dropout: Classifier dropout rate (default 0.3).
        freeze_stems: Whether to freeze backbone stems.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        use_frequency_filter: bool = True,
        use_cross_attention: bool = True,
        use_moe: bool = True,
        use_dann: bool = True,
        num_domains: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
        d_model: int = 256,
        num_heads: int = 8,
        moe_lambda: float = 0.5,
        dropout: float = 0.3,
        freeze_stems: bool = False,
        # Legacy aliases for backward compatibility
        use_freq_filter: Optional[bool] = None,
    ):
        super().__init__()

        # Handle legacy parameter name
        if use_freq_filter is not None:
            use_frequency_filter = use_freq_filter

        self.use_frequency_filter = use_frequency_filter
        self.use_cross_attention = use_cross_attention
        self.use_moe = use_moe
        self.use_dann = use_dann and use_moe  # DANN requires MoE

        # ---- Feature Extractors ----
        self.mlep_extractor = VectorizedMLEPExtractor()
        self.lota_extractor = TopKLOTAExtractor(k_patches=1)

        # Optional frequency pre-filter for MLEP branch
        self.freq_filter = None
        if use_frequency_filter:
            self.freq_filter = LearnableFrequencyPreFilter()

        # ---- Backbone Stems ----
        self.mlep_stem = ChannelAdaptedResNet(
            in_channels=9, backbone_name=backbone_name, pretrained=pretrained
        )
        self.lota_stem = ChannelAdaptedResNet(
            in_channels=3, backbone_name=backbone_name, pretrained=pretrained
        )

        feat_dim = self.mlep_stem.stage_channels["stage4"]  # 2048 for ResNet-50

        if freeze_stems:
            self._freeze_module(self.mlep_stem)
            self._freeze_module(self.lota_stem)
            if self.freq_filter is not None:
                self._freeze_module(self.freq_filter)

        # ---- Optional Cross-Attention (Architecture II) ----
        self.cross_attn = None
        self.cross_attn_pool = None
        self.cross_attn_proj = None
        if use_cross_attention:
            stage3_channels = self.mlep_stem.stage_channels["stage3"]
            self.cross_attn = PyramidCrossAttentionModule(
                in_channels_mlep=stage3_channels,
                in_channels_lota=stage3_channels,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.cross_attn_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.cross_attn_proj = nn.Linear(d_model, feat_dim)

        # ---- Fusion Head ----
        self.gating_head = CrossModalGatingFusionHead(feat_dim=feat_dim)

        # ---- Classification / MoE / DANN ----
        self.moe_detector = None
        self.classifier = None
        self.dann_head = None

        if use_moe:
            self.moe_detector = DomainAdversarialMoEDetector(
                in_dim=feat_dim,
                d_model=d_model * 2 if d_model < 512 else 512,
                num_experts=num_experts,
                top_k=top_k,
                num_domains=num_domains if self.use_dann else 1,
                lambda_coeff=moe_lambda if self.use_dann else 0.0,
            )
            if self.use_dann:
                self.dann_head = self.moe_detector
        else:
            # Simple binary classifier without MoE
            self.classifier = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DualCueAIGIDModel assembled: {backbone_name} | "
            f"FreqFilter={use_frequency_filter} | CrossAttn={use_cross_attention} | "
            f"MoE={use_moe} | DANN={self.use_dann} | "
            f"Trainable: {trainable:,} / Total: {total:,}"
        )

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        """Freeze all parameters of a module."""
        for param in module.parameters():
            param.requires_grad = False

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full end-to-end forward pass.

        Args:
            x: Raw RGB image tensor of shape (B, 3, H, W) in [0.0, 255.0].

        Returns:
            dict containing class_logits, domain_logits, aux_loss, gate_weights.
        """
        # ---- MLEP Branch ----
        mlep_input = x
        if self.freq_filter is not None:
            mlep_input = self.freq_filter(mlep_input)

        mlep_dict = self.mlep_extractor(mlep_input)
        entropy_map = mlep_dict["entropy_map"]

        # ---- LOTA Branch ----
        lota_dict = self.lota_extractor(x)
        noise_tensor = lota_dict["noise_tensor"]

        # ---- Backbone Feature Extraction ----
        if self.cross_attn is not None:
            mlep_spatial = self.mlep_stem.forward_features(entropy_map)
            lota_spatial = self.lota_stem.forward_features(noise_tensor)

            f_mlep = self.mlep_stem.avgpool(mlep_spatial["stage4"]).flatten(1)
            f_lota = self.lota_stem.avgpool(lota_spatial["stage4"]).flatten(1)

            cross_fused = self.cross_attn(
                mlep_spatial["stage3"], lota_spatial["stage3"]
            )
            cross_pooled = self.cross_attn_pool(cross_fused).flatten(1)
            cross_proj = self.cross_attn_proj(cross_pooled)

            f_mlep = f_mlep + cross_proj
            f_lota = f_lota + cross_proj
        else:
            f_mlep = self.mlep_stem(entropy_map)
            f_lota = self.lota_stem(noise_tensor)

        # ---- Cross-Modal Gating Fusion ----
        z_fused, gate_weights = self.gating_head(f_mlep, f_lota)

        # ---- Classification ----
        if self.moe_detector is not None:
            class_logits, domain_logits, aux_loss = self.moe_detector(z_fused)
        else:
            class_logits = self.classifier(z_fused)
            domain_logits = None
            aux_loss = torch.tensor(0.0, device=x.device)

        return {
            "class_logits": class_logits,
            "domain_logits": domain_logits,
            "aux_loss": aux_loss,
            "gate_weights": gate_weights,
        }


__all__ = ["DualCueAIGIDModel"]
