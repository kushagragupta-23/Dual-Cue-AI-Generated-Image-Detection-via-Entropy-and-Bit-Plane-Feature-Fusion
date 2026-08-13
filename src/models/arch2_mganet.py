"""
Architecture II: MGA-Net Multi-Head Cross-Attention — Roadmap Alias.

Exposes SpatialCrossAttentionHead, ChannelSEFusionHead, FrequencyCorrelationHead, and MultiHeadFusionModule.
"""

from src.models.fusion_heads import (
    SpatialCrossAttentionHead,
    ChannelSEFusionHead,
    FrequencyCorrelationHead,
    MultiHeadFusionModule,
    PyramidCrossAttentionModule,
)

__all__ = [
    "SpatialCrossAttentionHead",
    "ChannelSEFusionHead",
    "FrequencyCorrelationHead",
    "MultiHeadFusionModule",
    "PyramidCrossAttentionModule",
]
