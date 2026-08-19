"""
Architecture II: Multi-Granularity Cross-Attention & Pyramid Feature Gating (MGA-Net).

Standalone model for spatial cross-modal attention between MLEP and LOTA features.
"""

from src.models.arch2_mganet.modules import PyramidCrossAttentionModule
from src.models.arch2_mganet.model import MGANetDualCueDetector

__all__ = [
    "PyramidCrossAttentionModule",
    "MGANetDualCueDetector",
]
