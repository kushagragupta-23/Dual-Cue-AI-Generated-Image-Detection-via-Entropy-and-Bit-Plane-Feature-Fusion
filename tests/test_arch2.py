"""
Unit Tests for Architecture II (MGA-Net Multi-Head Cross-Attention).
"""

from tests.test_fusion_heads import (
    TestSpatialCrossAttentionHead,
    TestChannelSEFusionHead,
    TestFrequencyCorrelationHead,
    TestMultiHeadFusionModule,
)


import torch
from src.models.arch2_mganet import PyramidCrossAttentionModule

class TestArchitecture2:
    """Roadmap test suite wrapper for Architecture II MGA-Net."""
    def test_pyramid_cross_attention(self):
        module = PyramidCrossAttentionModule(in_channels_mlep=1024, in_channels_lota=1024, d_model=512)
        f_mlep = torch.randn(2, 1024, 8, 8)
        f_lota = torch.randn(2, 1024, 8, 8)
        out = module(f_mlep, f_lota)
        assert out.shape == (2, 512, 8, 8)
        assert not torch.isnan(out).any()

