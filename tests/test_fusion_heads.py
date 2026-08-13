"""
Unit Tests for HydraFusion Multi-Head Fusion Module.

Validates:
  - SpatialCrossAttentionHead: Q-K-V attention between MLEP/LOTA features
  - ChannelSEFusionHead: Squeeze-Excitation channel gating
  - FrequencyCorrelationHead: rFFT2-based frequency correlation
  - AdaptiveGatingRouter: Softmax-weighted head combination
  - MultiHeadFusionModule: End-to-end 4-head fusion pipeline
"""

import pytest
import torch
from src.models.fusion_heads import (
    SpatialCrossAttentionHead,
    ChannelSEFusionHead,
    FrequencyCorrelationHead,
    MultiHeadFusionModule,
)
from src.models.gating_router import AdaptiveGatingRouter


class TestSpatialCrossAttentionHead:
    """Tests for the cross-attention fusion head."""

    @pytest.fixture
    def head(self):
        return SpatialCrossAttentionHead(in_channels=1024, dim=512)

    def test_output_shape(self, head):
        """Output should be (B, dim)."""
        q = torch.randn(2, 1024, 8, 8)
        kv = torch.randn(2, 1024, 8, 8)
        out = head(q, kv)
        assert out.shape == (2, 512)

    def test_gradient_flow(self, head):
        """Gradients should flow through the cross-attention."""
        q = torch.randn(2, 1024, 8, 8, requires_grad=True)
        kv = torch.randn(2, 1024, 8, 8, requires_grad=True)
        out = head(q, kv)
        out.sum().backward()
        assert q.grad is not None
        assert kv.grad is not None

    def test_no_nans(self, head):
        """No NaN in output."""
        q = torch.randn(2, 1024, 8, 8)
        kv = torch.randn(2, 1024, 8, 8)
        out = head(q, kv)
        assert not torch.isnan(out).any()


class TestChannelSEFusionHead:
    """Tests for the channel squeeze-excitation head."""

    @pytest.fixture
    def head(self):
        return ChannelSEFusionHead(
            in_channels_mlep=1024, in_channels_lota=1024, dim=512,
        )

    def test_output_shape(self, head):
        """Output should be (B, dim)."""
        f_mlep = torch.randn(2, 1024, 8, 8)
        f_lota = torch.randn(2, 1024, 8, 8)
        out = head(f_mlep, f_lota)
        assert out.shape == (2, 512)

    def test_gradient_flow(self, head):
        """Gradients should flow through SE block."""
        f_mlep = torch.randn(2, 1024, 8, 8, requires_grad=True)
        f_lota = torch.randn(2, 1024, 8, 8, requires_grad=True)
        out = head(f_mlep, f_lota)
        out.sum().backward()
        assert f_mlep.grad is not None


class TestFrequencyCorrelationHead:
    """Tests for the frequency-domain correlation head."""

    @pytest.fixture
    def head(self):
        return FrequencyCorrelationHead(
            in_channels_mlep=1024, in_channels_lota=1024, dim=512,
        )

    def test_output_shape(self, head):
        """Output should be (B, dim)."""
        f_mlep = torch.randn(2, 1024, 8, 8)
        f_lota = torch.randn(2, 1024, 8, 8)
        out = head(f_mlep, f_lota)
        assert out.shape == (2, 512)

    def test_gradient_flow(self, head):
        """Gradients should flow through frequency path."""
        f_mlep = torch.randn(2, 1024, 8, 8, requires_grad=True)
        f_lota = torch.randn(2, 1024, 8, 8, requires_grad=True)
        out = head(f_mlep, f_lota)
        out.sum().backward()
        assert f_mlep.grad is not None


class TestAdaptiveGatingRouter:
    """Tests for the adaptive gating router."""

    @pytest.fixture
    def router(self):
        return AdaptiveGatingRouter(
            in_channels_mlep=1024, in_channels_lota=1024, num_heads=4,
        )

    def test_output_shape(self, router):
        """Fused output should be (B, dim) and alpha should be (B, 4)."""
        f_mlep = torch.randn(2, 1024, 8, 8)
        f_lota = torch.randn(2, 1024, 8, 8)
        head_outputs = torch.randn(2, 4, 512)

        fused, alpha = router(f_mlep, f_lota, head_outputs)
        assert fused.shape == (2, 512)
        assert alpha.shape == (2, 4)

    def test_alpha_sums_to_one(self, router):
        """Gating weights should sum to 1.0 (softmax)."""
        f_mlep = torch.randn(2, 1024, 8, 8)
        f_lota = torch.randn(2, 1024, 8, 8)
        head_outputs = torch.randn(2, 4, 512)

        _, alpha = router(f_mlep, f_lota, head_outputs)
        alpha_sums = alpha.sum(dim=1)
        torch.testing.assert_close(
            alpha_sums, torch.ones(2), atol=1e-5, rtol=1e-5
        )

    def test_alpha_non_negative(self, router):
        """All gating weights should be non-negative."""
        f_mlep = torch.randn(2, 1024, 8, 8)
        f_lota = torch.randn(2, 1024, 8, 8)
        head_outputs = torch.randn(2, 4, 512)

        _, alpha = router(f_mlep, f_lota, head_outputs)
        assert (alpha >= 0).all()


class TestMultiHeadFusionModule:
    """Tests for the complete 4-head fusion module."""

    @pytest.fixture
    def fusion(self):
        return MultiHeadFusionModule(
            channels_mlep=1024, channels_lota=1024, dim=512,
        )

    def test_output_shape(self, fusion):
        """Stacked head outputs should be (B, 4, 512)."""
        f_mlep = torch.randn(2, 1024, 8, 8)
        f_lota = torch.randn(2, 1024, 8, 8)
        out = fusion(f_mlep, f_lota)
        assert out.shape == (2, 4, 512)

    def test_gradient_flow(self, fusion):
        """End-to-end gradient flow through all 4 heads."""
        f_mlep = torch.randn(2, 1024, 8, 8, requires_grad=True)
        f_lota = torch.randn(2, 1024, 8, 8, requires_grad=True)
        out = fusion(f_mlep, f_lota)
        # Non-linear loss to prevent zero-mean cancellation across FFT heads
        loss = (out ** 2).sum()
        loss.backward()
        assert f_mlep.grad is not None
        assert f_lota.grad is not None
        assert torch.abs(f_mlep.grad).sum() > 0
        assert torch.abs(f_lota.grad).sum() > 0
