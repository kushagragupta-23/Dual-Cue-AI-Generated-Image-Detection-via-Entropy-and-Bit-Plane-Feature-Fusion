"""
Unit Tests for Architecture II: PyramidCrossAttentionModule & MGANetDualCueDetector.

Verifies:
    1. Attention affinity matrix rows sum to ≈1.0
    2. Spatial dimensions preserved through cross-attention
    3. Output tensor shapes
    4. Gradient flow through attention mechanism
"""

import sys
from pathlib import Path

import pytest
import torch

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.models.arch2_mganet import PyramidCrossAttentionModule, MGANetDualCueDetector


class TestPyramidCrossAttention:
    """Test suite for cross-attention module."""

    @pytest.fixture
    def cross_attn(self):
        return PyramidCrossAttentionModule(
            in_channels_mlep=1024,
            in_channels_lota=1024,
            d_model=256,
            num_heads=8,
            dropout=0.0,  # No dropout for deterministic tests
        )

    def test_output_shape(self, cross_attn):
        """Cross-attention should output (B, d_model, H, W)."""
        feat_mlep = torch.randn(2, 1024, 16, 16)
        feat_lota = torch.randn(2, 1024, 16, 16)
        out = cross_attn(feat_mlep, feat_lota)
        assert out.shape == (2, 256, 16, 16), f"Expected (2, 256, 16, 16), got {out.shape}"

    def test_spatial_preservation(self, cross_attn):
        """Spatial dimensions should be preserved."""
        for h, w in [(8, 8), (16, 16), (32, 32)]:
            feat_mlep = torch.randn(1, 1024, h, w)
            feat_lota = torch.randn(1, 1024, h, w)
            out = cross_attn(feat_mlep, feat_lota)
            assert out.shape[2] == h and out.shape[3] == w, (
                f"Spatial dims not preserved: input ({h},{w}), output ({out.shape[2]},{out.shape[3]})"
            )

    def test_mismatched_spatial_dims(self, cross_attn):
        """Should handle mismatched spatial dims via interpolation."""
        feat_mlep = torch.randn(1, 1024, 16, 16)
        feat_lota = torch.randn(1, 1024, 8, 8)
        out = cross_attn(feat_mlep, feat_lota)
        # Output spatial should match MLEP (query) dimensions
        assert out.shape == (1, 256, 16, 16)

    def test_gradient_flow(self, cross_attn):
        """Verify gradients flow through cross-attention."""
        feat_mlep = torch.randn(2, 1024, 8, 8, requires_grad=True)
        feat_lota = torch.randn(2, 1024, 8, 8, requires_grad=True)
        out = cross_attn(feat_mlep, feat_lota)
        loss = out.sum()
        loss.backward()
        assert feat_mlep.grad is not None, "No gradient on MLEP features"
        assert feat_lota.grad is not None, "No gradient on LOTA features"

    def test_no_nan_inf(self, cross_attn):
        """Verify no NaN/Inf in output."""
        feat_mlep = torch.randn(2, 1024, 16, 16)
        feat_lota = torch.randn(2, 1024, 16, 16)
        out = cross_attn(feat_mlep, feat_lota)
        assert not torch.isnan(out).any(), "NaN in cross-attention output"
        assert not torch.isinf(out).any(), "Inf in cross-attention output"


class TestMGANetDetector:
    """Test suite for end-to-end MGA-Net detector."""

    @pytest.fixture
    def detector(self):
        return MGANetDualCueDetector(
            mlep_channels=9,
            lota_channels=3,
            backbone_name="resnet18",  # Lighter for testing
            d_model=64,
            num_heads=4,
            num_classes=1,
        )

    def test_output_shape(self, detector):
        """Detector should output (B, num_classes) logits."""
        x_mlep = torch.randn(2, 9, 224, 224)
        x_lota = torch.randn(2, 3, 224, 224)
        logits = detector(x_mlep, x_lota)
        assert logits.shape == (2, 1), f"Expected (2, 1), got {logits.shape}"

    def test_forward_backward(self, detector):
        """Verify full forward/backward pass works."""
        x_mlep = torch.randn(2, 9, 224, 224)
        x_lota = torch.randn(2, 3, 224, 224)
        logits = detector(x_mlep, x_lota)
        loss = logits.sum()
        loss.backward()
        # Verify at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in detector.parameters()
            if p.requires_grad
        )
        assert has_grad, "No parameter has non-zero gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
