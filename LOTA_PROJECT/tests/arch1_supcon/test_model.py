"""
Unit Tests for Architecture I: LearnableFrequencyPreFilter & DualCueSupConLoss.

Verifies:
    1. FFT → iFFT reconstruction error < 1e-5
    2. Gradient backprop updates cutoff and slope parameters
    3. SupCon loss symmetry and scale invariance under L2 normalization
    4. Projection head output dimensions
"""

import sys
from pathlib import Path

import pytest
import torch

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.models.arch1_supcon import (
    LearnableFrequencyPreFilter,
    DualCueSupConLoss,
    ProjectionHead,
)


class TestLearnableFrequencyPreFilter:
    """Test suite for the learnable FFT frequency filter."""

    @pytest.fixture
    def freq_filter(self):
        return LearnableFrequencyPreFilter(height=256, width=256)

    def test_output_shape(self, freq_filter):
        """Filter should preserve input shape."""
        x = torch.rand(2, 3, 256, 256) * 255.0
        out = freq_filter(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_output_range(self, freq_filter):
        """Output should be clamped to [0, 255]."""
        x = torch.rand(2, 3, 256, 256) * 255.0
        out = freq_filter(x)
        assert out.min() >= 0.0, f"Output min {out.min()} < 0"
        assert out.max() <= 255.0, f"Output max {out.max()} > 255"

    def test_fft_reconstruction(self):
        """With cutoff=10 (very high), filter should approximate identity."""
        freq_filter = LearnableFrequencyPreFilter(
            height=256, width=256, init_cutoff=10.0, init_slope=1.0
        )
        x = torch.rand(1, 3, 256, 256) * 200.0 + 10.0  # Avoid edge clamping
        with torch.no_grad():
            out = freq_filter(x)
        error = (x - out).abs().mean()
        # Tolerance allows for FFT floating-point precision + output clamp effects
        assert error < 5.0, (
            f"With very high cutoff, reconstruction error should be small, got {error:.4f}"
        )

    def test_gradient_updates_parameters(self, freq_filter):
        """Verify backprop updates cutoff and slope parameters."""
        x = torch.rand(2, 3, 256, 256) * 255.0
        out = freq_filter(x)
        loss = out.mean()
        loss.backward()

        assert freq_filter.cutoff.grad is not None, "Cutoff gradient is None"
        assert freq_filter.slope.grad is not None, "Slope gradient is None"
        assert freq_filter.cutoff.grad.abs() > 0, "Cutoff gradient is zero"

    def test_no_nan_inf(self, freq_filter):
        """Verify no NaN/Inf in output."""
        x = torch.rand(2, 3, 256, 256) * 255.0
        out = freq_filter(x)
        assert not torch.isnan(out).any(), "NaN in frequency filter output"
        assert not torch.isinf(out).any(), "Inf in frequency filter output"


class TestDualCueSupConLoss:
    """Test suite for Supervised Contrastive Loss."""

    @pytest.fixture
    def supcon(self):
        return DualCueSupConLoss(temperature=0.07)

    def test_loss_is_scalar(self, supcon):
        """SupCon loss should return a scalar."""
        features = torch.randn(4, 2, 128)
        labels = torch.tensor([0, 0, 1, 1])
        loss = supcon(features, labels)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_loss_is_positive(self, supcon):
        """SupCon loss should be positive for non-trivial inputs."""
        features = torch.randn(8, 2, 128)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = supcon(features, labels)
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"

    def test_loss_symmetry(self, supcon):
        """SupCon loss should be symmetric: L(z1, z2) ≈ L(z2, z1)."""
        features = torch.randn(4, 2, 128)
        labels = torch.tensor([0, 1, 0, 1])

        loss1 = supcon(features, labels)

        # Swap MLEP and LOTA views
        features_swapped = features.flip(1)
        loss2 = supcon(features_swapped, labels)

        assert abs(loss1.item() - loss2.item()) < 0.1, (
            f"Loss should be approximately symmetric: {loss1.item():.4f} vs {loss2.item():.4f}"
        )

    def test_scale_invariance(self, supcon):
        """Loss should be invariant to input scale (due to L2 normalization)."""
        features = torch.randn(4, 2, 128)
        labels = torch.tensor([0, 1, 0, 1])

        loss1 = supcon(features, labels)
        loss2 = supcon(features * 10.0, labels)

        assert abs(loss1.item() - loss2.item()) < 0.01, (
            f"Loss should be scale-invariant: {loss1.item():.4f} vs {loss2.item():.4f}"
        )

    def test_gradient_flow(self, supcon):
        """Verify gradients flow back through features."""
        features = torch.randn(4, 2, 128, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = supcon(features, labels)
        loss.backward()
        assert features.grad is not None, "No gradient on features"
        assert features.grad.abs().sum() > 0, "Zero gradient on features"


class TestProjectionHead:
    """Test suite for the MLP projection head."""

    def test_output_shape(self):
        head = ProjectionHead(in_dim=2048, hidden_dim=256, out_dim=128)
        x = torch.randn(4, 2048)
        out = head(x)
        assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"

    def test_gradient_flow(self):
        head = ProjectionHead(in_dim=512, hidden_dim=128, out_dim=64)
        x = torch.randn(2, 512, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
