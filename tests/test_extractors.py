"""
Unit Tests for MLEP and LOTA Differentiable Extractors.

Validates:
  - Output shapes for various input sizes
  - Gradient flow (differentiability) through both extractors
  - Numerical stability (no NaN/Inf outputs)
  - Multi-scale entropy map generation (MLEP)
  - Bit-plane soft extraction (LOTA)
"""

import pytest
import torch
import torch.nn as nn
from src.models.mlep_extractor import MLEPExtractor
from src.models.lota_extractor import TopKLOTAExtractor


class TestMLEPExtractor:
    """Tests for the differentiable MLEP extractor."""

    @pytest.fixture
    def mlep(self):
        return MLEPExtractor(
            scales=(1.0, 0.5, 0.25),
            window_size=2,
            macro_grid_size=16,
        )

    @pytest.fixture
    def sample_input(self):
        """Batch of 2 RGB images at 256x256, range [0, 255]."""
        return torch.randn(2, 3, 256, 256) * 128 + 128

    def test_output_shape(self, mlep, sample_input):
        """Output should have 9 channels (3 scales x 3 RGB) for 256x256 input."""
        out = mlep(sample_input)
        assert out.dim() == 4
        assert out.shape[0] == 2  # Batch size
        assert out.shape[2] == 256  # Height preserved
        assert out.shape[3] == 256  # Width preserved
        # Channel count = num_scales * input_channels
        assert out.shape[1] == 3 * 3  # 3 scales * 3 channels = 9

    def test_gradient_flow(self, mlep, sample_input):
        """Gradients must flow through MLEP (differentiability check)."""
        sample_input.requires_grad_(True)
        out = mlep(sample_input)
        loss = out.sum()
        loss.backward()
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert torch.abs(sample_input.grad).sum() > 0

    def test_no_nans(self, mlep, sample_input):
        """Output should contain no NaN or Inf values."""
        out = mlep(sample_input)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_zero_input(self, mlep):
        """Zero input should not produce NaN."""
        zero_input = torch.zeros(1, 3, 256, 256)
        out = mlep(zero_input)
        assert not torch.isnan(out).any()

    def test_different_batch_sizes(self, mlep):
        """Should work with various batch sizes."""
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 3, 256, 256) * 128 + 128
            out = mlep(x)
            assert out.shape[0] == bs


class TestTopKLOTAExtractor:
    """Tests for the differentiable LOTA extractor."""

    @pytest.fixture
    def lota(self):
        return TopKLOTAExtractor(
            k_patches=4,
            patch_size=32,
            grid_size=8,
        )

    @pytest.fixture
    def sample_input(self):
        """Batch of 2 RGB images at 256x256, range [0, 255]."""
        return torch.randn(2, 3, 256, 256) * 128 + 128

    def test_output_shape(self, lota, sample_input):
        """Output should be 3-channel (RGB) at 256x256."""
        out = lota(sample_input)
        assert out.dim() == 4
        assert out.shape[0] == 2  # Batch size
        assert out.shape[1] == 3  # 3 RGB channels
        assert out.shape[2] == 256  # Height
        assert out.shape[3] == 256  # Width

    def test_gradient_flow(self, lota, sample_input):
        """Gradients must flow through LOTA (differentiability check)."""
        sample_input.requires_grad_(True)
        out = lota(sample_input)
        loss = out.sum()
        loss.backward()
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()
        assert torch.abs(sample_input.grad).sum() > 0

    def test_no_nans(self, lota, sample_input):
        """Output should contain no NaN or Inf values."""
        out = lota(sample_input)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_range(self, lota, sample_input):
        """Output values should be bounded (not exploding)."""
        out = lota(sample_input)
        assert out.max() < 1e6
        assert out.min() > -1e6

    def test_zero_input(self, lota):
        """Zero input should not produce NaN."""
        zero_input = torch.zeros(1, 3, 256, 256)
        out = lota(zero_input)
        assert not torch.isnan(out).any()

    def test_different_batch_sizes(self, lota):
        """Should work with various batch sizes."""
        for bs in [1, 4]:
            x = torch.randn(bs, 3, 256, 256) * 128 + 128
            out = lota(x)
            assert out.shape[0] == bs
