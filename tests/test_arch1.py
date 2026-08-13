"""
Unit Tests for Architecture I (Learnable Frequency Filter & SupCon Loss).
"""

import pytest
import torch
from src.models.freq_prefilter import LearnableFrequencyPreFilter
from src.models.supcon_loss import DualCueSupConLoss


class TestArchitecture1:
    """Tests for Learnable Frequency Filter and DualCueSupConLoss."""

    def test_freq_prefilter_forward(self):
        filter_net = LearnableFrequencyPreFilter(img_size=256)
        x = torch.randn(2, 3, 256, 256) * 128 + 128
        out = filter_net(x)
        assert out.shape == (2, 3, 256, 256)
        assert not torch.isnan(out).any()

    def test_supcon_loss_forward(self):
        loss_fn = DualCueSupConLoss(temperature=0.1)
        p_mlep = torch.randn(4, 512)
        p_lota = torch.randn(4, 512)
        labels = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(p_mlep, p_lota, labels)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0.0
