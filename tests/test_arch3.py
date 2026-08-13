"""
Unit Tests for Architecture III (Adaptive Gating Router & Domain Adversarial Head).
"""

import pytest
import torch
from src.models.gating_router import AdaptiveGatingRouter
from src.models.domain_adversarial import GradientReversalLayer, DomainAdversarialHead


class TestArchitecture3:
    """Tests for Architecture III MoE Router and Domain Adversarial Head."""

    def test_gating_router(self):
        router = AdaptiveGatingRouter(in_channels_mlep=1024, in_channels_lota=1024, num_heads=4)
        f_mlep = torch.randn(2, 1024, 8, 8)
        f_lota = torch.randn(2, 1024, 8, 8)
        head_outputs = torch.randn(2, 4, 512)
        fused, alpha = router(f_mlep, f_lota, head_outputs)
        assert fused.shape == (2, 512)
        assert alpha.shape == (2, 4)
        assert torch.allclose(alpha.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_gradient_reversal_layer(self):
        grl = GradientReversalLayer(lambda_val=1.0)
        x = torch.randn(2, 512, requires_grad=True)
        out = grl(x)
        out.sum().backward()
        assert x.grad is not None
        # GRL negates gradients in backward pass
        assert torch.allclose(x.grad, -torch.ones_like(x))

    def test_domain_adversarial_head(self):
        head = DomainAdversarialHead(in_features=512, num_domains=8)
        x = torch.randn(2, 512)
        out = head(x)
        assert out.shape == (2, 8)
        assert not torch.isnan(out).any()

    def test_sparse_moe_forensic_module(self):
        from src.models.arch3_moe import SparseMoEForensicModule
        moe = SparseMoEForensicModule(d_model=512, num_experts=4, top_k=2)
        x = torch.randn(4, 512)
        out, aux_loss = moe(x)
        assert out.shape == (4, 512)
        assert aux_loss >= 0
        assert not torch.isnan(out).any()

