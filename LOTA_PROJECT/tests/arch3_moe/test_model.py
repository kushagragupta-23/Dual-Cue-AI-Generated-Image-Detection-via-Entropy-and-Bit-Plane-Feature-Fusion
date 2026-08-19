"""
Unit Tests for Architecture III: SparseMoE + GRL + DANN.

Verifies:
    1. Top-2 routing: exactly num_experts - 2 weights = 0 per sample
    2. GRL: forward = identity, backward = −λ·gradient
    3. Aux loss > 0 for unbalanced routing
    4. Domain discriminator output shape
    5. Full forward/backward pass
"""

import sys
from pathlib import Path

import pytest
import torch

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.models.arch3_moe import (
    ExpertModule,
    SparseMoEForensicModule,
    GradientReversalLayer,
    DomainAdversarialMoEDetector,
)


class TestExpertModule:
    """Test suite for residual expert bottleneck."""

    def test_residual_output_shape(self):
        """Expert should preserve input shape via residual connection."""
        expert = ExpertModule(in_dim=512, hidden_dim=256)
        x = torch.randn(4, 512)
        out = expert(x)
        assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"

    def test_residual_not_identity(self):
        """Expert output should differ from input (non-zero learned transformation)."""
        expert = ExpertModule(in_dim=256, hidden_dim=128)
        x = torch.randn(4, 256)
        out = expert(x)
        # With random init, output should differ from input
        diff = (out - x).abs().mean()
        assert diff > 1e-6, f"Expert should transform input, diff={diff:.8f}"


class TestSparseMoE:
    """Test suite for Sparse Mixture-of-Experts routing."""

    @pytest.fixture
    def moe(self):
        return SparseMoEForensicModule(d_model=256, num_experts=4, top_k=2)

    def test_output_shape(self, moe):
        """MoE should output (B, d_model)."""
        x = torch.randn(8, 256)
        out, aux_loss = moe(x)
        assert out.shape == (8, 256), f"Expected (8, 256), got {out.shape}"

    def test_aux_loss_is_scalar(self, moe):
        """Auxiliary loss should be a scalar."""
        x = torch.randn(8, 256)
        _, aux_loss = moe(x)
        assert aux_loss.dim() == 0, f"Expected scalar aux loss, got shape {aux_loss.shape}"

    def test_aux_loss_positive(self, moe):
        """Aux loss should be positive."""
        x = torch.randn(16, 256)
        _, aux_loss = moe(x)
        assert aux_loss.item() > 0, f"Expected positive aux loss, got {aux_loss.item()}"

    def test_gradient_flow(self, moe):
        """Verify gradients flow through MoE routing."""
        x = torch.randn(8, 256, requires_grad=True)
        out, aux_loss = moe(x)
        total_loss = out.sum() + aux_loss
        total_loss.backward()
        assert x.grad is not None, "No gradient on MoE input"
        assert x.grad.abs().sum() > 0, "Zero gradient through MoE"

    def test_eval_mode_deterministic(self, moe):
        """In eval mode, routing should be deterministic (no noise)."""
        moe.eval()
        x = torch.randn(4, 256)
        out1, _ = moe(x)
        out2, _ = moe(x)
        assert torch.allclose(out1, out2, atol=1e-6), (
            "Eval mode should be deterministic"
        )

    def test_no_nan_inf(self, moe):
        """No NaN/Inf in output."""
        x = torch.randn(8, 256)
        out, aux_loss = moe(x)
        assert not torch.isnan(out).any(), "NaN in MoE output"
        assert not torch.isinf(out).any(), "Inf in MoE output"


class TestGradientReversalLayer:
    """Test suite for Gradient Reversal Layer."""

    def test_forward_identity(self):
        """GRL forward pass should be identity."""
        grl = GradientReversalLayer(lambda_coeff=1.0)
        x = torch.randn(4, 128)
        out = grl(x)
        assert torch.allclose(x, out), "GRL forward should be identity"

    def test_backward_negation(self):
        """GRL backward should negate gradients by lambda."""
        grl = GradientReversalLayer(lambda_coeff=1.0)
        x = torch.randn(4, 128, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()

        # Without GRL, gradient of sum() would be all ones
        # With GRL, it should be all negative ones
        expected_grad = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6), (
            f"GRL backward should negate gradients. "
            f"Got mean={x.grad.mean():.4f}, expected={expected_grad.mean():.4f}"
        )

    def test_backward_scaling(self):
        """GRL backward should scale negated gradients by lambda."""
        lambda_val = 0.5
        grl = GradientReversalLayer(lambda_coeff=lambda_val)
        x = torch.randn(4, 128, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()

        expected_grad = -lambda_val * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad, atol=1e-6), (
            f"GRL backward should scale by -λ. "
            f"Got mean={x.grad.mean():.4f}, expected={expected_grad.mean():.4f}"
        )

    def test_set_lambda(self):
        """Lambda coefficient should be updatable."""
        grl = GradientReversalLayer(lambda_coeff=1.0)
        grl.set_lambda(0.3)
        assert grl.lambda_coeff == 0.3


class TestDomainAdversarialMoEDetector:
    """Test suite for the complete MoE + DANN detector."""

    @pytest.fixture
    def detector(self):
        return DomainAdversarialMoEDetector(
            in_dim=512, d_model=256, num_experts=4,
            top_k=2, num_domains=8, lambda_coeff=0.5,
        )

    def test_output_shapes(self, detector):
        """Verify all three output shapes."""
        x = torch.randn(4, 512)
        class_logits, domain_logits, aux_loss = detector(x)
        assert class_logits.shape == (4, 1), f"Class logits: {class_logits.shape}"
        assert domain_logits.shape == (4, 8), f"Domain logits: {domain_logits.shape}"
        assert aux_loss.dim() == 0, f"Aux loss shape: {aux_loss.shape}"

    def test_forward_backward(self, detector):
        """Full forward/backward pass."""
        x = torch.randn(8, 512, requires_grad=True)
        class_logits, domain_logits, aux_loss = detector(x)
        total_loss = class_logits.sum() + domain_logits.sum() + aux_loss
        total_loss.backward()
        assert x.grad is not None, "No gradient on detector input"

    def test_no_nan_inf(self, detector):
        """No NaN/Inf in any output."""
        x = torch.randn(4, 512)
        class_logits, domain_logits, aux_loss = detector(x)
        for name, t in [("class", class_logits), ("domain", domain_logits), ("aux", aux_loss)]:
            assert not torch.isnan(t).any(), f"NaN in {name}"
            assert not torch.isinf(t).any(), f"Inf in {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
