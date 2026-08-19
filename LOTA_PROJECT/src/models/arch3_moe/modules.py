"""
Architecture III Modules: Expert, MoE Router, Gradient Reversal, Domain Discriminator.

Implements:
    1. ExpertModule — Lightweight 2-layer residual bottleneck expert.
    2. SparseMoEForensicModule — 4-expert Sparse MoE with Top-2 noisy gating.
    3. GradientReversalFunction / GradientReversalLayer — Identity forward, -λ backward.
    4. DomainDiscriminator — Domain classification head for adversarial training.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from src.utils.logger import get_logger

logger = get_logger("arch3_moe.modules")


class ExpertModule(nn.Module):
    """
    Lightweight 2-layer residual bottleneck expert.

    Architecture: Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm + Residual

    Args:
        in_dim: Input and output feature dimension (default 512).
        hidden_dim: Bottleneck hidden dimension (default 256).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual expert computation: x + f(x)."""
        return x + self.net(x)


class SparseMoEForensicModule(nn.Module):
    """
    4-Expert Sparse Mixture-of-Experts with Top-2 Dynamic Routing.

    Gating formulation:
        G(h) = Softmax(TopK(W_g·h + ε·Softplus(W_noise·h), k=2))
        z_MoE = Σ_{i∈Top-2} G_i(h) · E_i(h)

    Load-balancing auxiliary loss (Shazeer et al., 2017):
        L_aux = N · Σ_i f_i · P_i

    Args:
        d_model: Feature dimension for experts (default 512).
        num_experts: Number of domain-specific experts (default 4).
        top_k: Number of experts activated per sample (default 2).
    """

    def __init__(self, d_model: int = 512, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Instantiate domain-specific experts
        self.experts = nn.ModuleList(
            [ExpertModule(in_dim=d_model) for _ in range(num_experts)]
        )

        # Gating router projections
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)
        self.w_noise = nn.Linear(d_model, num_experts, bias=False)

        logger.info(
            f"SparseMoEForensicModule: {num_experts} experts, Top-{top_k} routing, "
            f"d_model={d_model}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse Top-K expert routing.

        Args:
            x: Input feature tensor of shape (B, d_model).

        Returns:
            Tuple of (fused_out, aux_loss).
        """
        B, D = x.shape

        # 1. Compute routing logits with training exploration noise
        clean_logits = self.w_gate(x)
        if self.training:
            noise_std = F.softplus(self.w_noise(x))
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            noisy_logits = clean_logits

        # 2. Select Top-K experts per sample
        topk_logits, topk_indices = torch.topk(noisy_logits, k=self.top_k, dim=-1)

        # 3. Softmax strictly over selected Top-K experts
        topk_weights = F.softmax(topk_logits, dim=-1)

        # 4. Execute selected experts and accumulate weighted outputs
        fused_out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i)
            sample_indices, weight_positions = torch.where(mask)

            if len(sample_indices) > 0:
                expert_inputs = x[sample_indices]
                expert_outputs = expert(expert_inputs)
                routing_weights = topk_weights[sample_indices, weight_positions].unsqueeze(-1)
                fused_out.index_add_(0, sample_indices, expert_outputs * routing_weights)

        # 5. Load-Balancing Auxiliary Loss
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_usage[i] = (topk_indices == i).float().sum() / (B * self.top_k)
        gate_probs_mean = F.softmax(clean_logits, dim=-1).mean(dim=0)
        aux_loss = self.num_experts * torch.sum(expert_usage * gate_probs_mean)

        return fused_out, aux_loss


class GradientReversalFunction(Function):
    """Autograd Function: forward = identity, backward = -λ·gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_coeff: float) -> torch.Tensor:
        ctx.lambda_coeff = lambda_coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_coeff, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) for Domain-Adversarial Feature Alignment.

    Forward: identity. Backward: -λ·gradient.

    Args:
        lambda_coeff: Gradient reversal scaling factor (default 1.0).
    """

    def __init__(self, lambda_coeff: float = 1.0):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def set_lambda(self, lambda_coeff: float) -> None:
        """Update the reversal coefficient."""
        self.lambda_coeff = lambda_coeff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_coeff)


class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator Head for adversarial generator identity classification.

    Args:
        in_dim: Input feature dimension (default 512).
        hidden_dim: Hidden layer dimension (default 256).
        num_domains: Number of generator domain classes (default 8).
        dropout: Dropout rate (default 0.3).
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 256,
        num_domains: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = [
    "ExpertModule",
    "SparseMoEForensicModule",
    "GradientReversalFunction",
    "GradientReversalLayer",
    "DomainDiscriminator",
]
