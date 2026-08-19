"""
Architecture II Modules: Pyramid Cross-Attention Module.

Performs spatial cross-attention between MLEP and LOTA feature maps:
    Q = Conv1×1(MLEP_features) → GroupNorm
    K = Conv1×1(LOTA_features) → GroupNorm
    V = Conv1×1(LOTA_features)
    Attn = Softmax(Q^T K / √d_model)
    H_cross = Reshape(Attn · V) + Q  (residual)
    Output = GroupNorm(H_cross + FFN(H_cross))  (FFN residual)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger("arch2_mganet.modules")


class PyramidCrossAttentionModule(nn.Module):
    """
    Multi-Granularity Cross-Attention Module (MGA-Net).

    Performs spatial cross-attention between MLEP and LOTA feature maps.
    The attention affinity matrix A ∈ R^{N×N} (N = H×W) captures which MLEP
    entropy anomaly locations spatially coincide with LOTA noise spikes.

    Args:
        in_channels_mlep: Channel dim of MLEP spatial features (e.g., 1024 from layer3).
        in_channels_lota: Channel dim of LOTA spatial features (e.g., 1024 from layer3).
        d_model: Shared latent attention dimension (default 256).
        num_heads: Number of attention heads (default 8). Must divide d_model.
        dropout: Dropout rate for attention and FFN (default 0.1).
    """

    def __init__(
        self,
        in_channels_mlep: int = 1024,
        in_channels_lota: int = 1024,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        # 1×1 Convolutional Projections to shared latent dimension
        self.proj_q = nn.Conv2d(in_channels_mlep, d_model, kernel_size=1, bias=False)
        self.proj_k = nn.Conv2d(in_channels_lota, d_model, kernel_size=1, bias=False)
        self.proj_v = nn.Conv2d(in_channels_lota, d_model, kernel_size=1, bias=False)

        # GroupNorm across feature channels for Q and K
        num_groups = min(8, d_model)
        self.norm_q = nn.GroupNorm(num_groups, d_model)
        self.norm_k = nn.GroupNorm(num_groups, d_model)

        # Multi-Head Attention mechanism
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-Forward Network (FFN) with Residual Connection
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
        )

        self.out_norm = nn.GroupNorm(num_groups, d_model)

        logger.info(
            f"PyramidCrossAttentionModule: MLEP({in_channels_mlep}) × LOTA({in_channels_lota}) "
            f"→ d_model={d_model}, heads={num_heads}"
        )

    def forward(self, feat_mlep: torch.Tensor, feat_lota: torch.Tensor) -> torch.Tensor:
        """
        Execute spatial cross-modal attention.

        Args:
            feat_mlep: Spatial feature map from MLEP stem, shape (B, C_m, H, W).
            feat_lota: Spatial feature map from LOTA stem, shape (B, C_l, H_l, W_l).

        Returns:
            torch.Tensor: Spatio-modally fused feature map of shape (B, d_model, H, W).
        """
        B, C_m, H, W = feat_mlep.shape
        _, C_l, H_l, W_l = feat_lota.shape

        # Ensure spatial dimensions match via bilinear interpolation if necessary
        if (H != H_l) or (W != W_l):
            feat_lota = F.interpolate(
                feat_lota, size=(H, W), mode="bilinear", align_corners=False
            )

        # 1. Project to shared latent dimension d_model
        q_map = self.norm_q(self.proj_q(feat_mlep))  # (B, d_model, H, W)
        k_map = self.norm_k(self.proj_k(feat_lota))  # (B, d_model, H, W)
        v_map = self.proj_v(feat_lota)                # (B, d_model, H, W)

        # 2. Flatten spatial dimensions for sequence attention: (B, H*W, d_model)
        q_seq = q_map.flatten(2).transpose(1, 2)
        k_seq = k_map.flatten(2).transpose(1, 2)
        v_seq = v_map.flatten(2).transpose(1, 2)

        # 3. Spatial Cross-Attention: MLEP queries attend to LOTA keys/values
        attn_out, attn_weights = self.attn(
            query=q_seq, key=k_seq, value=v_seq
        )  # (B, H*W, d_model)

        # 4. Reshape back to spatial map and apply residual connection
        attn_map = attn_out.transpose(1, 2).reshape(B, self.d_model, H, W)
        h_res = q_map + attn_map

        # 5. FFN refinement with residual
        out = self.out_norm(h_res + self.ffn(h_res))

        return out


__all__ = ["PyramidCrossAttentionModule"]
