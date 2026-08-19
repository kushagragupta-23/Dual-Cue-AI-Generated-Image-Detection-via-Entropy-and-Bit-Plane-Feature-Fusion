"""
Supervised Contrastive Loss for Dual-Cue MLEP & LOTA Feature Alignment.

Pulls Real image embeddings together and Fake embeddings together across both
entropy and bit-plane modalities on the L2 unit hypersphere.

Mathematical formulation (Khosla et al., NeurIPS 2020):
    L_SupCon = Σ_i (-1/|P(i)|) Σ_{p∈P(i)} log[ exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ]

where:
    - τ is the temperature scaling parameter (default 0.07)
    - P(i) = positive pairs sharing the same label (excluding self)
    - A(i) = all indices except self

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualCueSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for Dual-Cue MLEP & LOTA Feature Alignment.

    Pulls Real image embeddings together and Fake embeddings together across both
    entropy and bit-plane modalities on the L2 unit hypersphere.

    Args:
        temperature: Contrastive temperature τ (default 0.07).
        base_temperature: Base temperature for loss scaling (default 0.07).
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute SupCon loss over dual-view embeddings.

        Args:
            features: Projection tensor of shape (B, 2, D), where
                      features[:, 0] is MLEP embedding and features[:, 1] is LOTA embedding.
            labels: Ground truth binary labels of shape (B,).

        Returns:
            torch.Tensor: Scalar SupCon loss value.
        """
        device = features.device
        B, n_views, D = features.shape

        # L2-normalize embeddings to unit hypersphere
        features = F.normalize(features, p=2, dim=-1)

        # Flatten: (B * 2, D)
        flat_features = features.reshape(B * n_views, D)

        # Replicate labels for both views: (B * 2, 1)
        flat_labels = labels.view(-1, 1).repeat(1, n_views).reshape(-1, 1)

        # Compute pairwise cosine similarity: (B*2, B*2)
        sim_matrix = torch.matmul(flat_features, flat_features.T) / self.temperature

        # Numerical stability: subtract max per row
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()

        # Positive pair mask: same label (B*2, B*2)
        label_mask = torch.eq(flat_labels, flat_labels.T).float()

        # Self-contrast mask: diagonal = 0
        n = B * n_views
        self_mask = 1.0 - torch.eye(n, device=device)
        mask_positives = label_mask * self_mask

        # Compute log probabilities
        exp_sim = torch.exp(sim_matrix) * self_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        pos_count = mask_positives.sum(dim=1)
        mean_log_prob_pos = (mask_positives * log_prob).sum(dim=1) / (pos_count + 1e-8)

        # Temperature scaling and final loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        return loss.mean()


__all__ = ["DualCueSupConLoss"]
