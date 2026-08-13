import torch
import torch.nn as nn
import torch.nn.functional as F

class DualCueSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for cross-modal alignment.
    Aligns MLEP and LOTA embeddings on an L2 hypersphere based on class labels.
    
    ALL computations forced to float32 to prevent FP16 overflow:
    At temperature=0.07, sim/temp values reach ~14, and exp(14)≈1.2M > FP16 max (65504).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, p_mlep: torch.Tensor, p_lota: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        p_mlep, p_lota: (B, D) L2-normalized embeddings
        labels: (B,) class labels (0=Real, 1=Fake)
        """
        device = p_mlep.device
        batch_size = p_mlep.shape[0]
        
        # ── Force float32 for entire loss computation ──
        features = torch.cat([p_mlep.float(), p_lota.float()], dim=0)  # (2B, D)
        
        # Re-normalize after cast (guards against any rounding drift)
        features = F.normalize(features, dim=1)
        
        # Expand labels: (2B,)
        labels = labels.contiguous().view(-1, 1)
        labels = torch.cat([labels, labels], dim=0)
        
        # Mask of positive pairs
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute similarities (float32 matmul → no overflow)
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature
        
        # Numerical stability: subtract max
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob with clamped denominator
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp(min=1e-8))
        
        # Compute mean of log-likelihood over positive (guard zero-div)
        pos_count = mask.sum(1).clamp(min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count
        
        # Loss
        loss = -mean_log_prob_pos
        return loss.mean()
