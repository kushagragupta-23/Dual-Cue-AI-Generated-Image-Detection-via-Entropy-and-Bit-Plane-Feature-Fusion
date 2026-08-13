import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGatingRouter(nn.Module):
    """
    Learns to dynamically route and weight the 4 fusion heads based on the 
    global context of the image modalities.
    """
    def __init__(self, in_channels_mlep=1024, in_channels_lota=1024, num_heads=4):
        super().__init__()
        total_in = in_channels_mlep + in_channels_lota
        
        self.router = nn.Sequential(
            nn.Linear(total_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_heads)
        )
        
    def forward(self, f_mlep: torch.Tensor, f_lota: torch.Tensor, head_outputs: torch.Tensor) -> torch.Tensor:
        """
        head_outputs: shape (B, 4, dim)
        returns: (B, dim) fused representation
        """
        B = f_mlep.shape[0]
        
        # Global context vector
        z_mlep = F.adaptive_avg_pool2d(f_mlep, 1).flatten(1)
        z_lota = F.adaptive_avg_pool2d(f_lota, 1).flatten(1)
        
        z_cat = torch.cat([z_mlep, z_lota], dim=1)
        
        # Predict weights alpha
        logits = self.router(z_cat)
        alpha = F.softmax(logits, dim=-1) # (B, 4)
        
        # Weighted sum of head outputs
        # head_outputs: (B, 4, dim)
        # alpha: (B, 4) -> (B, 4, 1)
        fused = (head_outputs * alpha.unsqueeze(-1)).sum(dim=1) # (B, dim)
        
        return fused, alpha


class ExpertModule(nn.Module):
    """A lightweight 2-layer residual bottleneck expert for MoE architecture."""
    def __init__(self, in_dim=512, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SparseMoEForensicModule(nn.Module):
    """
    4-Expert Sparse Mixture-of-Experts (MoE) with Top-2 Dynamic Routing and Load-Balancing Aux Loss.
    Routes image representations across Micro-Entropy, Macro-Entropy, LSB-Steganalysis,
    and Wavelet/DCT frequency experts to maximize zero-shot generalization.
    """
    def __init__(self, d_model=512, num_experts=4, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([ExpertModule(in_dim=d_model) for _ in range(num_experts)])
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)
        self.w_noise = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        B, D = x.shape
        clean_logits = self.w_gate(x)
        if self.training:
            noise_std = F.softplus(self.w_noise(x))
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            noisy_logits = clean_logits
            
        topk_logits, topk_indices = torch.topk(noisy_logits, k=self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        
        fused_out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i)
            sample_indices, weight_positions = torch.where(mask)
            
            if len(sample_indices) > 0:
                expert_inputs = x[sample_indices]
                expert_outputs = expert(expert_inputs)
                routing_weights = topk_weights[sample_indices, weight_positions].unsqueeze(-1)
                fused_out.index_add_(0, sample_indices, expert_outputs * routing_weights)
                
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_usage[i] = (topk_indices == i).float().sum() / (B * self.top_k)
        gate_probs_mean = F.softmax(clean_logits, dim=-1).mean(dim=0)
        aux_loss = self.num_experts * torch.sum(expert_usage * gate_probs_mean)
        
        return fused_out, aux_loss

