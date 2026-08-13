"""Quick GPU sanity check for HydraFusion model forward pass."""
import torch
import sys
sys.path.insert(0, '.')
from src.models.hydrafusion_net import HydraFusionNet

device = torch.device('cuda')
model = HydraFusionNet().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"GPU memory after model load: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

# Test Stage 1 forward pass
x = torch.randn(2, 3, 256, 256, device=device) * 255.0
with torch.amp.autocast('cuda', dtype=torch.float16):
    p_mlep, p_lota = model(x, stage=1)
print(f"Stage 1 OK: p_mlep={p_mlep.shape}, p_lota={p_lota.shape}")

# Test Stage 2 forward pass
with torch.amp.autocast('cuda', dtype=torch.float16):
    logits, domain_logits, alpha = model(x, stage=2)
print(f"Stage 2 OK: logits={logits.shape}, domain={domain_logits.shape}, alpha={alpha.shape}")
print(f"GPU memory peak: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
print("ALL CHECKS PASSED ✓")
