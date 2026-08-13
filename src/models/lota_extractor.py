import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKLOTAExtractor(nn.Module):
    """
    LOw-biT pAtch (LOTA) Extractor — DIFFERENTIABLE version.
    
    Original LOTA uses uint8 cast + bitwise AND + binary thresholding,
    all of which are non-differentiable and sever gradient flow.
    
    This version uses:
      1. Soft bit extraction via learned 1x1 convolutions (replaces uint8 + bitwise)
      2. Sigmoid-based soft thresholding (replaces hard binary mask)
      3. Differentiable top-k via soft attention (replaces hard index selection)
    
    The key insight: we don't need to literally extract LSB planes.
    We need a learned transformation that captures the same *information* as LSB
    patterns — namely, low-amplitude noise patterns that differ between real and
    forged images. A 1x1 conv with small initialization naturally learns to
    amplify these subtle signals.
    """
    def __init__(self, k_patches=4, patch_size=32, grid_size=8):
        super().__init__()
        self.k_patches = k_patches
        self.patch_size = patch_size
        self.grid_size = grid_size
        
        # Learnable "LSB-like" feature extractor: 3 → 3 channels
        # Small init ensures it starts by looking at low-amplitude patterns
        self.lsb_conv = nn.Conv2d(3, 3, kernel_size=1, bias=True)
        nn.init.normal_(self.lsb_conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lsb_conv.bias)
        
        # Gradient scoring network (replaces non-differentiable MGPS)
        self.score_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(grid_size),  # → (B, 16, 8, 8)
            nn.Conv2d(16, 1, kernel_size=1),  # → (B, 1, 8, 8) = 64 patch scores
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Step 1: Differentiable "LSB extraction"
        # Small-weight conv naturally amplifies low-amplitude noise patterns
        lsb_features = self.lsb_conv(x)  # (B, 3, H, W)
        
        # Step 2: Compute patch importance scores (differentiable MGPS)
        scores = self.score_net(lsb_features)  # (B, 1, 8, 8)
        scores_flat = scores.view(B, -1)       # (B, 64)
        
        # Step 3: Soft top-k via temperature-scaled softmax
        # High temperature → uniform; low temperature → hard top-k
        # Use temperature=0.1 for relatively sharp selection
        soft_weights = F.softmax(scores_flat / 0.1, dim=-1)  # (B, 64)
        
        # Step 4: Create spatial attention mask from soft weights
        # Reshape to (B, 1, 8, 8) and upsample to full resolution
        mask = soft_weights.view(B, 1, self.grid_size, self.grid_size)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')  # (B, 1, H, W)
        
        # Step 5: Apply mask to LSB features
        # Normalize mask to preserve scale (multiply by num_patches / k)
        mask = mask * (self.grid_size * self.grid_size / self.k_patches)
        
        output = lsb_features * mask  # (B, 3, H, W)
        return output
