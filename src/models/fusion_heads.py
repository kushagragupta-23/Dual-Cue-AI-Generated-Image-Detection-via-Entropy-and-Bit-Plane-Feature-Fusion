import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialCrossAttentionHead(nn.Module):
    """
    Head 1 & 2: Spatial Cross-Attention.
    Queries from modality A attend to Keys/Values from modality B.
    """
    def __init__(self, in_channels: int, dim: int = 512):
        super().__init__()
        self.q_conv = nn.Conv2d(in_channels, dim, 1)
        self.k_conv = nn.Conv2d(in_channels, dim, 1)
        self.v_conv = nn.Conv2d(in_channels, dim, 1)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_feat: torch.Tensor, key_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = query_feat.shape
        orig_dtype = query_feat.dtype
        
        Q = self.q_conv(query_feat).view(B, -1, H * W).permute(0, 2, 1) # (B, N, dim)
        K = self.k_conv(key_feat).view(B, -1, H * W)                    # (B, dim, N)
        V = self.v_conv(key_feat).view(B, -1, H * W).permute(0, 2, 1)   # (B, N, dim)
        
        # Upcast everything to float32 for numerically stable attention
        Q_f32, K_f32, V_f32 = Q.float(), K.float(), V.float()
        attn = (Q_f32 @ K_f32) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ V_f32).to(orig_dtype)
        
        out = out.permute(0, 2, 1).view(B, -1, H, W)
        
        # Pool to feature vector
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.norm(out.float()).to(orig_dtype)
        return out


class ChannelSEFusionHead(nn.Module):
    """
    Head 3: Channel Squeeze-Excitation Fusion.
    Learns which channels across both modalities are most informative.
    """
    def __init__(self, in_channels_mlep: int, in_channels_lota: int, dim: int = 512):
        super().__init__()
        total_channels = in_channels_mlep + in_channels_lota
        self.se = nn.Sequential(
            nn.Linear(total_channels, total_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(total_channels // 4, total_channels),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(total_channels, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_mlep: torch.Tensor, feat_lota: torch.Tensor) -> torch.Tensor:
        orig_dtype = feat_mlep.dtype
        # Concatenate on channel dim
        f_cat = torch.cat([feat_mlep, feat_lota], dim=1) # (B, C1+C2, H, W)
        
        # Squeeze
        z = F.adaptive_avg_pool2d(f_cat, 1).flatten(1)
        
        # Excite
        s = self.se(z)
        
        # Recalibrate
        f_calibrated = f_cat * s.unsqueeze(-1).unsqueeze(-1)
        
        # Pool and project
        out = F.adaptive_avg_pool2d(f_calibrated, 1).flatten(1)
        out = self.proj(out)
        out = self.norm(out.float()).to(orig_dtype)
        return out


class FrequencyCorrelationHead(nn.Module):
    """
    Head 4: Frequency-Domain Cross-Correlation Fusion.
    """
    def __init__(self, in_channels_mlep: int, in_channels_lota: int, dim: int = 512):
        super().__init__()
        self.proj_mlep = nn.Conv2d(in_channels_mlep, dim, 1)
        self.proj_lota = nn.Conv2d(in_channels_lota, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_mlep: torch.Tensor, feat_lota: torch.Tensor) -> torch.Tensor:
        orig_dtype = feat_mlep.dtype
        pm = self.proj_mlep(feat_mlep)
        pl = self.proj_lota(feat_lota)
        
        # Cast to float32 for FFT (ComplexHalf is experimental and buggy)
        pm_f32 = pm.float()
        pl_f32 = pl.float()
        
        # Normalize prior to FFT to keep spectrum bounded
        pm_f32 = (pm_f32 - pm_f32.mean(dim=(-2, -1), keepdim=True)) / (pm_f32.std(dim=(-2, -1), keepdim=True) + 1e-5)
        pl_f32 = (pl_f32 - pl_f32.mean(dim=(-2, -1), keepdim=True)) / (pl_f32.std(dim=(-2, -1), keepdim=True) + 1e-5)

        # To Frequency Domain
        fm = torch.fft.rfft2(pm_f32, norm='ortho')
        fl = torch.fft.rfft2(pl_f32, norm='ortho')
        
        # Cross-Correlation (Multiplication with complex conjugate)
        corr = fm * torch.conj(fl)
        
        # Back to Spatial
        out = torch.fft.irfft2(corr, s=(pm.shape[2], pm.shape[3]), norm='ortho')
        
        # Clamp extreme values before casting back
        out = out.clamp(-10.0, 10.0)
        out = out.to(orig_dtype)
        
        # Pool
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.norm(out.float()).to(orig_dtype)
        return out

class MultiHeadFusionModule(nn.Module):
    """Encapsulates all 4 heads."""
    def __init__(self, channels_mlep=1024, channels_lota=1024, dim=512):
        super().__init__()
        self.head1 = SpatialCrossAttentionHead(channels_mlep, dim) # MLEP -> LOTA
        self.head2 = SpatialCrossAttentionHead(channels_lota, dim) # LOTA -> MLEP
        self.head3 = ChannelSEFusionHead(channels_mlep, channels_lota, dim)
        self.head4 = FrequencyCorrelationHead(channels_mlep, channels_lota, dim)
        
    def forward(self, f_mlep, f_lota):
        h1 = self.head1(f_mlep, f_lota)
        h2 = self.head2(f_lota, f_mlep)
        h3 = self.head3(f_mlep, f_lota)
        h4 = self.head4(f_mlep, f_lota)
        return torch.stack([h1, h2, h3, h4], dim=1) # (B, 4, dim)


class PyramidCrossAttentionModule(nn.Module):
    """
    Multi-Granularity Cross-Attention Module (MGA-Net).
    Performs spatial cross-attention between MLEP multi-scale entropy pyramids
    and LOTA LSB quantization noise maps without losing spatial coordinate geometry.
    """
    def __init__(self, in_channels_mlep=1024, in_channels_lota=1024, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.proj_q = nn.Conv2d(in_channels_mlep, d_model, kernel_size=1, bias=False)
        self.proj_k = nn.Conv2d(in_channels_lota, d_model, kernel_size=1, bias=False)
        self.proj_v = nn.Conv2d(in_channels_lota, d_model, kernel_size=1, bias=False)
        
        self.norm_q = nn.GroupNorm(8, d_model)
        self.norm_k = nn.GroupNorm(8, d_model)
        
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.GroupNorm(8, d_model)

    def forward(self, feat_mlep: torch.Tensor, feat_lota: torch.Tensor) -> torch.Tensor:
        B, C_m, H, W = feat_mlep.shape
        _, C_l, H_l, W_l = feat_lota.shape
        
        if (H != H_l) or (W != W_l):
            feat_lota = F.interpolate(feat_lota, size=(H, W), mode='bilinear', align_corners=False)
            
        q_map = self.norm_q(self.proj_q(feat_mlep)) # (B, d_model, H, W)
        k_map = self.norm_k(self.proj_k(feat_lota)) # (B, d_model, H, W)
        v_map = self.proj_v(feat_lota)              # (B, d_model, H, W)
        
        q_flat = q_map.flatten(2).permute(0, 2, 1) # (B, H*W, d_model)
        k_flat = k_map.flatten(2).permute(0, 2, 1) # (B, H*W, d_model)
        v_flat = v_map.flatten(2).permute(0, 2, 1) # (B, H*W, d_model)
        
        # Upcast for stable attention calculation
        q_f32, k_f32, v_f32 = q_flat.float(), k_flat.float(), v_flat.float()
        attn_out, _ = self.attn(q_f32, k_f32, v_f32)
        attn_out = attn_out.to(feat_mlep.dtype)
        
        out = attn_out.permute(0, 2, 1).view(B, self.d_model, H, W) + q_map
        out = self.out_norm(out)
        return out

