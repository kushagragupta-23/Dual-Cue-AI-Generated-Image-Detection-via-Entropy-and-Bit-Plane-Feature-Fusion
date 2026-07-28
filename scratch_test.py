import torch

def shuffle_patches(x, patch_size=2, macro_window_size=16, seed=42):
    B, C, H, W = x.shape
    L = patch_size
    M = macro_window_size
    
    macro_grid_h = H // M
    macro_grid_w = W // M
    micro_grid_h = M // L
    micro_grid_w = M // L
    num_micro_patches = micro_grid_h * micro_grid_w
    
    x_macro = x.view(B, C, macro_grid_h, micro_grid_h, L, macro_grid_w, micro_grid_w, L)
    x_micro = x_macro.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
    x_micro = x_micro.view(B, C, macro_grid_h, macro_grid_w, num_micro_patches, L, L)
    
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    
    perm = torch.randperm(num_micro_patches, generator=generator, device=x.device)
    
    x_shuffled = x_micro[:, :, :, :, perm, :, :]
    
    x_shuffled = x_shuffled.view(B, C, macro_grid_h, macro_grid_w, micro_grid_h, micro_grid_w, L, L)
    x_shuffled = x_shuffled.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
    x_shuffled = x_shuffled.view(B, C, H, W)
    
    return x_shuffled

x = torch.arange(256 * 256, dtype=torch.float32).view(1, 1, 256, 256)
x_out = shuffle_patches(x)
print("Output shape:", x_out.shape)

orig_macro_0_0 = x[0, 0, 0:16, 0:16].flatten().sort()[0]
shuf_macro_0_0 = x_out[0, 0, 0:16, 0:16].flatten().sort()[0]
print("Macro Window 0,0 preserved:", torch.all(orig_macro_0_0 == shuf_macro_0_0).item())

orig_macro_0_1 = x[0, 0, 0:16, 16:32].flatten().sort()[0]
shuf_macro_0_1 = x_out[0, 0, 0:16, 16:32].flatten().sort()[0]
print("Macro Window 0,1 preserved:", torch.all(orig_macro_0_1 == shuf_macro_0_1).item())
