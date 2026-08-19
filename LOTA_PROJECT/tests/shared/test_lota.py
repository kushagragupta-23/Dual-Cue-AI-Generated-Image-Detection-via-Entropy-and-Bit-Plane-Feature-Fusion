"""
Unit tests for the LOTA preprocessing pipeline, MGPS scoring, and Top-K extraction.
"""

import pytest
import torch
from src.shared.extractors.lota import TopKLOTAExtractor


def test_bit_plane_reconstruction():
    """Verify that summing 2^k * x_k across k=0..7 exactly reconstructs original uint8 tensor."""
    extractor = TopKLOTAExtractor()
    # Create random integer tensor in [0, 255]
    x = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.float32)
    
    planes = extractor.extract_all_bit_planes(x)  # Shape: (2, 3, 8, 32, 32)
    assert planes.shape == (2, 3, 8, 32, 32)
    
    # Reconstruct integer values
    reconstructed = torch.zeros_like(x, dtype=torch.int32)
    for k in range(8):
        reconstructed = reconstructed + (planes[:, :, k, :, :].to(torch.int32) << k)
        
    assert torch.equal(reconstructed, x.to(torch.int32))


def test_lsb_composition_and_thresholding():
    """Verify LSB formula z = 4*x2 + 2*x1 + x0 and LOTA-scl min-max normalization."""
    extractor = TopKLOTAExtractor(bit_planes=[0, 1, 2], threshold_val=255.0)
    
    # Create synthetic input tensor with known bit patterns
    # Pixel 0 (0,0): val=0 -> LSB=0 -> z=0 -> min
    # Pixel 1 (0,1): val=1 -> LSB=1 -> z=1
    # Pixel 2 (1,0): val=8 (binary 1000) -> LSB=0 -> z=0 -> min
    # Pixel 3 (1,1): val=7 (binary 0111) -> LSB=7 -> z=7 -> max
    x_vals = torch.tensor([[[[0.0, 1.0], [8.0, 7.0]]]], dtype=torch.float32) # (1, 1, 2, 2)
    x = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    x[:, :, 0:2, 0:2] = x_vals
    
    z_norm = extractor._extract_lsb_threshold(x)
    
    # z_min=0, z_max=7.
    # z_norm for pixel (0,0): 255 * 0/7 = 0.0
    # z_norm for pixel (1,1): 255 * 7/7 = 255.0
    assert torch.isclose(z_norm[0, 0, 0, 0], torch.tensor(0.0), atol=1e-3)
    assert torch.isclose(z_norm[0, 0, 1, 1], torch.tensor(255.0), atol=1e-3)


def test_mgps_scoring_on_flat_and_edge_images():
    """Verify that flat noise maps yield 0 divergence and textured quadrants yield high scores."""
    extractor = TopKLOTAExtractor(grid_size=8, patch_size=32)
    
    # Flat image -> all gradients zero -> all scores zero
    flat_img = torch.ones((1, 3, 256, 256), dtype=torch.float32) * 255.0
    scores_flat = extractor._compute_mgps_scores(flat_img)
    assert torch.allclose(scores_flat, torch.zeros_like(scores_flat))
    
    # Create textured noise map in top-left quadrant patch 0 (rows 0..31, cols 0..31)
    textured_img = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    # Checkerboard edge pattern in patch 0
    textured_img[0, :, 0:32:2, 0:32:2] = 255.0
    
    scores_textured = extractor._compute_mgps_scores(textured_img)
    assert scores_textured[0, 0].item() > 0.0  # Patch 0 should have high divergence score
    assert scores_textured[0, 63].item() == 0.0  # Bottom-right patch should be 0.0


def test_max_gradient_patch_selection():
    """Verify that selected patch index corresponds to maximum MGPS score (Eq 6 in paper)."""
    extractor = TopKLOTAExtractor(grid_size=8)
    
    scores = torch.zeros((1, 64), dtype=torch.float32)
    scores[0, 42] = 150.0  # Set maximum score at patch index 42
    
    top1_idx = extractor._select_max_gradient_patch(scores)  # (1, 1)
    assert top1_idx.shape == (1, 1)
    assert top1_idx[0, 0].item() == 42


def test_forward_pipeline_output_shapes():
    """Verify all returned output shapes and data types from forward pass."""
    extractor = TopKLOTAExtractor(patch_size=32, grid_size=8)
    x = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)
    
    out = extractor(x)
    
    assert isinstance(out, dict)
    assert out["z_norm"].shape == (2, 3, 256, 256)
    assert out["mgps_scores"].shape == (2, 64)
    assert out["top1_index"].shape == (2, 1)
    assert out["top1_patch"].shape == (2, 1, 3, 32, 32)
    assert out["noise_tensor"].shape == (2, 3, 256, 256)

