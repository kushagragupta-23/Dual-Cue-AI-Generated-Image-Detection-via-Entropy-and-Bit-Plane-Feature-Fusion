"""
Unit tests for the LOTA preprocessing pipeline, MGPS scoring, and Top-K extraction.
"""

import pytest
import torch
from src.models.lota import TopKLOTAExtractor


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
    """Verify LSB formula z = 4*x2 + 2*x1 + x0 and binarized threshold normalization."""
    extractor = TopKLOTAExtractor(bit_planes=[0, 1, 2], threshold_val=255.0)
    
    # Create synthetic input tensor with known bit patterns
    # Pixel 0: val=0 -> LSB=0 -> z=0 -> norm=0.0
    # Pixel 1: val=1 -> LSB=1 -> z=1 -> norm=255.0
    # Pixel 2: val=8 (binary 1000) -> LSB=0 -> z=0 -> norm=0.0
    # Pixel 3: val=7 (binary 0111) -> LSB=7 -> z=7 -> norm=255.0
    x_vals = torch.tensor([[[[0.0, 1.0], [8.0, 7.0]]]], dtype=torch.float32) # (1, 1, 2, 2)
    x = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    x[:, :, 0:2, 0:2] = x_vals
    
    z_norm = extractor._extract_lsb_threshold(x)
    
    assert z_norm[0, 0, 0, 0].item() == 0.0
    assert z_norm[0, 0, 0, 1].item() == 255.0
    assert z_norm[0, 0, 1, 0].item() == 0.0
    assert z_norm[0, 0, 1, 1].item() == 255.0


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


def test_topk_quadrant_diversity():
    """Verify that selected K=4 indices belong to 4 distinct spatial quadrants."""
    extractor = TopKLOTAExtractor(k_patches=4, grid_size=8)
    
    # Create artificial score vector where multiple peaks exist in quadrant 0
    # but quadrant diverse selection forces picking 1 peak per quadrant
    scores = torch.zeros((1, 64), dtype=torch.float32)
    
    # Quadrant 0 (rows 0-3, cols 0-3): set indices 0 and 1 very high
    scores[0, 0] = 100.0
    scores[0, 1] = 90.0
    
    # Quadrant 1 (rows 0-3, cols 4-7): set index 4 high
    scores[0, 4] = 50.0
    
    # Quadrant 2 (rows 4-7, cols 0-3): set index 32 high
    scores[0, 32] = 60.0
    
    # Quadrant 3 (rows 4-7, cols 4-7): set index 36 high
    scores[0, 36] = 70.0
    
    indices = extractor._select_topk_quadrant_diverse(scores)  # (1, 4)
    selected = set(indices[0].tolist())
    
    # Despite index 1 having a higher score (90) than indices 4, 32, 36,
    # quadrant diversity must select exactly one from each quadrant: {0, 4, 32, 36}
    assert selected == {0, 4, 32, 36}


def test_forward_pipeline_output_shapes():
    """Verify all returned output shapes and data types from forward pass."""
    extractor = TopKLOTAExtractor(k_patches=4, patch_size=32, grid_size=8)
    x = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)
    
    out = extractor(x)
    
    assert isinstance(out, dict)
    assert out["z_norm"].shape == (2, 3, 256, 256)
    assert out["mgps_scores"].shape == (2, 64)
    assert out["topk_indices"].shape == (2, 4)
    assert out["topk_patches"].shape == (2, 4, 3, 32, 32)
    assert out["noise_tensor"].shape == (2, 12, 32, 32)
