"""
Unit tests for the MLEP preprocessing pipeline, patch shuffling, pyramid, and entropy extraction.
"""

import pytest
import torch
from src.models.mlep import MLEPExtractor


def test_patch_shuffling_preserves_pixel_values():
    """Verify that shuffling rearranges spatial positions but preserves all pixel values."""
    extractor = MLEPExtractor(patch_size=2, seed=42)
    x = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)

    x_shuffled = extractor.shuffle_patches(x)
    assert x_shuffled.shape == x.shape

    # All original pixel values must still exist (multiset equality per channel)
    for b in range(2):
        for c in range(3):
            orig_sorted = x[b, c].flatten().sort()[0]
            shuf_sorted = x_shuffled[b, c].flatten().sort()[0]
            assert torch.equal(orig_sorted, shuf_sorted)


def test_multiscale_pyramid_shapes():
    """Verify multi-scale pyramid outputs correct concatenated channel dimensions."""
    extractor = MLEPExtractor(scales=[1.0, 0.5, 0.25])
    x = torch.rand((2, 3, 256, 256), dtype=torch.float32) * 255.0

    pyramid = extractor.build_multiscale_pyramid(x)

    # 3 scales × 3 RGB channels = 9 output channels
    assert pyramid.shape == (2, 9, 256, 256)
    # Identity scale channels should match input exactly
    assert torch.equal(pyramid[:, 0:3, :, :], x)


def test_entropy_known_distributions():
    """Verify Shannon entropy computation against known 4-pixel distributions."""
    extractor = MLEPExtractor(patch_size=2, scales=[1.0], window_size=2)

    # Create a small 4×4 single-channel image with known patterns
    # Window at (0,0) = [100, 100, 100, 100] -> all identical -> entropy = 0.0
    # Window at (0,2) = [10, 20, 30, 40] -> all unique -> entropy = 2.0
    img = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    img[0, 0, 0:2, 0:2] = 100.0  # All same -> entropy 0.0
    img[0, 0, 0, 2] = 10.0
    img[0, 0, 0, 3] = 20.0
    img[0, 0, 1, 2] = 30.0
    img[0, 0, 1, 3] = 40.0  # All unique -> entropy 2.0

    entropy = extractor.compute_shannon_entropy(img)

    # Output shape: (1, 1, 3, 3) for 4×4 input with 2×2 window
    assert entropy.shape == (1, 1, 3, 3)

    # Top-left window (all identical): entropy must be 0.0
    assert abs(entropy[0, 0, 0, 0].item() - 0.0) < 1e-4

    # Top-right window at (0, 2) -> {10, 20, 30, 40} all unique: entropy must be 2.0
    assert abs(entropy[0, 0, 0, 2].item() - 2.0) < 1e-4


def test_forward_pipeline_output_shapes():
    """Verify all returned output shapes and data types from full MLEP forward pass."""
    extractor = MLEPExtractor(patch_size=2, scales=[1.0, 0.5, 0.25], window_size=2)
    x = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)

    out = extractor(x)

    assert isinstance(out, dict)
    assert out["shuffled"].shape == (2, 3, 256, 256)
    assert out["pyramid"].shape == (2, 9, 256, 256)
    assert out["mlep_features"].shape == (2, 9, 255, 255)
    assert len(out["entropy_maps"]) == 3
    for emap in out["entropy_maps"]:
        assert emap.shape == (2, 3, 255, 255)


def test_entropy_value_set_membership():
    """Verify that all computed entropy values belong to V = {0.0, ~0.8113, 1.0, 1.5, 2.0}."""
    extractor = MLEPExtractor(patch_size=2, scales=[1.0], window_size=2)

    # Use integer-valued image to produce exact discrete distributions
    x = torch.randint(0, 4, (1, 1, 32, 32), dtype=torch.float32)
    entropy = extractor.compute_shannon_entropy(x)

    valid_values = {0.0, 0.8113, 1.0, 1.5, 2.0}
    tolerance = 0.02

    unique_vals = entropy.unique().tolist()
    for v in unique_vals:
        matched = any(abs(v - valid) < tolerance for valid in valid_values)
        assert matched, f"Entropy value {v:.4f} not in valid set V = {valid_values}"
