"""
Unit tests for the Dual-Cue Fusion Architecture.
Verifies the integration of MLEP, LOTA, backbones, and the fusion head.
"""

import pytest
import torch
from src.models.fusion_head import CrossModalGatingFusionHead
from src.models.backbones import SharedFeatureExtractor
from src.models.dual_cue_detector import DualCueDetector

@pytest.fixture
def dummy_mlep_features():
    # Batch size 2, 512 dimensions
    return torch.randn(2, 512)

@pytest.fixture
def dummy_lota_features():
    # Batch size 2, 512 dimensions
    return torch.randn(2, 512)

@pytest.fixture
def dummy_image_batch():
    # Batch size 2, 3 channels, 256x256 resolution
    return torch.rand(2, 3, 256, 256) * 255.0

def test_shared_feature_extractor():
    # Test standard 3-channel input
    model_3c = SharedFeatureExtractor(in_channels=3, pretrained=False)
    out_3c = model_3c(torch.randn(2, 3, 32, 32))
    assert out_3c.shape == (2, 512), "Feature extractor should output 512-D vector"

    # Test custom 9-channel input (MLEP)
    model_9c = SharedFeatureExtractor(in_channels=9, pretrained=False)
    out_9c = model_9c(torch.randn(2, 9, 32, 32))
    assert out_9c.shape == (2, 512), "Feature extractor should support arbitrary input channels"

def test_cross_modal_fusion_head(dummy_mlep_features, dummy_lota_features):
    fusion_head = CrossModalGatingFusionHead(
        in_channels_mlep=512,
        in_channels_lota=512,
        latent_dim=256
    )
    
    logits = fusion_head(dummy_mlep_features, dummy_lota_features)
    assert logits.shape == (2, 1), "Fusion head should output a single logit per batch item"
    
def test_dual_cue_detector(dummy_image_batch):
    # Initialize with small backbones (pretrained=False) for speed
    detector = DualCueDetector(pretrained_backbones=False)
    
    # Forward pass
    logits = detector(dummy_image_batch)
    assert logits.shape == (2, 1), "DualCueDetector should output (B, 1) logits"
    
    # Forward pass with return_features=True
    outputs = detector(dummy_image_batch, return_features=True)
    assert "logits" in outputs
    assert "entropy_map" in outputs
    assert "noise_tensor" in outputs
    assert "feat_mlep" in outputs
    assert "feat_lota" in outputs
    assert outputs["feat_mlep"].shape == (2, 512)
    assert outputs["feat_lota"].shape == (2, 512)
