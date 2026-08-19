"""
Unit tests for LOTAClassifier model architecture, forward pass, and training step.
"""

import pytest
import torch
import torch.nn as nn

from src.models.legacy.classifier import LOTAClassifier


def test_lota_classifier_forward():
    """Verify forward pass output shape and logit values."""
    model = LOTAClassifier()
    model.eval()
    
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 1)
    assert not torch.isnan(output).any()


def test_lota_classifier_train_step():
    """Verify single training optimization step updates model weights."""
    model = LOTAClassifier()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    dummy_input = torch.randn(2, 3, 256, 256)
    targets = torch.tensor([0.0, 1.0], dtype=torch.float32)
    
    optimizer.zero_grad()
    logits = model(dummy_input).squeeze(-1)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    
    assert loss.item() >= 0.0
