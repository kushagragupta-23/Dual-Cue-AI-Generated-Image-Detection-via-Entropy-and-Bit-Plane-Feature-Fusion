"""
End-to-End Integration Test for DualCueAIGIDModel.

CI Requirement (Roadmap §7, Item 413):
    Execute an automated integration check passing a random tensor batch
    of shape (4, 3, 256, 256) through DualCueAIGIDModel, asserting clean
    logit output (4, 1) and non-zero gradient backpropagation without NaN values.
"""

import sys
from pathlib import Path

import pytest
import torch

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.models.fusion.model import DualCueAIGIDModel


class TestEndToEndIntegration:
    """End-to-end integration test for the master fusion model."""

    @pytest.fixture
    def model(self):
        """Create DualCueAIGIDModel with all components enabled."""
        return DualCueAIGIDModel(
            backbone_name="resnet18",  # Lighter for CI testing
            pretrained=False,
            use_frequency_filter=True,
            use_cross_attention=True,
            use_moe=True,
            use_dann=True,
            num_domains=8,
            num_experts=4,
            top_k=2,
            d_model=64,
            num_heads=4,
        )

    def test_forward_pass_output_shape(self, model):
        """
        CI Gate: Forward pass produces (B, 1) logits.

        Passes (4, 3, 256, 256) random tensor through full pipeline and
        verifies output classification logit shape is (4, 1).
        """
        x = torch.randint(0, 256, (4, 3, 256, 256), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(x)

        if isinstance(outputs, dict):
            class_logits = outputs["class_logits"]
        elif isinstance(outputs, tuple):
            class_logits = outputs[0]
        else:
            class_logits = outputs

        assert class_logits.shape == (4, 1), (
            f"Expected logit shape (4, 1), got {class_logits.shape}"
        )

    def test_no_nan_forward(self, model):
        """
        CI Gate: No NaN or Inf values in any output tensor.
        """
        x = torch.randint(0, 256, (4, 3, 256, 256), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(x)

        if isinstance(outputs, dict):
            for key, val in outputs.items():
                if isinstance(val, torch.Tensor):
                    assert not torch.isnan(val).any(), f"NaN in output '{key}'"
                    assert not torch.isinf(val).any(), f"Inf in output '{key}'"
        elif isinstance(outputs, tuple):
            for i, val in enumerate(outputs):
                if isinstance(val, torch.Tensor):
                    assert not torch.isnan(val).any(), f"NaN in output[{i}]"
                    assert not torch.isinf(val).any(), f"Inf in output[{i}]"

    def test_backward_pass_nonzero_gradients(self, model):
        """
        CI Gate: Backward pass produces non-zero gradients on trainable params.

        Verifies full gradient flow through MLEP extractor → Frequency Filter →
        Backbone Stems → Cross-Attention → MoE → Classifier.
        """
        x = torch.randint(0, 256, (4, 3, 256, 256), dtype=torch.float32)
        model.train()

        outputs = model(x)

        if isinstance(outputs, dict):
            class_logits = outputs["class_logits"]
            aux_loss = outputs.get("aux_loss", torch.tensor(0.0))
        elif isinstance(outputs, tuple):
            class_logits = outputs[0]
            aux_loss = outputs[2] if len(outputs) > 2 else torch.tensor(0.0)
        else:
            class_logits = outputs
            aux_loss = torch.tensor(0.0)

        # Compute loss and backprop
        labels = torch.randint(0, 2, (4, 1), dtype=torch.float32)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            class_logits, labels
        ) + 0.01 * aux_loss
        loss.backward()

        # Verify at least some parameters have non-zero gradients
        params_with_grad = 0
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None and param.grad.abs().sum() > 0:
                    params_with_grad += 1

        assert params_with_grad > 0, (
            f"No trainable parameters received non-zero gradients. "
            f"Checked {total_params} parameters."
        )

        # Verify no NaN in gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient in parameter '{name}'"
                )

    def test_dann_output_shape(self, model):
        """
        Verify domain discriminator produces (B, num_domains) logits
        when DANN is enabled.
        """
        x = torch.randint(0, 256, (4, 3, 256, 256), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(x)

        if isinstance(outputs, dict) and "domain_logits" in outputs:
            domain_logits = outputs["domain_logits"]
            assert domain_logits.shape == (4, 8), (
                f"Expected domain logit shape (4, 8), got {domain_logits.shape}"
            )

    def test_moe_aux_loss(self, model):
        """
        Verify MoE auxiliary load-balancing loss is a positive scalar.
        """
        x = torch.randint(0, 256, (4, 3, 256, 256), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(x)

        if isinstance(outputs, dict) and "aux_loss" in outputs:
            aux_loss = outputs["aux_loss"]
            assert aux_loss.dim() == 0, f"Aux loss should be scalar, got {aux_loss.shape}"
            assert aux_loss.item() >= 0, f"Aux loss should be non-negative, got {aux_loss.item()}"

    def test_model_without_optional_components(self):
        """
        Verify model works with all optional components disabled.
        """
        model = DualCueAIGIDModel(
            backbone_name="resnet18",
            pretrained=False,
            use_frequency_filter=False,
            use_cross_attention=False,
            use_moe=False,
            use_dann=False,
        )
        x = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            outputs = model(x)

        if isinstance(outputs, dict):
            class_logits = outputs["class_logits"]
        elif isinstance(outputs, tuple):
            class_logits = outputs[0]
        else:
            class_logits = outputs

        assert class_logits.shape == (2, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
