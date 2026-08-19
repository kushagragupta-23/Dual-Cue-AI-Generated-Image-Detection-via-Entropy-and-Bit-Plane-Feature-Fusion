"""
Grad-CAM Explainability Module for Dual-Cue AIGID.

Provides visual explainability via Gradient-weighted Class Activation Mapping:
    1. GradCAM — Hooks into specified convolutional layers to compute saliency maps.
    2. AttentionOverlayGenerator — Overlays cross-attention weights onto original images.
    3. Export utilities for publication-ready saliency visualizations.

Grad-CAM formula:
    α_k = (1/Z) Σ_i Σ_j (∂y_c / ∂A_k_{i,j})     [global average pooled gradients]
    L_GradCAM = ReLU(Σ_k α_k · A_k)                [weighted combination + ReLU]
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger("explainability")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explainability.

    Hooks into a target convolutional layer to capture activations and gradients,
    then computes pixel-level saliency maps showing which spatial regions contributed
    most to the classification decision.

    Args:
        model: The neural network model.
        target_layer: The convolutional layer to hook (e.g., model.layer4[-1]).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: save activation maps."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save gradient maps."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Model input of shape (1, C, H, W).
            target_class: Target class index for gradient computation.
                          If None, uses the predicted class.

        Returns:
            np.ndarray: Normalized heatmap of shape (H, W) in [0, 1].
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=-1).item() if logits.shape[-1] > 1 else 0

        # Zero gradients and backward pass
        self.model.zero_grad()
        if logits.shape[-1] == 1:
            # Binary classification: use the single logit
            target_score = logits[0, 0]
        else:
            target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        if self._activations is None or self._gradients is None:
            logger.error("Grad-CAM hooks failed to capture activations/gradients.")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Compute channel weights: global average pooled gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Apply ReLU

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        """Remove registered hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


class AttentionOverlayGenerator:
    """
    Generates attention weight overlays from cross-attention modules.

    Hooks into PyramidCrossAttentionModule's MultiheadAttention to capture
    and visualize spatial attention patterns.

    Args:
        model: Model containing cross-attention module.
        cross_attn_module: The PyramidCrossAttentionModule to hook.
    """

    def __init__(self, model: nn.Module, cross_attn_module: nn.Module):
        self.model = model
        self.cross_attn = cross_attn_module
        self._attn_weights: Optional[torch.Tensor] = None

        # Hook into the MultiheadAttention layer
        if hasattr(cross_attn_module, "attn"):
            self._hook = cross_attn_module.attn.register_forward_hook(
                self._save_attn_weights
            )
        else:
            logger.warning("Cross-attention module has no 'attn' attribute.")
            self._hook = None

    def _save_attn_weights(self, module, input, output):
        """Forward hook to capture attention weights."""
        if isinstance(output, tuple) and len(output) > 1:
            self._attn_weights = output[1].detach() if output[1] is not None else None

    @torch.no_grad()
    def generate_overlay(
        self,
        input_tensor: torch.Tensor,
        spatial_size: Tuple[int, int] = (16, 16),
    ) -> Optional[np.ndarray]:
        """
        Generate attention weight heatmap.

        Args:
            input_tensor: Model input.
            spatial_size: Expected spatial grid size of attention (default 16×16).

        Returns:
            np.ndarray: Attention heatmap of shape spatial_size, or None.
        """
        self.model.eval()
        _ = self.model(input_tensor)

        if self._attn_weights is None:
            logger.warning("No attention weights captured.")
            return None

        # Average across heads and batch: (H*W,) → (H, W)
        attn = self._attn_weights.mean(dim=(0, 1))  # (N, N)
        # Average query attention: which keys each query attends to
        attn_map = attn.mean(dim=0).cpu().numpy()  # (N,)

        h, w = spatial_size
        if len(attn_map) == h * w:
            return attn_map.reshape(h, w)
        else:
            logger.warning(
                f"Attention map size {len(attn_map)} doesn't match "
                f"expected {h}×{w}={h*w}"
            )
            return None

    def remove_hooks(self):
        if self._hook is not None:
            self._hook.remove()


def plot_gradcam_overlay(
    image: Union[torch.Tensor, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Grad-CAM Saliency",
) -> plt.Figure:
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        image: Original image as tensor (C, H, W) in [0, 255] or numpy (H, W, 3).
        heatmap: Grad-CAM heatmap of shape (H, W) in [0, 1].
        alpha: Overlay transparency (default 0.5).
        colormap: Matplotlib colormap name (default 'jet').
        save_path: Optional path to save the figure.
        title: Figure title.

    Returns:
        matplotlib.figure.Figure: Rendered overlay figure.
    """
    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            image = image[0]
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        img_np = image

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    # Heatmap
    im = axes[1].imshow(heatmap, cmap=colormap, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(img_np)
    heatmap_resized = np.array(
        F.interpolate(
            torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float(),
            size=img_np.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
    ) if heatmap.shape != img_np.shape[:2] else heatmap
    axes[2].imshow(
        heatmap_resized, cmap=colormap, alpha=alpha,
        extent=(0, img_np.shape[1], img_np.shape[0], 0),
    )
    axes[2].set_title(title, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved Grad-CAM figure to {path}")
        plt.close(fig)

    return fig


__all__ = [
    "GradCAM",
    "AttentionOverlayGenerator",
    "plot_gradcam_overlay",
]
