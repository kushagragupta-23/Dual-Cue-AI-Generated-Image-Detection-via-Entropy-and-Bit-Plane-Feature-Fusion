"""
Grad-CAM Explainability Module for HydraFusion-Net.

Implements:
  - Grad-CAM (Gradient-weighted Class Activation Mapping) backward hooks
    on the un-scrambled spatial feature branches
  - Attention weight overlay generation for fusion heads
  - Composite visualization: original image + MLEP heatmap + LOTA heatmap + overlay

Hooks into model.mlep_stem.layer3 and model.lota_stem.layer3 to capture
the spatial feature maps that drive the classification decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    raise ImportError("matplotlib is required for explainability visualizations.")


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for HydraFusion-Net.

    Registers forward and backward hooks on a target convolutional layer
    to capture activations and gradients, then computes the weighted
    combination to produce a class-discriminative saliency heatmap.

    Args:
        model: The HydraFusion-Net model.
        target_layer: The nn.Module to hook into (e.g., model.mlep_stem.layer3).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Forward hook: save the output activation map."""
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        """Backward hook: save the gradient of the output."""
        self._gradients = grad_output[0].detach()

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        use_amp: bool = True,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Input image tensor (1, C, H, W) on the model's device.
            target_class: Class index to compute CAM for (0=Real, 1=Fake).
                         If None, uses the predicted class.
            use_amp: Whether to use automatic mixed precision.

        Returns:
            numpy array (H_input, W_input) with values in [0, 1] representing
            the saliency heatmap.
        """
        self.model.eval()

        # Enable gradients for the forward pass
        input_tensor.requires_grad_(True)

        # Forward pass
        if use_amp and input_tensor.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, _, _ = self.model(input_tensor, stage=2)
        else:
            logits, _, _ = self.model(input_tensor, stage=2)

        logits = logits.float()

        # Determine target
        if target_class is None:
            target_class = 1 if torch.sigmoid(logits).item() > 0.5 else 0

        # Compute the score for the target class
        if target_class == 1:
            score = logits.squeeze()
        else:
            score = -logits.squeeze()

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Compute Grad-CAM weights: global average pool of gradients
        gradients = self._gradients  # (1, C, H, W)
        activations = self._activations  # (1, C, H, W)

        if gradients is None or activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Ensure the target layer is part of the forward graph."
            )

        # Global average pooling of gradients → importance weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU: only positive influence
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self) -> None:
        """Remove registered hooks to avoid memory leaks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


class DualBranchExplainer:
    """
    Generates combined Grad-CAM visualizations for both MLEP and LOTA branches.

    Creates a 4-panel visualization:
      [Original Image] | [MLEP Grad-CAM] | [LOTA Grad-CAM] | [Combined Overlay]

    Usage:
        explainer = DualBranchExplainer(model, device)
        explainer.explain_image(image_tensor, save_path="outputs/gradcam_sample.png")
        explainer.cleanup()
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.use_amp = use_amp

        # Create Grad-CAM instances for both branches
        self.cam_mlep = GradCAM(model, model.mlep_stem.layer3)
        self.cam_lota = GradCAM(model, model.lota_stem.layer3)

    def explain_image(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        save_path: Optional[str] = None,
        title: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for both branches on a single image.

        Args:
            image_tensor: Single image tensor (1, 3, H, W) or (3, H, W).
                         Expected range [0, 255] (matching HydraFusion convention).
            target_class: Class to explain (0=Real, 1=Fake). None = predicted class.
            save_path: If provided, saves the 4-panel visualization to this path.
            title: Title for the visualization.

        Returns:
            Tuple of (mlep_heatmap, lota_heatmap), each numpy (H, W) in [0, 1].
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Compute both heatmaps (requires two forward+backward passes)
        heatmap_mlep = self.cam_mlep.compute(
            image_tensor.clone(), target_class, use_amp=self.use_amp
        )
        heatmap_lota = self.cam_lota.compute(
            image_tensor.clone(), target_class, use_amp=self.use_amp
        )

        # Save visualization if requested
        if save_path is not None:
            self._save_visualization(
                image_tensor, heatmap_mlep, heatmap_lota, save_path, title
            )

        return heatmap_mlep, heatmap_lota

    def explain_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        output_dir: str = "outputs/gradcam",
        max_images: int = 16,
    ) -> None:
        """
        Generate Grad-CAM visualizations for a batch of images.

        Args:
            images: Batch tensor (B, 3, H, W) in [0, 255].
            labels: Ground truth labels (B,).
            output_dir: Directory to save individual visualizations.
            max_images: Maximum number of images to process.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        num = min(images.shape[0], max_images)
        for i in range(num):
            img = images[i].unsqueeze(0)
            label = labels[i].item()
            class_name = "Fake" if label == 1 else "Real"
            save_file = out_path / f"gradcam_{i:03d}_{class_name}.png"
            self.explain_image(
                img,
                target_class=None,
                save_path=str(save_file),
                title=f"Sample {i} — Ground Truth: {class_name}",
            )

    def _save_visualization(
        self,
        image_tensor: torch.Tensor,
        heatmap_mlep: np.ndarray,
        heatmap_lota: np.ndarray,
        save_path: str,
        title: str = "",
    ) -> None:
        """Create and save a 4-panel Grad-CAM visualization."""
        # Convert image to displayable format
        img_np = image_tensor.squeeze(0).cpu().numpy()  # (3, H, W)
        img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, 3)
        img_np = np.clip(img_np / 255.0, 0.0, 1.0)

        # Create colorized heatmaps
        heatmap_mlep_color = cm.jet(heatmap_mlep)[:, :, :3]
        heatmap_lota_color = cm.jet(heatmap_lota)[:, :, :3]

        # Create overlay (average of both heatmaps)
        combined_heatmap = np.clip((heatmap_mlep + heatmap_lota) / 2.0, 0, 1)
        overlay = 0.5 * img_np + 0.5 * cm.jet(combined_heatmap)[:, :, :3]
        overlay = np.clip(overlay, 0, 1)

        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img_np)
        axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(img_np)
        axes[1].imshow(heatmap_mlep_color, alpha=0.5)
        axes[1].set_title("MLEP Branch Grad-CAM", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(img_np)
        axes[2].imshow(heatmap_lota_color, alpha=0.5)
        axes[2].set_title("LOTA Branch Grad-CAM", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title("Combined Overlay", fontsize=12, fontweight="bold")
        axes[3].axis("off")

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    def cleanup(self) -> None:
        """Remove all hooks to prevent memory leaks."""
        self.cam_mlep.remove_hooks()
        self.cam_lota.remove_hooks()
