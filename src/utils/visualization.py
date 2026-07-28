"""
Visualization suite for LOTA bit-planes, MGPS divergence heatmaps, and Top-K patch overlays.
"""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from src.utils.logger import get_logger

logger = get_logger("visualization")


def _tensor_to_rgb_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Helper to convert PyTorch image tensor (C, H, W) in [0, 255] to HWC uint8 numpy array."""
    if tensor.ndim == 4:
        tensor = tensor[0]  # Take first sample in batch
    np_img = tensor.permute(1, 2, 0).cpu().numpy()
    np_img = np.clip(np_img, 0.0, 255.0).astype(np.uint8)
    return np_img


def plot_bit_planes(
    img_tensor: torch.Tensor,
    planes_tensor: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Render a comparison figure showing the original RGB image alongside all 8 bit-planes.

    Args:
        img_tensor: RGB image tensor of shape (3, H, W) or (1, 3, H, W) in [0, 255].
        planes_tensor: Binary bit-planes of shape (3, 8, H, W) or (1, 3, 8, H, W).
        save_path: Optional path to save exported PNG figure.

    Returns:
        matplotlib.figure.Figure: Rendered figure instance.
    """
    if planes_tensor.ndim == 5:
        planes_tensor = planes_tensor[0]
    
    rgb_img = _tensor_to_rgb_numpy(img_tensor)
    
    # Average across RGB channels for clean monochrome bit-plane visualization
    mono_planes = planes_tensor.mean(dim=0).cpu().numpy()  # (8, H, W)

    fig, axes = plt.subplots(1, 9, figsize=(20, 3), gridspec_kw={"wspace": 0.05})
    
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original RGB", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for k in range(8):
        ax = axes[k + 1]
        ax.imshow(mono_planes[k], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Bit-Plane {k}\n{' (LSB)' if k <= 2 else ''}", fontsize=9)
        ax.axis("off")
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved bit-planes figure to: {path}")
        plt.close(fig)

    return fig


def plot_mgps_heatmap(
    img_tensor: torch.Tensor,
    scores: torch.Tensor,
    grid_size: int = 8,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Overlay the 8x8 Multi-Grid Patch Scoring (MGPS) divergence heatmap onto the original image.

    Args:
        img_tensor: RGB image tensor of shape (3, H, W) or (1, 3, H, W).
        scores: Flattened divergence scores of shape (64,) or (1, 64).
        grid_size: Number of grid divisions along each spatial axis (default 8).
        save_path: Optional path to save exported PNG figure.

    Returns:
        matplotlib.figure.Figure: Rendered figure instance.
    """
    if scores.ndim == 2:
        scores = scores[0]
        
    rgb_img = _tensor_to_rgb_numpy(img_tensor)
    score_grid = scores.view(grid_size, grid_size).cpu().numpy()
    
    # Normalize score grid for colormap display
    s_min, s_max = score_grid.min(), score_grid.max()
    if s_max > s_min:
        score_norm = (score_grid - s_min) / (s_max - s_min)
    else:
        score_norm = np.zeros_like(score_grid)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"wspace": 0.15})
    
    ax1.imshow(rgb_img)
    ax1.set_title("Original Image", fontsize=11, fontweight="bold")
    ax1.axis("off")

    im2 = ax2.imshow(score_norm, cmap="inferno", interpolation="nearest")
    ax2.set_title("MGPS Divergence Scores (8x8 Grid)", fontsize=11, fontweight="bold")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Overlay
    ax3.imshow(rgb_img)
    ax3.imshow(
        score_norm,
        cmap="inferno",
        interpolation="bilinear",
        alpha=0.5,
        extent=(0, rgb_img.shape[1], rgb_img.shape[0], 0),
    )
    ax3.set_title("MGPS Heatmap Overlay", fontsize=11, fontweight="bold")
    ax3.axis("off")
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved MGPS heatmap figure to: {path}")
        plt.close(fig)

    return fig


def plot_topk_patches(
    img_tensor: torch.Tensor,
    z_norm: torch.Tensor,
    topk_indices: torch.Tensor,
    grid_size: int = 8,
    patch_size: int = 32,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Highlight selected Top-K diverse quadrant patches with bounding boxes on RGB image and LSB map.

    Args:
        img_tensor: RGB image tensor of shape (3, H, W) or (1, 3, H, W).
        z_norm: Thresholded LSB noise map of shape (3, H, W) or (1, 3, H, W).
        topk_indices: Selected grid indices of shape (K,) or (1, K).
        grid_size: Grid dimensions (default 8).
        patch_size: Spatial patch dimensions (default 32).
        save_path: Optional path to save exported PNG figure.

    Returns:
        matplotlib.figure.Figure: Rendered figure instance.
    """
    if topk_indices.ndim == 2:
        topk_indices = topk_indices[0]
    if z_norm.ndim == 4:
        z_norm = z_norm[0]

    rgb_img = _tensor_to_rgb_numpy(img_tensor)
    lsb_img = _tensor_to_rgb_numpy(z_norm)

    indices_list = topk_indices.cpu().tolist()
    K = len(indices_list)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, K + 2, width_ratios=[2, 2] + [1] * K, hspace=0.25, wspace=0.15)

    ax_rgb = fig.add_subplot(gs[:, 0])
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("Top-K Patches (RGB)", fontsize=11, fontweight="bold")
    ax_rgb.axis("off")

    ax_lsb = fig.add_subplot(gs[:, 1])
    ax_lsb.imshow(lsb_img)
    ax_lsb.set_title("Top-K Patches (LSB Noise Map)", fontsize=11, fontweight="bold")
    ax_lsb.axis("off")

    colors = ["red", "cyan", "lime", "yellow", "magenta", "orange", "white", "blue"]

    for idx_pos, grid_idx in enumerate(indices_list):
        r = grid_idx // grid_size
        c = grid_idx % grid_size
        
        y0 = r * patch_size
        x0 = c * patch_size
        
        color = colors[idx_pos % len(colors)]
        
        # Add bounding boxes to RGB and LSB axes
        rect_rgb = patches.Rectangle(
            (x0, y0), patch_size, patch_size, linewidth=2, edgecolor=color, facecolor="none"
        )
        rect_lsb = patches.Rectangle(
            (x0, y0), patch_size, patch_size, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax_rgb.add_patch(rect_rgb)
        ax_lsb.add_patch(rect_lsb)
        
        ax_rgb.text(x0 + 2, y0 + 12, f"#{idx_pos+1}", color=color, fontweight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.1", facecolor="black", alpha=0.6))
        ax_lsb.text(x0 + 2, y0 + 12, f"#{idx_pos+1}", color=color, fontweight="bold", fontsize=10, bbox=dict(boxstyle="square,pad=0.1", facecolor="black", alpha=0.6))

        # Crop patch for zoom view
        patch_rgb = rgb_img[y0 : y0 + patch_size, x0 : x0 + patch_size]
        patch_lsb = lsb_img[y0 : y0 + patch_size, x0 : x0 + patch_size]

        ax_zoom_rgb = fig.add_subplot(gs[0, 2 + idx_pos])
        ax_zoom_rgb.imshow(patch_rgb)
        ax_zoom_rgb.set_title(f"Patch #{idx_pos+1} RGB\n(Grid {grid_idx})", fontsize=9, color=color, fontweight="bold")
        ax_zoom_rgb.axis("off")
        for spine in ax_zoom_rgb.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        ax_zoom_lsb = fig.add_subplot(gs[1, 2 + idx_pos])
        ax_zoom_lsb.imshow(patch_lsb)
        ax_zoom_lsb.set_title(f"Patch #{idx_pos+1} LSB", fontsize=9, color=color, fontweight="bold")
        ax_zoom_lsb.axis("off")
        for spine in ax_zoom_lsb.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved Top-K patches figure to: {path}")
        plt.close(fig)

    return fig


# =====================================================================
# MLEP Visualization Functions
# =====================================================================


def plot_entropy_heatmap(
    img_tensor: torch.Tensor,
    mlep_features: torch.Tensor,
    scale_idx: int = 0,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Render MLEP Shannon entropy heatmap for a specific pyramid scale alongside the original image.

    Args:
        img_tensor: RGB image tensor of shape (3, H, W) or (1, 3, H, W) in [0, 255].
        mlep_features: MLEP feature tensor of shape (9, H-1, W-1) or (1, 9, H-1, W-1).
        scale_idx: Pyramid scale index to visualize (0=1.0x, 1=0.5x, 2=0.25x).
        save_path: Optional path to save exported PNG figure.

    Returns:
        matplotlib.figure.Figure: Rendered figure instance.
    """
    if mlep_features.ndim == 4:
        mlep_features = mlep_features[0]

    rgb_img = _tensor_to_rgb_numpy(img_tensor)

    # Extract 3 channels for the requested scale
    c_start = scale_idx * 3
    c_end = c_start + 3
    scale_entropy = mlep_features[c_start:c_end].cpu().numpy()
    heatmap = np.mean(scale_entropy, axis=0)

    scale_names = {0: "1.0x (Identity)", 1: "0.5x (Half)", 2: "0.25x (Quarter)"}
    scale_label = scale_names.get(scale_idx, f"Scale {scale_idx}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"wspace": 0.15})

    ax1.imshow(rgb_img)
    ax1.set_title("Original Image", fontsize=11, fontweight="bold")
    ax1.axis("off")

    im2 = ax2.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=2.0)
    ax2.set_title(f"MLEP Entropy Map ({scale_label})", fontsize=11, fontweight="bold")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Shannon Entropy (bits)")

    # Overlay
    ax3.imshow(rgb_img)
    ax3.imshow(
        heatmap,
        cmap="viridis",
        interpolation="bilinear",
        alpha=0.5,
        extent=(0, rgb_img.shape[1], rgb_img.shape[0], 0),
    )
    ax3.set_title("Entropy Heatmap Overlay", fontsize=11, fontweight="bold")
    ax3.axis("off")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved MLEP entropy heatmap to: {path}")
        plt.close(fig)

    return fig


def plot_multiscale_entropy(
    mlep_features: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Render side-by-side entropy heatmaps across all 3 pyramid scales for comparison.

    Args:
        mlep_features: MLEP feature tensor of shape (9, H-1, W-1) or (1, 9, H-1, W-1).
        save_path: Optional path to save exported PNG figure.

    Returns:
        matplotlib.figure.Figure: Rendered figure instance.
    """
    if mlep_features.ndim == 4:
        mlep_features = mlep_features[0]

    scale_names = ["Scale 1.0x", "Scale 0.5x", "Scale 0.25x"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"wspace": 0.1})

    for s_idx in range(3):
        c_start = s_idx * 3
        c_end = c_start + 3
        scale_entropy = mlep_features[c_start:c_end].cpu().numpy()
        heatmap = np.mean(scale_entropy, axis=0)

        im = axes[s_idx].imshow(heatmap, cmap="viridis", vmin=0.0, vmax=2.0)
        axes[s_idx].set_title(scale_names[s_idx], fontsize=11, fontweight="bold")
        axes[s_idx].axis("off")

    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.04, label="Shannon Entropy (bits)")
    fig.suptitle("MLEP Multi-Scale Entropy Comparison", fontsize=13, fontweight="bold", y=1.02)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved MLEP multi-scale entropy comparison to: {path}")
        plt.close(fig)

    return fig
