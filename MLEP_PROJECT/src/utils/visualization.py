"""
Visualization suite for MLEP entropy heatmaps.
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


def plot_training_curves(history_data: list, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Render training and validation curves for Loss and Accuracy.
    
    Args:
        history_data: List of dictionaries containing epoch metrics.
        save_path: Optional path to save exported PNG figure.
    """
    epochs = [d['epoch'] for d in history_data]
    train_loss = [d['train_loss'] for d in history_data]
    val_loss = [d['val_loss'] for d in history_data]
    train_acc = [d['train_acc'] for d in history_data]
    val_acc = [d['val_acc'] for d in history_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', marker='o')
    ax1.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', marker='s')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Accuracy plot
    ax2.plot(epochs, train_acc, label='Train Accuracy', color='#2ca02c', marker='o')
    ax2.plot(epochs, val_acc, label='Validation Accuracy', color='#d62728', marker='s')
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved training curves to: {path}")
        plt.close(fig)
        
    return fig


def plot_confusion_matrix(labels: list, preds: list, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Render a clean confusion matrix.
    
    Args:
        labels: Ground truth binary labels.
        preds: Predicted binary labels.
        save_path: Optional path to save exported PNG figure.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(labels, preds)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, 
                annot_kws={"size": 14, "weight": "bold"})
    
    ax.set_title('Test Set Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    
    # Assuming 0 is Real, 1 is AI-Generated as per dataset conventions
    ax.set_xticklabels(['Real', 'AI-Generated'])
    ax.set_yticklabels(['Real', 'AI-Generated'], rotation=0)
    
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to: {path}")
        plt.close(fig)
        
    return fig
