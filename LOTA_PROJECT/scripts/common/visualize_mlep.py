#!/usr/bin/env python3
"""
MLEP Spatial Entropy Heatmap Diagnostic Visualizer.

Generates visual diagnostic outputs for the VectorizedMLEPExtractor:
    1. Multi-scale entropy pyramid heatmaps (scales 1.0, 0.5, 0.25)
    2. Per-channel RGB entropy distributions
    3. Original vs. shuffled comparison grids
    4. Composite overlay visualizations

Usage:
    python scripts/visualize_mlep.py --image path/to/image.png --output outputs/vis_mlep/
    python scripts/visualize_mlep.py --image-dir path/to/images/ --output outputs/vis_mlep/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

from src.utils.device import get_compute_device
from src.utils.logger import get_logger
from src.shared.extractors.mlep import VectorizedMLEPExtractor

logger = get_logger("visualize_mlep")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLEP Spatial Entropy Heatmap Diagnostic Visualizer"
    )
    parser.add_argument("--image", type=str, help="Path to single input image.")
    parser.add_argument("--image-dir", type=str, help="Directory of input images.")
    parser.add_argument(
        "--output", type=str, default="outputs/vis_mlep",
        help="Output directory for visualizations."
    )
    parser.add_argument(
        "--scales", nargs="+", type=float, default=[1.0, 0.5, 0.25],
        help="Multi-scale pyramid scales."
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    return parser.parse_args()


def load_image_tensor(image_path: str) -> torch.Tensor:
    """Load image as a (1, 3, H, W) tensor in [0, 255]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256), Image.BILINEAR)
    tensor = TF.to_tensor(img) * 255.0  # (3, 256, 256)
    return tensor.unsqueeze(0)  # (1, 3, 256, 256)


def visualize_entropy_heatmaps(
    image_tensor: torch.Tensor,
    extractor: VectorizedMLEPExtractor,
    save_path: Path,
    image_name: str = "image",
    dpi: int = 200,
):
    """
    Generate multi-panel entropy heatmap visualization.

    Creates a figure showing:
        Row 1: Original image + shuffled image
        Row 2: Per-scale mean entropy heatmaps (1.0, 0.5, 0.25)
        Row 3: Per-channel RGB entropy at scale 1.0

    Args:
        image_tensor: (1, 3, 256, 256) tensor in [0, 255].
        extractor: VectorizedMLEPExtractor instance.
        save_path: Directory to save the output figure.
        image_name: Filename stem for the output.
        dpi: Output resolution.
    """
    with torch.no_grad():
        result = extractor(image_tensor)

    entropy_map = result["entropy_map"][0].cpu().numpy()  # (9, 256, 256)
    pyramid = result["pyramid"][0].cpu().numpy()           # (9, 256, 256)
    shuffled = result["shuffled"][0].cpu().numpy()          # (3, 256, 256)
    original = image_tensor[0].cpu().numpy()                # (3, 256, 256)

    num_scales = len(extractor.scales)
    fig, axes = plt.subplots(3, max(num_scales, 3), figsize=(5 * max(num_scales, 3), 14))

    # Row 1: Original + Shuffled + Difference
    ax = axes[0, 0]
    ax.imshow(np.transpose(original / 255.0, (1, 2, 0)).clip(0, 1))
    ax.set_title("Original Image", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(np.transpose(shuffled / 255.0, (1, 2, 0)).clip(0, 1))
    ax.set_title("Locally Shuffled (16×16 grid)", fontsize=12, fontweight="bold")
    ax.axis("off")

    ax = axes[0, 2]
    diff = np.abs(original - shuffled).mean(axis=0)
    ax.imshow(diff, cmap="hot", vmin=0)
    ax.set_title("Shuffle Difference Map", fontsize=12, fontweight="bold")
    ax.axis("off")

    # Row 2: Per-scale mean entropy
    for i in range(num_scales):
        ax = axes[1, i]
        # Mean across RGB channels for this scale
        scale_entropy = entropy_map[i * 3:(i + 1) * 3].mean(axis=0)
        im = ax.imshow(scale_entropy, cmap="inferno", vmin=0, vmax=2.0)
        ax.set_title(f"Entropy @ Scale {extractor.scales[i]}", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3: Per-channel entropy at scale 1.0
    channel_names = ["Red", "Green", "Blue"]
    for c in range(3):
        ax = axes[2, c]
        im = ax.imshow(entropy_map[c], cmap="viridis", vmin=0, vmax=2.0)
        ax.set_title(f"{channel_names[c]} Channel Entropy", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"MLEP Entropy Diagnostic: {image_name}",
        fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path.mkdir(parents=True, exist_ok=True)
    fig_path = save_path / f"{image_name}_mlep_entropy.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")


def visualize_entropy_distribution(
    image_tensor: torch.Tensor,
    extractor: VectorizedMLEPExtractor,
    save_path: Path,
    image_name: str = "image",
    dpi: int = 200,
):
    """
    Generate entropy value distribution histogram.

    Args:
        image_tensor: (1, 3, 256, 256) tensor.
        extractor: Extractor instance.
        save_path: Output directory.
        image_name: Filename stem.
        dpi: Resolution.
    """
    with torch.no_grad():
        result = extractor(image_tensor)

    entropy = result["entropy_map"][0].cpu().numpy().flatten()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(entropy, bins=50, color="#2196F3", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Shannon Entropy (bits)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"MLEP Entropy Distribution: {image_name}", fontsize=13, fontweight="bold")
    ax.axvline(entropy.mean(), color="red", linestyle="--", label=f"Mean={entropy.mean():.3f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path.mkdir(parents=True, exist_ok=True)
    fig_path = save_path / f"{image_name}_entropy_hist.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")


def main():
    args = parse_args()
    output_dir = Path(args.output)

    extractor = VectorizedMLEPExtractor(
        scales=args.scales, grid_size=16, patch_size=16
    )

    # Collect images
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        image_paths = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
    else:
        # Demo mode: generate synthetic images
        logger.info("No image specified — generating synthetic demo images")
        demo_dir = output_dir / "demo_inputs"
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Synthetic flat image
        flat = Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 128)
        flat_path = demo_dir / "flat.png"
        flat.save(flat_path)
        image_paths.append(flat_path)

        # Synthetic noisy image
        noisy = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        noisy_path = demo_dir / "noisy.png"
        noisy.save(noisy_path)
        image_paths.append(noisy_path)

    for img_path in image_paths:
        logger.info(f"Processing: {img_path.name}")
        tensor = load_image_tensor(str(img_path))
        name = img_path.stem

        visualize_entropy_heatmaps(tensor, extractor, output_dir, name, args.dpi)
        visualize_entropy_distribution(tensor, extractor, output_dir, name, args.dpi)

    logger.info(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
