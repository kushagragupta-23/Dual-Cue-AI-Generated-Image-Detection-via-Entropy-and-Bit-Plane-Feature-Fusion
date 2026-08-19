#!/usr/bin/env python3
"""
CLI Script for executing LOTA preprocessing and exporting diagnostic visualization figures.
"""

import argparse
from pathlib import Path
import sys
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional as TF

# Ensure root directory is on python path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.shared.extractors.lota import TopKLOTAExtractor
from src.data.transforms import LOTAPreprocessingTransform
from src.utils.visualization import (
    plot_bit_planes,
    plot_mgps_heatmap,
    plot_topk_patches,
)
from src.utils.logger import get_logger

logger = get_logger("visualize_lota_cli")


def create_synthetic_test_image(size: int = 256) -> Image.Image:
    """Generate a synthetic image with varying textures and edges for visual testing."""
    img = Image.new("RGB", (size, size), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    
    # Quadrant 0 (Top-Left): Checkerboard noise
    for y in range(0, size // 2, 8):
        for x in range(0, size // 2, 8):
            if (y // 8 + x // 8) % 2 == 0:
                draw.rectangle([x, y, x + 7, y + 7], fill=(255, 50, 50))
                
    # Quadrant 1 (Top-Right): Concentric circles
    for r in range(size // 2, 10, -15):
        draw.ellipse([size * 3 // 4 - r // 2, size // 4 - r // 2, size * 3 // 4 + r // 2, size // 4 + r // 2], outline=(50, 255, 50), width=3)
        
    # Quadrant 2 (Bottom-Left): Diagonal stripes
    for i in range(0, size, 12):
        draw.line([(0, size // 2 + i), (i, size)], fill=(255, 255, 50), width=4)
        
    # Quadrant 3 (Bottom-Right): High frequency speckles
    import random
    rng = random.Random(42)
    for _ in range(500):
        rx = rng.randint(size // 2, size - 1)
        ry = rng.randint(size // 2, size - 1)
        draw.point((rx, ry), fill=(255, 255, 255))

    return img


def main():
    parser = argparse.ArgumentParser(description="Run LOTA Preprocessing and Export Visualizations.")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input RGB image.")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Directory to save figures.")
    parser.add_argument("--k_patches", type=int, default=4, help="Number of Top-K diverse patches to select.")
    parser.add_argument("--patch_size", type=int, default=32, help="MGPS patch size.")
    parser.add_argument("--grid_size", type=int, default=8, help="MGPS grid size.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path is not None:
        img_path = Path(args.image_path)
        if not img_path.exists():
            logger.error(f"Image not found at: {img_path}")
            return
        logger.info(f"Loading input image from: {img_path}")
        img_pil = Image.open(img_path).convert("RGB")
        filename_prefix = img_path.stem
    else:
        logger.info("No --image_path provided. Generating synthetic multi-texture test image.")
        img_pil = create_synthetic_test_image(size=args.grid_size * args.patch_size)
        filename_prefix = "synthetic_test"

    # Transform to tensor (3, H, W) in [0.0, 255.0]
    transform = LOTAPreprocessingTransform(image_size=args.grid_size * args.patch_size, crop_to_square=True)
    tensor_img = transform(img_pil).unsqueeze(0)  # Add batch dimension (1, 3, H, W)

    # Initialize LOTA Extractor
    extractor = TopKLOTAExtractor(
        k_patches=args.k_patches,
        patch_size=args.patch_size,
        grid_size=args.grid_size,
    )
    extractor.eval()

    with torch.no_grad():
        # 1. Extract bit-planes
        planes = extractor.extract_all_bit_planes(tensor_img)
        plot_bit_planes(
            img_tensor=tensor_img,
            planes_tensor=planes,
            save_path=out_dir / f"{filename_prefix}_bit_planes.png",
        )

        # 2. Run Forward Pipeline
        out = extractor(tensor_img)
        z_norm = out["z_norm"]
        scores = out["mgps_scores"]
        topk_indices = out["topk_indices"]

        # 3. MGPS Heatmap
        plot_mgps_heatmap(
            img_tensor=tensor_img,
            scores=scores,
            grid_size=args.grid_size,
            save_path=out_dir / f"{filename_prefix}_mgps_heatmap.png",
        )

        # 4. Top-K Patches
        plot_topk_patches(
            img_tensor=tensor_img,
            z_norm=z_norm,
            topk_indices=topk_indices,
            grid_size=args.grid_size,
            patch_size=args.patch_size,
            save_path=out_dir / f"{filename_prefix}_topk_patches.png",
        )

    logger.info(f"Successfully generated and saved all LOTA visual diagnostic figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
