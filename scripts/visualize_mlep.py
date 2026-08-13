"""
MLEP Diagnostic Visualizer Script for HydraFusion-Net.

Visualizes:
  - Input RGB Image
  - Multi-scale Pyramid Entropy Maps (Scale 1.0, Scale 0.5, Scale 0.25)
  - Composite Multi-Granularity Local Entropy Pattern (MLEP) Heatmap

Usage:
    python scripts/visualize_mlep.py --image_path path/to/image.jpg --save_path outputs/figures/mlep_visualization.png
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.mlep_extractor import MLEPExtractor

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    raise ImportError("matplotlib is required for visualizer.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize MLEP Entropy Patterns")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to an input image. If None, uses a random test set image.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/figures/mlep_visualization.png",
        help="Path to save the output visualization image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load image
    if args.image_path and Path(args.image_path).exists():
        img_path = Path(args.image_path)
    else:
        # Fallback to test dataset image if exists
        test_dir = Path("C:/Users/Eldoria/Music/project main cl dv/DL AND CV PROJECT (1)/dataset10000/test/fake")
        if test_dir.exists():
            img_path = next(test_dir.glob("*.png"), next(test_dir.glob("*.jpg"), None))
        else:
            img_path = None

    if img_path is None or not img_path.exists():
        print("No image path provided and default dataset not found. Generating synthetic image.")
        pil_img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))
        img_name = "Synthetic Random Image"
    else:
        pil_img = Image.open(img_path).convert("RGB")
        img_name = img_path.name

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    tensor = transform(pil_img) * 255.0  # (3, 256, 256) in [0, 255]
    batch = tensor.unsqueeze(0)  # (1, 3, 256, 256)

    mlep_extractor = MLEPExtractor(scales=(1.0, 0.5, 0.25))
    mlep_extractor.eval()

    with torch.no_grad():
        mlep_out = mlep_extractor(batch)  # (1, 9, 256, 256)

    mlep_np = mlep_out.squeeze(0).cpu().numpy()  # (9, 256, 256)

    # Extract 3 scales (channels 0..2 = Scale 1.0, 3..5 = Scale 0.5, 6..8 = Scale 0.25)
    s1_map = mlep_np[0:3].mean(axis=0)
    s2_map = mlep_np[3:6].mean(axis=0)
    s3_map = mlep_np[6:9].mean(axis=0)
    composite_map = mlep_np.mean(axis=0)

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    img_display = np.array(pil_img.resize((256, 256))) / 255.0

    axes[0].imshow(img_display)
    axes[0].set_title(f"Input: {img_name[:15]}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    im1 = axes[1].imshow(s1_map, cmap="inferno")
    axes[1].set_title("MLEP Scale 1.0 (Full Res)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    im2 = axes[2].imshow(s2_map, cmap="inferno")
    axes[2].set_title("MLEP Scale 0.5 (Half Res)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    im3 = axes[3].imshow(s3_map, cmap="inferno")
    axes[3].set_title("MLEP Scale 0.25 (Quarter Res)", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    im4 = axes[4].imshow(composite_map, cmap="magma")
    axes[4].set_title("Composite MLEP Feature", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    plt.suptitle("MLEP (Multi-granularity Local Entropy Patterns) Feature Diagnostic", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = Path(args.save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"MLEP visualization saved to {out_path}")


if __name__ == "__main__":
    main()
