"""
LOTA Diagnostic Visualizer Script for HydraFusion-Net.

Visualizes:
  - Input RGB Image
  - Learned Soft Bit-Plane Noise Features (LSB Steganalysis)
  - Gradient Divergence Patch Score Map
  - Final LOTA Feature Map Overlay

Usage:
    python scripts/visualize_lota.py --image_path path/to/image.jpg --save_path outputs/figures/lota_visualization.png
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lota_extractor import TopKLOTAExtractor

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    raise ImportError("matplotlib is required for visualizer.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LOTA Bit-Plane Noise Patterns")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to an input image. If None, uses a random test set image.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/figures/lota_visualization.png",
        help="Path to save the output visualization image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load image
    if args.image_path and Path(args.image_path).exists():
        img_path = Path(args.image_path)
    else:
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

    lota_extractor = TopKLOTAExtractor(k_patches=4, patch_size=32, grid_size=8)
    lota_extractor.eval()

    with torch.no_grad():
        # Extractor output
        lota_out = lota_extractor(batch)  # (1, 3, 256, 256)
        # Intermediate LSB conv
        lsb_feat = lota_extractor.lsb_conv(batch)  # (1, 3, 256, 256)
        # Patch score map
        score_map = lota_extractor.score_net(batch)  # (1, 1, 8, 8)
        score_map_upsampled = F.interpolate(score_map, size=(256, 256), mode="nearest")

    img_display = np.array(pil_img.resize((256, 256))) / 255.0
    lsb_np = np.clip(lsb_feat.squeeze(0).cpu().numpy().transpose(1, 2, 0) / 255.0, 0, 1)
    score_np = score_map_upsampled.squeeze().cpu().numpy()
    lota_np = np.clip(lota_out.squeeze(0).cpu().numpy().transpose(1, 2, 0) / 255.0, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(img_display)
    axes[0].set_title(f"Input: {img_name[:15]}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(lsb_np)
    axes[1].set_title("Soft Bit-Plane LSB Residuals", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    im2 = axes[2].imshow(score_np, cmap="viridis")
    axes[2].set_title("MGPS Patch Anomaly Map (8x8 Grid)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(lota_np)
    axes[3].set_title("Final LOTA Gated Representation", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle("LOTA (LOw-biT pAtch) Steganalysis & Patch Divergence Visualizer", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = Path(args.save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"LOTA visualization saved to {out_path}")


if __name__ == "__main__":
    main()
