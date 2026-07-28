#!/usr/bin/env python3
"""
Automated Dataset Downloader & Scale-Up Utility for MLEP & LOTA Fusion
Downloads real vs. AI-generated benchmark datasets from HuggingFace or generates large-scale local benchmark datasets.
Compatible with SharedImageDataset and the LOTA Steganalysis pipeline.
"""

import argparse
import os
from pathlib import Path
import random
import sys
import time
from PIL import Image, ImageDraw

# Ensure root path is accessible
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.logger import get_logger

logger = get_logger("dataset_downloader")


def download_huggingface_dataset(target_dir: Path, num_samples_per_class: int = 500) -> bool:
    """
    Download open-access Real vs. AI-generated image datasets from HuggingFace Hub.
    Requires: pip install datasets
    """
    try:
        import datasets
        logger.info(f"Connecting to HuggingFace Hub to download 'dima806/ai_vs_real_image_detection'...")
        ds = datasets.load_dataset("dima806/ai_vs_real_image_detection", split="train", streaming=True)
        
        real_dir = target_dir / "0_real"
        ai_dir = target_dir / "1_stylegan2"  # Using StyleGAN2/Diffusion domain tag
        real_dir.mkdir(parents=True, exist_ok=True)
        ai_dir.mkdir(parents=True, exist_ok=True)

        real_count, ai_count = 0, 0
        logger.info(f"Downloading and structuring up to {num_samples_per_class} images per class...")

        for item in ds:
            label = item.get("label", -1)
            img = item.get("image")
            if img is None:
                continue

            # In dima806 dataset: label 0 is Real, label 1 is Fake/AI
            if label == 0 and real_count < num_samples_per_class:
                img.convert("RGB").save(real_dir / f"hf_real_{real_count}.png")
                real_count += 1
            elif label == 1 and ai_count < num_samples_per_class:
                img.convert("RGB").save(ai_dir / f"hf_ai_{ai_count}.png")
                ai_count += 1

            if real_count >= num_samples_per_class and ai_count >= num_samples_per_class:
                break

        logger.info(f"Successfully downloaded {real_count} Real images and {ai_count} AI images to: {target_dir}")
        return True

    except ImportError:
        logger.warning("The 'datasets' library is not installed. Run 'pip install datasets' to download from HuggingFace.")
        return False
    except Exception as e:
        logger.error(f"Error downloading from HuggingFace: {e}")
        return False


def generate_large_scale_benchmark(target_dir: Path, total_samples: int = 1400) -> None:
    """
    Generate a large-scale 1,400+ image benchmark dataset locally for stress testing throughput and GPU training loops.
    """
    logger.info(f"Generating large-scale local benchmark dataset ({total_samples} images) at: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    domains = {
        "0_real": {"count": total_samples // 2, "color_base": (70, 130, 80), "noise_lvl": 6},
        "1_stylegan2": {"count": total_samples // 8, "color_base": (210, 70, 70), "noise_lvl": 28},
        "1_midjourney": {"count": total_samples // 8, "color_base": (70, 190, 210), "noise_lvl": 16},
        "1_flux": {"count": total_samples // 8, "color_base": (190, 70, 210), "noise_lvl": 32},
        "1_progan": {"count": total_samples // 8, "color_base": (210, 170, 40), "noise_lvl": 38},
    }

    rng = random.Random(12345)
    start_time = time.time()
    count = 0

    for folder_name, cfg in domains.items():
        domain_dir = target_dir / folder_name
        domain_dir.mkdir(exist_ok=True)
        for i in range(cfg["count"]):
            w = rng.choice([256, 280, 320])
            h = rng.choice([256, 280, 320])
            img = Image.new("RGB", (w, h), color=cfg["color_base"])
            draw = ImageDraw.Draw(img)

            step = rng.choice([10, 16, 20])
            for y in range(0, h, step):
                for x in range(0, w, step):
                    if (x // step + y // step) % 2 == 0:
                        c = (
                            min(255, max(0, cfg["color_base"][0] + rng.randint(-25, 25))),
                            min(255, max(0, cfg["color_base"][1] + rng.randint(-25, 25))),
                            min(255, max(0, cfg["color_base"][2] + rng.randint(-25, 25))),
                        )
                        draw.rectangle([x, y, x + step - 1, y + step - 1], fill=c)

            noise_lvl = cfg["noise_lvl"]
            for _ in range(w * h // 12):
                nx = rng.randint(0, w - 1)
                ny = rng.randint(0, h - 1)
                draw.point((nx, ny), fill=(rng.randint(0, noise_lvl), rng.randint(0, noise_lvl), rng.randint(0, noise_lvl)))

            img.save(domain_dir / f"img_{i}.png")
            count += 1
            if count % 100 == 0:
                logger.info(f"Generated {count}/{total_samples} images...")

    logger.info(f"Successfully generated {count} benchmark images in {time.time() - start_time:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="Download or Generate Large-Scale Training Dataset.")
    parser.add_argument("--target_dir", type=str, default="outputs/big_dataset", help="Directory to save downloaded/generated images.")
    parser.add_argument("--source", type=str, choices=["huggingface", "local"], default="local", help="Source: download from HuggingFace or generate locally.")
    parser.add_argument("--num_samples", type=int, default=1400, help="Total number of images to generate or download.")
    args = parser.parse_args()

    target_path = root_path / args.target_dir

    if args.source == "huggingface":
        success = download_huggingface_dataset(target_path, num_samples_per_class=args.num_samples // 2)
        if not success:
            logger.info("Falling back to large-scale local benchmark generation...")
            generate_large_scale_benchmark(target_path, total_samples=args.num_samples)
    else:
        generate_large_scale_benchmark(target_path, total_samples=args.num_samples)

    print("\n" + "=" * 80)
    print(f"SUCCESS! Large-scale dataset is ready at: {target_path.resolve()}")
    print("To run the LOTA pipeline on this new big dataset, execute:")
    print(f"    python scripts/run_project.py --data_dir {args.target_dir} --batch_size 16")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
