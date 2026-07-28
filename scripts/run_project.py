#!/usr/bin/env python3
"""
Master Project Execution Script: MLEP Fusion
Connects the Shared Dataset Infrastructure directly into the MLEP Steganalysis & Preprocessing Engine.
Demonstrates end-to-end data ingestion, stratified splitting, balanced batch sampling, and vectorized Top-K patch extraction.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List
import numpy as np
from PIL import Image, ImageDraw
import torch

# Ensure root directory is on python path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.dataset import SharedImageDataset
from src.data.dataloader import create_dataloader
from src.models.mlep import MLEPExtractor
from src.utils.visualization import plot_entropy_heatmap, plot_multiscale_entropy
from src.utils.logger import get_logger

logger = get_logger("run_project")


def generate_benchmark_dataset(target_dir: Path, total_samples: int = 40) -> None:
    """
    Generate a structured multi-domain synthetic dataset for benchmark testing when no external dataset is provided.
    Simulates Real images (nature/authentic) and AI-generated images (StyleGAN2, Midjourney, FLUX, ProGAN).
    """
    logger.info(f"Generating synthetic multi-domain benchmark dataset at: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    domains = {
        "0_real": {"count": total_samples // 2, "color_base": (80, 140, 60), "noise_lvl": 5},
        "1_stylegan2": {"count": total_samples // 8, "color_base": (200, 80, 80), "noise_lvl": 25},
        "1_midjourney": {"count": total_samples // 8, "color_base": (80, 180, 200), "noise_lvl": 15},
        "1_flux": {"count": total_samples // 8, "color_base": (180, 80, 200), "noise_lvl": 30},
        "1_progan": {"count": total_samples // 8, "color_base": (200, 160, 50), "noise_lvl": 35},
    }

    rng = random.Random(42)
    for folder_name, cfg in domains.items():
        domain_dir = target_dir / folder_name
        domain_dir.mkdir(exist_ok=True)
        for i in range(cfg["count"]):
            # Generate varying image dimensions to test online resizing in SharedImageTransform
            w = rng.choice([256, 300, 320])
            h = rng.choice([256, 280, 300])
            img = Image.new("RGB", (w, h), color=cfg["color_base"])
            draw = ImageDraw.Draw(img)

            # Draw structured geometric textures
            step = rng.choice([12, 16, 24])
            for y in range(0, h, step):
                for x in range(0, w, step):
                    if (x // step + y // step) % 2 == 0:
                        c = (
                            min(255, max(0, cfg["color_base"][0] + rng.randint(-30, 30))),
                            min(255, max(0, cfg["color_base"][1] + rng.randint(-30, 30))),
                            min(255, max(0, cfg["color_base"][2] + rng.randint(-30, 30))),
                        )
                        draw.rectangle([x, y, x + step - 1, y + step - 1], fill=c)

            # Inject LSB high-frequency noise simulating generator Steganalysis artifacts
            noise_lvl = cfg["noise_lvl"]
            for _ in range(w * h // 10):
                nx = rng.randint(0, w - 1)
                ny = rng.randint(0, h - 1)
                draw.point((nx, ny), fill=(rng.randint(0, noise_lvl), rng.randint(0, noise_lvl), rng.randint(0, noise_lvl)))

            img.save(domain_dir / f"sample_{i}.png")

    logger.info(f"Successfully generated {total_samples} benchmark samples across {len(domains)} domains.")


def main():
    parser = argparse.ArgumentParser(description="Run Complete MLEP Project Pipeline.")
    parser.add_argument("--data_dir", type=str, default="outputs/dataset_1400", help="Path to input dataset directory.")
    parser.add_argument("--output_dir", type=str, default="outputs/project_run", help="Directory to store run artifacts and logs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading subprocess workers.")
    parser.add_argument("--patch_size", type=int, default=2, help="Micro-patch size for spatial shuffling.")
    parser.add_argument("--export_visualizations", action="store_true", default=True, help="Export diagnostic sample visualizations.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)

    data_path = Path(args.data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        logger.warning(f"Data directory '{data_path}' not found or empty. Generating benchmark dataset...")
        generate_benchmark_dataset(data_path, total_samples=1400)

    # ==================== STEP 1: INITIALIZE SHARED DATASET INFRASTRUCTURE ====================
    print("\n" + "=" * 80)
    print("STEP 1: INITIALIZING SHARED DATASET INFRASTRUCTURE (src.data)")
    print("=" * 80)
    
    start_time = time.time()
    train_ds = SharedImageDataset(
        root_dir=data_path,
        split="train",
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        validate_integrity=True,
        split_manifest_dir=manifest_dir,
    )
    val_ds = SharedImageDataset(
        root_dir=data_path,
        split="val",
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        validate_integrity=True,
        split_manifest_dir=manifest_dir,
    )
    test_ds = SharedImageDataset(
        root_dir=data_path,
        split="test",
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        validate_integrity=True,
        split_manifest_dir=manifest_dir,
    )

    total_indexed = len(train_ds) + len(val_ds) + len(test_ds)
    logger.info(f"Dataset indexed in {time.time() - start_time:.2f}s. Total valid images: {total_indexed}")
    logger.info(f"Split Stratification -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    loader = create_dataloader(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampling=True,  # Guarantee 50/50 Real vs AI within every mini-batch
        drop_last=False,
    )

    # ==================== STEP 2: INITIALIZE MLEP PREPROCESSING ENGINE ====================
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZING MLEP PREPROCESSING ENGINE (src.models)")
    print("=" * 80)

    mlep_extractor = MLEPExtractor(
        patch_size=args.patch_size,
        scales=[1.0, 0.5, 0.25],
        window_size=2,
    )
    mlep_extractor.eval()
    logger.info(
        f"MLEP Extractor Configured -> Patch Size: {args.patch_size}x{args.patch_size}, "
        f"Scales: [1.0, 0.5, 0.25], Window: 2x2"
    )

    # ==================== STEP 3: EXECUTING END-TO-END PIPELINE ====================
    print("\n" + "=" * 80)
    print("STEP 3: EXECUTING END-TO-END DATASET -> MLEP BATCH PIPELINE")
    print("=" * 80)

    total_images_processed = 0
    real_entropy_scores: List[float] = []
    ai_entropy_scores: List[float] = []
    batch_latencies: List[float] = []

    with torch.no_grad():
        for batch_idx, (images, labels, metas) in enumerate(loader):
            b_start = time.time()
            batch_size_curr = images.shape[0]

            # Execute MLEP Forward Pipeline on batch
            mlep_out = mlep_extractor(images)
            entropy_maps = mlep_out["mlep_features"]  # Shape: (B, 9, H-1, W-1)

            b_latency = (time.time() - b_start) * 1000.0
            batch_latencies.append(b_latency)
            total_images_processed += batch_size_curr

            # Record divergence metrics across classes
            for i in range(batch_size_curr):
                mean_score = entropy_maps[i].mean().item()
                if labels[i].item() == 0:
                    real_entropy_scores.append(mean_score)
                else:
                    ai_entropy_scores.append(mean_score)

            logger.info(
                f"Batch [{batch_idx + 1}/{len(loader)}] processed in {b_latency:.1f}ms "
                f"({batch_size_curr / (b_latency / 1000.0):.1f} img/s) | "
                f"Input: {tuple(images.shape)} -> MLEP Features: {tuple(entropy_maps.shape)}"
            )

            # Export visualizations for the first batch
            if args.export_visualizations and batch_idx == 0:
                logger.info("Exporting sample diagnostic figures for Batch 1...")
                sample_img = images[0:1]  # Slice first sample (1, 3, 256, 256)
                sample_out = mlep_extractor(sample_img)

                plot_entropy_heatmap(sample_img, sample_out["mlep_features"], scale_idx=0, save_path=vis_dir / "batch1_sample0_mlep_heatmap.png")
                plot_multiscale_entropy(sample_out["mlep_features"], save_path=vis_dir / "batch1_sample0_mlep_multiscale.png")
                logger.info(f"Batch 1 visualizations exported to: {vis_dir.resolve()}")

    # ==================== STEP 4: PIPELINE EXECUTION SUMMARY REPORT ====================
    print("\n" + "=" * 80)
    print("STEP 4: PIPELINE EXECUTION SUMMARY REPORT")
    print("=" * 80)

    avg_latency = np.mean(batch_latencies) if batch_latencies else 0.0
    throughput = (total_images_processed / (sum(batch_latencies) / 1000.0)) if batch_latencies else 0.0
    mean_real_entropy = np.mean(real_entropy_scores) if real_entropy_scores else 0.0
    mean_ai_entropy = np.mean(ai_entropy_scores) if ai_entropy_scores else 0.0

    summary_data = {
        "dataset_root": str(data_path.resolve()),
        "total_images_processed": total_images_processed,
        "batches_processed": len(loader),
        "performance": {
            "avg_batch_latency_ms": round(float(avg_latency), 2),
            "throughput_images_per_sec": round(float(throughput), 2),
        },
        "steganalysis_metrics": {
            "mean_entropy_real": round(float(mean_real_entropy), 4),
            "mean_entropy_ai_generated": round(float(mean_ai_entropy), 4),
            "divergence_contrast_ratio": round(float(mean_ai_entropy / (mean_real_entropy + 1e-8)), 2),
        },
        "artifacts_generated": {
            "split_manifests_dir": str(manifest_dir.resolve()),
            "visualizations_dir": str(vis_dir.resolve()),
        },
    }

    report_path = out_dir / "execution_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Execution report successfully saved to: {report_path.resolve()}")
    print("\n" + "-" * 50)
    print(f"Total Images Processed  : {total_images_processed}")
    print(f"Throughput              : {throughput:.1f} images/second ({avg_latency:.1f} ms/batch)")
    print(f"Real Mean Entropy Score : {mean_real_entropy:.4f}")
    print(f"AI Mean Entropy Score   : {mean_ai_entropy:.4f}")
    print(f"Divergence Contrast     : {summary_data['steganalysis_metrics']['divergence_contrast_ratio']}x")
    print(f"Visualizations Saved To : {vis_dir.resolve()}")
    print("-" * 50 + "\n")
    logger.info("Project execution finished successfully.")

    if args.export_visualizations:
        print("\n[5] Automatically opening the generated project previews...")
        try:
            mlep_img = vis_dir / "batch1_sample0_mlep_heatmap.png"
            mlep_multi = vis_dir / "batch1_sample0_mlep_multiscale.png"
            
            import subprocess
            def open_file(filepath):
                path_str = str(filepath.resolve())
                file_url = f"file:///{path_str.replace(chr(92), '/')}"
                
                brave_path = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
                librewolf_path = r"C:\Program Files\LibreWolf\librewolf.exe"
                
                if os.name == 'nt' and os.path.exists(brave_path):
                    subprocess.Popen([brave_path, path_str])
                elif os.name == 'nt' and os.path.exists(librewolf_path):
                    subprocess.Popen([librewolf_path, path_str])
                else:
                    import webbrowser
                    webbrowser.open(file_url)
                    
                print(f"    [Fallback] If it didn't pop up, please Ctrl+Click the URL below:\n    {file_url}")
            
            if mlep_img.exists():
                open_file(mlep_img)
            if mlep_multi.exists():
                open_file(mlep_multi)
        except Exception as e:
            print(f"    Failed to auto-open project previews: {e}")



if __name__ == "__main__":
    main()
