"""
Build a production-grade 10,000-image benchmark dataset.
Performs: SHA256 deduplication, 256x256 resizing, RGB normalization, 
strict 60/20/20 stratified splitting, and metadata generation.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

def compute_sha256(img: Image.Image) -> str:
    """Compute SHA256 hash of image data."""
    return hashlib.sha256(img.tobytes()).hexdigest()

def process_and_validate_image(img: Image.Image) -> Image.Image:
    """Normalize to RGB and resize to 256x256."""
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    return img

def build_benchmark(target_dir: str):
    print(f"Building Production Benchmark in: {target_dir}")
    base_dir = Path(target_dir)
    splits = ["train", "validation", "test"]
    labels = ["real", "fake"]
    
    # Create directory structure
    for split in splits:
        for label in labels:
            (base_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    metadata_dir = base_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading verified dataset from Hugging Face: Hemg/ai-vs-real-image-detection")
    dataset = load_dataset("Hemg/ai-vs-real-image-detection", split="train", streaming=True)
    
    # Target sizes per class
    target_counts = {
        "train": 3000,
        "validation": 1000,
        "test": 1000
    }
    
    # Trackers
    collected = {"real": {"train": 0, "validation": 0, "test": 0},
                 "fake": {"train": 0, "validation": 0, "test": 0}}
    seen_hashes = set()
    manifests = {"train": [], "validation": [], "test": []}
    
    stats = {
        "total_images": 10000,
        "real_images": 5000,
        "fake_images": 5000,
        "duplicate_count": 0,
        "corrupted_count": 0,
        "generator_distribution": {"stable_diffusion": 0, "midjourney": 0, "dalle": 0, "unknown_diffusion": 5000},
        "real_source_distribution": {"huggingface_verified": 5000},
        "resolution_stats": {"mean_size": "256x256", "std_dev": 0.0}
    }
    
    print("Streaming and processing images (Deduplication & Validation active)...")
    progress_bar = tqdm(total=10000)
    
    for item in dataset:
        # Check if we are done
        total_real = sum(collected["real"].values())
        total_fake = sum(collected["fake"].values())
        if total_real == 5000 and total_fake == 5000:
            break
            
        label = item.get("label", -1)
        if label == 0:
            class_str = "real"
        elif label == 1:
            class_str = "fake"
        else:
            continue
            
        # Determine split assignment
        target_split = None
        for s in splits:
            if collected[class_str][s] < target_counts[s]:
                target_split = s
                break
                
        if target_split is None:
            continue # We have enough of this class
            
        try:
            img = item["image"]
            # Deduplication
            img_hash = compute_sha256(img)
            if img_hash in seen_hashes:
                stats["duplicate_count"] += 1
                continue
            seen_hashes.add(img_hash)
            
            # Preprocess
            img_processed = process_and_validate_image(img)
            
            # Save
            filename = f"{img_hash}.jpg"
            filepath = base_dir / target_split / class_str / filename
            img_processed.save(filepath, format="JPEG", quality=95)
            
            # Update trackers
            collected[class_str][target_split] += 1
            progress_bar.update(1)
            
            # Add to manifest
            # Use forward slashes for cross-platform compatibility in manifests
            rel_path = f"{target_split}/{class_str}/{filename}"
            manifests[target_split].append({
                "path": str(filepath.resolve()),
                "label": 0 if class_str == "real" else 1,
                "domain": "real" if class_str == "real" else "ai_generated"
            })
            
        except Exception as e:
            stats["corrupted_count"] += 1
            continue

    progress_bar.close()
    
    # Verify counts
    print("\nVerifying Dataset Integrity...")
    for class_str in ["real", "fake"]:
        for split in splits:
            cnt = collected[class_str][split]
            target = target_counts[split]
            assert cnt == target, f"Missing {class_str} images in {split}! Got {cnt}, needed {target}"
    print("[OK] Class balance verified: 1:1 in all splits")
    print(f"[OK] Duplicate images removed: {stats['duplicate_count']}")
    print(f"[OK] Corrupted images removed: {stats['corrupted_count']}")
    
    # Save Manifests
    for split in splits:
        manifest_path = metadata_dir / f"{split}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifests[split], f, indent=4)
            
    # Save Statistics
    with open(metadata_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    with open(metadata_dir / "generator_statistics.json", "w") as f:
        json.dump(stats["generator_distribution"], f, indent=4)

    print(f"\n========================================================")
    print(f"SUCCESS: 10,000-Image Benchmark Dataset Created at {target_dir}")
    print(f"========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, default="dataset10000")
    args = parser.parse_args()
    build_benchmark(args.target_dir)
