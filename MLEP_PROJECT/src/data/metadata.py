"""
Dataset integrity checking, metadata scanning, and statistical summary generation.
Supports GenImage, ForenSynths, and custom dataset directory structures.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
from src.utils.logger import get_logger

logger = get_logger("dataset_metadata")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def validate_image_file(file_path: Union[str, Path]) -> bool:
    """
    Verify image header readability and RGB compatibility.

    Args:
        file_path: Path to target image file.

    Returns:
        bool: True if image is readable and uncorrupted, False otherwise.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Corrupted or unreadable image file ignored: {path} ({e})")
        return False


def _infer_label_and_domain(file_path: Path) -> Tuple[int, str]:
    """
    Infer Real (0) vs AI-Generated (1) class label and generator domain from file path.

    Args:
        file_path: Absolute path to the image file.

    Returns:
        tuple: (label: int, domain: str)
    """
    parts_lower = [p.lower() for p in file_path.parts]

    # Check standard indicators for real images
    if any(w in parts_lower for w in ["real", "0_real", "nature", "authentic", "original", "imagenet", "lsun", "val_real", "train_real"]):
        label = 0
        domain = "real"
    else:
        label = 1
        domain = "ai_custom"
        # Match standard generative model domains (ForenSynths & GenImage)
        known_generators = [
            "stylegan3", "stylegan2", "stylegan", "progan", "biggan", "cyclegan",
            "stargan", "gaugan", "sdv15", "sdv14", "sdxl", "midjourney", "flux",
            "adm", "vqdm", "wukong", "glide", "dalle3", "dalle2", "dalle", "imagen"
        ]
        for gen in known_generators:
            if any(gen in part for part in parts_lower):
                domain = gen
                break

    return label, domain


def scan_dataset_directory(
    root_dir: Union[str, Path],
    validate_integrity: bool = True,
) -> List[Dict[str, Any]]:
    """
    Scan dataset root directory for image files, infer metadata, and filter corrupted files.

    Args:
        root_dir: Root path of the dataset.
        validate_integrity: If True, performs header integrity verification on every file.

    Returns:
        list of dicts: List of valid sample entries with keys 'path', 'label', and 'domain'.
    """
    root = Path(root_dir)
    if not root.exists():
        logger.error(f"Dataset root directory does not exist: {root}")
        return []

    samples: List[Dict[str, Any]] = []
    logger.info(f"Scanning dataset directory: {root.resolve()} (integrity_check={validate_integrity})")

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            if validate_integrity and not validate_image_file(path):
                continue
            
            label, domain = _infer_label_and_domain(path)
            samples.append({
                "path": str(path.resolve()),
                "label": label,
                "domain": domain,
            })

    logger.info(f"Successfully scanned {len(samples)} valid image samples.")
    return samples


def generate_metadata_summary(
    samples: List[Dict[str, Any]],
    root_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Calculate statistical distributions and class balance summaries for scanned samples.

    Args:
        samples: List of sample dictionaries returned by scan_dataset_directory.
        root_dir: Optional root directory string to include in metadata.

    Returns:
        dict: Summary statistics dictionary.
    """
    total = len(samples)
    real_count = sum(1 for s in samples if s.get("label") == 0)
    fake_count = sum(1 for s in samples if s.get("label") == 1)

    domain_counts: Dict[str, int] = {}
    for s in samples:
        dom = s.get("domain", "unknown")
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    summary = {
        "root_directory": str(Path(root_dir).resolve()) if root_dir is not None else "N/A",
        "total_samples": total,
        "class_distribution": {
            "real_count": real_count,
            "ai_generated_count": fake_count,
            "real_ratio": round(real_count / total, 4) if total > 0 else 0.0,
            "ai_generated_ratio": round(fake_count / total, 4) if total > 0 else 0.0,
        },
        "generator_domain_counts": domain_counts,
    }
    return summary


def export_metadata_summary(
    summary: Dict[str, Any],
    export_path: Union[str, Path],
) -> None:
    """
    Export generated metadata summary to a formatted JSON file.

    Args:
        summary: Statistical summary dictionary.
        export_path: Destination JSON file path.
    """
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    logger.info(f"Dataset metadata summary exported to: {path}")
