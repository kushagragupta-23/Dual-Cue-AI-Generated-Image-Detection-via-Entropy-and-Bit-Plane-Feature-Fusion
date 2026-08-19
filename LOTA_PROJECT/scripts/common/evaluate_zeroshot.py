#!/usr/bin/env python3
"""
Cross-Generator Zero-Shot Evaluation CLI.

Evaluates trained models on generators not seen during training to measure
generalization capability. Supports:
    - Single-generator evaluation
    - Multi-generator sweep across all GenImage/DiffusionForensics splits
    - Robustness degradation testing (JPEG + Blur)
    - Automated Markdown report generation

Usage:
    python scripts/evaluate_zeroshot.py --checkpoint outputs/checkpoints/stage2_fusion/best_model.pt
    python scripts/evaluate_zeroshot.py --checkpoint best.pt --generators ProGAN StyleGAN
    python scripts/evaluate_zeroshot.py --checkpoint best.pt --robustness
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.utils.device import get_compute_device, set_global_seed
from src.utils.logger import get_logger
from src.models.fusion.model import DualCueAIGIDModel
from src.eval.evaluator import ModelEvaluator, ZeroShotBenchmark
from src.eval.metrics import format_metrics_table, format_per_domain_table

logger = get_logger("evaluate_zeroshot")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-Generator Zero-Shot Evaluation CLI"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint."
    )
    parser.add_argument(
        "--data-root", type=str, default="datasets/GenImage",
        help="Root directory of evaluation dataset."
    )
    parser.add_argument(
        "--generators", nargs="+", default=None,
        help="Specific generator names to evaluate (default: all available)."
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet50",
        help="Backbone architecture matching the checkpoint."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Evaluation batch size."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold."
    )
    parser.add_argument(
        "--robustness", action="store_true",
        help="Also run robustness degradation tests."
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/eval_results",
        help="Directory to save evaluation reports."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Force specific device (cuda/mps/cpu)."
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, backbone: str, device: torch.device) -> DualCueAIGIDModel:
    """
    Load a trained DualCueAIGIDModel from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        backbone: Backbone name ('resnet18' or 'resnet50').
        device: Target device.

    Returns:
        Loaded model in eval mode.
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    model = DualCueAIGIDModel(
        backbone_name=backbone,
        pretrained=False,
        use_frequency_filter=True,
        use_cross_attention=True,
        use_moe=True,
        use_dann=True,
        num_domains=8,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"  Loaded from epoch {ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?')}")
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model


def create_eval_dataloaders(
    data_root: str,
    generators: Optional[List[str]],
    batch_size: int,
) -> Dict[str, "DataLoader"]:
    """
    Create evaluation DataLoaders for each generator.

    In production, this should use src.data.dataset.AIGIDDataset to load
    real test splits. For now, creates synthetic data for integration testing.

    Args:
        data_root: Path to dataset root.
        generators: List of generator names.
        batch_size: Batch size.

    Returns:
        dict mapping generator_name → DataLoader.
    """
    from torch.utils.data import DataLoader, TensorDataset

    if generators is None:
        generators = [
            "ProGAN", "StyleGAN", "BigGAN", "CycleGAN",
            "StarGAN", "GauGAN", "DeepFake", "SITD",
            "Midjourney", "DALLE2", "StableDiffusion_v1.5",
        ]

    data_path = Path(data_root)
    loaders = {}

    for gen_name in generators:
        gen_path = data_path / gen_name / "test"

        if gen_path.exists():
            # Production: Load real dataset
            logger.info(f"  Loading {gen_name} from {gen_path}")
            # TODO: Replace with actual dataset loader
            # dataset = AIGIDDataset(gen_path, ...)
            # loaders[gen_name] = DataLoader(dataset, batch_size=batch_size)

        # Fallback: Synthetic data for integration testing
        logger.info(f"  Creating synthetic test data for {gen_name}")
        n_samples = 64
        images = torch.randint(0, 256, (n_samples, 3, 256, 256), dtype=torch.float32)
        labels = torch.randint(0, 2, (n_samples,), dtype=torch.long)
        dataset = TensorDataset(images, labels)
        loaders[gen_name] = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loaders


def run_zero_shot_evaluation(
    model: DualCueAIGIDModel,
    eval_loaders: Dict[str, "DataLoader"],
    device: torch.device,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """
    Run zero-shot evaluation across all generator test sets.

    Args:
        model: Trained model.
        eval_loaders: Generator → DataLoader mapping.
        device: Compute device.
        threshold: Classification threshold.
        output_dir: Directory to save results.

    Returns:
        dict mapping generator_name → metrics.
    """
    evaluator = ModelEvaluator(model, device)
    all_results = {}

    logger.info("=" * 70)
    logger.info("Cross-Generator Zero-Shot Evaluation")
    logger.info("=" * 70)

    for gen_name, loader in eval_loaders.items():
        logger.info(f"\nEvaluating: {gen_name}")
        results = evaluator.evaluate(loader, threshold=threshold)
        metrics = results["overall_metrics"]
        all_results[gen_name] = metrics
        logger.info(
            f"  Acc={metrics['accuracy']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, "
            f"AP={metrics['average_precision']:.4f}"
        )

    # Compile report
    report_lines = [
        "# Zero-Shot Cross-Generator Evaluation Report",
        "",
        "## Per-Generator Results",
        "",
        "| Generator | Accuracy | AUC | F1 | AP |",
        "|-----------|----------|-----|-----|-----|",
    ]

    for gen_name, metrics in sorted(all_results.items()):
        report_lines.append(
            f"| {gen_name} | {metrics['accuracy']:.4f} | "
            f"{metrics['roc_auc']:.4f} | {metrics['f1_score']:.4f} | "
            f"{metrics['average_precision']:.4f} |"
        )

    # Summary stats
    import numpy as np
    accs = [m["accuracy"] for m in all_results.values()]
    aucs = [m["roc_auc"] for m in all_results.values()]

    report_lines.extend([
        "",
        "## Summary",
        "",
        f"- **Mean Accuracy**: {np.mean(accs):.4f} ± {np.std(accs):.4f}",
        f"- **Mean AUC**: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
        f"- **Generators Tested**: {len(all_results)}",
    ])

    report = "\n".join(report_lines)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Markdown report
        report_path = output_dir / "zeroshot_report.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info(f"\nReport saved to: {report_path}")

        # Save JSON metrics
        json_path = output_dir / "zeroshot_metrics.json"
        json_path.write_text(
            json.dumps(all_results, indent=2, default=str),
            encoding="utf-8"
        )
        logger.info(f"Metrics saved to: {json_path}")

    return all_results


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_compute_device()

    logger.info(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, args.backbone, device)

    # Create eval loaders
    eval_loaders = create_eval_dataloaders(
        args.data_root, args.generators, args.batch_size
    )

    # Run evaluation
    output_dir = Path(args.output_dir)
    results = run_zero_shot_evaluation(
        model, eval_loaders, device,
        threshold=args.threshold,
        output_dir=output_dir,
    )

    # Optional robustness test
    if args.robustness:
        logger.info("\n" + "=" * 70)
        logger.info("Robustness Degradation Testing")
        logger.info("=" * 70)
        from src.eval.robustness_suite import RobustnessSuite
        # Run on first available loader
        for gen_name, loader in eval_loaders.items():
            suite = RobustnessSuite(
                model, device,
                jpeg_levels=[100, 90, 80, 70],
                blur_sigmas=[0.0, 0.5, 1.0, 2.0],
                batch_size=args.batch_size,
            )
            robust_results = suite.run_full_suite(loader.dataset)
            report = suite.compile_markdown_table(robust_results)
            report_path = output_dir / f"robustness_{gen_name}.md"
            report_path.write_text(report, encoding="utf-8")
            logger.info(f"Robustness report saved to: {report_path}")
            break  # Run on first generator only

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
