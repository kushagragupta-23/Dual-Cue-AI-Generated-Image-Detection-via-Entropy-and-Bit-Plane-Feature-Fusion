"""
HydraFusion-Net: Cross-Generator Zero-Shot Evaluation CLI.

Usage:
    python scripts/evaluate_zeroshot.py --checkpoint outputs/checkpoints/hydrafusion_best.pt
    python scripts/evaluate_zeroshot.py --checkpoint outputs/checkpoints/hydrafusion_best.pt --robustness
    python scripts/evaluate_zeroshot.py --checkpoint outputs/checkpoints/hydrafusion_best.pt --gradcam --num_gradcam 8

This script:
  1. Loads the trained HydraFusion-Net checkpoint
  2. Evaluates on the test set with full academic metrics (Accuracy, AP, ROC-AUC, F1)
  3. Optionally runs robustness degradation sweep (JPEG + Blur)
  4. Optionally generates Grad-CAM visualizations
  5. Exports results as JSON and Markdown reports
"""

import os
import sys
import json
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hydrafusion_net import HydraFusionNet
from src.data.dataset import get_dataloaders
from src.utils.device import get_compute_device, set_global_seed
from src.utils.logger import get_logger
from src.eval.metrics import ForensicMetricsCalculator, format_metrics_table
from src.eval.evaluator import HydraFusionEvaluator
from src.eval.robustness_suite import RobustnessSuite

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HydraFusion-Net Zero-Shot Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/hydrafusion_best.pt",
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/results",
        help="Directory to save evaluation reports",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run robustness degradation sweep (JPEG + Blur)",
    )
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate Grad-CAM visualizations for sample images",
    )
    parser.add_argument(
        "--num_gradcam",
        type=int,
        default=16,
        help="Number of Grad-CAM visualizations to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (default: from config)",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> HydraFusionNet:
    """Load HydraFusion-Net from checkpoint."""
    model = HydraFusionNet(freeze_backbones=True).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return model


def main() -> None:
    # Windows console encoding fix
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args()
    set_global_seed(42)
    device = get_compute_device()
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size

    # Load data
    _, _, test_loader = get_dataloaders(config)
    logger.info(f"Test set: {len(test_loader.dataset)} images, {len(test_loader)} batches")

    # Load model
    model = load_model(args.checkpoint, device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════
    #  1. Standard Test-Set Evaluation
    # ═══════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("STANDARD TEST-SET EVALUATION")
    logger.info("=" * 60)

    evaluator = HydraFusionEvaluator(model, device, use_amp=True)
    eval_result = evaluator.evaluate(test_loader, desc="Test Evaluation", collect_alphas=True)

    metrics = eval_result["metrics"]
    logger.info(f"  {metrics.summary_string()}")

    # Export reports
    report_path = evaluator.export_report(
        eval_result, output_dir=str(out_dir), report_name="test_evaluation"
    )
    logger.info(f"  Report: {report_path}")

    # Print confusion matrix
    if metrics.confusion_matrix_raw is not None:
        cm = metrics.confusion_matrix_raw
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    Predicted:    Real   Fake")
        logger.info(f"    Actual Real:  {cm[0][0]:5d}  {cm[0][1]:5d}")
        logger.info(f"    Actual Fake:  {cm[1][0]:5d}  {cm[1][1]:5d}")

    # Print gating weights
    if eval_result["alphas"] is not None:
        alphas = eval_result["alphas"]
        head_names = [
            "SpatialAttn MLEP->LOTA",
            "SpatialAttn LOTA->MLEP",
            "Channel SE",
            "Freq Correlation",
        ]
        logger.info("  Gating Weight Distribution:")
        for name, mean, std in zip(
            head_names, alphas.mean(axis=0), alphas.std(axis=0)
        ):
            logger.info(f"    {name}: mean={mean:.4f}, std={std:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    #  2. Robustness Degradation Sweep (Optional)
    # ═══════════════════════════════════════════════════════════════════
    if args.robustness:
        logger.info("")
        logger.info("=" * 60)
        logger.info("ROBUSTNESS DEGRADATION SWEEP")
        logger.info("=" * 60)

        suite = RobustnessSuite(
            model, device, use_amp=True,
            jpeg_qualities=[100, 90, 80, 70],
            blur_sigmas=[0.0, 0.5, 1.0, 2.0],
        )
        robustness_results = suite.run(test_loader)

        # Print and save
        table = RobustnessSuite.format_table(robustness_results)
        logger.info(f"\n{table}")

        # Save JSON
        with open(out_dir / "robustness_results.json", "w") as f:
            json.dump(robustness_results, f, indent=2)

        # Save Markdown
        with open(out_dir / "robustness_results.md", "w", encoding="utf-8") as f:
            f.write(table)

        logger.info(f"  Robustness results saved to {out_dir}")

    # ═══════════════════════════════════════════════════════════════════
    #  3. Grad-CAM Visualizations (Optional)
    # ═══════════════════════════════════════════════════════════════════
    if args.gradcam:
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"GRAD-CAM VISUALIZATIONS (top {args.num_gradcam} images)")
        logger.info("=" * 60)

        from src.eval.explainability import DualBranchExplainer

        explainer = DualBranchExplainer(model, device, use_amp=True)
        gradcam_dir = out_dir / "gradcam"

        # Grab a batch from test loader
        images_batch, labels_batch = next(iter(test_loader))

        explainer.explain_batch(
            images=images_batch,
            labels=labels_batch,
            output_dir=str(gradcam_dir),
            max_images=args.num_gradcam,
        )
        explainer.cleanup()

        logger.info(f"  Grad-CAM images saved to {gradcam_dir}")

    # ═══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Test Accuracy     : {metrics.accuracy*100:.2f}%")
    logger.info(f"  ROC-AUC           : {metrics.roc_auc:.4f}")
    logger.info(f"  Average Precision : {metrics.average_precision:.4f}")
    logger.info(f"  F1 Score          : {metrics.f1*100:.2f}%")
    logger.info(f"  Latency           : {eval_result['latency_ms']:.2f} ms/image")
    logger.info(f"  Throughput        : {eval_result['throughput_ips']:.1f} img/sec")
    logger.info(f"  Reports           : {out_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
