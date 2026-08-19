"""
Fusion Model: Master 2-Stage Training Loop.

Stage 1: Supervised Contrastive Pre-Training (SupCon Alignment)
Stage 2: Gated Classification Fine-Tuning with MoE + DANN

Usage:
    python scripts/fusion/train.py --config configs/fusion/train_fusion_genimage.yaml
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.device import get_compute_device, set_global_seed
from src.utils.logger import get_logger
from src.models.arch1_supcon.trainer import train as train_stage1_supcon
from src.models.fusion.model import DualCueAIGIDModel

logger = get_logger("fusion.trainer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fusion Model: 2-Stage Training (SupCon -> Gated Fine-Tuning)"
    )
    parser.add_argument("--config", type=str, default="configs/fusion/train_fusion_genimage.yaml")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=None, help="Run only specific stage.")
    parser.add_argument("--stage1-ckpt", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--epochs-stage1", type=int, default=50)
    parser.add_argument("--epochs-stage2", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/fusion")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def create_synthetic_loader(batch_size: int = 32, num_batches: int = 5):
    """Create a synthetic DataLoader for integration testing."""
    total = batch_size * num_batches
    images = torch.randint(0, 256, (total, 3, 256, 256), dtype=torch.float32)
    labels = torch.randint(0, 2, (total,), dtype=torch.long)
    domains = torch.randint(0, 8, (total,), dtype=torch.long)
    dataset = TensorDataset(images, labels, domains)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def train_stage2(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    device: torch.device,
    stage1_ckpt: Optional[Path] = None,
) -> Path:
    """Stage 2: Gated classification fine-tuning."""
    logger.info("=" * 70)
    logger.info("FUSION STAGE 2: Gated Classification Fine-Tuning")
    logger.info("=" * 70)

    model = DualCueAIGIDModel(
        backbone_name=args.backbone,
        pretrained=True,
        use_frequency_filter=True,
        use_cross_attention=True,
        use_moe=True,
        use_dann=True,
        num_domains=8,
        num_experts=4,
        top_k=2,
        d_model=256,
        num_heads=8,
    ).to(device)

    # Load Stage 1 stems if checkpoint provided
    if stage1_ckpt is not None and Path(stage1_ckpt).exists():
        logger.info(f"Loading Stage 1 checkpoint: {stage1_ckpt}")
        ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=False)
        stage1_state = ckpt["model_state_dict"]

        model_state = model.state_dict()
        transferred = 0
        for key, value in stage1_state.items():
            for prefix_src, prefix_dst in [
                ("mlep_backbone.", "mlep_stem."),
                ("lota_backbone.", "lota_stem."),
            ]:
                if key.startswith(prefix_src):
                    new_key = key.replace(prefix_src, prefix_dst)
                    if new_key in model_state and model_state[new_key].shape == value.shape:
                        model_state[new_key] = value
                        transferred += 1

        model.load_state_dict(model_state, strict=False)
        logger.info(f"  Transferred {transferred} parameters from Stage 1")

    bce_criterion = nn.BCEWithLogitsLoss()
    domain_criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr * 0.5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_stage2, eta_min=1e-6
    )

    use_amp = (device.type == "cuda") and (args.precision != "fp32")
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = Path(args.output_dir) / "checkpoints" / "stage2_fusion"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_ckpt_path = ckpt_dir / "best_model.pt"

    lambda_max = 0.5
    warmup_epochs = 5

    for epoch in range(1, args.epochs_stage2 + 1):
        model.train()

        current_lambda = lambda_max * min(epoch / warmup_epochs, 1.0)
        if hasattr(model, 'dann_head') and model.dann_head is not None:
            if hasattr(model.dann_head, 'grl'):
                model.dann_head.grl.set_lambda(current_lambda)

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels, *meta) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
            domain_labels = meta[0].to(device) if len(meta) > 0 and isinstance(meta[0], torch.Tensor) else torch.zeros(images.size(0), dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = model(images)
                class_logits = outputs["class_logits"]
                domain_logits = outputs.get("domain_logits")
                aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=device))

                loss_cls = bce_criterion(class_logits, labels)
                loss_domain = domain_criterion(domain_logits, domain_labels) if domain_logits is not None else torch.tensor(0.0, device=device)
                total_loss = loss_cls + 0.1 * loss_domain + 0.01 * aux_loss

            scaler.scale(total_loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"  [Epoch {epoch}/{args.epochs_stage2}] Batch {batch_idx}/{len(train_loader)} "
                    f"Loss: {total_loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        scheduler.step()
        logger.info(f"Epoch {epoch}/{args.epochs_stage2} — Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
            }, best_ckpt_path)
            logger.info(f"  ✓ Saved best checkpoint (loss={best_loss:.4f})")

    logger.info(f"Stage 2 complete. Best loss: {best_loss:.4f}")
    return best_ckpt_path


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = get_compute_device()

    logger.info(f"Device: {device} | Backbone: {args.backbone}")

    train_loader = create_synthetic_loader(args.batch_size)
    val_loader = create_synthetic_loader(args.batch_size, num_batches=2)

    stage1_ckpt = args.stage1_ckpt

    if args.stage is None or args.stage == 1:
        # Delegate to Arch1 trainer for Stage 1
        from types import SimpleNamespace
        stage1_args = SimpleNamespace(
            backbone=args.backbone, epochs=args.epochs_stage1, batch_size=args.batch_size,
            lr=args.lr, proj_dim=128, temperature=0.07, grad_clip=args.grad_clip,
            precision=args.precision, seed=args.seed,
            output_dir=str(Path(args.output_dir) / "stage1_supcon"), num_workers=args.num_workers,
        )
        stage1_ckpt = train_stage1_supcon(train_loader, val_loader, stage1_args, device)
        logger.info(f"Stage 1 checkpoint: {stage1_ckpt}")

    if args.stage is None or args.stage == 2:
        stage2_ckpt = train_stage2(
            train_loader, val_loader, args, device,
            stage1_ckpt=Path(stage1_ckpt) if stage1_ckpt else None,
        )
        logger.info(f"Stage 2 checkpoint: {stage2_ckpt}")

    logger.info("Fusion training complete.")


if __name__ == "__main__":
    main()
