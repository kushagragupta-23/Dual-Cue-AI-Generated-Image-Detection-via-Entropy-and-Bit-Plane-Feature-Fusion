"""
Architecture I: Standalone SupCon Pre-Training Loop.

Trains LearnableFreqSupConNet to align MLEP and LOTA representations
on the L2 hypersphere using supervised contrastive loss.
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.device import get_compute_device, set_global_seed
from src.utils.logger import get_logger
from src.shared.extractors import VectorizedMLEPExtractor, TopKLOTAExtractor
from src.models.arch1_supcon.model import LearnableFreqSupConNet

logger = get_logger("arch1_supcon.trainer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Architecture I: SupCon Contrastive Pre-Training"
    )
    parser.add_argument("--config", type=str, default="configs/arch1_supcon/train.yaml")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/arch1_supcon")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def create_synthetic_loader(batch_size: int = 32, num_batches: int = 5):
    """Create a synthetic DataLoader for integration testing."""
    total = batch_size * num_batches
    images = torch.randint(0, 256, (total, 3, 256, 256), dtype=torch.float32)
    labels = torch.randint(0, 2, (total,), dtype=torch.long)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def train(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    args,
    device: torch.device,
) -> Path:
    """
    Train Architecture I (SupCon contrastive pre-training).

    Returns:
        Path to the best checkpoint.
    """
    logger.info("=" * 70)
    logger.info("ARCHITECTURE I: SupCon Contrastive Pre-Training")
    logger.info("=" * 70)

    # Initialize extractors
    mlep_extractor = VectorizedMLEPExtractor().to(device)
    lota_extractor = TopKLOTAExtractor().to(device)

    # Build model
    model = LearnableFreqSupConNet(
        mlep_extractor=mlep_extractor,
        lota_extractor=lota_extractor,
        backbone_name=args.backbone,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # AMP setup
    use_amp = (device.type == "cuda") and (args.precision != "fp32")
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Checkpointing
    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_ckpt_path = ckpt_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels, *_) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                z_mlep, z_lota, loss = model(images, labels)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"  [Epoch {epoch}/{args.epochs}] "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{args.epochs} — "
            f"Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "backbone": args.backbone,
            }, best_ckpt_path)
            logger.info(f"  ✓ Saved best checkpoint (loss={best_loss:.4f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
            }, ckpt_dir / f"epoch_{epoch}.pt")

    logger.info(f"Arch1 SupCon training complete. Best loss: {best_loss:.4f}")
    return best_ckpt_path


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = get_compute_device()

    logger.info(f"Device: {device} | Backbone: {args.backbone} | Seed: {args.seed}")

    train_loader = create_synthetic_loader(args.batch_size)
    val_loader = create_synthetic_loader(args.batch_size, num_batches=2)

    best_ckpt = train(train_loader, val_loader, args, device)
    logger.info(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
