#!/usr/bin/env python3
"""
Dual-Cue Feature Fusion GPU Training Script
Combines RGB Spatial Semantics & LOTA LSB Bit-Plane Noise via Dual ResNet-50 Streams.
Optimized for high generalization accuracy with lr=1e-4, Cosine Annealing, and Dropout(0.5).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.dataset import SharedImageDataset
from src.data.dataloader import create_dataloader
from src.models.legacy.dual_cue import DualCueClassifier
from src.utils.logger import get_logger


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    all_targets, all_probs = [], []

    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device, dtype=torch.float32)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images).squeeze(-1)
            # Label smoothing to prevent over-confident memorization
            labels_smoothed = labels * 0.95 + 0.025
            loss = criterion(logits, labels_smoothed)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, np.array(all_probs) >= 0.5)
    epoch_auc = roc_auc_score(all_targets, all_probs)
    return epoch_loss, epoch_acc, epoch_auc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets, all_probs = [], []

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device, dtype=torch.float32)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images).squeeze(-1)
                loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred_prob = np.array(all_probs)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "loss": float(loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }
    return loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Cue Feature Fusion Model on GPU")
    parser.add_argument("--data_dir", type=str, default="dataset10000", help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "rmsprop"], help="Optimizer choice")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("dual_cue_train", log_dir=output_dir, log_filename="training.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Activated 100% GPU Capacity: cuDNN Benchmark + TF32 Matrix Math Cores on CUDA.")

    # Datasets & DataLoaders
    logger.info(f"Loading dataset splits from {args.data_dir}...")
    train_ds = SharedImageDataset(root_dir=args.data_dir, split="train")
    val_ds = SharedImageDataset(root_dir=args.data_dir, split="validation")
    test_ds = SharedImageDataset(root_dir=args.data_dir, split="test")

    train_loader = create_dataloader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    logger.info(f"Dataset Loaded | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model & Optimization Setup (Frozen early layers, dropout=0.6)
    model = DualCueClassifier(dropout_rate=0.6, freeze_early_layers=True).to(device)
    criterion = nn.BCEWithLogitsLoss()

    opt_choice = args.optimizer.lower()
    if opt_choice == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_choice == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_choice == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * 5, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    elif opt_choice == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.99, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info(f"Configured Optimizer: {opt_choice.upper()} (LR: {args.lr}, Weight Decay: {args.weight_decay})")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_auc = 0.0
    patience_counter = 0
    history = {"epochs": [], "train": [], "val": []}

    logger.info(f"Starting Dual-Cue Model Training for {args.epochs} Epochs (lr={args.lr})...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        val_acc = val_metrics["accuracy"]
        val_auc = val_metrics["roc_auc"]

        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
        )

        history["epochs"].append(epoch)
        history["train"].append({"loss": train_loss, "accuracy": train_acc, "roc_auc": train_auc})
        history["val"].append(val_metrics)

        # Checkpoint Best Model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            checkpoint_path = output_dir / "best_dual_cue_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            logger.info(f"--> Saved best model checkpoint to {checkpoint_path} (Val AUC: {val_auc:.4f} | Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            logger.info(f"--> Early stopping patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered! Validation AUC did not improve.")
                break

    # Save History
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    logger.info("Training Complete!")


if __name__ == "__main__":
    main()
