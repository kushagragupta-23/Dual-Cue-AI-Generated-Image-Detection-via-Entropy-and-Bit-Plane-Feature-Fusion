#!/usr/bin/env python3
"""
Multi-Optimizer Benchmark Suite for Dual-Cue AI-Generated Image Detection
Trains and compares AdamW, Adam, SGD, and RMSprop on 10,000-image benchmark dataset (dataset10000) using 100% GPU Capacity.
Saves comprehensive metrics report to outputs/optimizer_benchmark_results.json.
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.dataset import SharedImageDataset
from src.data.dataloader import create_dataloader
from src.models.legacy.dual_cue import DualCueClassifier
from src.utils.logger import get_logger

logger = get_logger("optimizer_benchmark")


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        auc = 0.5
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    all_targets, all_probs = [], []

    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        smoothed_targets = labels * 0.95 + 0.025

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images).squeeze(-1)
            loss = criterion(logits, smoothed_targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = float(running_loss / len(loader.dataset))
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    return epoch_loss, metrics["accuracy"], metrics["roc_auc"]


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

    eval_loss = float(running_loss / len(loader.dataset))
    metrics = compute_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = eval_loss
    return eval_loss, metrics


def benchmark_optimizers(
    data_dir: str = "dataset10000",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    output_dir: str = "outputs/benchmark_optimizers",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info("Loading dataset splits...")
    train_ds = SharedImageDataset(root_dir=data_dir, split="train")
    val_ds = SharedImageDataset(root_dir=data_dir, split="validation")
    test_ds = SharedImageDataset(root_dir=data_dir, split="test")

    train_loader = create_dataloader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, num_workers=0, shuffle=False)
    test_loader = create_dataloader(test_ds, batch_size=batch_size, num_workers=0, shuffle=False)

    optimizers_to_test = ["AdamW", "Adam", "SGD", "RMSprop"]
    results_summary = {}

    for opt_name in optimizers_to_test:
        logger.info("=" * 80)
        logger.info(f"BENCHMARKING OPTIMIZER: {opt_name.upper()} on 100% GPU Capacity")
        logger.info("=" * 80)

        model = DualCueClassifier(dropout_rate=0.6, freeze_early_layers=True).to(device)
        criterion = nn.BCEWithLogitsLoss()

        if opt_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr * 5, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        elif opt_name == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, weight_decay=weight_decay)

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        opt_history = {"epochs": [], "train_loss": [], "train_acc": [], "train_auc": [], "val_loss": [], "val_acc": [], "val_auc": [], "epoch_time_sec": []}
        best_val_auc = 0.0
        best_checkpoint_path = out_path / f"best_model_{opt_name.lower()}.pth"

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_auc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            v_loss, v_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            dt = time.time() - t0

            v_acc = v_metrics["accuracy"]
            v_auc = v_metrics["roc_auc"]

            opt_history["epochs"].append(epoch)
            opt_history["train_loss"].append(round(tr_loss, 4))
            opt_history["train_acc"].append(round(tr_acc, 4))
            opt_history["train_auc"].append(round(tr_auc, 4))
            opt_history["val_loss"].append(round(v_loss, 4))
            opt_history["val_acc"].append(round(v_acc, 4))
            opt_history["val_auc"].append(round(v_auc, 4))
            opt_history["epoch_time_sec"].append(round(dt, 2))

            logger.info(
                f"[{opt_name}] Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} AUC: {tr_auc:.4f} | "
                f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} AUC: {v_auc:.4f}"
            )

            if v_auc > best_val_auc:
                best_val_auc = v_auc
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_auc": v_auc, "val_acc": v_acc}, best_checkpoint_path)

        # Test Set Evaluation for this optimizer
        logger.info(f"Evaluating {opt_name} Best Checkpoint on Test Set (2,000 images)...")
        best_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        t_loss, test_metrics = evaluate(model, test_loader, criterion, device)

        results_summary[opt_name] = {
            "trajectory": opt_history,
            "test_metrics": {
                "loss": round(t_loss, 4),
                "accuracy": round(test_metrics["accuracy"], 4),
                "precision": round(test_metrics["precision"], 4),
                "recall": round(test_metrics["recall"], 4),
                "f1_score": round(test_metrics["f1_score"], 4),
                "roc_auc": round(test_metrics["roc_auc"], 4),
                "tn": test_metrics["tn"],
                "fp": test_metrics["fp"],
                "fn": test_metrics["fn"],
                "tp": test_metrics["tp"],
            },
            "avg_epoch_sec": round(float(np.mean(opt_history["epoch_time_sec"])), 2),
        }

    results_file = Path("outputs") / "optimizer_benchmark_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)
    logger.info(f"Successfully saved all optimizer benchmark results to {results_file}")
    return results_summary


if __name__ == "__main__":
    benchmark_optimizers()
