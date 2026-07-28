#!/usr/bin/env python3
"""
Training Pipeline for the Dual-Cue AI-Generated Image Detector.
Trains the unified DualCueDetector (MLEP + LOTA branches) on the provided dataset.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure root directory is on python path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.dataset import SharedImageDataset
from src.data.dataloader import create_dataloader
from src.models.dual_cue_detector import DualCueDetector
from src.utils.logger import get_logger

logger = get_logger("train_pipeline")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, labels, _) in enumerate(loader):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Calculate accuracy (sigmoid > 0.5 is predicted as class 1)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(loader)}] | "
                f"Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.2f}%"
            )
            
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    epoch_time = time.time() - start_time
    
    return epoch_loss, epoch_acc, epoch_time


def evaluate(model, loader, criterion, device, epoch, split_name="Validation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    logger.info(f"--- Starting {split_name} Evaluation for Epoch {epoch} ---")
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    logger.info(
        f">>> {split_name} Results | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%"
    )
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train DualCueDetector")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints", help="Where to save model weights")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    data_path = root_path / args.data_dir
    output_path = root_path / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_path / "manifests"

    print("=" * 80)
    print("STEP 1: INITIALIZING DATASETS")
    print("=" * 80)

    train_ds = SharedImageDataset(
        root_dir=data_path, split="train", val_ratio=0.2, test_ratio=0.2,
        validate_integrity=False, split_manifest_dir=manifest_dir
    )
    val_ds = SharedImageDataset(
        root_dir=data_path, split="val", val_ratio=0.2, test_ratio=0.2,
        validate_integrity=False, split_manifest_dir=manifest_dir
    )
    test_ds = SharedImageDataset(
        root_dir=data_path, split="test", val_ratio=0.2, test_ratio=0.2,
        validate_integrity=False, split_manifest_dir=manifest_dir
    )
    
    train_loader = create_dataloader(train_ds, batch_size=args.batch_size, balanced_sampling=True)
    val_loader = create_dataloader(val_ds, batch_size=args.batch_size, balanced_sampling=False)
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, balanced_sampling=False)

    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZING DUAL-CUE DETECTOR")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize the complete model (MLEP + LOTA Fusion)
    model = DualCueDetector(pretrained_backbones=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    print("\n" + "=" * 80)
    print(f"STEP 3: STARTING TRAINING LOOP (Epochs: {args.epochs})")
    print("=" * 80)

    best_val_acc = 0.0
    best_model_path = output_path / "dualcue_best.pth"
    history_path = output_path.parent / "training_history.json"
    
    training_history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss, train_acc, epoch_time = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logger.info(f"Epoch {epoch} Training Completed in {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch, "Validation")
        scheduler.step()
        
        # Save history
        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
        })
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2)
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"*** New best validation accuracy: {best_val_acc:.2f}%. Saving checkpoint... ***")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, best_model_path)
            
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("STEP 4: FINAL TEST EVALUATION")
    print("=" * 80)
    
    # Load the best model for test evaluation
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, "FINAL", "Test")
    
    # Save final test metrics
    test_results_path = output_path.parent / "test_results.json"
    test_results = {
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 2)
    }
    with open(test_results_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

if __name__ == "__main__":
    main()
