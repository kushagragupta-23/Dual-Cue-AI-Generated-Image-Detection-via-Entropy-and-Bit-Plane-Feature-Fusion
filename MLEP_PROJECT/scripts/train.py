#!/usr/bin/env python3
"""
Training Pipeline for the MLEP (Macro-Texture Analyzer) Branch.
Trains the Entropy Pyramids to detect Generative Oversmoothing in the provided dataset.
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
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Ensure root directory is on python path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.dataset import SharedImageDataset
from src.data.dataloader import create_dataloader
from src.data.transforms import MLEPPreprocessingTransform
from src.models.mlep_detector import MLEPDetector
from src.utils.logger import get_logger

logger = get_logger("train_pipeline")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler, use_amp=True):
    model.train()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels_orig = labels.to(device).float().unsqueeze(1)
        # Apply Label Smoothing: 0 -> 0.05, 1 -> 0.95
        labels_smooth = labels_orig * 0.9 + 0.05
        
        optimizer.zero_grad()
        
        # Forward pass with AMP (only on CUDA)
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels_smooth)
            # Backward pass with scaler
            scaler.scale(loss).backward()
            # Gradient clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels_smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Calculate accuracy
        preds = (torch.sigmoid(logits.float()) > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_orig.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            current_acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).mean()
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(loader)}] | "
                f"Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%"
            )
            
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).mean()
    epoch_precision = 100.0 * precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = 100.0 * recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = 100.0 * f1_score(all_labels, all_preds, zero_division=0)
    epoch_time = time.time() - start_time
    
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_time


def evaluate(model, loader, criterion, device, epoch, split_name="Validation", use_amp=True):
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    logger.info(f"--- Starting {split_name} Evaluation for Epoch {epoch} ---")
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits.float()) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(all_labels)
    epoch_acc = 100.0 * (np.array(all_preds) == np.array(all_labels)).mean()
    epoch_precision = 100.0 * precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = 100.0 * recall_score(all_labels, all_preds, zero_division=0)
    epoch_f1 = 100.0 * f1_score(all_labels, all_preds, zero_division=0)
    
    logger.info(
        f">>> {split_name} Results | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | "
        f"Prec: {epoch_precision:.2f}% | Rec: {epoch_recall:.2f}% | F1: {epoch_f1:.2f}%"
    )
    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Train MLEPDetector")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints", help="Where to save model weights")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "SGD", "RMSprop"], help="Optimizer to use")
    args = parser.parse_args()

    data_path = root_path / args.data_dir
    output_path = root_path / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_path / "manifests"

    print("=" * 80)
    print("STEP 1: INITIALIZING DATASETS")
    print("=" * 80)

    train_tf = MLEPPreprocessingTransform(image_size=256, crop_to_square=True, enable_augmentations=True)
    eval_tf = MLEPPreprocessingTransform(image_size=256, crop_to_square=True, enable_augmentations=False)

    train_ds = SharedImageDataset(
        root_dir=data_path, split="train", val_ratio=0.2, test_ratio=0.2,
        transform=train_tf, validate_integrity=False, split_manifest_dir=manifest_dir
    )
    val_ds = SharedImageDataset(
        root_dir=data_path, split="val", val_ratio=0.2, test_ratio=0.2,
        transform=eval_tf, validate_integrity=False, split_manifest_dir=manifest_dir
    )
    test_ds = SharedImageDataset(
        root_dir=data_path, split="test", val_ratio=0.2, test_ratio=0.2,
        transform=eval_tf, validate_integrity=False, split_manifest_dir=manifest_dir
    )
    
    # Windows-safe num_workers (>0 causes multiprocessing hangs on Windows)
    n_workers = 0 
    train_loader = create_dataloader(train_ds, batch_size=args.batch_size, num_workers=n_workers, balanced_sampling=True, pin_memory=torch.cuda.is_available())
    val_loader = create_dataloader(val_ds, batch_size=args.batch_size, num_workers=n_workers, balanced_sampling=False, pin_memory=torch.cuda.is_available())
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, num_workers=n_workers, balanced_sampling=False, pin_memory=torch.cuda.is_available())

    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZING MLEP DETECTOR")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == 'cuda'
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # RTX 4050 Full Utilization: cuDNN benchmark + TF32 Tensor Core acceleration
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB) | cuDNN benchmark=ON | TF32=ON")
    
    # Initialize the complete model (MLEP) with Pre-trained ImageNet weights
    model = MLEPDetector(pretrained_backbones=True).to(device)
    
    # Differential Learning Rates: lower LR for pretrained backbones, higher for new heads
    backbone_params = list(model.mlep_backbone.parameters())
    head_params = list(model.classifier.parameters())
    extractor_params = list(model.mlep_extractor.parameters())
    
    param_groups = [
        {'params': backbone_params, 'lr': args.lr * 0.5},      # Pretrained: train at half base LR
        {'params': head_params, 'lr': args.lr * 5.0},           # Classifier head: learn fast
        {'params': extractor_params, 'lr': args.lr},            # Extractors: standard rate
    ]
    
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(param_groups, weight_decay=0.05)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(param_groups, weight_decay=0.05)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=0.05)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(param_groups, weight_decay=0.05)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    
    # Enable Automatic Mixed Precision (AMP) only on CUDA
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        logger.info("Enabled Automatic Mixed Precision (AMP) GradScaler.")
    else:
        logger.info("Running on CPU — AMP disabled.")

    print("\n" + "=" * 80)
    print(f"STEP 3: STARTING TRAINING LOOP (Epochs: {args.epochs})")
    print("=" * 80)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = output_path / f"mlep_best_{args.optimizer}.pth"
    history_path = output_path.parent / f"training_history_{args.optimizer}.json"
    
    training_history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{args.epochs} ---")
        
        train_loss, train_acc, train_prec, train_rec, train_f1, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, use_amp=use_amp
        )
        logger.info(
            f"Epoch {epoch} Training Completed in {epoch_time:.1f}s | "
            f"Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Prec: {train_prec:.2f}% | Rec: {train_rec:.2f}% | F1: {train_f1:.2f}%"
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device, epoch, "Validation", use_amp=use_amp
        )
        
        # Compute current learning rates for logging
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        logger.info(f"LR Schedule: backbone={current_lrs[0]:.6f}, head={current_lrs[1]:.6f}, extractor={current_lrs[2]:.6f}")
        scheduler.step()
        
        # Overfitting gap diagnosis
        overfit_gap = train_acc - val_acc
        gap_status = "OVERFITTING" if overfit_gap > 10.0 else "Healthy"
        logger.info(f"Overfit Gap: {overfit_gap:.2f}% ({gap_status})")
        
        # Save FULL history (including recall, F1, LR, overfit gap)
        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "train_prec": round(train_prec, 2),
            "train_rec": round(train_rec, 2),
            "train_f1": round(train_f1, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_prec": round(val_prec, 2),
            "val_rec": round(val_rec, 2),
            "val_f1": round(val_f1, 2),
            "lr_backbone": round(current_lrs[0], 8),
            "lr_head": round(current_lrs[1], 8),
            "overfit_gap": round(overfit_gap, 2),
        })
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2)
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0
            logger.info(f"*** New best validation accuracy: {best_val_acc:.2f}%. Saving checkpoint... ***")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'train_prec': train_prec,
                'train_rec': train_rec,
                'train_f1': train_f1,
                'val_prec': val_prec,
                'val_rec': val_rec,
                'val_f1': val_f1,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve}/{args.patience} epochs.")
            if epochs_no_improve >= args.patience:
                logger.info(f"[EARLY STOPPING] Validation accuracy did not improve for {args.patience} epochs. Stopping training.")
                break
            
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("STEP 4: FINAL TEST EVALUATION")
    print("=" * 80)
    
    # Load the best model for test evaluation
    # PyTorch 2.6 defaults to weights_only=True which crashes when loading dicts with numpy scalars
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device, "FINAL", "Test", use_amp=use_amp
    )
    
    # Save final test metrics
    test_results_path = output_path.parent / f"test_results_{args.optimizer}.json"
    test_results = {
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 2),
        "test_prec": round(test_prec, 2),
        "test_rec": round(test_rec, 2),
        "test_f1": round(test_f1, 2)
    }
    with open(test_results_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    # -------------------------------------------------------------
    # Generate Visualizations (Training Curves & Confusion Matrix)
    # -------------------------------------------------------------
    from src.utils.visualization import plot_training_curves, plot_confusion_matrix
    
    vis_dir = output_path.parent / "project_run" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating Training Curves...")
    plot_training_curves(training_history, save_path=vis_dir / "training_curves.png")
    
    logger.info("Generating Test Set Confusion Matrix...")
    plot_confusion_matrix(test_labels, test_preds, save_path=vis_dir / "confusion_matrix.png")

    print("\n" + "=" * 80)
    print(f"FINAL TEST RESULTS: Acc: {test_acc:.2f}% | Prec: {test_prec:.2f}%")
    print("=" * 80)
    
    # Retrieve best train/val metrics for final validation check
    best_train_acc = checkpoint.get('train_acc', 0.0)
    best_train_prec = checkpoint.get('train_prec', 0.0)
    best_val_prec = checkpoint.get('val_prec', 0.0)
    
    # Check if ALL splits are > 90%
    if (best_train_acc > 90.0 and best_train_prec > 90.0 and
        best_val_acc > 90.0 and best_val_prec > 90.0 and
        test_acc > 90.0 and test_prec > 90.0):
        
        logger.info("[SUCCESS] TARGET REACHED ACROSS ALL 3 SPLITS (Train, Val, Test) > 90%!")
        logger.info("Committing changes to git as requested...")
        try:
            import subprocess
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"feat: MLEP >90% across all splits! Test Acc: {test_acc:.2f}%, Test Prec: {test_prec:.2f}%"], check=True)
            logger.info("[OK] Successfully committed >90% milestone to git.")
        except Exception as e:
            logger.error(f"[ERROR] Failed to commit to git: {e}")
            
    # Fallback: Check if ALL splits are > 80%
    elif (best_train_acc > 80.0 and best_train_prec > 80.0 and
        best_val_acc > 80.0 and best_val_prec > 80.0 and
        test_acc > 80.0 and test_prec > 80.0):
        
        logger.info("[SUCCESS] TARGET REACHED ACROSS ALL 3 SPLITS (Train, Val, Test) > 80%!")
        logger.info("Committing changes to git as requested...")
        try:
            import subprocess
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", f"feat: MLEP >80% across all splits! Test Acc: {test_acc:.2f}%, Test Prec: {test_prec:.2f}%"], check=True)
            logger.info("[OK] Successfully committed >80% milestone to git.")
        except Exception as e:
            logger.error(f"[ERROR] Failed to commit to git: {e}")
    else:
        logger.info("[INFO] Metrics did not exceed >80% across all 3 splits simultaneously. Skipping Git Commit.")

if __name__ == "__main__":
    main()
