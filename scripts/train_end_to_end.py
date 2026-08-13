"""
HydraFusion-Net End-to-End Training Script (v3 — Production)
=============================================================
2-Stage GPU-accelerated training with all critical fixes applied:

  Stage 1: Contrastive Pre-Training
    - SupCon loss in float32 (prevents FP16 overflow)
    - Higher temperature (τ=0.2) for gentler alignment
    - Only trains: extractors, projections (backbones frozen)
    
  Stage 2: Gated Fusion Fine-Tuning
    - All losses in float32
    - Only trains: fusion heads, router, classifier
    - Domain adversarial REMOVED (no real domain labels available)
    - OneCycleLR with warmup
    - Early stopping with patience=7
    
  Key fixes from v1/v2:
    1. LOTA extractor now fully differentiable (no uint8/bitwise ops)
    2. MLEP entropy uses proper differentiable approximation
    3. FreqPreFilter does FFT in float32
    4. Backbones frozen (prevents overfitting 47M params on 6k images)
    5. DataLoader workers capped at 2 (prevents Windows deadlocks)
    6. No domain adversarial gradient noise
"""
import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.models.hydrafusion_net import HydraFusionNet
from src.models.supcon_loss import DualCueSupConLoss
from src.data.dataset import get_dataloaders
from src.utils.device import get_compute_device, set_global_seed
from src.utils.logger import get_logger

logger = get_logger("train")

# ──────────────────────────── Optimiser factory ────────────────────────────
def get_optimizer(params, name: str, lr: float, weight_decay: float):
    """Create optimizer from the given parameter iterator."""
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    elif name == "nadam":
        return torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


# ──────────────────────────── Stage 1 ──────────────────────────────────────
def train_stage1(model, loader, optimizer, scaler, device, config, epoch, writer):
    """Contrastive Pre-Training — loss ALWAYS in float32."""
    model.train()
    criterion = DualCueSupConLoss(temperature=config['training']['supcon_temp'])
    total_loss = 0.0
    valid_batches = 0

    pbar = tqdm(loader, desc=f"S1 Epoch {epoch+1}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            p_mlep, p_lota = model(images, stage=1)

        # Loss in FULL float32
        loss = criterion(p_mlep.float(), p_lota.float(), labels)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        valid_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(valid_batches, 1)
    writer.add_scalar("Stage1/train_loss", avg_loss, epoch)
    return avg_loss


# ──────────────────────────── Stage 2 (train) ──────────────────────────────
def train_stage2(model, loader, optimizer, scaler, device, epoch, max_epochs, writer, scheduler=None):
    """Gated Fusion Fine-Tuning — stable label-smoothed BCE."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    nan_batches = 0
    valid_batches = 0

    pbar = tqdm(loader, desc=f"S2 Epoch {epoch+1}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels_int = labels.to(device, non_blocking=True)

        # Label smoothing: 0 → 0.05, 1 → 0.95
        labels_smooth = labels_int.float().unsqueeze(1) * 0.9 + 0.05

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _, alpha = model(images, stage=2)

        # ALL losses in float32
        logits_f32 = logits.float()

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits_f32, labels_smooth.float()
        )

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Step OneCycleLR per batch
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        valid_batches += 1

        preds = (torch.sigmoid(logits_f32) > 0.5).long().squeeze()
        correct += (preds == labels_int).sum().item()
        total += labels_int.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

    avg_loss = total_loss / max(valid_batches, 1)
    train_acc = 100. * correct / max(total, 1)
    writer.add_scalar("Stage2/train_loss", avg_loss, epoch)
    writer.add_scalar("Stage2/train_acc", train_acc, epoch)
    if nan_batches > 0:
        logger.warning(f"  ⚠ {nan_batches} NaN batches skipped")
    return avg_loss, train_acc


# ──────────────────────────── Stage 2 (validate) ───────────────────────────
@torch.no_grad()
def validate(model, loader, device, epoch, writer):
    """Validation pass."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_int = labels.to(device, non_blocking=True)
        labels_float = labels_int.float().unsqueeze(1)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _, _ = model(images, stage=2)

        loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), labels_float)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
        
        preds = (torch.sigmoid(logits.float()) > 0.5).long().squeeze()
        correct += (preds == labels_int).sum().item()
        total += labels_int.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    val_acc = 100. * correct / max(total, 1)
    writer.add_scalar("Stage2/val_loss", avg_loss, epoch)
    writer.add_scalar("Stage2/val_acc", val_acc, epoch)
    return avg_loss, val_acc


# ──────────────────────────── Main ─────────────────────────────────────────
def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    set_global_seed(42)
    device = get_compute_device()
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    config_path = Path("configs/default.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_loader, val_loader, test_loader = get_dataloaders(config)
    logger.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # Create model with frozen backbones
    model = HydraFusionNet(freeze_backbones=True).to(device)
    
    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,} ({100.*trainable_params/total_params:.1f}%)")

    # Outputs
    out_dir = Path(config['logging']['checkpoint_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=config['logging']['log_dir'])

    opt_name = config['training']['optimizer']
    epochs_1 = config['training']['stage1_epochs']
    weight_decay = config['training']['weight_decay']

    # ═══════════════════════════════════════════════════════════════════
    #  STAGE 1: Contrastive Pre-Training
    # ═══════════════════════════════════════════════════════════════════
    s1_params = [p for p in model.parameters() if p.requires_grad]
    lr_s1 = config['training']['lr_stage1']
    opt_1 = get_optimizer(s1_params, opt_name, lr_s1, weight_decay)
    
    sched_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_1, T_max=epochs_1, eta_min=1e-5
    )
    scaler_1 = torch.amp.GradScaler('cuda')

    logger.info(f"{'='*60}")
    logger.info(f"STAGE 1 — Contrastive Pre-Training ({opt_name.upper()}, {epochs_1} epochs)")
    logger.info(f"  Trainable params: {sum(p.numel() for p in s1_params):,}")
    logger.info(f"{'='*60}")

    best_s1_loss = float('inf')
    for epoch in range(epochs_1):
        loss = train_stage1(model, train_loader, opt_1, scaler_1, device, config, epoch, writer)
        sched_1.step()
        lr = sched_1.get_last_lr()[0]
        
        marker = " ★" if loss < best_s1_loss else ""
        if loss < best_s1_loss:
            best_s1_loss = loss
        logger.info(f"  Stage 1 | Epoch {epoch+1:02d}/{epochs_1} | Loss: {loss:.4f} | LR: {lr:.6f}{marker}")

    torch.save(model.state_dict(), out_dir / "hydrafusion_stage1.pt")
    logger.info("Stage 1 checkpoint saved.")

    # ═══════════════════════════════════════════════════════════════════
    #  STAGE 2: Gated Fusion Fine-Tuning (with Layer3 Fine-Tuning)
    # ═══════════════════════════════════════════════════════════════════
    epochs_2 = config['training']['stage2_epochs']
    scaler_2 = torch.amp.GradScaler('cuda')
    
    # Freeze Stage 1 projections & extractors
    for mod in [model.freq_filter, model.mlep_extractor, model.lota_extractor,
                model.mlep_proj, model.lota_proj]:
        for param in mod.parameters():
            param.requires_grad = False
            
    # Unfreeze ResNet Layer3 for deepfake feature adaptation
    for param in model.mlep_stem.layer3.parameters():
        param.requires_grad = True
    for param in model.lota_stem.layer3.parameters():
        param.requires_grad = True
    
    lr_s2 = config['training']['lr_stage2']
    lr_backbone = config['training'].get('lr_backbone', 0.00003)
    
    head_params = list(model.fusion_module.parameters()) + list(model.router.parameters()) + list(model.classifier.parameters())
    backbone_params = list(model.mlep_stem.layer3.parameters()) + list(model.lota_stem.layer3.parameters())
    
    param_groups = [
        {'params': head_params, 'lr': lr_s2, 'weight_decay': weight_decay},
        {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay * 0.5}
    ]
    
    opt_2 = torch.optim.AdamW(param_groups)
    
    # OneCycleLR with differential peak LRs
    sched_2 = torch.optim.lr_scheduler.OneCycleLR(
        opt_2,
        max_lr=[lr_s2 * 1.5, lr_backbone * 1.5],
        epochs=epochs_2,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        div_factor=2,
        final_div_factor=50
    )

    s2_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{'='*60}")
    logger.info(f"STAGE 2 — Gated Fusion Fine-Tuning ({opt_name.upper()}, {epochs_2} epochs)")
    logger.info(f"  Trainable params: {s2_params_count:,} (Heads: {sum(p.numel() for p in head_params):,}, Layer3: {sum(p.numel() for p in backbone_params):,})")
    logger.info(f"  Heads LR: {lr_s2:.6f} | Backbone LR: {lr_backbone:.6f}")
    logger.info(f"{'='*60}")

    best_val_acc = 0.0
    patience_counter = 0
    patience = 8

    for epoch in range(epochs_2):
        train_loss, train_acc = train_stage2(
            model, train_loader, opt_2, scaler_2, device, 
            epoch, epochs_2, writer, scheduler=sched_2
        )
        val_loss, val_acc = validate(model, val_loader, device, epoch, writer)
        
        lr_head = opt_2.param_groups[0]['lr']
        
        # Check for overfitting
        overfit_gap = train_acc - val_acc
        overfit_marker = f" ⚠ OVERFIT gap={overfit_gap:.1f}%" if overfit_gap > 10 else ""

        logger.info(
            f"  Stage 2 | Epoch {epoch+1:02d}/{epochs_2} | "
            f"Train: {train_loss:.4f} / {train_acc:.1f}% | "
            f"Val: {val_loss:.4f} / {val_acc:.1f}% | "
            f"LR: {lr:.7f}{overfit_marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "hydrafusion_best.pt")
            logger.info(f"  ★ New best val accuracy: {best_val_acc:.1f}% — saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  ■ Early stopping (no improvement for {patience} epochs)")
                break

    # ═══════════════════════════════════════════════════════════════════
    #  FINAL EVALUATION ON TEST SET
    # ═══════════════════════════════════════════════════════════════════
    logger.info(f"{'='*60}")
    logger.info("Loading best checkpoint for final test evaluation...")
    model.load_state_dict(torch.load(out_dir / "hydrafusion_best.pt", weights_only=True))
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test Eval", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, _, _ = model(images, stage=2)
            
            preds = (torch.sigmoid(logits.float()) > 0.5).long().squeeze()
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    test_acc = 100. * test_correct / test_total
    
    # Compute detailed metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "best_val_accuracy": round(best_val_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "test_correct": test_correct,
        "test_total": test_total,
        "optimizer": opt_name,
        "stage1_epochs": epochs_1,
        "stage2_epochs": epochs_2,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    writer.close()
    
    logger.info(f"{'='*60}")
    logger.info(f"  TRAINING COMPLETE — FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Best Val Accuracy : {best_val_acc:.2f}%")
    logger.info(f"  Test Accuracy     : {test_acc:.2f}%")
    logger.info(f"  Precision         : {precision*100:.2f}%")
    logger.info(f"  Recall            : {recall*100:.2f}%")
    logger.info(f"  F1 Score          : {f1*100:.2f}%")
    logger.info(f"  Checkpoint        : {out_dir / 'hydrafusion_best.pt'}")
    logger.info(f"  Metrics           : {results_dir / 'metrics.json'}")
    logger.info(f"{'='*60}")
    
    # Print verdict
    if test_acc >= 90:
        logger.info(f"  ✅ TARGET ACHIEVED: {test_acc:.1f}% > 90%")
    else:
        logger.info(f"  ❌ Below target: {test_acc:.1f}% < 90%. Consider Phase C optimizations.")


if __name__ == "__main__":
    main()
