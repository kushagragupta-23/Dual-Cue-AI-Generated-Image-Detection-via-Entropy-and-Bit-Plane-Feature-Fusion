import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.models.lota_extractor import TopKLOTAExtractor
from src.models.backbones import ResNet50SpatialStem
from src.data.dataset import get_dataloaders
from src.utils.device import get_compute_device, set_global_seed
from src.utils.logger import get_logger
from src.utils.regularization import cutmix_data, mixup_criterion, LabelSmoothingBCEWithLogitsLoss

logger = get_logger("train_lota_standalone")

class LOTAStandalone(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = TopKLOTAExtractor(k_patches=4)
        self.stem = ResNet50SpatialStem(in_channels=3, return_layer='layer4')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        feat = self.extractor(x)
        spatial = self.stem(feat)
        z = self.pool(spatial).flatten(1)
        return self.classifier(z)

def train_epoch(model, loader, optimizer, scaler, criterion, device, epoch, writer):
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(loader, desc=f"LOTA Epoch {epoch}"):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        # 50% chance of CutMix for strong regularization
        use_cutmix = np.random.rand() > 0.5
        if use_cutmix:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(images)
            
            if use_cutmix:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, labels)
                
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss

def train_lota_standalone(optimizer_name="adamw"):
    set_global_seed(42)
    device = get_compute_device()
    logger.info(f"Using device: {device} | Optimizer: {optimizer_name}")
    
    config_path = Path("configs/default.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    train_loader, _, _ = get_dataloaders(config)
    
    model = LOTAStandalone().to(device)
    scaler = torch.amp.GradScaler('cuda')
    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=0.1)
    
    lr = 0.001
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
    # Cosine Annealing Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
        
    epochs = config['training']['stage2_epochs']
    
    writer = SummaryWriter(log_dir=f"outputs/logs/lota_{optimizer_name}")
    
    logger.info("Starting LOTA Standalone Training...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch, writer)
        scheduler.step()
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        logger.info(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
    writer.close()
    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"lota_standalone_{optimizer_name}.pt")
    logger.info("LOTA Training complete.")

if __name__ == "__main__":
    train_lota_standalone()
