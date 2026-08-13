import yaml
import torch
from tqdm import tqdm
from pathlib import Path
import json

from src.models.hydrafusion_net import HydraFusionNet
from src.data.dataset import get_dataloaders
from src.utils.device import get_compute_device

def evaluate_model(model, loader, device):
    model.eval()
    
    y_true = []
    y_scores = []
    alpha_logs = []
    
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits, _, alpha = model(images, stage=2)
                probs = torch.sigmoid(logits).squeeze()
                
            y_true.extend(labels.cpu().numpy().tolist())
            y_scores.extend(probs.cpu().numpy().tolist())
            alpha_logs.extend(alpha.cpu().numpy().tolist())
            
    # Calculate real metrics using sklearn
    y_pred = [1 if p > 0.5 else 0 for p in y_scores]
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)), 
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    return results, y_true, y_scores, alpha_logs

def main():
    device = get_compute_device()
    config_path = Path("configs/default.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    _, _, test_loader = get_dataloaders(config)
    
    model = HydraFusionNet().to(device)
    ckpt_path = Path(config['logging']['checkpoint_dir']) / "hydrafusion_best.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path))
        
    results, y_true, y_scores, alpha_logs = evaluate_model(model, test_loader, device)
    
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    with open(out_dir / "alphas.json", "w") as f:
        json.dump(alpha_logs[:100], f, indent=4) # Save a sample of gating weights
        
    print(f"Evaluation complete. Results: {results}")

if __name__ == "__main__":
    main()
