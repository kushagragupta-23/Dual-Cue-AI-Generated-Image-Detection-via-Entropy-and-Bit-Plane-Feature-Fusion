import subprocess
import sys
import json
from pathlib import Path
import os

def run_training_for_optimizer(optimizer, epochs=1):
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING FOR OPTIMIZER: {optimizer}")
    print(f"{'='*80}\n")
    
    # Use venv Python to ensure CUDA torch is available
    python_exe = "venv\\Scripts\\python.exe" if os.name == 'nt' else "venv/bin/python"
    if not Path(python_exe).exists():
        print(f"Warning: {python_exe} not found. Falling back to sys.executable.")
        python_exe = sys.executable

    cmd = [
        python_exe, "scripts/train.py",
        "--data_dir", "dataset10000",
        "--optimizer", optimizer,
        "--epochs", str(epochs),
        "--batch_size", "64"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] Completed training for {optimizer}")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed for {optimizer}: {e}")
        sys.exit(1)

def main():
    optimizers = ["AdamW", "Adam", "SGD", "RMSprop"]
    epochs = 3 # Set to 3 epochs to finish faster on synchronous DataLoader
    
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    checkpoints_dir = outputs_dir / "checkpoints"
    
    # Run training
    for opt in optimizers:
        run_training_for_optimizer(opt, epochs)
        
    print("\n" + "="*80)
    print("ALL OPTIMIZERS TRAINED. COMPARING RESULTS...")
    print("="*80)
    
    opt_results = {}
    best_opt = None
    best_acc = -1
    
    # Read metrics
    for opt in optimizers:
        hist_path = outputs_dir / f"training_history_{opt}.json"
        test_path = outputs_dir / f"test_results_{opt}.json"
        
        if hist_path.exists() and test_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
            best_train_acc = max(float(row.get("train_acc", 0)) for row in history) if history else 0
            
            with open(test_path) as f:
                test_res = json.load(f)
            test_acc = float(test_res.get("test_acc", 0))
            
            opt_results[opt] = {"train_acc": best_train_acc, "test_acc": test_acc}
            
            print(f"Optimizer {opt:8} | Best Train Acc: {best_train_acc:5.2f}% | Test Acc: {test_acc:5.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_opt = opt
                
    if best_opt:
        print(f"\n[DECISION] We will be choosing {best_opt} because it achieved the best test accuracy ({best_acc}%).")
        
        # Keep only the best model
        print("Cleaning up suboptimal model files...")
        for opt in optimizers:
            model_path = checkpoints_dir / f"mlep_best_{opt}.pth"
            if opt != best_opt and model_path.exists():
                model_path.unlink()
                print(f"Deleted {model_path.name}")
            elif opt == best_opt:
                print(f"Kept best model: {model_path.name}")
    else:
        print("No valid results found to compare.")

    print("\nGenerating final HTML report...")
    
    python_exe = "venv\\Scripts\\python.exe" if os.name == 'nt' else "venv/bin/python"
    if not Path(python_exe).exists():
        python_exe = sys.executable
        
    try:
        subprocess.run([python_exe, "scripts/generate_html_report.py"], check=True)
        print("Report generated.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Report generation failed: {e}")
        
if __name__ == "__main__":
    main()
