import os
import subprocess
import time
from pathlib import Path

def run_experiment(script_name, env_vars=None):
    print(f"\n{'='*50}")
    print(f"LAUNCHING EXPERIMENT: {script_name}")
    print(f"{'='*50}")
    
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
        
    # Add project root to PYTHONPATH
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    
    start = time.time()
    try:
        # We use check_call to stream output to console
        subprocess.check_call(["python", f"scripts/{script_name}"], env=env)
    except subprocess.CalledProcessError as e:
        print(f"EXPERIMENT FAILED: {script_name}")
        return False
        
    duration = time.time() - start
    print(f"EXPERIMENT COMPLETED in {duration/60:.2f} minutes.")
    return True

def main():
    print("Beginning Multi-Model Experimental Suite for HydraFusion")
    print("Target Constraints: Standalones ~90%, Fusion >95%, GPU: RTX 4050")
    
    experiments = [
        "train_mlep_standalone.py",
        "train_lota_standalone.py",
        "train_end_to_end.py",        # HydraFusion training
        "evaluate.py",                # Generate metrics json
        "generate_html_report.py"     # Generate Final Dashboard
    ]
    
    for exp in experiments:
        success = run_experiment(exp)
        if not success:
            print("Aborting test suite due to failure.")
            break
            
    print("\nALL EXPERIMENTS FINISHED SUCCESSFULLY")
    print("Check outputs/HydraFusion_Dashboard.html for the final report.")

if __name__ == "__main__":
    main()
