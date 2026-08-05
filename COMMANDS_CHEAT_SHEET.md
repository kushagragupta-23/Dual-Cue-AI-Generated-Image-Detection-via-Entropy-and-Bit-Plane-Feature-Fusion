# 💻 MLEP AI Detection: Master Command Cheat Sheet

This document contains **all manual terminal commands** required to run, monitor, and deploy the MLEP AI-Generated Image Detection project across **any platform** (Windows, Linux, macOS) or interpreter (PowerShell, Bash, Zsh). 

Use this sheet if you need to present the project live in front of a professor, run it on a completely new machine, or manually trigger the Deep Learning pipeline.

---

## 1. Environment Setup (First-Time Only)
Before running any code, you must activate the dedicated Python virtual environment to ensure all Deep Learning dependencies (PyTorch, Albumentations, Streamlit) are loaded correctly.

**On Windows (PowerShell / CMD):**
```powershell
python -m venv env_mlep
.\env_mlep\Scripts\activate
pip install -r requirements.txt
```

**On Linux / macOS (Bash / Zsh):**
```bash
python3 -m venv env_mlep
source env_mlep/bin/activate
pip install -r requirements.txt
```

---

## 2. Core Execution Commands

### A. Run the Deep Learning Training Pipeline
This command triggers the end-to-end PyTorch training process on the 10,000-image benchmark dataset. It will automatically run all epochs, calculate the Train/Validation scores, and execute the Final Test Evaluation.

**Cross-Platform Command:**
```bash
python scripts/train.py --data_dir dataset10000 --output_dir outputs/checkpoints --epochs 15
```
*(Optional Flags: `--batch_size 16` or `--lr 0.0001` to tweak hyper-parameters live)*

### B. Run the Full Pipeline & Generate Visualizations
This command executes the end-to-end MLEP extraction pipeline, computes entropy metrics across the dataset, and exports diagnostic heatmaps.

**Cross-Platform Command:**
```bash
python scripts/run_project.py --data_dir dataset10000 --output_dir outputs/project_run --batch_size 8 --export_visualizations
```
### C. Generate the 15 Advanced Forensic Visuals
This standalone script runs the dataset through the LBP, FFT, and Noise Residual processors to mathematically map the Generative Oversmoothing flaws (creates the 15 charts used in the dashboard).

**Cross-Platform Command:**
```bash
python scripts/generate_extra_visuals.py
```
### D. Generate the Interactive HTML Dashboard
This generates a self-contained, premium glassmorphism HTML dashboard with embedded training metrics, entropy Pyramids, and the MLEP architectural pipeline.

**Cross-Platform Command:**
```bash
python scripts/generate_html_report.py --output outputs/MLEP_Dashboard.html
```

### E. Download the Verified Dataset
Streams the verified Real/AI images directly from HuggingFace and writes the `provenance_manifest.json`.

**Cross-Platform Command:**
```bash
python scripts/download_dataset.py --target_dir outputs/verified_dataset --num_images 10000 --source auto
```

### F. Build / Validate the Benchmark Dataset
If you ever move to a new machine and need to synthesize or reconstruct a structured benchmark dataset from scratch.

**Cross-Platform Command:**
```bash
python scripts/build_benchmark_dataset.py
```

### G. Visualize Core MLEP Algorithm Live
Generates a sample image in memory and runs it through the Multi-granularity Local Entropy Patterns extractor, printing the entropy maps and sizes directly to the terminal for debugging or demonstration.

**Cross-Platform Command:**
```bash
python scripts/visualize_mlep.py
```

---

## 3. Live Log Monitoring (Asynchronous Tracking)
If you launch the training pipeline on a remote server, or as a background task, you will want to stream the live logs (Loss and Accuracy metrics) to your terminal.

**On Windows (PowerShell):**
```powershell
# Streams the last 20 lines and waits for live updates
Get-Content -Wait -Tail 20 "outputs/checkpoints/training.log"
```
*(Note: Replace the file path with the exact `.log` file you are tracking)*

**On Linux / macOS (Bash / Zsh):**
```bash
# The standard UNIX tail command for real-time streaming
tail -f outputs/checkpoints/training.log
```

---

## 4. Git & Version Control Commands
If you need to sync changes, push new dataset images, or verify that your commits are correctly mapped to your identity on a new computer.

**Verify your exact Git Identity (Who gets the credit):**
```bash
git config user.name
git config user.email
```

**Fix your Git Identity to your correct GitHub Account:**
```bash
git config --global user.name "kushagragupta-23"
git config --global user.email "aakg2310@gmail.com"
```

**Standard Push to GitHub (After fixing identity):**
```bash
git add .
git commit -m "Update project files"
git push origin main
```
