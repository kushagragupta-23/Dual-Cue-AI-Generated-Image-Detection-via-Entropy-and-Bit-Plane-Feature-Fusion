# 💻 Dual-Cue AI Detection: Master Command Cheat Sheet

This document contains **all manual terminal commands** required to run, monitor, and deploy the Dual-Cue AI-Generated Image Detection project across **any platform** (Windows, Linux, macOS) or interpreter (PowerShell, Bash, Zsh). 

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
python scripts/train.py --data_dir dataset10000 --output_dir outputs/project_run_training --epochs 3
```
*(Optional Flags: `--batch_size 16` or `--lr 0.0001` to tweak hyper-parameters live)*

### B. Launch the Interactive Dashboard (UI)
This command fires up the local Streamlit web server. This is perfect for live presentations, as it provides a beautiful UI where you can upload an image and visually see the MLEP (Entropy) and LOTA (Bit-Plane) feature extraction in real-time.

**Cross-Platform Command:**
```bash
streamlit run scripts/run_project.py
```

### C. Build / Validate the Dataset
If you ever move to a new machine and need to reconstruct the verified 10,000-image dataset from scratch (assuming you have the raw images downloaded).

**Cross-Platform Command:**
```bash
python scripts/build_benchmark_dataset.py
```

---

## 3. Live Log Monitoring (Asynchronous Tracking)
If you launch the training pipeline on a remote server, or as a background task, you will want to stream the live logs (Loss and Accuracy metrics) to your terminal.

**On Windows (PowerShell):**
```powershell
# Streams the last 20 lines and waits for live updates
Get-Content -Wait -Tail 20 "outputs/project_run_training/training.log"
```
*(Note: Replace the file path with the exact `.log` file you are tracking)*

**On Linux / macOS (Bash / Zsh):**
```bash
# The standard UNIX tail command for real-time streaming
tail -f outputs/project_run_training/training.log
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
git config --global user.name "Your Name"
git config --global user.email "your.actual.github.email@example.com"
```

**Standard Push to GitHub (After fixing identity):**
```bash
git add .
git commit -m "Update project files"
git push origin main
```
