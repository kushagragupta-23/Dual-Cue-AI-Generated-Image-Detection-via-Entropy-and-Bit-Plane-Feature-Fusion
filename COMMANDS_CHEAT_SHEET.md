# MLEP AI Detection: Command Cheat Sheet

All commands needed to run the project on any platform.

---

## 1. Environment Setup (First-Time Only)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Core Commands

### A. Train the Model
Runs the full PyTorch training loop with Train/Validation/Test evaluation.
```bash
python scripts/train.py --data_dir dataset10000 --output_dir outputs/checkpoints --epochs 10
```
Optional flags: `--batch_size 32`, `--lr 0.0001`

### B. Run the Full MLEP Pipeline
Runs the entropy extraction pipeline and exports diagnostic heatmaps.
```bash
python scripts/run_project.py --data_dir dataset10000 --output_dir outputs/project_run --batch_size 32 --export_visualizations
```

### C. Generate Diagnostic Visualizations
Generates ROC, PR, t-SNE, FFT, LBP, and other charts from actual model predictions.
```bash
python scripts/generate_extra_visuals.py
```

### D. Generate the HTML Dashboard
Creates a self-contained HTML report with embedded metrics and charts.
```bash
python scripts/generate_html_report.py --output outputs/MLEP_Dashboard.html
```

### E. Download the Dataset
Streams verified images from HuggingFace and writes provenance metadata.
```bash
python scripts/download_dataset.py --target_dir dataset10000 --num_images 10000 --source auto
```

### F. Build the Benchmark Dataset
Reconstructs the structured 60/20/20 split dataset from scratch.
```bash
python scripts/build_benchmark_dataset.py
```

### G. Visualize MLEP Algorithm
Generates a sample image and runs the entropy extractor for debugging.
```bash
python scripts/visualize_mlep.py
```

---

## 3. Log Monitoring

**Windows (PowerShell):**
```powershell
Get-Content -Wait -Tail 20 "outputs/checkpoints/training.log"
```

**Linux / macOS:**
```bash
tail -f outputs/checkpoints/training.log
```

---

## 4. Git Commands

**Check your Git identity:**
```bash
git config user.name
git config user.email
```

**Set your Git identity:**
```bash
git config --global user.name "kushagragupta-23"
git config --global user.email "aakg2310@gmail.com"
```

**Push to GitHub:**
```bash
git add .
git commit -m "Update project files"
git push origin main
```
