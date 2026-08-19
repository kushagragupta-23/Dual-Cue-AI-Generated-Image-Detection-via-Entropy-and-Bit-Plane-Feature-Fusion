# MLEP Project: Commands & Live Demonstration Guide

All commands needed to run the MLEP AI-Generated Image Detection project, plus a step-by-step live demo walkthrough.

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

### A. Download the Dataset
Streams verified images from HuggingFace and writes provenance metadata.
```bash
python scripts/download_dataset.py --target_dir dataset10000 --num_images 10000 --source auto
```

### B. Build the Benchmark Dataset
Reconstructs the structured 60/20/20 split dataset from scratch.
```bash
python scripts/build_benchmark_dataset.py
```

### C. Train the Model
Runs the full PyTorch training loop with Train/Validation/Test evaluation on the RTX 4050.
```bash
python scripts/train.py --data_dir dataset10000 --output_dir outputs/checkpoints --epochs 25 --batch_size 32
```
Optional flags: `--lr 0.0002`

### D. Run the Full MLEP Pipeline
Runs the entropy extraction pipeline and exports diagnostic heatmaps.
```bash
python scripts/run_project.py --data_dir dataset10000 --output_dir outputs/project_run --batch_size 32 --export_visualizations
```

### E. Generate Diagnostic Visualizations
Generates ROC, PR, t-SNE, FFT, LBP, calibration, and other charts from actual model predictions.
```bash
python scripts/generate_extra_visuals.py
```

### F. Generate the HTML Dashboard
Creates a self-contained HTML report with embedded metrics and charts.
```bash
python scripts/generate_html_report.py --output outputs/MLEP_Dashboard.html
```

### G. Visualize MLEP Algorithm
Generates a sample image and runs the entropy extractor for debugging.
```bash
python scripts/visualize_mlep.py
```

### H. Run Unit Tests
```bash
python -m pytest tests/ -v
```

---

## 3. Live Demonstration Walkthrough

This section covers the exact steps to present the MLEP project live, such as for a professor or evaluation panel.

### Step 1: Activate Environment
```powershell
.\venv\Scripts\activate
```
Tip: Keep the `outputs/` folder open in your file explorer so you can show files being generated in real-time.

### Step 2: Run Live Training
```bash
python scripts/train.py --data_dir dataset10000 --output_dir outputs/checkpoints --epochs 25 --batch_size 32
```
**What to explain:**
1. The dataloader reads from the `dataset10000` folder (5,000 real + 5,000 AI images).
2. As epochs progress, the loss decreases and accuracy increases.
3. After each epoch, the model runs a validation pass on held-out data. At the end, it runs a final test evaluation.
4. Early stopping will halt training automatically if validation accuracy stops improving (patience=7 epochs).

### Step 3: Generate Forensic Visualizations
```bash
python scripts/generate_extra_visuals.py
```
**What to explain:**
1. The script loads the trained model and runs inference on the test set.
2. Charts are generated from actual model predictions, not synthetic data.
3. Open the generated `.png` files in `outputs/project_run/visualizations/` to show results.

### Step 4: Run the MLEP Extraction Pipeline
```bash
python scripts/run_project.py --data_dir dataset10000 --output_dir outputs/project_run --batch_size 32 --export_visualizations
```
**What to explain:**
1. This is the core MLEP pipeline — it computes Shannon entropy at multiple scales.
2. Point out the real-time processing speed printed in the console.
3. The entropy gap (Real: ~1.784 vs AI: ~1.766) demonstrates the generative oversmoothing effect.

### Step 5: Generate & Open the HTML Dashboard
```bash
python scripts/generate_html_report.py --output outputs/MLEP_Dashboard.html
```
**What to explain:**
1. Open `outputs/MLEP_Dashboard.html` in a browser.
2. Walk through the Architecture Pipeline diagram.
3. Show the training metrics, optimizer/overfitting analysis, diagnostic charts, and data provenance sections.

---

## 4. QA Tips

**"How do you know the dataset is valid?"**
The real images come from a curated HuggingFace dataset that aggregates photographs from pre-2020 benchmarks. Since modern generative AI didn't exist then, the labels are reliable. We also apply SHA256 deduplication.

**"Why not just use a standard CNN?"**
Standard CNNs tend to learn surface-level patterns (colors, textures) that don't generalize to unseen generators. We specifically target the entropy gap caused by generative oversmoothing, which is harder for AI to fake.

**"How do you prevent overfitting?"**
We use Dropout (50% + 30%), AdamW with weight decay 0.05 (L2 regularization), CosineAnnealingLR scheduler, label smoothing, gradient clipping, early stopping (patience=7), and balanced sampling during training.

