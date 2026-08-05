# Live Demonstration Guide

This guide covers the exact steps to present the MLEP AI-Generated Image Detection project live, such as for a professor or evaluation panel.

---

## Step 1: Environment Setup (Before Presenting)

Make sure the virtual environment is activated before starting.

**Windows:**
```powershell
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

Tip: Keep the `outputs/` folder open in your file explorer so you can show files being generated in real-time.

---

## Step 2: Run Live Training

This trains the MLEP model on the dataset, showing loss and accuracy updating in the terminal.

```bash
python scripts/train.py --data_dir dataset10000 --output_dir outputs/checkpoints --epochs 10 --batch_size 32
```

**What to explain:**
1. The dataloader reads from the `dataset10000` folder (5,000 real + 5,000 AI images).
2. As epochs progress, the loss should decrease and accuracy should increase.
3. After each epoch, the model runs a validation pass on held-out data. At the end, it runs a final test evaluation.

---

## Step 3: Generate Forensic Visualizations

This script generates the diagnostic charts (ROC, PR, t-SNE, FFT, LBP, etc.) used in the dashboard.

```bash
python scripts/generate_extra_visuals.py
```

**What to explain:**
1. The script loads the trained model and runs inference on the test set.
2. Charts are generated from actual model predictions, not synthetic data.
3. Open the generated `.png` files in `outputs/project_run/visualizations/` to show results.

---

## Step 4: Run the MLEP Extraction Pipeline

This runs the raw entropy extraction pipeline, computing the entropy gap between real and AI images.

```bash
python scripts/run_project.py --data_dir dataset10000 --output_dir outputs/project_run --batch_size 32 --export_visualizations
```

**What to explain:**
1. This is the core MLEP pipeline — it computes Shannon entropy at multiple scales.
2. Point out the real-time processing speed printed in the console.
3. The entropy gap (Real: ~1.911 vs AI: ~1.906) demonstrates the generative oversmoothing effect.

---

## Step 5: Generate the HTML Dashboard

```bash
python scripts/generate_html_report.py --output outputs/MLEP_Dashboard.html
```

**What to explain:**
1. Open `outputs/MLEP_Dashboard.html` in a browser.
2. Walk through the Architecture Pipeline diagram.
3. Show the training metrics, diagnostic charts, and data provenance sections.

---

## QA Tips

**"How do you know the dataset is valid?"**
The real images come from a curated HuggingFace dataset that aggregates photographs from pre-2020 benchmarks. Since modern generative AI didn't exist then, the labels are reliable. We also apply SHA256 deduplication.

**"Why not just use a standard CNN?"**
Standard CNNs tend to learn surface-level patterns (colors, textures) that don't generalize to unseen generators. We specifically target the entropy gap caused by generative oversmoothing, which is harder for AI to fake.
