# 🎤 Live Demonstration & Presentation Guide

This guide is designed specifically for presenting the MLEP AI-Generated Image Detection project to a live audience, such as a professor or an evaluation panel. It contains the exact sequence of commands to run to demonstrate the live training, validation, and generation of the final dashboard.

---

## Step 1: Environment Preparation (Before Presenting)

Before you share your screen or start the presentation, make sure your terminal is open in the project root folder and the virtual environment is activated.

**Windows:**
```powershell
.\env_mlep\Scripts\activate
```

**Linux/macOS:**
```bash
source env_mlep/bin/activate
```

*Tip: Have the `outputs/` folder open in your file explorer so you can show the audience the files being generated in real-time.*

---

## Step 2: Show Live Data Training, Testing, and Validating

To demonstrate the Deep Learning model actively learning, you will run the `train.py` script. This script ingests the verified dataset, splits it into Train/Validation/Test sets, and runs the backpropagation loop.

**Command to run live:**
```bash
python scripts/train.py --data_dir dataset10000 --output_dir outputs/checkpoints --epochs 5 --batch_size 16
```

**What to point out to the audience while this runs:**
1. **Data Ingestion:** Mention that the dataloader is reading the verified dataset from the `dataset10000` folder.
2. **Live Metrics:** As the epochs progress, point out the **Loss** decreasing and the **Accuracy** increasing in the terminal output.
3. **Validation & Testing:** Explain that after every epoch, the model runs a strict validation pass, and at the very end, it performs a final Test Evaluation on unseen data to prevent overfitting.

---

## Step 3: Generate the Core Forensic Visuals (Dashboard Outputs)

The dashboard relies on 15 advanced mathematical visualizations (LBP, FFT, Noise Residuals) to prove the "Generative Oversmoothing" hypothesis. You can generate these live.

**Command to run live:**
```bash
python scripts/generate_extra_visuals.py
```

**What to point out to the audience:**
1. Explain that this script is analyzing the images at the pixel-level to extract high-frequency structural chaos.
2. Open the newly generated `.png` files in the `outputs/visualizations` folder to show them the real differences between AI (smoothed) and Real (noisy) camera sensors.

---

## Step 4: Execute the Full MLEP Extraction Pipeline

This command runs the data through the `MLEPExtractor`, proving the Entropy Collapse (Real: 1.911 vs AI: 1.906).

**Command to run live:**
```bash
python scripts/run_project.py --data_dir dataset10000 --output_dir outputs/project_run --batch_size 8 --export_visualizations
```

**What to point out to the audience:**
1. This is the core pipeline (Multi-granularity Local Entropy Patterns).
2. It's shuffling micro-patches and computing Shannon Entropy.
3. Show them the real-time processing speed (e.g., 39 FPS) printed in the console.

---

## Step 5: Compile the Final HTML Dashboard Live

Finally, you want to show the audience the beautiful, self-contained dashboard that aggregates all the metrics, charts, and architectural diagrams.

**Command to run live:**
```bash
python scripts/generate_html_report.py --output outputs/MLEP_Dashboard.html
```

**What to point out to the audience:**
1. Once the script finishes, double-click the `outputs/MLEP_Dashboard.html` file to open it in your web browser.
2. Walk them through the **MLEP Architecture Pipeline** diagram.
3. Scroll down to show the **Deep Research Summary: Top 8 Metrics** and the embedded forensic visuals generated in Step 3.

---

## 💡 Quick QA Defense

If asked: *"How do you know the dataset is valid?"*
**Response:** "We rely on the Epistemological Chain of Trust. Our Real images are from 2009-2014, making AI-contamination chronologically impossible. We have a cryptographic provenance manifest to prove this."

If asked: *"Why not just use a standard CNN?"*
**Response:** "Standard CNNs learn trivial RGB semantic patterns and fail on unseen generators. We specifically target the mathematical noise gap (Generative Oversmoothing) using Shannon Entropy, which is much harder for AI to fake."
