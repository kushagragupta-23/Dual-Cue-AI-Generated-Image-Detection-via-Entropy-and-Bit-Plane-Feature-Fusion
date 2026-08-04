#!/usr/bin/env python3
"""
Interactive HTML Dashboard Generator for MLEP
Generates a self-contained, professional HTML report with Base64-embedded diagnostic figures.
Can be opened directly in Google Chrome or any web browser.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import webbrowser

# Ensure root path is accessible
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.logger import get_logger

logger = get_logger("html_report_generator")


def img_to_base64(img_path: Path) -> str:
    """Read an image file and return its data URI scheme Base64 string."""
    if not img_path.exists():
        logger.warning(f"Image not found for base64 encoding: {img_path}")
        return ""
    with open(img_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
        ext = img_path.suffix.lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{encoded}"


def load_json_safe(json_path: Path, default: dict) -> dict:
    """Load JSON file safely with default fallback."""
    if not json_path.exists():
        return default
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {json_path}: {e}")
        return default


def generate_html(output_file: Path, auto_open: bool = True) -> None:
    logger.info("Gathering diagnostic figures and execution metrics for HTML dashboard...")

    batch_vis_dir = root_path / "outputs" / "project_run" / "visualizations"
    summary_path = root_path / "outputs" / "project_run" / "execution_summary.json"
    history_path = root_path / "outputs" / "training_history.json"
    test_path = root_path / "outputs" / "test_results.json"

    default_summary = {
        "performance": {"avg_batch_latency_ms": "N/A", "throughput_images_per_sec": "N/A"},
        "steganalysis_metrics": {
            "mean_entropy_real": 0.0,
            "mean_entropy_ai_generated": 0.0,
            "divergence_contrast_ratio": 1.0
        }
    }
    summary_data = load_json_safe(summary_path, default_summary)
    perf = summary_data.get("performance", {})
    steg = summary_data.get("steganalysis_metrics", {})
    
    training_history = load_json_safe(history_path, [])
    best_val_acc = max([float(row["val_acc"]) for row in training_history]) if training_history else 0.0
    best_train_acc = max([float(row["train_acc"]) for row in training_history]) if training_history else 0.0

    default_test = {
        "test_loss": 0.0, "test_acc": 0.0, "test_prec": 0.0, "test_rec": 0.0, "test_f1": 0.0
    }
    test_results = load_json_safe(test_path, default_test)

    # Encode images to Base64 for 100% self-contained HTML portability
    images = {
        "batch_mlep_heatmap": img_to_base64(batch_vis_dir / "batch1_sample0_mlep_heatmap.png"),
        "batch_mlep_multiscale": img_to_base64(batch_vis_dir / "batch1_sample0_mlep_multiscale.png"),
        "training_curves": img_to_base64(batch_vis_dir / "training_curves.png"),
        "confusion_matrix": img_to_base64(batch_vis_dir / "confusion_matrix.png"),
    }

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLEP Steganalysis Research Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #f8fafc;
            --surface-color: #ffffff;
            --border-color: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --primary-color: #0f172a;
            --accent-blue: #2563eb;
            --accent-green: #059669;
            --accent-red: #dc2626;
            --font-sans: 'Inter', sans-serif;
            --font-mono: 'Roboto Mono', monospace;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: var(--font-sans);
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 2rem 1rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background-color: var(--primary-color);
            color: #ffffff;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }}

        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }}

        .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
            max-width: 800px;
            margin: 0 auto;
        }}

        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}

        .stat-card {{
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        .stat-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            font-family: var(--font-mono);
            color: var(--primary-color);
        }}

        .stat-value.blue {{ color: var(--accent-blue); }}
        .stat-value.green {{ color: var(--accent-green); }}
        .stat-value.red {{ color: var(--accent-red); }}

        .stat-subtext {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}

        /* Tabs */
        .tabs {{
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 2rem;
            gap: 1rem;
        }}

        .tab-btn {{
            background: none;
            border: none;
            color: var(--text-muted);
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
        }}

        .tab-btn:hover {{
            color: var(--text-main);
        }}

        .tab-btn.active {{
            color: var(--accent-blue);
            border-bottom-color: var(--accent-blue);
        }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; animation: fadeIn 0.3s ease; }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }}

        .section-desc {{
            color: var(--text-muted);
            margin-bottom: 2rem;
        }}

        /* Visualizations */
        .vis-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }}
        
        @media(min-width: 1024px) {{
            .vis-grid {{ grid-template-columns: 1fr 1fr; }}
        }}

        .vis-card {{
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        .vis-card h3 {{
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }}

        .vis-card p {{
            font-size: 0.95rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }}

        .img-wrapper {{
            background: #f1f5f9;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.5rem;
            text-align: center;
        }}

        .img-wrapper img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}

        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: #f8fafc;
            font-weight: 600;
            color: var(--text-muted);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        tr:last-child td {{ border-bottom: none; }}

        /* Pre/Code */
        pre {{
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            overflow-x: auto;
            font-family: var(--font-mono);
            font-size: 0.9rem;
            color: var(--text-main);
        }}

        footer {{
            text-align: center;
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <span class="badge">ICCV 2025 Architecture</span>
            <h1>MLEP Steganalysis Research Dashboard</h1>
            <p class="subtitle">Diagnostic overview of Multi-Level Entropy Pyramids evaluating structural chaos in AI-generated images versus real sensor captures.</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Throughput</div>
                <div class="stat-value">{perf.get('throughput_images_per_sec', 'N/A')}</div>
                <div class="stat-subtext">images / sec (NVIDIA RTX 4050)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Batch Latency</div>
                <div class="stat-value">{perf.get('avg_batch_latency_ms', 'N/A')} <span style="font-size: 1rem;">ms</span></div>
                <div class="stat-subtext">Batch Size: 8 images</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Real Entropy (Mean)</div>
                <div class="stat-value green">{steg.get('mean_entropy_real', 0.0):.3f}</div>
                <div class="stat-subtext">Natural Structural Chaos</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">AI Entropy (Mean)</div>
                <div class="stat-value red">{steg.get('mean_entropy_ai_generated', 0.0):.3f}</div>
                <div class="stat-subtext">Generator Oversmoothing</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Training Score (Best)</div>
                <div class="stat-value blue">{best_train_acc:.2f}%</div>
                <div class="stat-subtext">Peak Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Validation Score (Best)</div>
                <div class="stat-value blue">{best_val_acc:.2f}%</div>
                <div class="stat-subtext">Best Epoch</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Testing Score</div>
                <div class="stat-value blue">{test_results.get('test_acc', 0.0):.2f}%</div>
                <div class="stat-subtext">Hold-out Set</div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('tab-batch')">Data Visualization</button>
            <button class="tab-btn" onclick="switchTab('tab-analytics')">Model Analytics</button>
            <button class="tab-btn" onclick="switchTab('tab-test')">Test Results</button>
            <button class="tab-btn" onclick="switchTab('tab-training')">Training History</button>
            <button class="tab-btn" onclick="switchTab('tab-report')">Execution JSON</button>
        </div>

        <!-- TAB 1: BATCH RUN -->
        <div id="tab-batch" class="tab-content active">
            <h2 class="section-title">Dataset Integration Pipeline (Batch 1, Sample 0)</h2>
            <p class="section-desc">Visualizations from the DataLoader passing 256x256 RGB tensors through the MLEP steganalysis feature extractor.</p>
            
            <div class="vis-grid">
                <div class="vis-card">
                    <h3>1. High-Resolution Entropy Heatmap</h3>
                    <p>Local Shannon Entropy across shuffled image patches. Real images exhibit high-entropy chaos; AI models show over-smoothed structures.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["batch_mlep_heatmap"]}" alt="MLEP Entropy Heatmap">' if images["batch_mlep_heatmap"] else '<p style="padding: 2rem;">No heatmap found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>2. Multi-Scale Shannon Pyramid</h3>
                    <p>Downsampled 3-level pyramid evaluating semantic anomalies across multiple receptive fields before classification.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["batch_mlep_multiscale"]}" alt="MLEP Multi-Scale Pyramid">' if images["batch_mlep_multiscale"] else '<p style="padding: 2rem;">No pyramid found</p>'}
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 1.5: MODEL ANALYTICS -->
        <div id="tab-analytics" class="tab-content">
            <h2 class="section-title">Model Analytics</h2>
            <p class="section-desc">Training progression and final classification performance.</p>
            
            <div class="vis-grid">
                <div class="vis-card">
                    <h3>1. Training & Validation Curves</h3>
                    <p>Accuracy and Loss progression across training epochs.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["training_curves"]}" alt="Training Curves">' if images["training_curves"] else '<p style="padding: 2rem;">No training curves found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>2. Test Set Confusion Matrix</h3>
                    <p>Classification performance breakdown on the unseen hold-out set.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["confusion_matrix"]}" alt="Confusion Matrix">' if images["confusion_matrix"] else '<p style="padding: 2rem;">No confusion matrix found</p>'}
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: TEST RESULTS -->
        <div id="tab-test" class="tab-content">
            <h2 class="section-title">Final Model Evaluation (Test Set)</h2>
            <p class="section-desc">Results on the unseen hold-out test set generated by <code>scripts/train.py</code>.</p>
            
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Test Loss</td>
                            <td style="font-family: var(--font-mono);">{test_results.get('test_loss', 0.0):.4f}</td>
                        </tr>
                        <tr>
                            <td>Test Accuracy</td>
                            <td style="font-family: var(--font-mono); color: var(--accent-blue); font-weight: 600;">{test_results.get('test_acc', 0.0):.2f}%</td>
                        </tr>
                        <tr>
                            <td>Precision</td>
                            <td style="font-family: var(--font-mono);">{test_results.get('test_prec', 0.0):.2f}%</td>
                        </tr>
                        <tr>
                            <td>Recall</td>
                            <td style="font-family: var(--font-mono);">{test_results.get('test_rec', 0.0):.2f}%</td>
                        </tr>
                        <tr>
                            <td>F1 Score</td>
                            <td style="font-family: var(--font-mono);">{test_results.get('test_f1', 0.0):.2f}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- TAB 3: TRAINING HISTORY -->
        <div id="tab-training" class="tab-content">
            <h2 class="section-title">MLEP Detector Training History</h2>
            <p class="section-desc">Epoch-by-epoch tracking of Train vs Validation metrics.</p>
            
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Epoch</th>
                            <th>Train Loss</th>
                            <th>Train Acc</th>
                            <th>Val Loss</th>
                            <th>Val Acc</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([f'''<tr>
                            <td>{row["epoch"]}</td>
                            <td>{row["train_loss"]:.4f}</td>
                            <td style="color: var(--text-main);">{row["train_acc"]:.2f}%</td>
                            <td>{row["val_loss"]:.4f}</td>
                            <td style="color: var(--accent-green); font-weight: 500;">{row["val_acc"]:.2f}%</td>
                        </tr>''' for row in training_history]) if training_history else '<tr><td colspan="5" style="text-align: center;">No training history found.</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- TAB 4: ANALYTICS REPORT -->
        <div id="tab-report" class="tab-content">
            <h2 class="section-title">Pipeline Execution Analytics</h2>
            <p class="section-desc">Master JSON execution summary.</p>
            <pre><code>{json.dumps(summary_data, indent=2)}</code></pre>
        </div>
    </div>

    <footer>
        <p>MLEP Project | Optimized for Windows & NVIDIA RTX 4050</p>
    </footer>

    <script>
        function switchTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Successfully generated self-contained interactive dashboard at: {output_file.resolve()}")

    if auto_open:
        logger.info("Opening dashboard in your web browser...")
        path_str = str(output_file.resolve())
        
        import urllib.request
        file_url = f"file:{urllib.request.pathname2url(path_str)}"
        
        if os.name == 'nt':
            try:
                import subprocess
                subprocess.Popen([r"C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe", path_str])
            except Exception:
                os.startfile(path_str)
        else:
            import webbrowser
            webbrowser.open(file_url)


def main():
    parser = argparse.ArgumentParser(description="Generate & Open Interactive MLEP HTML Dashboard.")
    parser.add_argument("--output", type=str, default="outputs/MLEP_Dashboard.html", help="Path to save generated HTML file.")
    parser.add_argument("--no_browser", action="store_true", help="Do not open browser automatically.")
    args = parser.parse_args()

    out_file = root_path / args.output
    generate_html(out_file, auto_open=not args.no_browser)


if __name__ == "__main__":
    main()


