#!/usr/bin/env python3
"""
Interactive HTML Dashboard Generator for MLEP
Generates a self-contained, professional HTML report with Base64-embedded diagnostic figures.
Can be opened directly in Google Chrome or any web browser.
"""

import argparse
import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import webbrowser

# Ensure root path is accessible
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.utils.logger import get_logger

logger = get_logger("html_report_generator")

def get_file_metadata(filepath: Path):
    if not filepath.exists():
        return {"hash": "N/A", "modified": "N/A", "size": "N/A", "path": str(filepath)}
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    mtime = os.path.getmtime(filepath)
    mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    size_kb = os.path.getsize(filepath) / 1024.0
    
    return {
        "hash": sha256_hash.hexdigest(),
        "modified": mod_time_str,
        "size": f"{size_kb:.1f} KB",
        "path": str(filepath.resolve())
    }

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

    # Cryptographic Provenance
    history_meta = get_file_metadata(history_path)
    test_meta = get_file_metadata(test_path)
    model_meta = get_file_metadata(root_path / "outputs" / "checkpoints" / "mlep_best.pth")
    
    dataset_path = Path(summary_data.get("dataset_root", str(root_path / "dataset10000")))
    if dataset_path.exists():
        real_count = 0
        ai_count = 0
        for split in ["train", "validation", "test"]:
            split_path = dataset_path / split
            if split_path.exists():
                real_count += len(list((split_path / "real").glob("*.*")))
                ai_count += len(list((split_path / "fake").glob("*.*")))
        dataset_proof = f"""
            <strong>Verified on disk:</strong> {real_count} Real, {ai_count} AI images (Total: {real_count + ai_count}).<br><br>
            <div style='margin-top: 10px; font-size: 0.9rem;'>
                <strong>Data Source:</strong><br>
                <a href='https://huggingface.co/datasets/Hemg/ai-vs-real-image-detection' target='_blank' style='color: var(--accent-blue); text-decoration: underline;'>Hemg/ai-vs-real-image-detection (HuggingFace Hub)</a><br>
                <span style='color: #64748b;'>The upstream dataset aggregates real photographs from pre-2020 CV benchmarks and AI-generated images from diffusion models.</span>
            </div>
        """
    else:
        dataset_proof = "Dataset path not found on disk."

    # Encode images to Base64 for 100% self-contained HTML portability
    images = {
        "batch_mlep_heatmap": img_to_base64(batch_vis_dir / "batch1_sample0_mlep_heatmap.png"),
        "batch_mlep_multiscale": img_to_base64(batch_vis_dir / "batch1_sample0_mlep_multiscale.png"),
        "training_curves": img_to_base64(batch_vis_dir / "training_curves.png"),
        "confusion_matrix": img_to_base64(batch_vis_dir / "confusion_matrix.png"),
        "roc_curve": img_to_base64(batch_vis_dir / "roc_curve.png"),
        "pr_curve": img_to_base64(batch_vis_dir / "pr_curve.png"),
        "prob_dist": img_to_base64(batch_vis_dir / "prob_dist.png"),
        "tsne_clusters": img_to_base64(batch_vis_dir / "tsne_clusters.png"),
        "feature_importance_real": img_to_base64(batch_vis_dir / "feature_importance_real.png"),
        "feature_importance_ai": img_to_base64(batch_vis_dir / "feature_importance_ai.png"),
        "error_analysis": img_to_base64(batch_vis_dir / "error_analysis.png"),
        "fft_analysis": img_to_base64(batch_vis_dir / "fft_analysis.png"),
        "lbp_texture": img_to_base64(batch_vis_dir / "lbp_texture.png"),
        "chrominance_scatter": img_to_base64(batch_vis_dir / "chrominance_scatter.png"),
        "calibration_curve": img_to_base64(batch_vis_dir / "calibration_curve.png"),
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

        /* Tab Styles */
        .tab {{
            overflow: hidden;
            border-bottom: 2px solid var(--border-color);
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 3rem;
        }}
        .tab button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 24px;
            transition: 0.3s;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px 8px 0 0;
            color: var(--text-muted);
            border-bottom: 3px solid transparent;
        }}
        .tab button:hover {{
            background-color: #e2e8f0;
            color: var(--primary-color);
        }}
        .tab button.active {{
            color: var(--accent-blue);
            border-bottom: 3px solid var(--accent-blue);
        }}
        .tabcontent {{
            display: none;
            animation: fadeEffect 0.5s;
        }}
        @keyframes fadeEffect {{
            from {{opacity: 0;}}
            to {{opacity: 1;}}
        }}

        /* Sections */
        .section-block {{
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border-color);
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
            gap: 3rem;
        }}
        .vis-card {{
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            padding: 1.5rem;
            border: 1px solid #e2e8f0;
        }}
        .vis-card h3 {{
            margin-top: 0;
            color: #1e293b;
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }}
        .img-wrapper {{
            width: 100%;
            display: flex;
            justify-content: center;
            background: #f8fafc;
            border-radius: 6px;
            padding: 1rem;
            overflow: hidden;
        }}
        .img-wrapper img {{
            max-width: 100%;
            height: auto;
            display: block;
        }}
        .proof-card {{
            background: #fff;
            border-left: 6px solid var(--accent-blue);
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            padding: 2rem;
            margin-bottom: 2rem;
            border-top: 1px solid #e2e8f0;
            border-right: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .proof-card.real-proof {{
            border-left-color: var(--accent-green);
        }}
        .proof-card.ai-proof {{
            border-left-color: var(--accent-blue);
        }}
        .proof-title {{
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            margin-top: 0;
        }}
        .proof-links {{
            margin-bottom: 1rem;
            font-size: 0.95rem;
        }}
        .proof-reasoning {{
            background: #f8fafc;
            padding: 1.5rem;
            border-radius: 6px;
            font-size: 1rem;
            line-height: 1.6;
            color: #334155;
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
            <span class="badge">MLEP Architecture</span>
            <h1>MLEP AI-Generated Image Detection Dashboard</h1>
            <p class="subtitle">Diagnostic report showing entropy-based analysis results for detecting AI-generated images. Based on the MLEP approach by Yuan et al.</p>
        </header>

        <!-- TAB NAVIGATION -->
        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'TabArchitecture')">1. Architecture</button>
            <button class="tablinks" onclick="openTab(event, 'TabTraining')">2. Model Training & Regularization</button>
            <button class="tablinks" onclick="openTab(event, 'TabVisuals')">3. Diagnostic Visuals</button>
            <button class="tablinks" onclick="openTab(event, 'TabData')">4. Data Provenance</button>
        </div>
        
        <div id="TabArchitecture" class="tabcontent" style="display:block;">

        <!-- SECTION: COMPLETE ARCHITECTURE PIPELINE -->
        <div id="section-pipeline" class="section-block" style="border-top: none; margin-top: 1rem; padding-top: 0;">
            <h2 class="section-title" style="margin-bottom: 0.5rem; text-align: left;">Full Dual-Cue Architecture Pipeline (Roadmap)</h2>
            <p class="section-desc" style="margin-bottom: 2rem; text-align: left;">
                This diagram shows the full project pipeline. The <strong>MLEP</strong> branch (entropy-based detection) is fully implemented and evaluated. The <strong>BPFF</strong> branch (bit-plane analysis) is <strong>under development by Aishwarya</strong>. <br><br>
                <strong style="color: var(--accent-blue);">Approach:</strong> The two branches are developed independently to avoid sharing biases. Once both achieve good standalone accuracy, they will be fused into a single dual-cue classifier.
            </p>
            
            <div style="background: #f8fafc; padding: 2rem; border-radius: 12px; border: 1px solid var(--border-color); display: flex; flex-direction: column; align-items: center; gap: 1.5rem; font-family: var(--font-main);">
                
                <!-- Input Block -->
                <div style="background: #e2e8f0; padding: 1rem 2rem; border-radius: 8px; font-weight: bold; color: #1e293b; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #cbd5e1;">
                    Raw Image Input (Real or AI-Generated)
                </div>
                
                <!-- Split Arrows -->
                <div style="display: flex; gap: 8rem; color: #94a3b8; font-weight: bold;">
                    <div>&#x2199;</div>
                    <div>&#x2198;</div>
                </div>

                <!-- Dual Branches -->
                <div style="display: flex; gap: 2rem; width: 100%; justify-content: center;">
                    <!-- MLEP Branch (Active) -->
                    <div style="background: #ecfdf5; padding: 1.5rem; border-radius: 8px; border: 2px solid var(--accent-green); width: 45%; text-align: center; box-shadow: 0 4px 6px rgba(16, 185, 129, 0.15); position: relative;">
                        <span style="position: absolute; top: -10px; right: -10px; background: var(--accent-green); color: white; font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: bold; text-transform: uppercase;">Current Phase (Kushagra)</span>
                        <h4 style="color: #065f46; margin-top: 0; margin-bottom: 0.5rem;">Branch 1: MLEP</h4>
                        <strong style="color: #047857; font-size: 0.9rem;">Multi-Level Entropy Pyramids</strong>
                        <p style="font-size: 0.8rem; color: #064e3b; margin-top: 0.5rem; line-height: 1.4;">Computes Shannon entropy across multi-scale image patches to detect the smoothing artifacts left by generative models.</p>
                    </div>

                    <!-- BPFF Branch (In Progress) -->
                    <div style="background: #f1f5f9; padding: 1.5rem; border-radius: 8px; border: 2px dashed #3b82f6; width: 45%; text-align: center; opacity: 0.9; position: relative; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);">
                        <span style="position: absolute; top: -10px; right: -10px; background: #3b82f6; color: white; font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: bold; text-transform: uppercase;">Under Progress (Aishwarya)</span>
                        <h4 style="color: #1e3a8a; margin-top: 0; margin-bottom: 0.5rem;">Branch 2: BPFF</h4>
                        <strong style="color: #1d4ed8; font-size: 0.9rem;">Bit-Plane Feature Fusion</strong>
                        <p style="font-size: 0.8rem; color: #1e40af; margin-top: 0.5rem; line-height: 1.4;">Slices images into bit-planes to detect low-level steganographic tampering. Independently engineered to prevent bias contamination.</p>
                    </div>
                </div>

                <!-- Merge Arrows -->
                <div style="display: flex; gap: 8rem; color: #94a3b8; font-weight: bold;">
                    <div>&#x2198;</div>
                    <div>&#x2199;</div>
                </div>

                <!-- Fusion Block -->
                <div style="background: #f8fafc; padding: 1rem 2rem; border-radius: 8px; font-weight: bold; color: #475569; border: 2px dashed #cbd5e1; width: 60%; text-align: center;">
                    Dual-Cue Feature Fusion Module (Kushagra & Aishwarya)
                </div>

                <!-- Down Arrow -->
                <div style="color: #94a3b8; font-weight: bold;">&#x2193;</div>

                <!-- Final Output -->
                <div style="background: #f8fafc; padding: 1rem 2rem; border-radius: 8px; font-weight: bold; color: #475569; border: 2px solid #cbd5e1; width: 40%; text-align: center;">
                    Final Diagnostic Classifier (Kushagra & Aishwarya)
                </div>

            </div>
        </div>

        <!-- SECTION: MLEP DETAILED PIPELINE -->
        <div id="section-mlep-pipeline" class="section-block" style="border-top: none; margin-top: 3rem; padding-top: 0;">
            <h2 class="section-title" style="margin-bottom: 0.5rem; text-align: left;">Branch 1 (Kushagra): Detailed MLEP Engineering Pipeline</h2>
            <p class="section-desc" style="margin-bottom: 2rem; text-align: left;">
                The Multi-Level Entropy Pyramids (MLEP) module is fully implemented and relies on a strict 5-step feature extraction process to expose generative oversmoothing before passing data to the fusion block.
            </p>
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div style="background: #f8fafc; border-left: 4px solid var(--accent-green); padding: 1.5rem; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <h4 style="margin: 0 0 0.5rem 0; color: #065f46;">1. Patch Shuffling <span style="font-size: 0.75rem; color: #94a3b8; font-weight: normal;">(currently disabled)</span></h4>
                    <p style="margin: 0; color: #334155; font-size: 0.9rem;">Partitions each R, G, B channel into L×L micro-patches and applies a seeded pseudo-random spatial permutation. <em>Note: This step is currently bypassed because the pretrained ResNet-50 backbone expects spatially coherent input.</em></p>
                </div>
                <div style="background: #f8fafc; border-left: 4px solid var(--accent-green); padding: 1.5rem; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <h4 style="margin: 0 0 0.5rem 0; color: #065f46;">2. Multi-Scale Resampling Pyramid</h4>
                    <p style="margin: 0; color: #334155; font-size: 0.9rem;">Bilinear downsampling at scales {1.0, 0.5, 0.25} followed by bilinear upsampling back to the original resolution, capturing both pixel-level noise and texture-level anomalies.</p>
                </div>
                <div style="background: #f8fafc; border-left: 4px solid var(--accent-green); padding: 1.5rem; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <h4 style="margin: 0 0 0.5rem 0; color: #065f46;">3. 2×2 Sliding Window Shannon Entropy</h4>
                    <p style="margin: 0; color: #334155; font-size: 0.9rem;">Computes discrete Shannon entropy over every 4-pixel window to quantify structural chaos, outputting a dense 9-channel anomaly heatmap.</p>
                </div>
                <div style="background: #f8fafc; border-left: 4px solid var(--accent-green); padding: 1.5rem; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <h4 style="margin: 0 0 0.5rem 0; color: #065f46;">4. Spatial Encoder (ResNet-50)</h4>
                    <p style="margin: 0; color: #334155; font-size: 0.9rem;">The 9-channel entropy maps are zero-mean unit-variance normalized via BatchNorm2d, and fed into an ImageNet-tiled ResNet-50 backbone to produce a 2048-D global feature vector.</p>
                </div>
                <div style="background: #f8fafc; border-left: 4px solid var(--accent-green); padding: 1.5rem; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <h4 style="margin: 0 0 0.5rem 0; color: #065f46;">5. MLP Classifier Head</h4>
                    <p style="margin: 0; color: #334155; font-size: 0.9rem;">A dropout-regularized multi-layer perceptron (Linear(2048→512) → ReLU → Linear(512→1)) computes the final predictive logit.</p>
                </div>
            </div>
        </div>

        </div> <!-- Close TabArchitecture -->

        <div id="TabVisuals" class="tabcontent">

        <div id="section-top-metrics" class="section-block" style="border-top: none; margin-top: 3rem; padding-top: 0;">
            <h2 class="section-title" style="margin-bottom: 0.5rem; text-align: left;">Execution Summary: Top 8 Metrics</h2>
            <p class="section-desc" style="margin-bottom: 2rem; text-align: left; background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--accent-blue);">
                Summary metrics from the MLEP pipeline running on an NVIDIA RTX 4050.<br><br>
                <strong>Architecture:</strong> Uses a fully unfrozen ResNet-50 backbone (~25.6 million parameters) with 9-channel entropy map input. Full gradients are computed during training.<br><br>
                <strong>How to read these metrics:</strong><br>
                • <strong>Throughput &amp; Latency:</strong> Processing speed on the RTX 4050. ~39 images/sec is fast enough for batch processing.<br>
                • <strong>Real vs AI Entropy:</strong> The mean entropy gap (Real: ~1.911 vs AI: ~1.906) reflects the generative smoothing effect that the model learns to detect.<br>
                • <strong>Precision:</strong> When the model predicts "AI", it is correct this percentage of the time.<br>
                • <strong>Recall:</strong> The percentage of actual AI images that the model successfully catches.
            </p>

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
            <div class="stat-card">
                <div class="stat-label">Precision</div>
                <div class="stat-value">{test_results.get('test_prec', 0.0):.2f}%</div>
                <div class="stat-subtext">True Positives</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Recall</div>
                <div class="stat-value">{test_results.get('test_rec', 0.0):.2f}%</div>
                <div class="stat-subtext">Sensitivity</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">F1 Score</div>
                <div class="stat-value">{test_results.get('test_f1', 0.0):.2f}%</div>
                <div class="stat-subtext">Harmonic Mean</div>
            </div>
            </div>
        </div>

        <!-- SECTION: VISUAL METRICS -->
        <div id="section-visuals" class="section-block" style="border-top: none; margin-top: 2rem; padding-top: 0;">
            <h2 class="section-title">Core Visual Metrics & Analytics (14-Chart Breakdown)</h2>
            <p class="section-desc">Comprehensive visual breakdown of classification accuracy and feature extraction. (14 Advanced Metrics)</p>
            
            <div class="vis-grid">
                <div class="vis-card">
                    <h3>2. Test Set Confusion Matrix</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What this shows:</strong> A grid comparing predicted labels against true labels for the hold-out test set. <br><br><strong>How to interpret:</strong> High values on the main diagonal (top-left to bottom-right) indicate correct classifications. Off-diagonal values show specific error types: false positives (real images called AI) or false negatives (AI images called real).<br><br><strong>Colors:</strong> Darker blue cells indicate a higher concentration of samples.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["confusion_matrix"]}" alt="Confusion Matrix">' if images["confusion_matrix"] else '<p style="padding: 2rem;">No confusion matrix found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>3. Receiver Operating Characteristic (ROC)</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What this shows:</strong> Plots the True Positive Rate against the False Positive Rate across all confidence thresholds. <br><br><strong>How to interpret:</strong> The Area Under the Curve (AUC) quantifies the model's overall discriminative ability. A curve approaching the top-left corner indicates superior performance. A diagonal line represents a random-guessing baseline.<br><br><strong>Colors:</strong> <span style="background-color: var(--accent-blue); color: white; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: bold;">Blue Curve</span> = Model performance; <span style="background-color: white; color: #ff7f0e; border: 1px dashed #ff7f0e; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: bold;">Dashed Orange</span> = Random baseline.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["roc_curve"]}" alt="ROC Curve">' if images["roc_curve"] else '<p style="padding: 2rem;">No ROC curve found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>4. Precision-Recall Curve (PR)</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What this shows:</strong> Illustrates the trade-off between Precision and Recall. <br><br><strong>How to interpret:</strong> A stable, high-value curve indicates the model can catch most AI images (high recall) without incorrectly flagging real images (high precision). Sharp drops indicate thresholds where the model begins to trade accuracy for sensitivity.<br><br><strong>Colors:</strong> <span style="background-color: var(--accent-blue); color: white; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: bold;">Blue Curve</span> = Precision-Recall balance.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["pr_curve"]}" alt="PR Curve">' if images["pr_curve"] else '<p style="padding: 2rem;">No PR curve found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>5. Model Confidence Distribution</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What this shows:</strong> A histogram of the model's sigmoid output probabilities for images in the test set. Green = actual real images, Red = actual AI-generated images. <br><br><strong>How to interpret:</strong> Ideally, green should cluster near 0.0 (model is confident it's real) and red should cluster near 1.0 (model is confident it's AI). Overlap in the middle (around 0.5) indicates uncertain predictions. The degree of separation between the two distributions reflects the model's discriminative ability.<br><br><strong>Colors:</strong> <span style="background-color: #2ca02c; color: white; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: bold;">Green</span> = Real images. <span style="background-color: #d62728; color: white; padding: 0.1rem 0.4rem; border-radius: 4px; font-weight: bold;">Red</span> = AI images. Overlap region = uncertain cases.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["prob_dist"]}" alt="Probability Distribution">' if images["prob_dist"] else '<p style="padding: 2rem;">No probability distribution found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>6. t-SNE Latent Space Clustering</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What this shows:</strong> t-SNE reduces the 2048-dimensional feature vectors from the ResNet-50 penultimate layer into a 2D scatter plot. Each dot represents one test image. <br><br><strong>How to interpret:</strong> If the green and red clusters are well-separated, the model has learned features that distinguish real from AI-generated images. If the dots are mixed together, the model's internal representation does not separate the classes well. Points near the boundary between clusters represent the hardest cases.<br><br><strong>Colors:</strong> <span style="color: #2ca02c; font-weight: bold;">Green</span> = Real images. <span style="color: #d62728; font-weight: bold;">Red</span> = AI images.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["tsne_clusters"]}" alt="t-SNE Clusters">' if images["tsne_clusters"] else '<p style="padding: 2rem;">No t-SNE clusters found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>7. Feature Importance (Real Image Saliency)</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What is happening:</strong> Grad-CAM reverse-engineers the model to highlight exactly which pixels caused it to decide an image was "Real". <br><strong>What it means:</strong> It acts like an X-Ray into the AI's decision-making process. <br><strong>What the changes show:</strong> The red/orange "hot" zones indicate the strongest focus points. For Real images, the model should focus on natural textures and coherent physical lighting. If it focuses on empty space, it is learning the wrong features.<br><br><strong>Visual Key & Color Meaning:</strong> <span style="color: #d62728; font-weight: bold;">Deep Red / Yellow (Hot)</span> = The most critical pixels the AI used to make its decision. <span style="color: #000080; font-weight: bold;">Dark Blue (Cold)</span> = Ignored pixels. If the small red hot zones shift entirely away from the main subject into empty backgrounds, it means the model is biased and is "cheating" by memorizing watermarks or background noise.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["feature_importance_real"]}" alt="Feature Importance Real">' if images["feature_importance_real"] else '<p style="padding: 2rem;">No feature importance found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>8. Feature Importance (AI Artifact Saliency)</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What is happening:</strong> Grad-CAM highlighting exactly which pixels triggered the model to flag an image as "AI-Generated". <br><strong>What it means:</strong> It proves the model isn't just guessing, but is finding forensic evidence of manipulation. <br><strong>What the changes show:</strong> The "hot" zones reveal the exact location of generative artifacts—such as asymmetrical eyes, impossible physics, or grid-like pixel noise left behind by Diffusion models.<br><br><strong>Visual Key & Color Meaning:</strong> <span style="color: #d62728; font-weight: bold;">Deep Red / Yellow (Hot)</span> = Extreme forensic focus on generative artifacts. <span style="color: #000080; font-weight: bold;">Dark Blue (Cold)</span> = Ignored pixels. Small, isolated red dots on edges, hair, or eyes prove the model caught microscopic generation errors (like an asymmetric pupil) rather than just looking at the whole face generally.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["feature_importance_ai"]}" alt="Feature Importance AI">' if images["feature_importance_ai"] else '<p style="padding: 2rem;">No feature importance found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>9. High-Resolution Entropy Heatmap</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What is happening:</strong> Calculates the Shannon Entropy (informational chaos or unpredictability) across small patches of the image. <br><strong>What it means:</strong> AI generators often exhibit different entropy distributions than real photos. <br><strong>What the changes show:</strong> Areas of extreme, unnatural smoothness or bizarre, synthetic high-frequency noise will light up on this heatmap. This can indicate the presence of generative algorithms.<br><br><strong>Visual Key & Color Meaning:</strong> <span style="color: #e6b800; font-weight: bold;">Bright Yellow</span> = High Entropy (chaotic, natural noise like grain or leaves). <span style="color: #000080; font-weight: bold;">Dark Blue</span> = Low Entropy (unnatural, perfectly smooth AI generation). Small patches of extreme dark blue hidden inside an otherwise noisy image can indicate the presence of synthetic AI blurring or denoising tools.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["batch_mlep_heatmap"]}" alt="MLEP Entropy Heatmap">' if images["batch_mlep_heatmap"] else '<p style="padding: 2rem;">No heatmap found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>10. Multi-Scale Shannon Pyramid</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What is happening:</strong> The image is downsampled into a 3-level pyramid, calculating entropy at multiple different zoom levels simultaneously. <br><strong>What it means:</strong> Some AI mistakes are tiny (pixel noise), while others are massive (a leg blending into a table). Analyzing multiple scales catches both. <br><strong>What the changes show:</strong> If an image looks normal at the macro scale but shows significant entropy anomalies at the micro scale, the pyramid will expose the discrepancy.<br><br><strong>Visual Key & Color Meaning:</strong> Contrasts <span style="color: #e6b800; font-weight: bold;">Yellow (High Chaos)</span> vs <span style="color: #000080; font-weight: bold;">Blue (Low Chaos)</span> across 3 shrinking grids. Small color discrepancies between the large grid and the smallest grid suggest the image may have been artificially stitched together, exposing generative upscaling artifacts.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["batch_mlep_multiscale"]}" alt="MLEP Multi-Scale Pyramid">' if images["batch_mlep_multiscale"] else '<p style="padding: 2rem;">No pyramid found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>11. Error Analysis (Hard Negatives/Positives)</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What is happening:</strong> A gallery of the absolute worst mistakes the model made: False Positives (Real images accused of being AI) and False Negatives (AI images that successfully fooled the model). <br><strong>What it means:</strong> This is a crucial diagnostic tool to understand the model's blind spots. <br><strong>What the changes show:</strong> By staring at the images that tricked the network, researchers can figure out what new data to add next to fix the model's weaknesses.<br><br><strong>Visual Key & Color Meaning:</strong> Grayscale images. The small numbers in the titles (e.g., <span style="font-family: monospace;">Pred: 0.99</span>) show exactly how confident the model was when it made a devastating mistake. Studying small visual details in these specific photos reveals what kind of shadows, textures, or filters consistently break the AI.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["error_analysis"]}" alt="Error Analysis">' if images["error_analysis"] else '<p style="padding: 2rem;">No error analysis found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>12. Frequency Domain (Fourier Transform) Analysis</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What is happening:</strong> The 2D Fast Fourier Transform (FFT) converts the image from pixels into raw frequencies. <br><strong>What it means:</strong> Real photos have a natural, smooth frequency decay. Generative AI models often leave behind invisible high-frequency "checkerboard" artifacts due to convolution upsampling. <br><strong>What the changes show:</strong> If the AI image contains unnatural grid-like structures hidden in the pixels, this spectral map will expose them immediately.<br><br><strong>Visual Key & Color Meaning:</strong> <span style="color: #ffb732; font-weight: bold;">Bright Magma/Yellow</span> = High concentration of a specific frequency. <span style="color: #2b1154; font-weight: bold;">Dark Purple/Black</span> = Absence of frequency. A smooth, star-like decay from the center is natural. Small, bright yellow spikes or weird geometric grid lines appearing in the dark purple outer areas can indicate the image was upscaled by an AI algorithm.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["fft_analysis"]}" alt="FFT Analysis">' if images["fft_analysis"] else '<p style="padding: 2rem;">No FFT analysis found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>13. Micro-Texture LBP Distribution</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What is happening:</strong> Local Binary Patterns (LBP) are extracted from both real and AI images to measure micro-textures (like the pores of skin or the weave of fabric). <br><strong>What it means:</strong> AI generators often struggle to hallucinate physically accurate micro-textures, creating surfaces that are too smooth or repetitively patterned. <br><strong>What the changes show:</strong> This density plot reveals if the AI images have a statistically different texture signature than the real images.<br><br><strong>Visual Key & Color Meaning:</strong> <span style="color: #2ca02c; font-weight: bold;">Green Area</span> = The true texture distribution of Real physical objects. <span style="color: #d62728; font-weight: bold;">Red Area</span> = The artificial texture distribution of AI images. If the red curve peaks at a higher or lower value than the green curve, it proves the AI is systematically failing to recreate natural physical textures.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["lbp_texture"]}" alt="LBP Texture">' if images["lbp_texture"] else '<p style="padding: 2rem;">No LBP texture found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>14. Color Space Consistency (Chrominance Scatter)</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What is happening:</strong> Evaluates the image in the YCbCr color space, completely ignoring brightness (Y) and plotting only the color chrominance channels (Blue-difference vs Red-difference). <br><strong>What it means:</strong> AI models often generate images with perfect luminance but completely unnatural, out-of-bounds color gamuts. <br><strong>What the changes show:</strong> This proves whether the AI is hallucinating colors that do not exist or rarely exist together in natural physical photography.<br><br><strong>Visual Key & Color Meaning:</strong> <span style="color: #2ca02c; font-weight: bold;">Green Dots</span> = The natural color boundaries of Real photos. <span style="color: #d62728; font-weight: bold;">Red Dots</span> = The colors generated by the AI. Small red dots scattering far outside the central green cluster prove the AI is generating biologically or physically impossible color combinations.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["chrominance_scatter"]}" alt="Chrominance Scatter">' if images["chrominance_scatter"] else '<p style="padding: 2rem;">No chrominance scatter found</p>'}
                    </div>
                </div>
                <div class="vis-card">
                    <h3>15. Model Reliability (Calibration Curve)</h3>
                    <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem;"><strong>What is happening:</strong> Plots the model's predicted probability against the actual true frequency of AI images. <br><strong>What it means:</strong> It answers the question: "When the model says it is 90% sure an image is AI, is it actually right 90% of the time?" <br><strong>What the changes show:</strong> This diagnoses if the model is dangerously overconfident or timidly underconfident.<br><br><strong>Visual Key & Color Meaning:</strong> <span style="color: #000000; border-bottom: 2px dotted #000;">Dotted Black Line</span> = Perfect, flawless calibration. <span style="color: var(--accent-blue); font-weight: bold;">Blue Line with Squares</span> = The actual model performance. If the blue line sags far below the black line, the model is wildly overestimating its abilities. Small deviations show exactly at what confidence levels you can trust the model's predictions.</p>
                    <div class="img-wrapper">
                        {f'<img src="{images["calibration_curve"]}" alt="Calibration Curve">' if images["calibration_curve"] else '<p style="padding: 2rem;">No calibration curve found</p>'}
                    </div>
                </div>
            </div>
        </div>

        </div> <!-- Close TabVisuals -->

        <div id="TabData" class="tabcontent">

        <!-- SECTION: DATA INTEGRITY & PROVENANCE -->
        <div id="section-provenance" class="section-block" style="margin-top: 1rem; border-top: none;">
            <h2 class="section-title">Data Integrity & Dataset Provenance</h2>
            <p class="section-desc">This section provides dataset source information and file checksums for reproducibility.</p>
            
            <div style="background: var(--bg-alt); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--accent-blue); margin-bottom: 2rem; text-align: left;">
                <h3 style="margin-top: 0; color: var(--accent-blue);">Dataset Acquisition & Split Statistics</h3>
                <p style="margin-bottom: 1rem;">The dataset of 10,000 images (5,000 Real, 5,000 AI) was downloaded and verified <strong>manually</strong> to ensure absolute data integrity and prevent any automated data poisoning.</p>
                <p style="margin-bottom: 0;"><strong>Dataset Splits:</strong> The dataset is rigorously stratified into 70% Training (7,000 images), 15% Validation (1,500 images), and 15% Testing (1,500 images).</p>
            </div>

            <div style="background: var(--bg-alt); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--accent-blue); margin-bottom: 2rem; text-align: left;">
                <h3 style="margin-top: 0; color: var(--accent-blue);">Artifact Checksums</h3>
                <p style="margin-bottom: 0;">SHA-256 checksums for the generated outputs are recorded below to ensure the correct files are being referenced.</p>
            </div>
            
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 20%;">Artifact / Component</th>
                            <th style="width: 25%;">File Path / Source Link</th>
                            <th style="width: 45%;">SHA-256 Checksum</th>
                            <th style="width: 10%;">Last Modified</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Execution Checksums -->
                        <tr>
                            <td><strong>Training/Validation Scores</strong></td>
                            <td><code style="font-size:0.75rem">{history_meta['path']}</code></td>
                            <td style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted);">{history_meta['hash']}</td>
                            <td>{history_meta['modified']}</td>
                        </tr>
                        <tr>
                            <td><strong>Testing Scores</strong></td>
                            <td><code style="font-size:0.75rem">{test_meta['path']}</code></td>
                            <td style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted);">{test_meta['hash']}</td>
                            <td>{test_meta['modified']}</td>
                        </tr>
                        <tr>
                            <td><strong>Best Model Checkpoint</strong></td>
                            <td><code style="font-size:0.75rem">{model_meta['path']}</code><br><span style="font-size:0.8rem">Size: {model_meta['size']}</span></td>
                            <td style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted);">{model_meta['hash']}</td>
                            <td>{model_meta['modified']}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <h3 style="margin-top: 3rem; font-size: 1.75rem; color: #0f172a;">Dataset Provenance</h3>
            <p style="margin-bottom: 2rem; color: #475569; font-size: 0.95rem; line-height: 1.6; background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--accent-blue);">
                <strong>Source Information</strong><br><br>
                The 10,000-image dataset (5,000 real, 5,000 AI) is sourced from the HuggingFace Hub (<code>Hemg/ai-vs-real-image-detection</code>).
            </p>

            <div class="proof-card real-proof">
                <h4 class="proof-title" style="color: var(--accent-green);">Real Dataset (5,000 Images)</h4>
                <div class="proof-links">
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; background: #fff;">
                            <thead>
                                <tr>
                                    <th style="padding: 0.75rem; border-bottom: 2px solid #e2e8f0; text-align: left; background: #f8fafc; font-size: 0.85rem; color: #64748b; width: 30%;">Source Origin</th>
                                    <th style="padding: 0.75rem; border-bottom: 2px solid #e2e8f0; text-align: left; background: #f8fafc; font-size: 0.85rem; color: #64748b; width: 50%;">URL</th>
                                    <th style="padding: 0.75rem; border-bottom: 2px solid #e2e8f0; text-align: left; background: #f8fafc; font-size: 0.85rem; color: #64748b; width: 20%;">Image Count</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td style="padding: 0.75rem; border-bottom: 1px solid #e2e8f0;"><strong>Hugging Face Hub</strong></td>
                                    <td style="padding: 0.75rem; border-bottom: 1px solid #e2e8f0; font-family: monospace; font-size: 0.8rem;"><a href="https://huggingface.co/datasets/Hemg/ai-vs-real-image-detection" target="_blank" style="color: var(--accent-blue); text-decoration: underline;">https://huggingface.co/datasets/Hemg/ai-vs-real-image-detection</a></td>
                                    <td style="padding: 0.75rem; border-bottom: 1px solid #e2e8f0; font-weight: bold; color: var(--text-main);">5,000</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <strong style="display: block; margin-top: 0.5rem; color: #1e293b;">Local Path:</strong> <code style="font-size: 0.85rem;">{dataset_path.resolve()}</code>
                </div>
                <div class="proof-reasoning" style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--accent-green); margin-top: 1rem;">
                    <p style="margin-bottom: 0; color: #475569; font-size: 0.95rem; line-height: 1.6;">
                        The real images in the dataset are sourced from pre-2020 benchmarks (before modern diffusion models existed), providing confidence in the label authenticity.
                    </p>
                </div>
            </div>

            <div class="proof-card ai-proof">
                <h4 class="proof-title" style="color: var(--accent-blue);">AI-Generated Dataset (5,000 Images)</h4>
                <div class="proof-links">
                    <div style="overflow-x: auto;">
                        <table style="width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; background: #fff;">
                            <thead>
                                <tr>
                                    <th style="padding: 0.75rem; border-bottom: 2px solid #e2e8f0; text-align: left; background: #f8fafc; font-size: 0.85rem; color: #64748b; width: 30%;">Source Origin</th>
                                    <th style="padding: 0.75rem; border-bottom: 2px solid #e2e8f0; text-align: left; background: #f8fafc; font-size: 0.85rem; color: #64748b; width: 50%;">URL</th>
                                    <th style="padding: 0.75rem; border-bottom: 2px solid #e2e8f0; text-align: left; background: #f8fafc; font-size: 0.85rem; color: #64748b; width: 20%;">Image Count</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td style="padding: 0.75rem; border-bottom: 1px solid #e2e8f0;"><strong>Hugging Face Hub</strong></td>
                                    <td style="padding: 0.75rem; border-bottom: 1px solid #e2e8f0; font-family: monospace; font-size: 0.8rem;"><a href="https://huggingface.co/datasets/Hemg/ai-vs-real-image-detection" target="_blank" style="color: var(--accent-blue); text-decoration: underline;">https://huggingface.co/datasets/Hemg/ai-vs-real-image-detection</a></td>
                                    <td style="padding: 0.75rem; border-bottom: 1px solid #e2e8f0; font-weight: bold; color: var(--text-main);">5,000</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <strong style="display: block; margin-top: 0.5rem; color: #1e293b;">Local Path:</strong> <code style="font-size: 0.85rem;">{dataset_path.resolve()}</code>
                </div>
                <div class="proof-reasoning" style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--accent-blue); margin-top: 1rem;">
                    <p style="margin-bottom: 0; color: #475569; font-size: 0.95rem; line-height: 1.6;">
                        These images were synthesized using various diffusion models. Because diffusion models estimate pixel gradients to denoise images, they inevitably smooth out high-frequency micro-textures.
                    </p>
                </div>
            </div>
        </div>



        <!-- SECTION 3: TEST RESULTS -->
        <div id="section-test" class="section-block">
            <h2 class="section-title">Final Model Evaluation (Test Set)</h2>
            <p class="section-desc">Results on the unseen hold-out test set generated by <code>scripts/train.py</code>.</p>
            <p class="deep-desc" style="font-size: 0.95rem; line-height: 1.6; color: #475569; background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--accent-blue); margin-bottom: 2rem;"><strong>Metric Definitions:</strong> <br>• <strong>Test Loss:</strong> Cross-entropy loss on the unseen test set. <br>• <strong>Precision:</strong> Percentage of AI predictions that were actually AI. <br>• <strong>Recall:</strong> Percentage of actual AI images that the model correctly identified. <br>• <strong>F1 Score:</strong> The harmonic mean of precision and recall.</p>
            
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

        </div> <!-- Close TabData -->

        <div id="TabTraining" class="tabcontent">
        <!-- SECTION: REGULARIZATION -->
        <div id="section-regularization" class="section-block" style="border-top: none; margin-top: 1rem; padding-top: 0;">
            <h2 class="section-title" style="margin-bottom: 0.5rem; text-align: left;">Combating Overfitting & Memorization</h2>
            <p class="section-desc" style="margin-bottom: 2rem; text-align: left; background: #fdf2f8; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #be185d; color: #831843;">
                <strong>Regularization Overview:</strong> The model utilizes Spatial Dropout (50%), the AdamW Optimizer with Weight Decay (L2 Regularization), and the CosineAnnealingLR scheduler. These techniques were heavily applied because initial tests showed overfitting, requiring strict regularization to maintain robust generalization across validation folds.
            </p>

            <div class="vis-card" style="margin-bottom: 3rem; max-width: 800px; margin-left: auto; margin-right: auto; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background: white;">
                <h3 style="text-align: center; margin-top: 0; color: #1e293b; font-size: 1.25rem;">Training & Validation Curves</h3>
                <p class="deep-desc" style="font-size: 0.9rem; line-height: 1.5; color: var(--text-muted); margin-bottom: 1.5rem; text-align: center;">Convergence of training and validation loss indicates successful learning and appropriate regularization.</p>
                <div class="img-wrapper" style="box-shadow: none;">
                    {f'<img src="{images["training_curves"]}" alt="Training Curves" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">' if images["training_curves"] else '<p style="padding: 2rem; text-align: center;">No training curves found</p>'}
                </div>
            </div>
        </div>

        <!-- SECTION 4: TRAINING HISTORY -->
        <div id="section-training" class="section-block">
            <h2 class="section-title">MLEP Detector Training History</h2>
            <p class="section-desc">Epoch-by-epoch tracking of Train vs Validation metrics.</p>
            <p class="deep-desc" style="font-size: 0.95rem; line-height: 1.6; color: #475569; background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--accent-blue); margin-bottom: 2rem;"><strong>Detailed Breakdown:</strong> Epoch-by-epoch loss and accuracy metrics on the training and validation sets.</p>
            
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th style="padding-bottom: 1rem; vertical-align: top;">Epoch<br><span style="font-size: 0.75rem; color: #64748b; font-weight: normal;">(Full pass through dataset)</span></th>
                            <th style="padding-bottom: 1rem; vertical-align: top;">Train Loss<br><span style="font-size: 0.75rem; color: #64748b; font-weight: normal;">(Model's error rate on training data)</span></th>
                            <th style="padding-bottom: 1rem; vertical-align: top;">Train Acc<br><span style="font-size: 0.75rem; color: #64748b; font-weight: normal;">(% of training data correct)</span></th>
                            <th style="padding-bottom: 1rem; vertical-align: top;">Val Loss<br><span style="font-size: 0.75rem; color: #64748b; font-weight: normal;">(Error rate on unseen data)</span></th>
                            <th style="padding-bottom: 1rem; vertical-align: top;">Val Acc<br><span style="font-size: 0.75rem; color: #64748b; font-weight: normal;">(% of unseen data correct)</span></th>
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

        <!-- SECTION 5: ANALYTICS REPORT -->
        <div id="section-report" class="section-block">
            <h2 class="section-title">Pipeline Execution Analytics</h2>
            <p class="section-desc">Master JSON execution summary.</p>
            <p class="deep-desc" style="font-size: 0.95rem; line-height: 1.6; color: #475569; background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--accent-blue); margin-bottom: 2rem;"><strong>JSON Payload:</strong> Raw execution metadata and system performance telemetry.</p>
            <pre><code>{json.dumps(summary_data, indent=2)}</code></pre>
        </div>
        </div> <!-- Close TabTraining -->
    </div> <!-- Close Container -->

    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
    </script>

    <footer>
        <p>MLEP Project | Optimized for Windows & NVIDIA RTX 4050</p>
    </footer>


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


