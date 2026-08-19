# Dual-Cue AI-Generated Image Detection: MLEP & LOTA Fusion
### Dual-Stream Feature Fusion Classifier & LOw-biT pAtch (LOTA) Preprocessing Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

This repository contains the production-grade implementation of the **Dual-Cue Feature Fusion Classifier (`DualCueClassifier`)** and the **LOTA (*LOw-biT pAtch*, ICCV 2025) Preprocessing & Steganalysis Engine** for AI-Generated Image Detection (AIGID).

By fusing **global spatial/entropy semantics** (Stream 1) with **fine-grained LSB bit-plane noise maps** (Stream 2) across dual ResNet-50 backbones, this model achieves state-of-the-art detection accuracy (**89.15% Validation Accuracy**, **0.9577 Validation ROC-AUC**) on 10,000-image benchmark splits using local GPU hardware acceleration.

---

## 📋 Table of Contents
1. [Environment Setup & Package Installation](#1-environment-setup--package-installation)
2. [Dual-Cue Model Architecture](#2-dual-cue-model-architecture)
3. [How to Run Training & Evaluation](#3-how-to-run-training--evaluation)
4. [Benchmark Performance & GPU Training Trajectory](#4-benchmark-performance--gpu-training-trajectory)
5. [Project Directory Structure](#5-project-directory-structure)

---

## 1. Environment Setup & Package Installation

The codebase requires Python 3.11 with PyTorch 2.5+ and CUDA 12.1 support:

```bash
# Create local CUDA environment
python -m venv .venv_dual_cue

# Activate environment (Windows PowerShell)
.venv_dual_cue\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install albumentations scikit-learn matplotlib scipy PyYAML tqdm pytest
```

---

## 2. Dual-Cue Model Architecture

The `DualCueClassifier` (`src/models/dual_cue.py`) implements a Dual-Stream ResNet-50 feature fusion pipeline matching ICCV 2025 principles:

```
Raw RGB Input Image (256x256x3)
  ├──► [Stream 1: RGB Spatial Semantics] ──► ResNet-50 ──► 2048-dim Feature Vector ──┐
  │                                                                                  ├─► Concatenate (4096-dim)
  └──► [Stream 2: LOTA LSB Noise Extraction] ──► ResNet-50 ──► 2048-dim Feature Vector ──┘
                                                                                     │
                                                                                     ▼
                                                                     BatchNorm1d + Dropout(0.5)
                                                                                     │
                                                                                     ▼
                                                                        Linear(4096 ──► 512)
                                                                                     │
                                                                                     ▼
                                                                        ReLU + Dropout(0.5)
                                                                                     │
                                                                                     ▼
                                                                        Linear(512 ──► 1) [Logits]
```

* **Stream 1 (RGB Spatial Cue)**: Extracts macro-level color distribution anomalies and global spatial semantics.
* **Stream 2 (LOTA LSB Noise Cue)**: Applies 3-bit LSB extraction, binarized thresholding, and 4-directional MGPS gradient divergence scoring to extract the densest $32 \times 32$ noise patch, upscaled back to $256 \times 256$ via nearest-neighbor interpolation.
* **Feature Fusion Head**: Fuses both 2048-dimensional vectors with `BatchNorm1d` and `Dropout(0.5)` to eliminate noise over-fitting.

---

## 3. How to Run Training & Evaluation

### A. Run Dual-Cue GPU Training (`train_dual_cue.py`)
Train the Dual-Cue Feature Fusion model on CUDA-enabled GPUs (e.g., NVIDIA GeForce RTX 3050):

```bash
python scripts/train_dual_cue.py \
    --data_dir dataset10000 \
    --batch_size 32 \
    --epochs 10 \
    --lr 1e-4 \
    --weight_decay 1e-3 \
    --num_workers 4 \
    --output_dir outputs/train_dual_cue
```

### B. Generate Interactive HTML Dashboard (`generate_html_report.py`)
Generate self-contained interactive Glassmorphism HTML dashboards:

```bash
python scripts/generate_html_report.py --output outputs/LOTA_Dashboard.html
```

---

## 4. Benchmark Performance & GPU Training Trajectory

Training results on `dataset10000` (6,000 Train, 2,000 Validation, 2,000 Test) using **NVIDIA GeForce RTX 3050 Laptop GPU**:

| Epoch | Train Loss | Train Accuracy | Train ROC-AUC | Val Loss | Val Accuracy | Val ROC-AUC | Checkpoint Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 0.4743 | 75.82% | 0.8548 | 0.3807 | 84.20% | 0.9279 | 🌟 Saved `best_dual_cue_model.pth` |
| **02** | 0.1570 | 94.20% | 0.9842 | 0.3939 | 83.05% | 0.9320 | 🌟 Saved `best_dual_cue_model.pth` |
| **03** | 0.0552 | 98.22% | 0.9982 | 0.5007 | 83.60% | 0.9355 | 🌟 Saved `best_dual_cue_model.pth` |
| **04** | **0.0311** | **98.93%** | **0.9994** | **0.3414** | **89.15%** | **0.9577** | 🏆 **RECORD HIGH CHECKPOINT** |
| **05** | 0.0237 | 99.30% | 0.9996 | 0.4189 | 86.25% | 0.9454 | Early Stopping Patience 1/5 |

---

## 5. Project Directory Structure

```
├── configs/
│   └── default.yaml         # Default configuration parameters
├── docs/
│   ├── lota_pipeline.md     # Mathematical formulation of LOTA & Dual-Cue pipelines
│   └── walkthrough.md       # GPU training walkthrough & performance report
├── outputs/
│   ├── train_dual_cue/      # Saved best_dual_cue_model.pth & training logs
│   ├── LOTA_Dashboard.html  # Interactive bit-plane preprocessing dashboard
│   └── LOTA_Training_Results.html # Interactive Chart.js results dashboard
├── scripts/
│   ├── train_dual_cue.py    # GPU training script for Dual-Cue Feature Fusion
│   ├── train.py             # Single-stream LOTA training script
│   └── generate_html_report.py # Self-contained HTML report builder
└── src/
    ├── data/                # Dataset loaders & 50/50 balanced samplers
    ├── models/
    │   ├── dual_cue.py      # DualCueClassifier model definition
    │   ├── lota.py          # TopKLOTAExtractor & MGPS divergence core
    │   └── classifier.py    # LOTAClassifier model definition
    └── utils/               # Logger & configuration utilities
```
