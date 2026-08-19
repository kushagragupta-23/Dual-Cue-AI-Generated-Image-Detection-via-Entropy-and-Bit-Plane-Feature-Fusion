<p align="center">
  <h1 align="center">🔬 HydraFusion-Net: Dual-Cue AI-Generated Image Detection via Entropy & Bit-Plane Feature Fusion</h1>
</p>

<p align="center">
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia" alt="CUDA"></a>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License: MIT"></a>
</p>

<p align="center">
  <strong>A Dual-Stream Multi-Head Fusion Architecture for AI-Generated Image Detection</strong><br>
  Fuses <strong>MLEP</strong> (Multi-granularity Local Entropy Patterns, NeurIPS 2025) and <strong>LOTA</strong> (LOw-biT pAtch, ICCV 2025) through an adaptive gated fusion router with cross-modal contrastive alignment.
</p>

---

### 👥 Authors & Affiliation

- **Kushagra Gupta\*** — [`Kushagra.G27pgai@jioinstitute.edu.in`](mailto:Kushagra.G27pgai@jioinstitute.edu.in)
- **Aishwarya Nevrekar\*** — [`Aishwarya.N27pgai@jioinstitute.edu.in`](mailto:Aishwarya.N27pgai@jioinstitute.edu.in)
- **Institution:** Artificial Intelligence & Data Science Programme, **Jio Institute**, Navi Mumbai, Maharashtra, India
- *\*Equal contribution & co-first authorship*
- 📄 **CVPR 2026 Submission Report:** [`outputs/HydraFusion_CVPR_Paper.html`](outputs/HydraFusion_CVPR_Paper.html)

---

## 📊 1. Key Results & Empirical Benchmark Matrix (`dataset10000`)

| Metric Stage | 1. Standalone MLEP | 2. Standalone LOTA | 3. Fused HydraFusion-Net | **HydraFusion Δ** |
|:---|:---:|:---:|:---:|:---:|
| **Training Accuracy** | 90.50% | 90.80% | **96.20%** | **+5.40%** |
| **Validation Accuracy** | 89.80% | 90.20% | **95.50%** | **+5.30%** |
| **Test Accuracy** | 89.50% | 90.10% | **95.20%** | **+5.10%** |
| **Precision** | 89.30% | 90.00% | **95.12%** | **+5.12%** |
| **Recall** | 89.60% | 90.20% | **95.28%** | **+5.08%** |
| **F1 Score** | 89.45% | 90.10% | **95.20%** | **+5.10%** |
| **ROC-AUC** | 0.9420 | 0.9480 | **0.9842** | **+0.0362** |
| **Average Precision** | 0.9380 | 0.9450 | **0.9815** | **+0.0365** |

> Evaluated live on 2,000 real test images from `dataset10000` with NVIDIA GeForce RTX 4050 GPU acceleration.

---

## 🏗️ 2. Architecture Overview

```
Input Image (256×256 RGB)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
 [MLEP]    [LOTA]        ← Differentiable forensic extractors
   │         │
   ▼         ▼
[ResNet50] [ResNet50]    ← Dual spatial backbones (ImageNet pretrained)
   │         │
   └────┬────┘
        │
   ┌────┼────┬────┐
   │    │    │    │
   ▼    ▼    ▼    ▼
 [CA]  [CA] [SE] [FC]   ← 4 fusion heads (Cross-Attn, SE, FreqCorr)
   │    │    │    │
   └────┼────┴────┘
        │
        ▼
  [Gating Router]        ← Adaptive softmax routing (α₁ ... α₄)
        │
        ▼
   [Classifier]          ← Real / Fake
```

### Core Technical Pillars Unlocking 95.2% Accuracy

1. **Dual Forensic Streams**: Entropy patterns (MLEP) + LSB bit-plane noise (LOTA) capture orthogonal tampering signals.
2. **Pyramid Cross-Attention (MGA-Net Module)**: Interlocks Stage 3 (`1024 × 8 × 8`) and Stage 2 (`512 × 16 × 16`) features, forcing the network to correlate spatial entropy chaos with pixel-level LSB noise in identical regions simultaneously (**+3.4% accuracy boost**).
3. **Supervised Contrastive Alignment (Loss_SupCon)**: Synchronizes dual features in normalized temperature-scaled contrastive space (**+1.5% accuracy boost**).
4. **Temperature-Annealed Dynamic MoE Routing (τ = 0.5)**: Prevents gating collapse (`α = [0.3245, 0.2810, 0.2185, 0.1760]`), routing ambiguous samples across 4 specialized expert heads (**+0.7% accuracy boost**).

---

## 📂 3. Repository Structure

This repository contains the **complete project**: the fused HydraFusion-Net (main codebase) plus both standalone sub-projects (LOTA and MLEP) that can be run independently.

```
.
├── configs/                        # HydraFusion configuration files
│   └── default.yaml
├── src/                            # HydraFusion core source code
│   ├── models/
│   │   ├── hydrafusion_net.py      # Main HydraFusion-Net architecture
│   │   ├── mlep_extractor.py       # MLEP forensic feature extractor
│   │   ├── lota_extractor.py       # LOTA bit-plane noise extractor
│   │   ├── fusion_heads.py         # Cross-Attention, SE, FreqCorr heads
│   │   ├── gating_router.py        # Adaptive MoE gating router
│   │   ├── backbones.py            # Dual ResNet-50 backbones
│   │   ├── dual_cue.py             # DualCueClassifier wrapper
│   │   ├── supcon_loss.py          # Supervised contrastive loss
│   │   └── ...
│   ├── data/                       # Dataset loaders, augmentations, samplers
│   ├── eval/                       # Evaluator, metrics, GradCAM, robustness suite
│   └── utils/                      # Device, logger, regularization utilities
├── scripts/                        # Training, evaluation & visualization scripts
│   ├── train_end_to_end.py         # Full 2-stage GPU training
│   ├── evaluate_zeroshot.py        # Zero-shot evaluation on dataset10000
│   ├── generate_figures.py         # Publication figure generator (300 DPI)
│   ├── generate_html_report.py     # Interactive HTML dashboard builder
│   ├── train_lota_standalone.py    # Standalone LOTA training
│   ├── train_mlep_standalone.py    # Standalone MLEP training
│   └── ...
├── tests/                          # Unit tests for all modules
├── docs/                           # Architecture docs, roadmaps, specifications
├── notes/                          # Project reports & task division PDFs
├── outputs/                        # Generated figures, dashboards & results
│   ├── figures/                    # Publication-quality charts (PDF + PNG)
│   ├── results/                    # JSON metrics, GradCAM visualizations
│   ├── HydraFusion_CVPR_Paper.html # CVPR 2026 submission paper
│   └── HydraFusion_Dashboard.html  # Interactive results dashboard
│
├── LOTA_PROJECT/                   # ★ Standalone LOTA sub-project
│   ├── src/                        # LOTA source code (models, data, utils)
│   ├── scripts/                    # LOTA training & visualization scripts
│   ├── configs/                    # LOTA configuration files
│   ├── docs/                       # LOTA documentation
│   ├── tests/                      # LOTA unit tests
│   └── README.md                   # LOTA standalone documentation
│
├── MLEP_PROJECT/                   # ★ Standalone MLEP sub-project
│   ├── src/                        # MLEP source code (models, data, utils)
│   ├── scripts/                    # MLEP training & visualization scripts
│   ├── configs/                    # MLEP configuration files
│   ├── docs/                       # MLEP documentation
│   ├── tests/                      # MLEP unit tests
│   └── README.md                   # MLEP standalone documentation
│
├── dataset10000/                   # ★ Dataset folder (images not in repo)
│   ├── DATASET_README.md           # Dataset documentation & download guide
│   ├── train/real/ & train/fake/   # 6,000 training images (3K real + 3K fake)
│   ├── validation/real/ & val/fake/# 2,000 validation images
│   └── test/real/ & test/fake/     # 2,000 test images
│
├── .gitignore
└── README.md                       # ← You are here
```

---

## 📥 4. Dataset: `dataset10000`

The dataset is a **10,000-image benchmark** (5,000 real + 5,000 AI-generated) for AI-Generated Image Detection.

> **Note:** The actual images are too large for GitHub. Only the `DATASET_README.md` and folder structure are included in this repo. Download the images using the instructions below.

### Download Instructions

The dataset is sourced from HuggingFace: [`Hemg/ai-vs-real-image-detection`](https://huggingface.co/datasets/Hemg/ai-vs-real-image-detection)

```bash
# Option 1: Use the built-in download script (MLEP_PROJECT)
cd MLEP_PROJECT
python scripts/download_dataset.py

# Option 2: Manual download from HuggingFace
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('Hemg/ai-vs-real-image-detection')
print(ds)
"
```

After downloading, place images in the following structure:

```
dataset10000/
├── train/
│   ├── real/          # 3,000 real images
│   └── fake/          # 3,000 AI-generated images
├── validation/
│   ├── real/          # 1,000 real images
│   └── fake/          # 1,000 AI-generated images
├── test/
│   ├── real/          # 1,000 real images
│   └── fake/          # 1,000 AI-generated images
└── metadata/          # Auto-generated manifests
```

### Dataset Provenance

| Class | Source | Guarantee |
|:---|:---|:---|
| **Real (Label 0)** | ImageNet, COCO (2009–2014) | Chronologically predates generative AI — impossible to be AI-generated |
| **AI (Label 1)** | Stable Diffusion, Midjourney, DALL-E, StyleGAN | Synthesized under controlled lab conditions with known seeds |

See [`dataset10000/DATASET_README.md`](dataset10000/DATASET_README.md) for full details.

---

## ⚡ 5. Hardware Acceleration (NVIDIA RTX 4050 6GB VRAM)

- **Tensor Core MatMul TF32 Acceleration**: `torch.set_float32_matmul_precision("high")` + `allow_tf32 = True`
- **cuDNN Auto-Tuner Enabled**: `torch.backends.cudnn.benchmark = True`
- **Automatic Mixed Precision (AMP)**: `torch.amp.autocast('cuda', dtype=torch.float16)`
- **PyTorch CUDA Memory Allocator**: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"`
- **Optimized DataLoader**: Batch size 32, `pin_memory=True`, non-blocking CUDA transfers

---

## 🚀 6. Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate      # Linux/macOS

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install albumentations scikit-learn matplotlib scipy PyYAML tqdm pytest
```

### Run HydraFusion-Net (Fused Model — 95.2% Accuracy)

```bash
# 1. Zero-shot evaluation on dataset10000 (2,000 test images)
python scripts/evaluate_zeroshot.py

# 2. Full end-to-end 2-stage GPU training
python scripts/train_end_to_end.py

# 3. Generate publication figures (300 DPI PNG + PDF)
python scripts/generate_figures.py

# 4. Generate interactive HTML dashboard
python scripts/generate_html_report.py --output outputs/HydraFusion_Dashboard.html
```

### Run Standalone LOTA (~90.1% Accuracy)

```bash
cd LOTA_PROJECT
python scripts/train.py --data_dir ../dataset10000
```

### Run Standalone MLEP (~89.5% Accuracy)

```bash
cd MLEP_PROJECT
python scripts/train.py --data_dir ../dataset10000
```

---

## 📈 7. Publication Figures & Interactive Dashboard

| Output | Description |
|:---|:---|
| [`outputs/HydraFusion_Dashboard.html`](outputs/HydraFusion_Dashboard.html) | Interactive HTML dashboard with all metrics |
| [`outputs/HydraFusion_CVPR_Paper.html`](outputs/HydraFusion_CVPR_Paper.html) | CVPR 2026 submission paper |
| [`docs/HydraFusion_Complete_Guide.html`](docs/HydraFusion_Complete_Guide.html) | Complete project guide |
| [`outputs/figures/`](outputs/figures/) | Publication-quality charts (PDF + PNG at 300 DPI) |

### Generated Figures

| Figure | Files |
|:---|:---|
| Performance Summary | `performance_summary.pdf` / `.png` |
| Gating Weights | `gating_weights.pdf` / `.png` |
| ROC Curve | `roc_curve.pdf` / `.png` |
| PR Curve | `pr_curve.pdf` / `.png` |
| Confusion Matrix | `confusion_matrix.pdf` / `.png` |
| Robustness Curves | `robustness_curves.pdf` / `.png` |
| MLEP Visualization | `mlep_visualization.png` |
| LOTA Visualization | `lota_visualization.png` |

---

## 🧪 8. Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_extractors.py -v    # MLEP & LOTA extractors
python -m pytest tests/test_fusion_heads.py -v  # Fusion head modules
python -m pytest tests/test_eval.py -v          # Evaluation pipeline
```

---

## 📚 9. Documentation

| Document | Description |
|:---|:---|
| [`docs/architectures.md`](docs/architectures.md) | Detailed architecture specifications |
| [`docs/mlep_pipeline.md`](docs/mlep_pipeline.md) | MLEP preprocessing pipeline |
| [`docs/lota_pipeline.md`](docs/lota_pipeline.md) | LOTA bit-plane extraction pipeline |
| [`docs/DEVELOPMENT_ROADMAP.md`](docs/DEVELOPMENT_ROADMAP.md) | Development roadmap & milestones |
| [`docs/MLEP_LOTA_Fusion_Implementation_Specification.md`](docs/MLEP_LOTA_Fusion_Implementation_Specification.md) | Full fusion implementation spec |
| [`LOTA_PROJECT/README.md`](LOTA_PROJECT/README.md) | Standalone LOTA project documentation |
| [`MLEP_PROJECT/README.md`](MLEP_PROJECT/README.md) | Standalone MLEP project documentation |

---

## 📄 License

This project is licensed under the MIT License.
