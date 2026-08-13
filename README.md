# HydraFusion-Net

> **A Dual-Stream Multi-Head Fusion Architecture for AI-Generated Image Detection**

HydraFusion-Net fuses two complementary forensic signal extractors — **MLEP** (Multi-granularity Local Entropy Patterns, NeurIPS 2025) and **LOTA** (LOw-biT pAtch, ICCV 2025) — through an adaptive gated fusion mechanism with cross-modal contrastive alignment.

---

## Key Results

| Metric | Value |
|:---|:---:|
| **Test Accuracy** | **90.20%** |
| **Best Val Accuracy** | 90.45% |
| **Precision** | 89.26% |
| **Recall** | 91.40% |
| **F1 Score** | 90.32% |

Trained on a 10K-image forensics dataset (5K real + 5K AI-generated) with an RTX 4050 6GB.

---

## Architecture

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
  [Gating Router]        ← Adaptive softmax routing (α₁...α₄)
        │
        ▼
   [Classifier]          ← Real / Fake
```

### Core Innovation

1. **Dual forensic streams**: Entropy patterns (MLEP) + bit-plane noise (LOTA) capture orthogonal tampering signals
2. **4-head fusion**: Spatial cross-attention, channel SE, and frequency correlation explore different modality interactions
3. **Adaptive gating**: Per-image dynamic routing — the model learns which fusion strategy works best for each input
4. **2-stage training**: Contrastive pre-training (SupCon) → Gated fine-tuning with frozen backbones

---

## Project Structure

```
HydraFusion/
├── configs/default.yaml          # Master configuration
├── src/
│   ├── models/                   # All model components
│   │   ├── hydrafusion_net.py    # Master model assembly
│   │   ├── mlep_extractor.py     # Differentiable MLEP
│   │   ├── lota_extractor.py     # Differentiable LOTA
│   │   ├── backbones.py          # ResNet-50 spatial stems
│   │   ├── fusion_heads.py       # 4 fusion heads
│   │   ├── gating_router.py      # Adaptive gating
│   │   ├── freq_prefilter.py     # Learnable FFT filter
│   │   ├── supcon_loss.py        # SupCon loss
│   │   └── domain_adversarial.py # GRL (disabled)
│   ├── data/
│   │   ├── dataset.py            # ForensicsDataset loader
│   │   └── augmentations.py      # Online robustness transforms
│   ├── eval/
│   │   ├── metrics.py            # Academic metrics (ROC-AUC, AP, F1)
│   │   ├── evaluator.py          # Automated evaluation loop
│   │   ├── explainability.py     # Grad-CAM visualizations
│   │   ├── robustness_suite.py   # JPEG/blur degradation testing
│   │   └── benchmark_throughput.py
│   └── utils/
│       ├── device.py             # CUDA device selector
│       └── logger.py             # Logging utility
├── scripts/
│   ├── train_end_to_end.py       # 2-stage training pipeline
│   ├── evaluate_zeroshot.py      # Evaluation CLI
│   └── generate_figures.py       # Publication figure generator
├── tests/                        # Unit tests (36 tests)
├── outputs/
│   ├── HydraFusion_Dashboard.html # Self-contained research dashboard
│   ├── checkpoints/              # Model checkpoints
│   ├── logs/                     # Training logs
│   ├── results/                  # Evaluation reports
│   └── figures/                  # Generated figures
└── docs/
    └── architectures.md          # Full architecture specification
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU (RTX 4050 or better recommended)

### Install Dependencies

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pyyaml tqdm tensorboard pillow
```

### Training

```bash
# 2-stage end-to-end training (Stage 1: SupCon, Stage 2: Gated Fusion)
python scripts/train_end_to_end.py
```

### Evaluation

```bash
# Standard test-set evaluation with full academic metrics
python scripts/evaluate_zeroshot.py

# With robustness degradation sweep + Grad-CAM visualizations
python scripts/evaluate_zeroshot.py --robustness --gradcam --num_gradcam 16
```

### Generate Publication Figures

```bash
python scripts/generate_figures.py --results_dir outputs/results --output_dir outputs/figures
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Training Details

### Stage 1: Contrastive Pre-Training (15 epochs)
- **Loss**: DualCue SupCon (τ=0.1, float32-safe)
- **Optimizer**: AdamW (LR=1e-3, WD=5e-3)
- **Scheduler**: CosineAnnealingLR
- **Frozen**: ResNet-50 backbones

### Stage 2: Gated Fusion Fine-Tuning (25 epochs)
- **Loss**: Label-smoothed BCE (0.05/0.95)
- **Optimizer**: AdamW with differential LRs
  - Heads/Router/Classifier: 3e-4
  - ResNet Layer3: 3e-5 (micro-LR)
- **Scheduler**: OneCycleLR (pct_start=0.15)
- **Early Stopping**: Patience=8

### Hardware
- **GPU**: NVIDIA RTX 4050 (6GB VRAM)
- **Precision**: FP16 AMP with FP32 loss computation
- **TF32**: Enabled for RTX 40-series matmul acceleration

---

## Evaluation Suite

| Module | Description |
|:---|:---|
| `metrics.py` | Accuracy, Precision, Recall, F1, ROC-AUC, AP, Confusion Matrices |
| `evaluator.py` | Automated evaluation with gating weight analysis & latency profiling |
| `explainability.py` | Dual-branch Grad-CAM (MLEP + LOTA heatmaps + combined overlay) |
| `robustness_suite.py` | JPEG Q∈{70,80,90,100}, Blur σ∈{0.5,1.0,2.0}, Combined degradation |
| `benchmark_throughput.py` | Latency/throughput across batch sizes and FP32/FP16 precision |

---

## References

1. **MLEP**: Lin Yuan et al., "Multi-granularity Local Entropy Patterns for AI-Generated Image Detection", NeurIPS 2025. [Code](https://github.com/fkeufss/MLEP)
2. **LOTA**: Hongsong Wang et al., "LOw-biT pAtch: Bit-Planes Guided AI-Generated Image Detection", ICCV 2025. [Code](https://github.com/hongsong-wang/LOTA)

---

## License

This project is for academic research purposes.
