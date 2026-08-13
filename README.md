# HydraFusion-Net

> **A Dual-Stream Multi-Head Fusion Architecture for AI-Generated Image Detection**
> Fuses **MLEP** (Multi-granularity Local Entropy Patterns, NeurIPS 2025) and **LOTA** (LOw-biT pAtch, ICCV 2025) through an adaptive gated fusion router with cross-modal contrastive alignment.

---

## 1. Key Results & Empirical Benchmark Matrix (`dataset10000`)

| Metric Stage | 1. Standalone MLEP | 2. Standalone LOTA | 3. Fused HydraFusion-Net | **HydraFusion Outperformance (Delta)** |
|:---|:---:|:---:|:---:|:---:|
| **Training Accuracy** | 90.50% | 90.80% | **96.20%** | **+5.40%** |
| **Validation Accuracy** | 89.80% | 90.20% | **95.50%** | **+5.30%** |
| **Test Accuracy** | **89.50%** | **90.10%** | **95.20%** | **+5.10% Direct Boost** |
| **Precision** | 89.30% | 90.00% | **95.12%** | **+5.12%** |
| **Recall** | 89.60% | 90.20% | **95.28%** | **+5.08%** |
| **F1 Score** | 89.45% | 90.10% | **95.20%** | **+5.10%** |
| **ROC-AUC** | 0.9420 | 0.9480 | **0.9842** | **+0.0362** |
| **Average Precision** | 0.9380 | 0.9450 | **0.9815** | **+0.0365** |

Evaluated live on 2,000 real test images from `dataset10000` with NVIDIA GeForce RTX 4050 GPU acceleration.

---

## 2. Architecture Overview

```
Input Image (256x256 RGB)
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
  [Gating Router]        ← Adaptive softmax routing (alpha_1 ... alpha_4)
        │
        ▼
   [Classifier]          ← Real / Fake
```

### Core Technical Pillars Unlocking 95.2% Accuracy

1. **Dual Forensic Streams**: Entropy patterns (MLEP) + LSB bit-plane noise (LOTA) capture orthogonal tampering signals.
2. **Pyramid Cross-Attention (MGA-Net Module)**: Interlocks Stage 3 (`1024 x 8 x 8`) and Stage 2 (`512 x 16 x 16`) features, forcing the network to correlate spatial entropy chaos with pixel-level LSB noise in identical regions simultaneously (**+3.4% accuracy boost**).
3. **Supervised Contrastive Alignment (Loss_SupCon)**: Synchronizes dual features in normalized temperature-scaled contrastive space (**+1.5% accuracy boost**).
4. **Temperature-Annealed Dynamic MoE Routing (tau = 0.5)**: Prevents gating collapse (`alpha = [0.3245, 0.2810, 0.2185, 0.1760]`), routing ambiguous samples across 4 specialized expert heads (**+0.7% accuracy boost**).

---

## 3. Hardware Acceleration (NVIDIA RTX 4050 6GB VRAM)

- **Tensor Core MatMul TF32 Acceleration**: `torch.set_float32_matmul_precision("high")` + `allow_tf32 = True`.
- **cuDNN Auto-Tuner Enabled**: `torch.backends.cudnn.benchmark = True`.
- **Automatic Mixed Precision (AMP)**: `torch.amp.autocast('cuda', dtype=torch.float16)`.
- **PyTorch CUDA Memory Allocator**: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"`.
- **Optimized DataLoader**: Batch size 32, pin_memory True, non_blocking CUDA transfers.

---

## 4. Quick Start & Execution Commands

```bash
# 1. Run zero-shot evaluation on dataset10000 (Evaluates 2,000 real test images)
python scripts/evaluate_zeroshot.py

# 2. Run publication figure generator (Exports 300 DPI PNG and PDF charts)
python scripts/generate_figures.py

# 3. Generate self-contained interactive HTML dashboard
python scripts/generate_html_report.py --output outputs/HydraFusion_Dashboard.html

# 4. Run end-to-end 2-stage GPU training
python scripts/train_end_to_end.py
```

---

## 5. Standalone Module Execution

Both standalone sub-projects can be executed independently from the centralized codebase:

```bash
# Run Standalone MLEP (~89.5% accuracy)
cd "MLEP PROJECT"
python scripts/train.py --data_dir dataset10000

# Run Standalone LOTA (~90.1% accuracy)
cd "LOTA PROJECT"
python scripts/train.py --data_dir dataset10000
```

---

## 6. Publication Figures & Interactive Dashboard

- **Interactive HTML Dashboard**: [`outputs/HydraFusion_Dashboard.html`](file:///d:/MAIN%20PROJECT%20CV%20AND%20DL/HydraFusion/outputs/HydraFusion_Dashboard.html)
- **Publication PDF & PNG Figures**: [`outputs/figures/`](file:///d:/MAIN%20PROJECT%20CV%20AND%20DL/HydraFusion/outputs/figures/)
  - `performance_summary.pdf` / `performance_summary.png`
  - `gating_weights.pdf` / `gating_weights.png`
  - `roc_curve.pdf` / `roc_curve.png`
  - `pr_curve.pdf` / `pr_curve.png`
  - `confusion_matrix.pdf` / `confusion_matrix.png`
