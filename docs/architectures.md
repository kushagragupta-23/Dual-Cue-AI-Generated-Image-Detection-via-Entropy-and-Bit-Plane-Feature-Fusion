# HydraFusion-Net: Architecture Specification

## Overview

**HydraFusion-Net** is a dual-stream, multi-head fusion architecture for AI-Generated Image Detection (AIGID) that combines two complementary forensic signal extractors—**MLEP** (Multi-granularity Local Entropy Patterns, NeurIPS 2025) and **LOTA** (LOw-biT pAtch, ICCV 2025)—through a learnable gated fusion mechanism with cross-modal alignment.

---

## Architecture Diagram

```
Input Image (B, 3, 256, 256) — range [0, 255]
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
┌──────────────────┐                        ┌──────────────────┐
│ LearnableFreq    │                        │                  │
│ PreFilter (rFFT2)│                        │ Differentiable   │
│ Butterworth mask │                        │ LOTA Extractor   │
└────────┬─────────┘                        │ (Soft bit-plane  │
         │                                  │  + attention)    │
         ▼                                  └────────┬─────────┘
┌──────────────────┐                                 │
│ Differentiable   │                                 │
│ MLEP Extractor   │                                 │
│ (Multi-scale     │                                 │
│  entropy proxy)  │                                 │
└────────┬─────────┘                                 │
         │                                           │
    (B, 9, 256, 256)                          (B, 3, 256, 256)
         │                                           │
         ▼                                           ▼
┌──────────────────┐                        ┌──────────────────┐
│ ResNet-50 Stem   │                        │ ResNet-50 Stem   │
│ (MLEP branch)    │                        │ (LOTA branch)    │
│ 9-ch input       │                        │ 3-ch input       │
│ → layer3 output  │                        │ → layer3 output  │
└────────┬─────────┘                        └────────┬─────────┘
         │                                           │
    (B, 1024, 8, 8)                          (B, 1024, 8, 8)
         │                                           │
         └─────────────────┬─────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Multi-Head Fusion    │
              │                        │
              │  Head 1: SpatialCross  │
              │    Attn (MLEP→LOTA)    │
              │  Head 2: SpatialCross  │
              │    Attn (LOTA→MLEP)    │
              │  Head 3: Channel SE    │
              │  Head 4: FreqCorr      │
              │                        │
              │  Each → (B, 512)       │
              └──────────┬─────────────┘
                         │
                    (B, 4, 512)
                         │
                         ▼
              ┌────────────────────────┐
              │  Adaptive Gating      │
              │  Router               │
              │  α = softmax(MLP(     │
              │    [GAP(f_m);GAP(f_l)]│
              │  ))                   │
              │  fused = Σ αᵢ · hᵢ   │
              └──────────┬─────────────┘
                         │
                    (B, 512)
                         │
                         ▼
              ┌────────────────────────┐
              │  Binary Classifier     │
              │  Linear(512→256)→      │
              │  LayerNorm→GELU→       │
              │  Dropout(0.3)→         │
              │  Linear(256→1)         │
              └──────────┬─────────────┘
                         │
                    logit (B, 1)
                         │
                         ▼
              sigmoid → p(fake) → {Real, Fake}
```

---

## Component Details

### 1. Learnable Frequency PreFilter

**File:** `src/models/freq_prefilter.py`

Applies a differentiable Butterworth low-pass filter in the frequency domain using rFFT2 to strip JPEG quantization blockiness before entropy computation.

**Parameters:**
- `ω_c` (learnable): Cutoff frequency, initialized at 0.8
- `σ` (learnable): Slope steepness, initialized at 10.0

**Formula:**
```
H(u,v) = 1 / (1 + (D(u,v) / ω_c)^(2σ))
```

Where `D(u,v)` is the normalized Euclidean distance from the DC component.

---

### 2. Differentiable MLEP Extractor

**File:** `src/models/mlep_extractor.py`

Produces multi-scale entropy-proxy feature maps that capture statistical randomness at different granularities. This is a differentiable approximation of the original discrete Shannon entropy computation from the MLEP paper.

**Key Design Decision:** The original MLEP uses discrete entropy values V ∈ {0, 0.8113, 1.0, 1.5, 2.0} computed from windowed pixel distributions. Since this is non-differentiable, HydraFusion replaces it with a **soft entropy proxy** using local variance statistics passed through sigmoid activation, preserving the same information-theoretic signal while enabling gradient flow.

**Multi-Scale Processing:**
- Scale 1.0: Full resolution (256×256)
- Scale 0.5: Half resolution (128×128) → bicubic upsampled
- Scale 0.25: Quarter resolution (64×64) → bicubic upsampled

**Output:** `(B, 9, 256, 256)` — 3 scales × 3 RGB channels

---

### 3. Differentiable LOTA Extractor

**File:** `src/models/lota_extractor.py`

Extracts low-bit-plane noise patterns that expose intrinsic generation artifacts invisible to human perception. This is a differentiable approximation of the original uint8 bitwise AND + hard thresholding operations.

**Key Design Decision:** The original LOTA performs `pixel & (1 << bit_plane)` which is non-differentiable. HydraFusion replaces this with learned 1×1 convolutions + sigmoid-based soft thresholding that learns to extract equivalent noise patterns while maintaining gradient flow.

**Bit-Plane Configuration:**
- Bit-plane 0 (LSB): Strongest noise signal
- Bit-plane 1: Secondary noise
- Bit-plane 2: Tertiary noise

**Maximum Gradient Patch Selection:** Uses a soft attention mechanism instead of hard top-k selection to maintain differentiability.

**Output:** `(B, 3, 256, 256)`

---

### 4. ResNet-50 Spatial Stems (Dual Backbone)

**File:** `src/models/backbones.py`

Two independent ResNet-50 backbones (pre-trained on ImageNet) that extract deep spatial features from the MLEP and LOTA modalities.

| Stem | Input Channels | Output Shape |
|:---|:---:|:---:|
| MLEP Stem | 9 (3 scales × 3 RGB) | (B, 1024, 8, 8) |
| LOTA Stem | 3 (RGB bit-plane map) | (B, 1024, 8, 8) |

**Frozen during Stage 1.** Layer3 unfrozen with micro-LR (3e-5) during Stage 2.

---

### 5. Multi-Head Fusion Module

**File:** `src/models/fusion_heads.py`

Four complementary fusion heads process the dual spatial feature maps from different perspectives:

| Head | Mechanism | What It Captures |
|:---|:---|:---|
| **SpatialCrossAttn (MLEP→LOTA)** | Q-K-V attention with MLEP as query | Where MLEP entropy patterns align with LOTA noise |
| **SpatialCrossAttn (LOTA→MLEP)** | Q-K-V attention with LOTA as query | Where LOTA noise maps correlate with entropy anomalies |
| **ChannelSE** | Squeeze-Excitation on concatenated features | Which channel combinations are most discriminative |
| **FreqCorrelation** | rFFT2 → element-wise product → iFFT2 | Frequency-domain cross-correlation between modalities |

Each head outputs a `(B, 512)` vector. Stacked: `(B, 4, 512)`.

---

### 6. Adaptive Gating Router

**File:** `src/models/gating_router.py`

Dynamically weights the 4 fusion heads based on the global context of both modalities:

```
z = [GAP(f_mlep) ; GAP(f_lota)]  → (B, 2048)
α = softmax(MLP(z))              → (B, 4)
fused = Σᵢ αᵢ · hᵢ              → (B, 512)
```

This allows the model to emphasize different fusion strategies per image—e.g., relying more on frequency correlation for JPEG-compressed inputs, or spatial cross-attention for high-quality inputs.

---

### 7. Binary Classifier

**In:** `src/models/hydrafusion_net.py`

```python
Linear(512 → 256) → LayerNorm → GELU → Dropout(0.4)
 → Linear(256 → 128) → LayerNorm → GELU → Dropout(0.3)
 → Linear(128 → 1)
```

Output is a raw logit; sigmoid applied for probability.

---

### 8. Domain Adversarial Head (Disabled)

**File:** `src/models/domain_adversarial.py`

A Gradient Reversal Layer (GRL) + domain classifier intended to force generator-agnostic representations. **Currently disabled** because domain labels (which generator produced each fake image) are not available in the training dataset.

---

## Training Pipeline

### 2-Stage Training Strategy

| Stage | Objective | What Trains | LR | Epochs |
|:---|:---|:---|:---:|:---:|
| **Stage 1** | Contrastive Pre-Training (SupCon) | Extractors + Projections | 1e-3 | 15 |
| **Stage 2** | Gated Fusion Fine-Tuning (BCE) | Fusion heads + Router + Classifier + Layer3 (micro-LR) | 3e-4 / 3e-5 | 25 |

### Key Training Details

- **Label Smoothing:** 0 → 0.05, 1 → 0.95
- **AMP:** FP16 autocast with FP32 loss computation (prevents SupCon overflow)
- **Gradient Clipping:** max_norm=1.0
- **Scheduler Stage 1:** CosineAnnealingLR
- **Scheduler Stage 2:** OneCycleLR with differential peak LRs
- **Early Stopping:** Patience=8 on validation accuracy
- **Weight Decay:** 0.005 for heads, 0.0025 for backbone layers

---

## Current Performance

### Best Trained Model (2-Stage Pipeline — `metrics.json`)

| Metric | Value |
|:---:|:---:|
| **Best Val Accuracy** | 95.50% |
| **Test Accuracy** | 95.20% |
| **Precision** | 95.12% |
| **Recall** | 95.28% |
| **F1 Score** | 95.20% |
| **ROC-AUC** | 0.9842 |
| **Average Precision** | 0.9815 |

### Zero-Shot Evaluation (Pre-trained Checkpoint — `test_evaluation.json`)

| Metric | Value |
|:---:|:---:|
| **Test Accuracy** | 90.20% |
| **Precision** | 89.26% |
| **Recall** | 91.40% |
| **F1 Score** | 90.32% |
| **ROC-AUC** | 0.9576 |
| **Average Precision** | 0.9466 |
| **Latency** | 5.68 ms/image |
| **Throughput** | 176.0 images/sec |

**Dataset:** 10,000 images (5K real + 5K AI-generated), split 60/20/20.

---

## File Structure

```
HydraFusion/
├── configs/
│   └── default.yaml              # Master configuration
├── src/
│   ├── models/
│   │   ├── hydrafusion_net.py    # Master model assembly
│   │   ├── mlep_extractor.py     # Differentiable MLEP
│   │   ├── lota_extractor.py     # Differentiable LOTA
│   │   ├── backbones.py          # ResNet-50 spatial stems
│   │   ├── fusion_heads.py       # 4 fusion heads
│   │   ├── gating_router.py      # Adaptive gating
│   │   ├── freq_prefilter.py     # Learnable frequency filter
│   │   ├── supcon_loss.py        # SupCon contrastive loss
│   │   └── domain_adversarial.py # GRL + domain head (disabled)
│   ├── data/
│   │   ├── dataset.py            # ForensicsDataset loader
│   │   └── augmentations.py      # Online robustness transforms
│   ├── eval/
│   │   ├── metrics.py            # Academic metric calculators
│   │   ├── evaluator.py          # Automated evaluation loop
│   │   ├── explainability.py     # Grad-CAM dual-branch visualizer
│   │   ├── robustness_suite.py   # JPEG/blur degradation testing
│   │   └── benchmark_throughput.py # GPU throughput benchmarks
│   └── utils/
│       ├── device.py             # CUDA device selector
│       └── logger.py             # Logging utility
├── scripts/
│   ├── train_end_to_end.py       # 2-stage training pipeline
│   ├── evaluate_zeroshot.py      # Evaluation CLI
│   └── generate_figures.py       # Publication figure generator
├── tests/
│   ├── test_eval.py              # Metrics unit tests
│   ├── test_extractors.py        # MLEP/LOTA unit tests
│   └── test_fusion_heads.py      # Fusion head unit tests
├── outputs/
│   ├── checkpoints/              # Model checkpoints
│   ├── logs/                     # Training logs
│   └── results/                  # Evaluation reports
└── docs/
    └── architectures.md          # This document
```
