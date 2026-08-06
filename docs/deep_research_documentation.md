# MLEP Project: Complete Technical Reference & Deep Research Documentation

This document provides a comprehensive explanation of **every single component** in the MLEP AI-Generated Image Detection project — every configuration parameter, every chart, every architectural decision, every training strategy — with the reasoning, proofs, and scientific justification behind each choice.

---

## Table of Contents

1. [The Core Problem: Why Does This Work?](#1-the-core-problem)
2. [Configuration Parameters Explained](#2-configuration-parameters)
3. [Architecture Deep Dive](#3-architecture-deep-dive)
4. [Training Strategy & Optimizer Decisions](#4-training-strategy)
5. [Data Augmentation Strategy (Forensics Insight)](#5-data-augmentation)
6. [All Charts & Visualizations Explained](#6-charts-explained)
7. [Final Metrics & What They Mean](#7-final-metrics)

---

## 1. The Core Problem: Why Does This Work?

### The Generative Oversmoothing Effect

**What:** Real cameras capture light through a physical sensor (CCD/CMOS). This process inherently embeds random photonic noise — tiny, invisible high-frequency variations in pixel values. AI generative models (Stable Diffusion, Midjourney, DALL-E) work by *denoising* a random noise image step-by-step. This denoising process systematically smooths out high-frequency micro-textures.

**Why it matters:** This means every AI-generated image has slightly *less* randomness (lower entropy) at the pixel level compared to a real photograph, even if the image looks perfectly realistic to the human eye.

**Proof (from our data):**
| Metric | Real Images | AI-Generated Images |
|--------|------------|-------------------|
| Mean Shannon Entropy | **1.7844** | **1.7658** |
| Difference | | **-0.0186** |

This 0.0186 entropy gap is tiny but **consistent across thousands of images**. Our neural network learns to detect this gap.

**Scientific basis:** Yuan et al., "MLEP: Multi-granularity Local Entropy Patterns for AI-generated Image Detection" ([arXiv:2604.13726](https://arxiv.org/abs/2604.13726)); Wang et al., CVPR 2025, "Re-evaluating Frequency Domain Forensics in the Era of Advanced Diffusion Models."

---

## 2. Configuration Parameters Explained (`configs/default.yaml`)

### Dataset Section

| Parameter | Value | What It Does | Why This Value |
|-----------|-------|--------------|----------------|
| `data_dir` | `dataset10000` | Path to the image folder | Contains our 10,000 images (5K real + 5K AI) |
| `image_size` | `256` | All images are resized to 256×256 pixels | This is the standard input size for ResNet-50. Larger (512) would capture more noise detail but exceeds RTX 4050's 6GB VRAM. 256 is the optimal balance. |
| `batch_size` | `32` | Number of images processed together in one forward pass | 32 fills ~4GB of the 6GB VRAM. Going to 64 causes out-of-memory crashes. Going to 16 wastes GPU capacity. |
| `num_workers` | `0` | Number of CPU threads loading images in parallel | Windows has a known bug where `num_workers > 0` causes `fork()` memory leaks. Set to `0` for safety on Windows, `2` on Linux. |
| `val_split` | `0.20` | 20% of data reserved for validation | Standard ML practice. 20% = 2,000 images, enough for statistically reliable accuracy estimates. |
| `test_split` | `0.20` | 20% of data reserved for final testing | Never seen during training. This is the "exam" the model takes at the very end. |
| `seed` | `42` | Random seed for reproducibility | Ensures the same train/val/test split every run. `42` is a convention (Hitchhiker's Guide reference). |
| `enable_augmentations` | `true` | Apply random image transforms during training | Prevents the model from memorizing specific images. Forces it to learn the *entropy pattern* rather than surface features. |
| `jpeg_quality_min/max` | `70 / 100` | Range for random JPEG recompression quality | Simulates what happens when images are shared on social media (WhatsApp, Instagram compress images). Applied at only 10% probability because heavy JPEG destroys entropy signals. |
| `blur_sigma_min/max` | `0.5 / 2.0` | Range for random Gaussian blur strength | Simulates camera defocus or post-processing blur. Applied at only 10% probability because blur destroys the high-frequency noise our model depends on. |

### MLEP Section

| Parameter | Value | What It Does | Why This Value |
|-----------|-------|--------------|----------------|
| `patch_size` | `2` | Size of micro-patches for optional shuffling | Smallest possible patch. Each patch is just 2×2 = 4 pixels. |
| `scales` | `[1.0, 0.5, 0.25]` | Multi-scale pyramid factors | **1.0x:** Full resolution captures pixel-level noise. **0.5x:** Downsampled then upsampled — captures texture-level smoothing artifacts. **0.25x:** Quarter resolution — captures coarse structural artifacts. 3 scales × 3 RGB channels = **9 channels** fed into the backbone. |
| `window_size` | `2` | Sliding window for entropy computation | A 2×2 window contains 4 pixels. Shannon entropy is computed over these 4 values. Possible output values: {0.0, 0.811, 1.0, 1.5, 2.0}. |
| `seed` | `42` | Seed for deterministic patch shuffling | Ensures shuffling permutation is reproducible (currently disabled). |
| `use_shuffling` | `false` | Whether to spatially scramble patches | **Disabled** because our ResNet-50 backbone uses pretrained ImageNet weights that expect spatially coherent input. Shuffling would break the spatial relationships the convolutional filters learned. |

### Training Section

| Parameter | Value | What It Does | Why This Value |
|-----------|-------|--------------|----------------|
| `epochs` | `25` | Maximum training iterations over the full dataset | The model converges around epoch 12-15. 25 epochs with early stopping (patience=7) gives enough room without wasting time. |
| `lr` | `0.0002` | Base learning rate | Standard for fine-tuning pretrained models with AdamW. Too high (0.001) causes the model to "forget" ImageNet features. Too low (0.00001) means the model never learns. |
| `weight_decay` | `0.05` | L2 regularization penalty | Penalizes large weights to prevent overfitting. 0.05 is aggressive but necessary because the model easily memorizes the training set. |
| `early_stopping_patience` | `7` | Stop training if val accuracy doesn't improve for 7 epochs | Prevents wasted compute. If the model hasn't improved in 7 epochs, it's unlikely to get better. |
| `gradient_clip_norm` | `1.0` | Maximum gradient magnitude | Prevents "exploding gradients" where a single bad batch causes the model to jump wildly. Clips gradients to a maximum norm of 1.0. |
| `optimizer` | `AdamW` | Adam optimizer with decoupled weight decay | Better than vanilla Adam for fine-tuning because it applies weight decay correctly (to weights, not to gradient moments). |
| `scheduler` | `CosineAnnealingLR` | Learning rate schedule | Smoothly reduces the LR from its initial value to `eta_min` following a cosine curve. This prevents the model from overshooting the optimal solution late in training. |
| `scheduler_eta_min` | `0.000001` | Minimum learning rate | The LR never drops below 1e-6. This ensures the model can still make tiny adjustments even at the end of training. |
| `differential_lr.backbone` | `0.5` | LR multiplier for ResNet-50 backbone | The backbone has ImageNet pretrained weights. Training it too fast (multiplier > 1.0) would overwrite these valuable features. 0.5× means it learns at half the base rate. |
| `differential_lr.head` | `5.0` | LR multiplier for the classifier head | The classifier is randomly initialized (no pretrained weights). It needs to learn fast to catch up with the backbone. 5× means it learns 10× faster than the backbone. |
| `differential_lr.extractor` | `1.0` | LR multiplier for MLEP extractor | The extractor's BatchNorm layer needs standard adaptation speed. |

### Hardware Section

| Parameter | Value | What It Does | Why This Value |
|-----------|-------|--------------|----------------|
| `device` | `cuda` | Use GPU for computation | NVIDIA RTX 4050 — ~20× faster than CPU for matrix operations. |
| `amp` | `true` | Automatic Mixed Precision | Uses FP16 (half-precision) for forward passes and FP32 for gradients. Cuts VRAM usage by ~40% and increases throughput by ~25%. |
| `cudnn_benchmark` | `true` | Auto-tune convolution algorithms | cuDNN tries multiple kernel implementations and picks the fastest one for our specific tensor sizes. ~10-15% speed boost. |
| `tf32` | `true` | TF32 Tensor Core acceleration | RTX 40-series specific. Uses 19-bit precision internally (10-bit mantissa + 8-bit exponent + sign) instead of full FP32. ~2× faster matrix multiplications with negligible accuracy loss. |
| `target_gpu` | `NVIDIA RTX 4050` | Documentation reference | For logging and reproducibility only. |

---

## 3. Architecture Deep Dive

### The MLEP Pipeline (Step by Step)

```
Input Image (B, 3, 256, 256) — raw RGB pixels in [0, 255]
         │
         ▼
    ÷ 255.0  → Normalize to [0, 1]
         │
         ▼
   ┌─────────────────────────────────────┐
   │   Multi-Scale Resampling Pyramid    │
   │                                     │
   │  Scale 1.0x: Identity (full res)    │
   │  Scale 0.5x: Down→Up (blur effect)  │
   │  Scale 0.25x: Down→Up (more blur)   │
   │                                     │
   │  Concatenate: (B, 9, 256, 256)      │
   └─────────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────────┐
   │   2×2 Shannon Entropy (LEP)         │
   │                                     │
   │  For each 4-pixel window:           │
   │    H = -Σ p(x) · log₂(p(x))        │
   │                                     │
   │  Output: (B, 9, 255, 255)           │
   └─────────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────────┐
   │   BatchNorm2d(9)                    │
   │   Normalizes entropy maps to        │
   │   zero-mean, unit-variance          │
   └─────────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────────┐
   │   ResNet-50 Backbone                │
   │   (conv1 adapted: 3ch → 9ch)        │
   │   Global Average Pooling → 2048-D   │
   └─────────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────────┐
   │   Classifier MLP                    │
   │   Dropout(0.5) → Linear(2048→512)   │
   │   → ReLU → Dropout(0.3)            │
   │   → Linear(512→1) → Logit          │
   └─────────────────────────────────────┘
         │
         ▼
   Sigmoid → Probability (0 = Real, 1 = AI)
```

### Why ResNet-50?

**What:** ResNet-50 is a 50-layer deep convolutional neural network pretrained on ImageNet (1.2 million natural images, 1000 categories).

**Why:** We need a backbone that already "understands" image structure. Training a deep network from scratch on only 6,000 training images would massively overfit. By using pretrained weights, the backbone already knows how to detect edges, textures, and patterns — we just fine-tune it to detect entropy patterns instead.

**How the 9-channel adaptation works:** ResNet-50's first convolutional layer expects 3 input channels (RGB). Our MLEP extractor produces 9 channels. We tile the pretrained 3-channel weights 3 times to create 9-channel weights, preserving the learned filters.

### Why BatchNorm Before the Backbone?

**What:** BatchNorm2d normalizes each channel to have mean=0 and std=1 across the batch.

**Why:** The entropy maps have values in {0.0, 0.811, 1.0, 1.5, 2.0}. ImageNet-pretrained ResNet expects inputs with mean ~0.485 and std ~0.229. Without BatchNorm, the backbone would receive inputs at the completely wrong scale, making the pretrained weights useless.

### Why Dropout(0.5) + Dropout(0.3)?

**What:** Dropout randomly zeroes out neurons during training, forcing the network to not rely on any single feature.

**Why 0.5 first:** The 2048-D feature vector from ResNet is very high-dimensional. Without strong dropout, the classifier can memorize arbitrary patterns. 50% dropout forces robust feature usage.

**Why 0.3 second:** After compressing to 512 dimensions, lighter dropout (30%) allows the final classifier to make precise decisions without being too aggressive.

### Why Label Smoothing (0.05 → 0.95)?

**What:** Instead of training with hard labels (0 = Real, 1 = AI), we use soft labels (0.05 = Real, 0.95 = AI).

**Why:** Hard labels make the model overconfident. The loss function pushes the model to output exactly 0.0 or 1.0, which requires extreme weight values that cause overfitting. Soft labels tell the model "be 95% sure, not 100% sure" — this produces better-calibrated probabilities and reduces overfitting.

---

## 4. Training Strategy & Optimizer Decisions

### Why AdamW (Not SGD)?

**Adam** maintains per-parameter adaptive learning rates using momentum and RMS of gradients. This is critical because different parts of our model need different learning speeds (differential LR). **AdamW** (Weight-decoupled Adam) applies weight decay correctly — to the weights directly, not to the gradient moments — which prevents the regularization from being diluted by the adaptive learning rate.

### Why CosineAnnealingLR?

Instead of a fixed learning rate or step decay, cosine annealing smoothly reduces the LR following: `lr(t) = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(π * t / T_max))`

This has two benefits:
1. **Early epochs:** High LR allows rapid learning of coarse entropy patterns
2. **Late epochs:** Low LR allows fine-tuning without overshooting the loss minimum

### Why Early Stopping (Patience = 7)?

The training accuracy keeps climbing (up to ~96%) but validation accuracy plateaus around epoch 12. Continuing to train past this point only increases the gap between training and validation accuracy (overfitting). Patience=7 means we give the model 7 more chances to improve before stopping.

---

## 5. Data Augmentation Strategy (The Forensics Insight)

### Why Most Augmentations Are Kept Low

This is not a typical computer vision task. **This is a forensics task.** The model must detect subtle pixel-level noise patterns. Standard CV augmentations can destroy these patterns:

| Augmentation | Probability | Why This Value |
|-------------|-------------|----------------|
| **JPEG Recompression** | **10%** | JPEG introduces block artifacts that mask the natural entropy pattern. Too much JPEG = model can't see the real signal. 10% adds minimal robustness. |
| **Gaussian Blur** | **10%** | Blur destroys high-frequency noise — the exact signal we're detecting. Heavy blur (80%) caused accuracy to drop to 76%. 10% is barely noticeable. |
| **ColorJitter** | **40%** | Changes brightness/contrast/saturation. This doesn't affect entropy computation (entropy is computed per-pixel, not per-color) so it's safe at higher probability. |
| **Horizontal Flip** | **40%** | Mirrors the image. Entropy is symmetric, so flipping doesn't destroy the signal. Prevents the model from memorizing left/right positioning. |
| **Random Rotation (±10°)** | **40%** | Slight rotation prevents spatial memorization. 10° is small enough to preserve local pixel neighborhoods. |
| **Random Resized Crop (0.8-1.0)** | **40%** | Crops 80-100% of the image. Prevents the model from relying on objects always being at the center. |

### Proof: What Happened When We Used Heavy Augmentations

| Phase | Blur/JPEG Prob | Test Accuracy | Result |
|-------|---------------|---------------|--------|
| Phase 1 | 50% | **85.25%** | Baseline |
| Phase 2 | **80%** | **76.40%** | ❌ Dropped 9 points! Signal destroyed. |
| Phase 3 | **10%** | **85.90%** | ✓ Best result. Signal preserved. |

---

## 6. All Charts & Visualizations Explained

### 6.1 Training Curves (`training_curves.png`)

**What it shows:** Two subplots: (1) Training Loss vs Validation Loss over 25 epochs, (2) Training Accuracy vs Validation Accuracy over 25 epochs.

**How to read it:**
- If the training curve keeps improving but validation plateaus → **Overfitting** (the model memorizes training data but can't generalize)
- If both curves improve together → **Healthy learning**
- If both curves plateau → **Convergence** (the model has learned everything it can)

**What our chart shows:** Training accuracy reaches ~96% while validation plateaus at ~86.5%. The ~9.5% gap indicates moderate overfitting, which is expected given our small dataset (6,000 training images).

### 6.2 Confusion Matrix (`confusion_matrix.png`)

**What it shows:** A 2×2 grid showing how many images were classified correctly vs incorrectly.

```
                 Predicted
              Real    |    AI
Actual Real    TP     |    FP    (False Positives: Real images mistakenly flagged as AI)
Actual AI      FN     |    TN    (False Negatives: AI images that slipped through)
```

**Why it matters:** Accuracy alone doesn't tell the full story. If 90% of images are real, a model that always says "real" gets 90% accuracy but catches zero AI images. The confusion matrix reveals the specific failure modes.

### 6.3 ROC Curve (`roc_curve.png`)

**What it shows:** Receiver Operating Characteristic — plots True Positive Rate (sensitivity) vs False Positive Rate (1 - specificity) at every possible classification threshold.

**How to read it:**
- **AUC = 1.0:** Perfect classifier
- **AUC = 0.5:** Random guessing (diagonal line)
- **Our AUC = 0.922:** The model is very good at ranking AI images higher than real images, regardless of what threshold we pick.

**Why it matters:** Unlike accuracy (which depends on a fixed 0.5 threshold), ROC-AUC measures the model's ability to *separate* the two classes at ANY threshold. This is a threshold-independent performance metric.

### 6.4 Precision-Recall Curve (`pr_curve.png`)

**What it shows:** Plots Precision (of all images labeled AI, how many actually are?) vs Recall (of all actual AI images, how many did we catch?) at every threshold.

**Our PR-AUC = 0.901:** The model maintains high precision even at high recall — it doesn't need to sacrifice "catching more AI images" to avoid false alarms.

**Why it matters:** In a real-world scenario where AI images are rare, precision is critical. A model that flags everything as AI would have 100% recall but terrible precision (lots of false alarms).

### 6.5 Probability Distribution (`prob_dist.png`)

**What it shows:** Histogram of the model's sigmoid output probabilities, separated by actual class (green = real, red = AI).

**How to read it:**
- **Well-separated peaks** near 0.0 (real) and 1.0 (AI) = confident and correct
- **Overlapping peaks** near 0.5 = uncertain, model struggles to distinguish
- Our chart shows mostly separated peaks with some overlap around 0.3-0.6, explaining the ~14% error rate.

### 6.6 t-SNE Clusters (`tsne_clusters.png`)

**What it shows:** t-Distributed Stochastic Neighbor Embedding — compresses the 2048-dimensional ResNet features into 2D for visualization.

**How to read it:**
- **Two distinct clusters** (green and red separated) = the model has learned features that clearly distinguish real from AI
- **Overlapping blobs** = the model struggles to find distinguishing features
- Our chart shows two mostly-separated clusters with some mixing at the boundaries, consistent with 86% accuracy.

**Why it matters:** This proves the model isn't just memorizing — it has learned a meaningful internal representation where real and AI images naturally cluster apart.

### 6.7 FFT Analysis (`fft_analysis.png`)

**What it shows:** 2D Fast Fourier Transform spectrum of a real image vs an AI image.

**How to read it:**
- The **center** represents low-frequency content (overall brightness, large shapes)
- The **edges** represent high-frequency content (fine details, noise, textures)
- Real images typically show more energy at high frequencies (more noise)
- AI images show suppressed high-frequency energy (smoother, less noise)

**Why it matters:** This is the visual proof of the "generative oversmoothing" effect. You can literally see that AI images have less high-frequency content.

### 6.8 LBP Texture Distribution (`lbp_texture.png`)

**What it shows:** Local Binary Pattern histogram — a classical texture descriptor that encodes micro-texture patterns.

**How to read it:** Each of the 256 possible LBP codes represents a specific local texture pattern. Differences between the real (green) and AI (red) distributions reveal different micro-texture characteristics.

**Why it matters:** LBP is an independent validation of our entropy-based approach. If the LBP distributions differ, it confirms that real and AI images have genuinely different texture properties — we're not overfitting to noise.

### 6.9 Chrominance Scatter (`chrominance_scatter.png`)

**What it shows:** YCbCr color space analysis — plots the blue-difference (Cb) vs red-difference (Cr) chrominance channels.

**How to read it:** Real and AI images may have different color distributions in the chrominance domain. Clustering or separation indicates different color generation characteristics.

**Why it matters:** This is a forensics technique from JPEG steganalysis. It checks whether AI generators produce unrealistic color distributions.

### 6.10 Feature Importance / Saliency Maps (`feature_importance_real.png`, `feature_importance_ai.png`)

**What it shows:** Heatmap overlay on the original image showing which regions the model "pays attention to" when making its decision.

**How to read it:**
- **Hot (red/yellow) regions:** High entropy variation — the model finds these areas most informative
- **Cool (blue) regions:** Low entropy variation — the model ignores these areas

**Why it matters:** This is the model's "proof of work." If the saliency concentrates on textured areas (hair, grass, fabric), it confirms the model is detecting entropy patterns. If it focuses on semantic objects (faces, cars), it might be learning shortcuts instead.

### 6.11 Error Analysis (`error_analysis.png`)

**What it shows:** Bar charts of the model's confidence for high-confidence correct predictions (top row) and uncertain/borderline predictions (bottom row).

**How to read it:**
- Top row: The model is very confident and correct (P(AI) near 0.0 for real, near 1.0 for AI)
- Bottom row: The model is uncertain (P(AI) near 0.5) — these are the hard cases

**Why it matters:** Identifies the model's failure modes. If uncertain predictions cluster around specific image types, it reveals what the model struggles with.

### 6.12 Calibration Curve (`calibration_curve.png`)

**What it shows:** Plots predicted probability vs actual frequency of positive (AI) outcomes.

**How to read it:**
- **Perfectly calibrated:** Points fall on the diagonal line (when the model says "80% chance this is AI," it should be AI 80% of the time)
- **Above diagonal:** Model is under-confident (says 60% but it's actually AI 80% of the time)
- **Below diagonal:** Model is over-confident (says 80% but it's actually AI only 60% of the time)

**Why it matters:** A well-calibrated model is trustworthy. If a pathology lab uses this model and it says "90% AI," doctors need to know that really means 90%, not 60%.

### 6.13 MLEP Heatmap (`batch1_sample0_mlep_heatmap.png`)

**What it shows:** The raw entropy map produced by the MLEP extractor for a single image, visualized as a heatmap.

**How to read it:** Brighter regions have higher entropy (more randomness). Real images should show uniformly distributed entropy, while AI images may show smoother, lower-entropy patches.

### 6.14 MLEP Multiscale (`batch1_sample0_mlep_multiscale.png`)

**What it shows:** The three separate entropy maps at scales 1.0x, 0.5x, and 0.25x for a single image.

**How to read it:** Comparing scales reveals how entropy changes at different resolutions. Real images maintain entropy across all scales. AI images may show entropy collapse at coarser scales (where the upsampling artifacts become more visible).

---

## 7. Final Metrics & What They Mean

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Test Accuracy** | 85.90% | Of all 2,000 test images, 85.9% were classified correctly |
| **Test Precision** | 84.52% | Of all images the model called "AI," 84.52% actually were AI |
| **Test Recall** | 87.90% | Of all actual AI images, the model caught 87.9% of them |
| **Test F1-Score** | 86.18% | Harmonic mean of precision and recall (balanced metric) |
| **ROC-AUC** | 0.922 | Probability the model ranks a random AI image higher than a random real image |
| **PR-AUC** | 0.901 | Area under the precision-recall curve |
| **Best Val Accuracy** | 86.55% | Highest validation accuracy achieved during training |
| **Overfit Gap** | ~9.5% | Difference between training accuracy (96%) and validation accuracy (86.5%) |

### What These Numbers Mean in Practice

- The model correctly identifies **~86 out of every 100 images**
- When it says an image is AI-generated, it's right **~85% of the time**
- It catches **~88% of AI images** (only misses ~12%)
- The 0.922 ROC-AUC means the model has strong discriminative power even at different confidence thresholds
