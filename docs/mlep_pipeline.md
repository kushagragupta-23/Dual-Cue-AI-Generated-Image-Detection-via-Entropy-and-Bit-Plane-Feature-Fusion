# MLEP (Multi-granularity Local Entropy Patterns) Pipeline Specification

## 1. Overview

**Multi-granularity Local Entropy Patterns (MLEP)** is a feature extraction framework for AI-Generated Image Detection (AIGID) introduced at **NeurIPS 2025** by Lin Yuan et al.

The core principle of MLEP is that generative AI models introduce subtle statistical and textural anomalies (such as unnatural smoothness or micro-pattern inconsistency) that are preserved even when semantic contents are shuffled. By calculating Shannon entropy over multi-scale sliding windows on patch-shuffled images, MLEP destroys macro-level semantic bias (objects, faces, scenes) while isolating low-level statistical forgery artifacts.

---

## 2. Mathematical & Algorithmic Formulation

### 2.1 Multi-Scale Pyramid Resampling

Given an input image `X` with shape `(3, H, W)` and pixel range `[0, 255]`, MLEP constructs a 3-level spatial pyramid:

```
X_s = Bicubic(X, scale = s),  where s in {1.0, 0.5, 0.25}
```

Downsampled scales (`s = 0.5, 0.25`) are upsampled back to original resolution `(H, W)` to maintain spatial alignment. The multi-scale representation concatenates all scales along channels:

```
X_pyr = Concatenate([X_1.0, X_0.5, X_0.25]) -> Shape: (9, H, W)
```

### 2.2 Local Windowed Shuffling

To destroy semantic context, `X_pyr` is partitioned into a grid of macro-patches of size `M x M` (default `M = 16`):

```
X_pyr -> Partition into grid {P_i,j} for i,j = 1..H/M, 1..W/M
```

Spatial positions of macro-patches are permuted within local neighborhoods, disrupting object boundaries while preserving local texture distributions.

### 2.3 Vectorized Differentiable Local Entropy Computation

Over `2 x 2` sliding windows (`window_size = 2`, `stride = 1`), local patch variance and range statistics are computed to form a continuous, differentiable entropy proxy `E`:

```
Variance(x, y) = (1 / 4) * Sum_{i=0..1, j=0..1} (x_i,j - Mean_x,y)^2

E(x, y) = Sigmoid( Variance(x, y) / tau_e ) * log2( 1 + Variance(x, y) / tau_e )
```

where `tau_e` is a temperature scaling factor.

---

## 3. Implementation in HydraFusion

In HydraFusion-Net, MLEP is implemented in [`src/models/mlep_extractor.py`](file:///d:/MAIN%20PROJECT%20CV%20AND%20DL/HydraFusion/src/models/mlep_extractor.py):

- **Input:** Tensor `(B, 3, 256, 256)` in `[0, 255]`
- **Output:** Tensor `(B, 9, 256, 256)` representing 3 scales x 3 RGB channels
- **Normalization:** InstanceNorm2d with learnable scale and bias to stabilize downstream ResNet-50 feature extraction

---

## 4. Operational Pipeline Flow

```
Input RGB Image (256x256)
       │
       ▼
Learnable Frequency PreFilter (rFFT2 Butterworth Mask)
       │
       ▼
Multi-Scale Resampling Pyramid {1.0, 0.5, 0.25} ──► 9 Channels
       │
       ▼
Local Window Shuffling (M=16 Grid Permutation)
       │
       ▼
2x2 Differentiable Local Entropy Slicing
       │
       ▼
Instance Normalization & ResNet-50 MLEP Spatial Stem
```
