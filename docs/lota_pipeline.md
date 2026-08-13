# LOTA (LOw-biT pAtch) Pipeline Specification

## 1. Overview

**LOw-biT pAtch (LOTA)** is a high-speed AI-Generated Image Detection (AIGID) framework presented at **ICCV 2025** by Hongsong Wang, Renxi Cheng, et al.

LOTA operates on the insight that generative AI models leave distinct, non-perceptual noise patterns concentrated in the **lower bit-planes (LSB)** of generated images. Unlike computationally heavy reconstruction-error detectors (e.g., DIRE, LaRE²), LOTA extracts low-bit-plane noise maps at zero computational cost and uses Maximum Gradient Patch Selection (MGPS) to isolate regions with the densest forgery signatures.

---

## 2. Mathematical Formulation

### 2.1 Bit-Plane Decomposition & LSB Composition

An 8-bit image channel `I(x,y)` in `[0, 255]` can be decomposed into 8 binary bit-planes `b_k(x,y)` in `{0, 1}` (`k = 0, 1, ..., 7`):

```
I(x,y) = Sum_{k=0..7} b_k(x,y) * 2^k
```

LOTA focuses on the lower bit-planes `k in {0, 1, 2}`. The LSB composite noise map `Z(x,y)` combines the 3 lowest bit-planes:

```
Z(x,y) = 4 * b_2(x,y) + 2 * b_1(x,y) + b_0(x,y) -> Value Range: [0, 7]
```

### 2.2 Threshold Binarization Normalization

To convert the continuous LSB composite map into a sharp binary error mask `B(x,y)`:

```
B(x,y) = 255 if Z(x,y) > 0 else 0
```

### 2.3 Maximum Gradient Patch Selection (MGPS)

To identify high-density anomaly regions, 4-directional gradient operators (`G_x`, `G_y`, `G_xy`, `G_yx`) measure local divergence:

```
D(x,y) = |G_x * B| + |G_y * B| + |G_xy * B| + |G_yx * B|
```

The image is partitioned into an `8 x 8` grid of patches. Each patch `P_i` (`i = 1..64`) receives an anomaly score `S(P_i)`:

```
S(P_i) = Sum_{(x,y) in P_i} D(x,y)
```

Top-`K` patches (`K = 4`) with the highest anomaly divergence scores are extracted for deep classification.

---

## 3. Differentiable Approximation in HydraFusion

To enable end-to-end backpropagation without breaking the gradient graph (which uint8 casting and hard top-`K` indexing cause), HydraFusion implements a **differentiable soft LOTA extractor** ([`src/models/lota_extractor.py`](file:///d:/MAIN%20PROJECT%20CV%20AND%20DL/HydraFusion/src/models/lota_extractor.py), aliased via [`src/models/lota.py`](file:///d:/MAIN%20PROJECT%20CV%20AND%20DL/HydraFusion/src/models/lota.py)):

1. **Soft Bit-Plane Extraction:** A learned `1 x 1` convolution `W_lsb` with small normal initialization (`sigma = 0.01`) isolates low-amplitude noise patterns.
2. **Sigmoid Soft Thresholding:** Replaces discrete step functions with smooth sigmoid gates:
   ```
   B_hat(x,y) = Sigmoid( (W_lsb * X - theta) / tau_t )
   ```
3. **Soft Attention Selection:** A scoring network predicts spatial patch weights `A = Softmax(ScoreNet(X))`, multiplying feature maps continuously instead of hard cropping.

---

## 4. Operational Pipeline Flow

```
Input RGB Image (256x256)
       │
       ▼
Soft 1x1 Convolutions (LSB Noise Pattern Extraction)
       │
       ▼
Bit-Plane Composition (k=0, 1, 2) ──► 3 LSB Channels
       │
       ▼
Differentiable Sigmoid Soft Thresholding
       │
       ▼
4-Directional Spatial Gradient Divergence (MGPS)
       │
       ▼
Spatial Soft Attention Weighting & ResNet-50 LOTA Noise Stem
```
