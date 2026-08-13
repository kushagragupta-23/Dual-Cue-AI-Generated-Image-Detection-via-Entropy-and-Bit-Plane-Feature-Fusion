# LOTA (LOw-biT pAtch) Pipeline Specification

## 1. Overview

**LOw-biT pAtch (LOTA)** is a high-speed AI-Generated Image Detection (AIGID) framework presented at **ICCV 2025** by Hongsong Wang, Renxi Cheng, et al.

LOTA operates on the insight that generative AI models leave distinct, non-perceptual noise patterns concentrated in the **lower bit-planes (LSB)** of generated images. Unlike computationally heavy reconstruction-error detectors (e.g., DIRE, LaRE²), LOTA extracts low-bit-plane noise maps at zero computational cost and uses Maximum Gradient Patch Selection (MGPS) to isolate regions with the densest forgery signatures.

---

## 2. Mathematical Formulation

### 2.1 Bit-Plane Decomposition & LSB Composition

An 8-bit image channel $I(x,y) \in [0, 255]$ can be decomposed into 8 binary bit-planes $b_k(x,y) \in \{0, 1\}$ ($k = 0, 1, \dots, 7$):

$$I(x,y) = \sum_{k=0}^7 b_k(x,y) \cdot 2^k$$

LOTA focuses on the lower bit-planes $k \in \{0, 1, 2\}$. The LSB composite noise map $Z(x,y)$ combines the 3 lowest bit-planes:

$$Z(x,y) = 4 b_2(x,y) + 2 b_1(x,y) + b_0(x,y) \in [0, 7]$$

### 2.2 Threshold Binarization Normalization

To convert the continuous LSB composite map into a sharp binary error mask $B(x,y)$:

$$B(x,y) = \begin{cases} 255 & \text{if } Z(x,y) > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 2.3 Maximum Gradient Patch Selection (MGPS)

To identify high-density anomaly regions, 4-directional gradient operators ($\mathbf{G}_x, \mathbf{G}_y, \mathbf{G}_{xy}, \mathbf{G}_{yx}$) measure local divergence:

$$\mathbf{D}(x,y) = |\mathbf{G}_x * B| + |\mathbf{G}_y * B| + |\mathbf{G}_{xy} * B| + |\mathbf{G}_{yx} * B|$$

The image is partitioned into an $8 \times 8$ grid of patches. Each patch $P_i$ ($i = 1 \dots 64$) receives an anomaly score $S(P_i)$:

$$S(P_i) = \sum_{(x,y) \in P_i} \mathbf{D}(x,y)$$

Top-$K$ patches ($K=4$) with the highest anomaly divergence scores are extracted for deep classification.

---

## 3. Differentiable Approximation in HydraFusion

To enable end-to-end backpropagation without breaking the gradient graph (which uint8 casting and hard top-$K$ indexing cause), HydraFusion implements a **differentiable soft LOTA extractor** (`src/models/lota_extractor.py`, aliased via `src/models/lota.py`):

1. **Soft Bit-Plane Extraction:** A learned $1 \times 1$ convolution $\mathbf{W}_{\text{lsb}}$ with small normal initialization ($\sigma=0.01$) isolates low-amplitude noise patterns.
2. **Sigmoid Soft Thresholding:** Replaces discrete step functions with smooth sigmoid gates:
   $$\hat{B}(x,y) = \text{Sigmoid}\left(\frac{\mathbf{W}_{\text{lsb}} * \mathbf{X} - \theta}{\tau_t}\right)$$
3. **Soft Attention Selection:** A scoring network predicts spatial patch weights $\mathbf{A} = \text{Softmax}(\text{ScoreNet}(\mathbf{X}))$, multiplying feature maps continuously instead of hard cropping.

---

## 4. Operational Pipeline Flow

```
Input RGB Image (256x256)
       │
       ▼
Soft 1x1 Convolutions (LSB Noise Pattern Extraction)
       │
       ▼
Sigmoid Soft Thresholding Gate
       │
       ▼
Gradient Divergence Scoring Network (8x8 Grid)
       │
       ▼
Soft Attention Patch Selection Mask
       │
       ▼
LOTA Feature Tensor (B, 3, 256, 256) → ResNet-50 Spatial Stem
```
