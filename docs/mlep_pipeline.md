# MLEP (Multi-granularity Local Entropy Patterns) Pipeline Specification

## 1. Overview

**Multi-granularity Local Entropy Patterns (MLEP)** is a feature extraction framework for AI-Generated Image Detection (AIGID) introduced at **NeurIPS 2025** by Lin Yuan et al.

The core principle of MLEP is that generative AI models introduce subtle statistical and textural anomalies (such as unnatural smoothness or micro-pattern inconsistency) that are preserved even when semantic contents are shuffled. By calculating Shannon entropy over multi-scale sliding windows on patch-shuffled images, MLEP destroys macro-level semantic bias (objects, faces, scenes) while isolating low-level statistical forgery artifacts.

---

## 2. Mathematical Formulation

### 2.1 Multi-Scale Pyramid Resampling

Given an input image $\mathbf{X} \in \mathbb{R}^{3 \times H \times W}$ (range $[0, 255]$), MLEP constructs a 3-level spatial pyramid:

$$\mathbf{X}_{s} = \text{Bicubic}(\mathbf{X}, \text{scale}=s), \quad s \in \{1.0, 0.5, 0.25\}$$

Downsampled scales ($s=0.5, 0.25$) are upsampled back to $(H, W)$ to maintain spatial alignment. The multi-scale representation concatenates all scales:

$$\mathbf{X}_{\text{pyr}} = [\mathbf{X}_{1.0} \,\|\, \mathbf{X}_{0.5} \,\|\, \mathbf{X}_{0.25}] \in \mathbb{R}^{9 \times H \times W}$$

### 2.2 Local Windowed Shuffling

To destroy semantic context, $\mathbf{X}_{\text{pyr}}$ is partitioned into a grid of macro-patches of size $M \times M$ ($M=16$):

$$\mathbf{X}_{\text{pyr}} \to \{\mathbf{P}_{i,j}\}_{i,j=1}^{H/M, W/M}$$

Spatial positions of macro-patches are permuted within local neighborhoods, disrupting object boundaries while preserving local texture distributions.

### 2.3 Vectorized Differentiable Local Entropy Computation

Over $2 \times 2$ sliding windows ($w=2, \text{stride}=1$), local patch variance and range statistics are computed to form a continuous, differentiable entropy proxy $\mathbf{E}$:

$$\sigma^2_{x,y} = \frac{1}{4} \sum_{i=0}^1 \sum_{j=0}^1 (x_{i,j} - \mu_{x,y})^2$$

$$\mathbf{E}(x,y) = \text{Sigmoid}\left(\frac{\sigma^2_{x,y}}{\tau_e}\right) \cdot \log_2\left(1 + \frac{\sigma^2_{x,y}}{\tau_e}\right)$$

where $\tau_e$ is a temperature scaling factor.

---

## 3. Implementation in HydraFusion

In HydraFusion-Net, MLEP is implemented in `src/models/mlep_extractor.py` (and aliased via `src/models/mlep.py`):

- **Input:** Tensor $(B, 3, 256, 256)$ in $[0, 255]$
- **Output:** Tensor $(B, 9, 256, 256)$ representing 3 scales $\times$ 3 RGB channels
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
Multi-Scale Pyramid Resampling {1.0, 0.5, 0.25}
       │
       ▼
Local Windowed Shuffling (16x16 Macro-Grid)
       │
       ▼
2x2 Sliding Window Differentiable Local Entropy Proxy
       │
       ▼
InstanceNorm2d Normalization
       │
       ▼
MLEP Feature Tensor (B, 9, 256, 256) → ResNet-50 Spatial Stem
```
