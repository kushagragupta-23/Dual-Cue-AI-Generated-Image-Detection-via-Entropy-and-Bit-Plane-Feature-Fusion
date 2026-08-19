# LOw-biT pAtch (LOTA) Preprocessing Pipeline & Dual-Cue Feature Fusion Architecture

## 1. Executive Summary & Architectural Role

The **LOw-biT pAtch (LOTA)** preprocessing pipeline is a specialized computer vision feature extraction architecture engineered for detecting AI-generated images (e.g., ProGAN, StyleGAN, Stable Diffusion, Midjourney, FLUX). 

While modern generative models excel at matching macro-level semantic distributions and RGB color fidelity, their upsampling operators (such as transposed convolutions and bilinear interpolation) inevitably leave structural quantization artifacts and subtle spatial correlations in the **least significant bit-planes (LSBs)**. The LOTA pipeline isolates these high-frequency quantization fingerprints from natural semantic image content, quantifying local texture entropy through **Multi-Grid Patch Scoring (MGPS)** to select the most informative spatial regions for downstream classification.

To solve the high variance of LSB noise on JPEG compressed images, we combine LOTA LSB noise extraction with global spatial semantics into a **Dual-Cue Dual-Stream Feature Fusion Architecture (`DualCueClassifier`)**. This dual-stream approach achieved a benchmark peak of **89.15% Validation Accuracy** and **0.9577 Validation ROC-AUC** on 10,000-image dataset splits using NVIDIA GPU hardware acceleration.

---

## 2. Mathematical Formulation

### 2.1 Bit-Plane Extraction & LSB Composition
Let an input RGB image be represented as an unsigned integer tensor $\mathbf{X} \in \{0, 1, \dots, 255\}^{B \times C \times H \times W}$. The binary representation of each pixel intensity at bit-plane $k \in \{0, \dots, 7\}$ is extracted via bitwise operations:
$$x_k^{b,c,i,j} = \left( \lfloor \mathbf{X}_{b,c,i,j} \cdot 2^{-k} \rfloor \right) \bmod 2, \quad x_k \in \{0, 1\}$$
where $k=0$ denotes the least significant bit (LSB) and $k=7$ denotes the most significant bit (MSB).

To capture subtle generative artifacts while stripping away dominant scene semantics, we compose a weighted integer representation $\mathbf{z}$ across the three lowest bit-planes ($k \in \{0, 1, 2\}$):
$$\mathbf{z}_{b,c,i,j} = 4 \cdot x_2^{b,c,i,j} + 2 \cdot x_1^{b,c,i,j} + x_0^{b,c,i,j}, \quad \mathbf{z} \in \{0, 1, \dots, 7\}$$

### 2.2 Binarized Threshold Normalization
To amplify non-zero LSB noise activations into standardized continuous feature representations suitable for neural network convolution, we apply a binarized threshold normalization mapping:
$$\tilde{\mathbf{z}}_{b,c,i,j} = \begin{cases} 255.0, & \text{if } \mathbf{z}_{b,c,i,j} > 0 \\ 0.0, & \text{if } \mathbf{z}_{b,c,i,j} = 0 \end{cases}$$

### 2.3 Multi-Grid Patch Scoring (MGPS)
To identify regions with the highest concentration of generative texture anomalies, we convolve $\tilde{\mathbf{z}}$ against four fixed $2 \times 2$ directional gradient kernels:
$$\mathbf{g}_x = \begin{bmatrix} -1 & 1 \\ 0 & 0 \end{bmatrix}, \quad \mathbf{g}_y = \begin{bmatrix} -1 & 0 \\ 1 & 0 \end{bmatrix}, \quad \mathbf{g}_{xy} = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{g}_{yx} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

For an image tiled into an $8 \times 8$ grid of $32 \times 32$ spatial patches $\mathcal{P}_{r,c}$ (where $r, c \in \{0, \dots, 7\}$), the local gradient divergence score $S(r,c)$ for patch $(r,c)$ is computed as the aggregated $L_1$ norm across all channels and directions:
$$S(r,c) = \sum_{c=1}^3 \sum_{d \in \{x, y, xy, yx\}} \sum_{(i,j) \in \mathcal{P}_{r,c}} \left| \left( \tilde{\mathbf{z}}_c * \mathbf{g}_d \right)_{i,j} \right|$$

### 2.4 Single Maximum Gradient Patch Selection
To capture the most concentrated region of generative artifacts while matching the paper's specification, we select the single patch $p^*$ that maximizes the gradient divergence score across the entire grid:
$$p^* = \arg\max_{r,c} S(r,c)$$
This guarantees the selection of exactly one $32 \times 32$ patch containing the densest distribution of structural quantization anomalies.

### 2.5 Nearest-Neighbor Upscaling
To feed the extracted $32 \times 32$ LSB noise patch into standard ImageNet-pretrained architectures (e.g., ResNet-50) without destroying the delicate binary noise structures, the patch is upscaled back to $256 \times 256$ using nearest-neighbor interpolation:
$$\mathbf{P}_{\text{upscaled}} = \text{NearestInterpolate}(\mathbf{P}_{p^*}, \text{size}=(256, 256))$$
This rigidly duplicates the discrete noise pixels, avoiding any smoothing or floating-point blending introduced by bilinear/bicubic methods.

### 2.6 Dual-Cue Feature Fusion
The upscaled LOTA noise patch $\mathbf{P}_{\text{upscaled}}$ is processed in parallel with the normalized RGB image $\mathbf{X}_{\text{norm}}$:
$$\mathbf{f}_{\text{spatial}} = \text{ResNet50}_{\text{RGB}}(\mathbf{X}_{\text{norm}}) \in \mathbb{R}^{2048}$$
$$\mathbf{f}_{\text{noise}} = \text{ResNet50}_{\text{LOTA}}(\mathbf{P}_{\text{upscaled}}) \in \mathbb{R}^{2048}$$
$$\mathbf{f}_{\text{fused}} = \text{Concat}(\mathbf{f}_{\text{spatial}}, \mathbf{f}_{\text{noise}}) \in \mathbb{R}^{4096}$$
$$\hat{y} = \text{FusionHead}(\mathbf{f}_{\text{fused}})$$

---

## 3. Package Structure & API Reference

```
src/
├── utils/
│   ├── config.py         # Dataclasses (ProjectConfig, LOTAConfig, DatasetConfig)
│   ├── logger.py         # Standardized console & file logging
│   └── visualization.py  # Diagnostic rendering (bit planes, MGPS heatmaps)
├── data/
│   ├── transforms.py     # LOTAPreprocessingTransform
│   ├── dataset.py        # SharedImageDataset loader
│   └── samplers.py       # BalancedRealFakeSampler (50/50 class balance)
└── models/
    ├── dual_cue.py       # DualCueClassifier (Dual ResNet-50 feature fusion)
    ├── lota.py           # TopKLOTAExtractor & vectorized MGPS core
    └── classifier.py    # Single-stream LOTAClassifier
```

### 3.1 `DualCueClassifier`
```python
from src.models.dual_cue import DualCueClassifier
import torch

model = DualCueClassifier(k_patches=1, patch_size=32, grid_size=8, dropout_rate=0.5)
x = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)

logits = model(x)  # Output shape: (2, 1) binary classification logits
```

### 3.2 `TopKLOTAExtractor`
```python
from src.models.lota import TopKLOTAExtractor
import torch

extractor = TopKLOTAExtractor(k_patches=1, patch_size=32, grid_size=8)
x = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)

output = extractor(x)
# output['z_norm']: (2, 3, 256, 256) thresholded LSB map
# output['mgps_scores']: (2, 64) spatial divergence grid
# output['top1_index']: (2, 1) selected patch index
# output['noise_tensor']: (2, 3, 256, 256) upscaled feature representation for ResNet-50
```
