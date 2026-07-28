# LOw-biT pAtch (LOTA) Preprocessing Pipeline & Dataset Ingestion Engine

## 1. Executive Summary & Architectural Role

The **LOw-biT pAtch (LOTA)** preprocessing pipeline is a specialized computer vision feature extraction architecture engineered for detecting AI-generated images (e.g., ProGAN, StyleGAN, Stable Diffusion, Midjourney, FLUX). 

While modern generative models excel at matching macro-level semantic distributions and RGB color fidelity, their upsampling operators (such as transposed convolutions and bilinear interpolation) inevitably leave structural quantization artifacts and subtle spatial correlations in the **least significant bit-planes (LSBs)**. The LOTA pipeline isolates these high-frequency quantization fingerprints from natural semantic image content, quantifying local texture entropy through **Multi-Grid Patch Scoring (MGPS)** to select the most informative spatial regions for downstream classification.

In accordance with project division specifications, this package implements the **complete standalone LOTA preprocessing module, multi-dataset ingestion pipeline, balanced sampling primitives, and diagnostic visualization suite** without implementing any MLEP or cross-modal fusion modules.

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

### 2.4 Top-K Quadrant Diverse Selection
To prevent spatial redundancy and ensure coverage across the entire visual field, we partition the $8 \times 8$ score grid $\mathbf{S} \in \mathbb{R}^{B \times 64}$ into four distinct $4 \times 4$ quadrants:
* **Quadrant 0 (Top-Left)**: $r \in [0, 3], c \in [0, 3]$
* **Quadrant 1 (Top-Right)**: $r \in [0, 3], c \in [4, 7]$
* **Quadrant 2 (Bottom-Left)**: $r \in [4, 7], c \in [0, 3]$
* **Quadrant 3 (Bottom-Right)**: $r \in [4, 7], c \in [4, 7]$

When extracting $K=4$ patches, the algorithm selects the single highest-scoring patch index $p^*_q = \arg\max_{p \in Q_q} S(p)$ from each quadrant $q \in \{0, 1, 2, 3\}$, guaranteeing zero spatial overlap.

---

## 3. Package Structure & API Reference

```
src/
├── utils/
│   ├── config.py         # Typed dataclasses (ProjectConfig, LOTAConfig, DatasetConfig)
│   ├── logger.py         # Standardized console & file logging (get_logger)
│   └── visualization.py  # Diagnostic rendering (plot_bit_planes, plot_mgps_heatmap, plot_topk_patches)
├── data/
│   ├── transforms.py     # LOTAPreprocessingTransform, JPEGRecompression, GaussianBlurDegradation
│   ├── dataset.py        # AIGIDDataset loader & statistics summary generator
│   └── samplers.py       # BalancedRealFakeSampler guaranteeing 50/50 mini-batch ratios
└── models/
    └── lota.py           # TopKLOTAExtractor module & vectorized MGPS core
```

### 3.1 `TopKLOTAExtractor`
```python
from src.models.lota import TopKLOTAExtractor
import torch

extractor = TopKLOTAExtractor(k_patches=4, patch_size=32, grid_size=8)
x = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.float32)

output = extractor(x)
# Returns dictionary containing:
# output['z_norm']: (2, 3, 256, 256) thresholded LSB map
# output['mgps_scores']: (2, 64) spatial divergence grid
# output['topk_indices']: (2, 4) selected quadrant patch indices
# output['topk_patches']: (2, 4, 3, 32, 32) 5D extracted patch tensor
# output['noise_tensor']: (2, 12, 32, 32) stacked feature representation
```

### 3.2 `AIGIDDataset` & `BalancedRealFakeSampler`
```python
from src.data.dataset import AIGIDDataset
from src.data.samplers import BalancedRealFakeSampler
from torch.utils.data import DataLoader

dataset = AIGIDDataset(
    root_dir="data/raw/forensynths",
    split="train",
    val_ratio=0.15,
    test_ratio=0.15,
    metadata_export_path="outputs/metadata.json"
)

sampler = BalancedRealFakeSampler(dataset=dataset, batch_size=32)
loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)

for images, labels, metadata in loader:
    # images: (32, 3, 256, 256) float32 in [0.0, 255.0]
    # labels: (32,) exactly 16 Real (0) and 16 AI (1)
    pass
```

---

## 4. Diagnostic Visualizations

To verify LSB noise extraction on custom images, use the included command-line utility:
```bash
python scripts/visualize_lota.py --image_path sample.png --output_dir outputs/visualizations
```
If no `--image_path` is provided, the script automatically generates a synthetic multi-texture test image and outputs three diagnostic PNG figures:
1. `*_bit_planes.png`: Side-by-side monochrome comparison of original RGB vs. bit-planes $0 \dots 7$.
2. `*_mgps_heatmap.png`: High-contrast colormap overlay showing $8 \times 8$ divergence scores.
3. `*_topk_patches.png`: Bounding box overlays highlighting selected Top-$K$ quadrant patches and zoomed crops.

---

## 5. Hardware Verification & Performance

All operations in `TopKLOTAExtractor` are implemented using native PyTorch vectorized operations (`torch.bitwise_and`, `torch.nn.functional.conv2d`, and `torch.topk`), eliminating Python loops. This ensures 100% execution compatibility and zero-copy device transfers across:
* **NVIDIA RTX GPU**: Accelerated matrix multiplications via PyTorch CUDA (e.g. Lenovo LOQ RTX 4050).
* **NVIDIA CUDA**: Standard Linux multi-GPU server environments.
