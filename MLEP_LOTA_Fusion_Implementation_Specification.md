# Dual-Cue AI-Generated Image Detection: MLEP & LOTA Fusion
## Technical Architecture, Implementation Specification, and System Design

This document provides an exhaustive technical analysis and implementation specification for reproducing and fusing two state-of-the-art AI-generated image detection (AIGID) frameworks: **MLEP** (*Multi-granularity Local Entropy Patterns*, NeurIPS 2025) and **LOTA** (*LOw-biT pAtch*, ICCV 2025). 

By synthesizing low-level statistical randomness (entropy) with bit-plane quantization noise (steganalysis/compression artifacts), this specification lays out the architectural foundation for building a robust, cross-generator forensic classifier capable of real-time execution on consumer hardware.

---

### Table of Contents
1. [Architecture of MLEP](#1-architecture-of-mlep)
2. [Architecture of LOTA](#2-architecture-of-lota)
3. [Input Pipeline & Preprocessing](#3-input-pipeline--preprocessing)
4. [Mathematical Equations & Formulations](#4-mathematical-equations--formulations)
5. [Feature Extraction Process](#5-feature-extraction-process)
6. [Training Strategy & Generalization Regime](#6-training-strategy--generalization-regime)
7. [Dataset Architecture & Evaluation Benchmarks](#7-dataset-architecture--evaluation-benchmarks)
8. [Hyperparameter Configurations](#8-hyperparameter-configurations)
9. [Architectural & Empirical Weaknesses](#9-architectural--empirical-weaknesses)
10. [Reusable Components](#10-reusable-components)
11. [Components That Must NOT Be Copied](#11-components-that-must-not-be-copied)
12. [Opportunities for Algorithmic & System Improvement](#12-opportunities-for-algorithmic--system-improvement)
13. [Module Synthesis: Reusable vs. Redesigned Components Table](#13-module-synthesis-reusable-vs-redesigned-components-table)
14. [Concrete PyTorch Implementation Architecture & Module Specifications](#14-concrete-pytorch-implementation-architecture--module-specifications)
15. [Step-by-Step Implementation Roadmap & Team Task Division](#15-step-by-step-implementation-roadmap--team-task-division)
16. [Extended Capability Matrix: What All Can Be Implemented](#16-extended-capability-matrix-what-all-can-be-implemented)

---

### 1. Architecture of MLEP
**MLEP** (*Multi-granularity Local Entropy Patterns*, Yuan et al., NeurIPS 2025) is a content-agnostic spatial forensics architecture designed to detect generative image artifacts while actively suppressing semantic content bias. 

```
[Input RGB Image X] ──► [Independent Channel Partitioning (L×L)] ──► [Spatial Patch Shuffling (π)]
                                                                               │
       ┌───────────────────────────────────────────────────────────────────────┘
       ▼
[Multi-Scale Resampling Pyramid: Down(s_k) -> Up(H,W)] ──► [Channel & Scale Concatenation (X^)]
                                                                               │
       ┌───────────────────────────────────────────────────────────────────────┘
       ▼
[2×2 Sliding Window Shannon Entropy (LEP)] ──► [MLEP Feature Map (X¯)] ──► [ResNet-50 Backbone] ──► [Real/Fake Logits]
```

#### Core Design Philosophy
Generative models (GANs and Diffusion pipelines) invariably rely on upsampling and interpolation blocks in their decoders, which introduce subtle local pixel dependencies and structural randomness anomalies. Standard CNN detectors inadvertently overfit to visual semantics (e.g., face structures, fur textures, object background coherence), leading to catastrophic failure when evaluated on unseen generator families. MLEP overcomes this by deliberately destroying macro-semantics through fine-grained patch shuffling while measuring local randomness across multi-scale resampling pyramids via Shannon entropy.

#### Detailed Module Breakdown
1. **Channel-Independent Shuffling Stem**: The input RGB image is processed independently across its $R, G, B$ color channels. Each channel is partitioned into tiny uniform patches of size $L \times L$ (with $L=2$ achieving optimal results). These patches undergo a spatial random permutation (shuffling) defined by a bijection $\pi$. This fine-grained scrambling completely disrupts visible semantic structures while preserving localized pixel value pairings.
2. **Multi-Scale Resampling Pyramid**: To expose generative interpolation artifacts, the scrambled image $\tilde{X}$ is projected across a multi-scale pyramid using scaling factors $\mathbb{S} = \{s_1, s_2, \ldots, s_K\}$ (specifically $\{1, 0.5, 0.25\}$). Each scale is generated via bilinear downsampling followed by bilinear upsampling back to the original spatial resolution $(H, W)$. All scales are concatenated along the channel dimension to form $\hat{X} \in \mathbb{R}^{H \times W \times (C \cdot K)}$.
3. **Local Entropy Pattern (LEP) Extraction**: A $2 \times 2$ sliding window with a stride of $1 \times 1$ traverses $\hat{X}$. For each 4-pixel window, Shannon entropy is calculated over the discrete pixel intensity distribution. Because the stride is smaller than the window size, the computation creates overlap, capturing *intra-patch* randomness within shuffled patches, *inter-patch* randomness across scrambled boundaries, and *inter-scale* entropy across the resampling pyramid. The resulting tensor is the MLEP feature map $\bar{X} \in \mathbb{V}^{(H-1) \times (W-1) \times (C \cdot K)}$.
4. **Classification Backbone**: The sparse, discrete MLEP tensor $\bar{X}$ (containing 9 feature channels for 3 scales $\times$ 3 colors) is ingested directly by a standard CNN classification backbone (ResNet-50), which predicts binary classification probabilities (Real vs. AI-Generated).

---

### 2. Architecture of LOTA
**LOTA** (*LOw-biT pAtch*, Wang et al., ICCV 2025) is an ultra-fast, bit-plane guided detection architecture that exploits imperceptible high-frequency noise and quantization artifacts inherently embedded in the lowest bit-planes of generated images.

```
[Input RGB Image x] ──► [Bit-Plane Slicing (k=0..7)] ──► [Extract 3 LSBs (k=0,1,2)] ──► [Weighted Composition (z^c)]
                                                                                                 │
       ┌─────────────────────────────────────────────────────────────────────────────────────────┘
       ▼
[Thresholding Normalization (z~^c)] ──► [64 Non-Overlapping Patches (32×32)] ──► [MGPS 4-Directional Gradient Scoring]
                                                                                                 │
       ┌─────────────────────────────────── Select Top-1 Patch (z~_p*) ──────────────────────────┘
       ▼
[Branch A: NBC] ──► [Resize to 256×256] ──► [ResNet-50 Backbone] ──► [Real/Fake Prediction]
       ▲
       │ (Or Branch B)
       ▼
[Branch B: NGC] ──► [Noise Patch z~_p* ──► Projection E] ──────┐
                                                               ▼
[Raw Image x] ──► [ResNet-50 Encoder] ──► [Query Q, Key K, Value V] ──► [Noise-Guided Attention] ──► [Real/Fake Prediction]
```

#### Core Design Philosophy
While diffusion reconstruction methods (e.g., DIRE, SeDID, LaRE$^2$) detect deepfakes by computing multi-step DDIM inversion errors, they suffer from extreme computational overhead (taking hundreds of milliseconds to seconds per image) and introduce random sampling noise. LOTA operates on the insight that the least significant bit (LSB) planes of RGB images act as natural, zero-cost error maps. In natural photographs, LSB brightness distributions are regular and smooth; in AI-generated images, low-order bit-planes exhibit chaotic brightness distributions, grid misalignment, and denoising residuals.

#### Detailed Module Breakdown
1. **Bit-Planes Guided Noisy Image Generation (BGNIG)**: Each RGB channel $x^c$ ($c \in \{R, G, B\}$) is sliced into 8 binary bit-planes. The 3 lowest-order bit-planes ($k=0, 1, 2$) are extracted and combined via weighted addition: $z^c = 4x_2^c + 2x_1^c + x_0^c$. Because composed values reside in the narrow integer range $[0, 7]$, the noise image undergoes *Thresholding normalization*, setting any nonzero pixel intensity directly to $255$. This dramatically amplifies sparse noise signals without introducing texture blur.
2. **Maximum Gradient Patch Selection (MGPS)**: The normalized noise image $\tilde{z}$ (at $256 \times 256$ resolution) is divided into an $8 \times 8$ grid of sixty-four non-overlapping $32 \times 32$ patches. To locate the region with the most intense structural irregularity, MGPS evaluates a multi-directional gradient divergence score $g_p$ across horizontal, vertical, diagonal, and anti-diagonal convolution kernels. The single patch exhibiting the maximum gradient score $\tilde{z}_{p^*}$ is selected as the representative forensic fingerprint.
3. **Classification Heads (NBC vs. NGC)**:
   - **Noise-Based Classifier (NBC)**: A lightweight, standalone architecture where the selected $32 \times 32$ noise patch $\tilde{z}_{p^*}$ is bilinearly upsampled to $256 \times 256$ and fed into an ImageNet-pretrained ResNet-50. Operating in 4.00 ms per image with only 23.6M parameters, NBC achieves 98.9% average accuracy on GenImage.
   - **Noise-Guided Classifier (NGC)**: A dual-branch feature alignment model (28.4M parameters, 4.71 ms execution). The raw RGB image $x$ passes through a ResNet-50 encoder to generate spatial query ($Q$), key ($K$), and value ($V$) feature maps. Concurrently, the selected noise patch $\tilde{z}_{p^*}$ is flattened and linearly projected into an error guidance tensor $E$. A custom *Noise-Guided Multi-Head Attention* mechanism injects $E$ directly into the attention affinity matrix, modulating raw visual features with low-level noise localization before classification.

---

### 3. Input Pipeline & Preprocessing

To ensure reproducibility across both baseline models and the proposed dual-cue fusion architecture, the input pipeline is structured into three distinct stages:

```
[Raw Image Ingestion] ──► [Stage 1: Semantic ROI Extraction (Face/Skin Isolation)]
                                      │
         ┌────────────────────────────┴────────────────────────────┐
         ▼                                                         ▼
[Stage 2A: MLEP Branch Pipeline]                          [Stage 2B: LOTA Branch Pipeline]
  ├─ Resize to 256×256 (or 224×224)                         ├─ Resize to 256×256
  ├─ Split R, G, B Channels                                 ├─ Bit-Plane Slicing (k=0..7)
  ├─ 2×2 Patch Partitioning & Shuffling (π)                 ├─ LSB Composition (k=0,1,2)
  ├─ Multi-Scale Pyramid Resampling {1, 0.5, 0.25}          ├─ Thresholding Normalization (0 or 255)
  ├─ 2×2 Sliding Window Shannon Entropy                     ├─ 64 Grid Partition (32×32 patches)
  └─ Output: X¯ ∈ ℝ^(255×255×9)                             └─ Output: MGPS Top-1 Patch z~_p* ∈ ℝ^(32×32×3)
```

#### Stage 1: Optional/Recommended Preprocessing (Semantic ROI Extraction)
When evaluating real-world benchmark images (such as social media recompression or web-scraped diffusion datasets), background clutter and post-processing artifacts can degrade statistical noise metrics. 
- **Face/Subject Isolation**: Run a lightweight, robust object/face bounding box and landmark detection model (e.g., RetinaFace, MediaPipe Face Mesh, or YOLOv8-face).
- **Standardized Cropping**: Crop the isolated region of interest (ROI), applying a 15% margin around facial boundaries to capture high-frequency skin textures, hair edges, and blending boundaries.
- **Color Normalization**: Ensure input tensors are formatted in RGB color space with pixel intensities represented as 8-bit unsigned integers $[0, 255]$.

#### Stage 2A: MLEP Branch Pipeline
1. **Resolution Standardization**: Resize the cropped ROI to $H \times W = 256 \times 256$ (or $224 \times 224$ for strict NeurIPS baseline reproduction) using bicubic interpolation.
2. **Channel Separation**: Disassemble the RGB tensor into three independent matrices: $X^R, X^G, X^B \in \mathbb{R}^{H \times W}$.
3. **Patch Shuffling**: Partition each channel into $2 \times 2$ micro-patches ($128 \times 128$ grid). Apply a seeded pseudo-random spatial permutation $\pi$ to shuffle patch locations across the grid identically or independently per channel.
4. **Resampling Pyramid Construction**: For each scrambled channel $\tilde{X}^c$, generate two downsampled-then-upsampled feature maps at scaling factors $s_2 = 0.5$ and $s_3 = 0.25$ using bilinear interpolation, retaining the unscaled image as $s_1 = 1.0$. Concatenate all 3 scales $\times$ 3 color channels into tensor $\hat{X} \in \mathbb{R}^{H \times W \times 9}$.
5. **Entropy Computation**: Pass a $2 \times 2$ sliding window with stride $1 \times 1$ across all channels of $\hat{X}$. Compute the discrete Shannon entropy value for each window, mapping intensities to $\mathbb{V} \in \{0, 0.8, 1.0, 1.5, 2.0\}$.
6. **Output Tensor**: A multi-granularity entropy feature map $\bar{X} \in \mathbb{R}^{(H-1) \times (W-1) \times 9}$, ready for CNN encoder ingestion.

#### Stage 2B: LOTA Branch Pipeline
1. **Resolution Standardization**: Resize the raw input image to $256 \times 256 \times 3$.
2. **Bit-Plane Slicing & Composition**: Extract integer bit-planes $k \in \{0, 1, 2\}$ for each RGB channel using bitwise right-shift and masking operations. Compose the low-bit tensor $z^c = (x^c \ \& \ 4) + (x^c \ \& \ 2) + (x^c \ \& \ 1)$.
3. **Thresholding Normalization**: Apply binarized thresholding: map all pixel values where $z_{i,j}^c > 0$ to $255$, leaving $0$ values strictly at $0$.
4. **MGPS Patch Selection**: Subdivide the $256 \times 256$ noise image $\tilde{z}$ into sixty-four non-overlapping $32 \times 32$ patches. Convolve each patch against 4 directional gradient kernels ($g_x, g_y, g_{xy}, g_{yx}$) and sum the $L_1$ norms.
5. **Output Tensor**: Extract the single patch $\tilde{z}_{p^*} \in \mathbb{R}^{32 \times 32 \times 3}$ achieving the maximum gradient divergence score. (For standalone NBC, resize this patch to $256 \times 256$ via bilinear interpolation).

---

### 4. Mathematical Equations & Formulations

#### 1. MLEP Patch Partitioning and Permutation
Given an input channel $X \in \mathbb{R}^{H \times W}$, partitioning into uniform patches of size $L \times L$ is defined as:
$$X = \left\{ X_{i,j} \in \mathbb{R}^{L \times L} \right\}_{1 \le i \le \frac{H}{L}, \; 1 \le j \le \frac{W}{L}}$$

Applying spatial shuffling via a bijective permutation mapping $\pi$:
$$\tilde{X} = \left\{ \tilde{X}_{\pi(i,j)} = X_{i,j} \right\}_{1 \le i \le \frac{H}{L}, \; 1 \le j \le \frac{W}{L}}$$

#### 2. MLEP Multi-Scale Resampling Pyramid
For each scaling factor $s_k \in \mathbb{S} = \{s_1, s_2, \ldots, s_K\}$, the downsampling and upsampling transformations are:
$$\tilde{X}^{\vee(k)} = \text{Down}\left(\tilde{X}, s_k\right) \in \mathbb{R}^{\lfloor s_k \cdot H \rfloor \times \lfloor s_k \cdot W \rfloor}$$
$$\tilde{X}^{\wedge(k)} = \text{Up}\left(\tilde{X}^{\vee(k)}, H, W\right) \in \mathbb{R}^{H \times W}$$

The concatenated multi-scale feature tensor across all channels $C$ is:
$$\hat{X} = \text{Concat}\left(\tilde{X}^{\wedge(1)}, \tilde{X}^{\wedge(2)}, \ldots, \tilde{X}^{\wedge(K)}\right) \in \mathbb{R}^{H \times W \times (C \cdot K)}$$

#### 3. MLEP Local Entropy Pattern (LEP)
For a $2 \times 2$ sliding window $\hat{X}_{i,j} = \{x_{m,n}\}_{m \in \{i, i+1\}, n \in \{j, j+1\}}$ traversing $\hat{X}$, the Shannon entropy is formulated as:
$$\text{LEP}\left(\hat{X}_{i,j}\right) = -\sum_{m,n} p(x_{m,n}) \log_2 p(x_{m,n})$$
where $p(x_{m,n})$ is the empirical probability of occurrence of pixel intensity $x_{m,n}$ within the 4-pixel window. Because the sample space size is exactly 4, the discrete probability distribution can only partition into five configurations, yielding a strictly bounded set of discrete entropy values $\mathbb{V}$:
$$\mathbb{V} = \left\{ 0, \; 0.8113, \; 1.0, \; 1.5, \; 2.0 \right\} \approx \left\{ 0, \; 0.8, \; 1.0, \; 1.5, \; 2.0 \right\}$$
- **$0.0$**: All 4 pixels identical $(1.0)$.
- **$0.8113$**: 3 identical pixels, 1 distinct $(0.75, 0.25) \implies -[0.75 \log_2 0.75 + 0.25 \log_2 0.25]$.
- **$1.0$**: 2 pairs of identical pixels, or 2 identical and 2 empty $(0.5, 0.5)$.
- **$1.5$**: 2 identical pixels, 2 distinct pixels $(0.5, 0.25, 0.25)$.
- **$2.0$**: All 4 pixels completely unique $(0.25, 0.25, 0.25, 0.25)$.

#### 4. LOTA Bit-Plane Slicing and Weighted Composition
An 8-bit RGB channel image $x^c$ is decomposed into binary bit-planes $x_k^c \in \{0, 1\}$:
$$x^c = \sum_{k=0}^7 2^k \cdot x_k^c$$

The low-bit noise representation $z^c$ is composed from the 3 least significant bits ($k \in \{0, 1, 2\}$):
$$z^c = 2^2 \cdot x_2^c + 2^1 \cdot x_1^c + x_0^c \in [0, 7]$$

#### 5. LOTA Normalization Strategies
- **Min-Max Scaling**:
  $$\tilde{z}^c = 255 \cdot \frac{z^c - z_{\min}^c}{z_{\max}^c - z_{\min}^c}$$
- **Binarized Thresholding (Default & Optimal)**:
  $$\tilde{z}_{i,j}^c = \begin{cases} 0, & \text{if } z_{i,j}^c = 0 \\ 255, & \text{if } z_{i,j}^c > 0 \end{cases}$$

#### 6. LOTA Maximum Gradient Patch Selection (MGPS) Score
For each candidate $32 \times 32$ noise patch $\tilde{z}_p$ (where $p \in \{1, \ldots, 64\}$), the directional divergence score $g_p$ is:
$$g_p = \|\tilde{z}_p * g_x\|_1 + \|\tilde{z}_p * g_y\|_1 + \|\tilde{z}_p * g_{xy}\|_1 + \|\tilde{z}_p * g_{yx}\|_1$$
where $*$ denotes 2D image convolution, $\|\cdot\|_1$ represents the matrix $L_1$ norm (sum of absolute values), and the directional gradient convolution kernels are:
$$g_x = \begin{bmatrix} -1 & 1 \end{bmatrix}, \quad g_y = \begin{bmatrix} -1 \\ 1 \end{bmatrix}, \quad g_{xy} = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}, \quad g_{yx} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

The optimal noise patch $\tilde{z}_{p^*}$ is selected via argmax:
$$\tilde{z}_{p^*} = \arg\max_{p} g_p$$

#### 7. LOTA Noise-Guided Multi-Head Attention (NGC)
Let $Q, K, V \in \mathbb{R}^{M \times d_k}$ represent the Query, Key, and Value projections derived from the raw image feature map $\tilde{x}$. Let $E \in \mathbb{R}^{M \times M}$ be the flattened and linearly projected error matrix obtained from the selected noise patch $\tilde{z}_{p^*}$. The noise-guided attention affinity formulation is:
$$U = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + E \right) V$$

#### 8. Unified Optimization Objective
Both architectures and the fused classifier are trained end-to-end using standard Binary Cross-Entropy (BCE) loss over $N$ training samples:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log\left(f(X_i)\right) + (1 - y_i) \log\left(1 - f(X_i)\right) \right]$$
where $y_i \in \{0, 1\}$ denotes ground-truth labels ($0 = \text{Real}, 1 = \text{AI-Generated}$) and $f(X_i) \in (0, 1)$ represents model predicted probability.

---

### 5. Feature Extraction Process

The feature extraction pathways of MLEP and LOTA represent two fundamentally complementary philosophies for low-level forensic analysis:

| Feature Dimension | MLEP (Local Entropy Patterns) | LOTA (Bit-Planes Guided Noise) |
| :--- | :--- | :--- |
| **Primary Domain** | Statistical randomness & multi-scale spatial inconsistency. | Quantization errors, LSB steganalysis & denoising step residuals. |
| **What is Suppressed?** | **Visual Semantics & Macro-Texture**: By shuffling $2 \times 2$ patches, object shapes, face geometries, and scene lighting are destroyed. | **Luminance & Color Magnitude**: By discarding bit-planes $k=3\ldots7$, 96.8% of visual energy and semantic contours are stripped away. |
| **What is Amplified?** | **Pixel Value Randomness**: Highlights where generative models produce abnormal local uniformity ($0.0$) or hyper-entropy ($2.0$) compared to natural camera sensor noise. | **High-Frequency Divergence**: Exposes underlying grid misalignments, checkerboard artifacts, and DDIM/GAN decoder upsampling noise. |
| **Dimensionality** | Dense 3D spatial tensor $\bar{X} \in \mathbb{V}^{255 \times 255 \times 9}$ preserving relative scrambled grid coordinates. | Highly compressed localized fingerprint: a single $32 \times 32 \times 3$ patch representing peak image divergence. |
| **Computational Footprint** | Moderate: Requires sliding window entropy operations across 9 pyramid channels. | Extremely Low: Bitwise shifts and linear filtering execute in $<2.0$ ms on GPU. |

---

### 6. Training Strategy & Generalization Regime

To achieve robust open-world deepfake detection that generalizes to generative models never seen during training, both baselines employ strict cross-domain evaluation protocols.

```
[Training Domain: Simple ProGAN + LSUN Real] (Cars, Cats, Chairs, Horses)
                     │
                     ├─────────────────────────────────────────┐
                     ▼                                         ▼
[In-Domain Evaluation: GAN-Set]               [Zero-Shot Generalization: Diffusion-Set]
 (16 GANs: StyleGAN, BigGAN, CycleGAN, etc.)   (16 Diffusion: SD v1/v2, Midjourney, DALL·E 2, etc.)
```

#### 1. MLEP Generalization Regime
- **Training Protocol**: Trained strictly on a narrow subset of 4 object categories (*car, cat, chair, horse*) generated by **ProGAN**, paired with an equal number of real photographs from the **LSUN** dataset (18,000 real + 18,000 fake images total).
- **Zero-Shot Evaluation Protocol**: Evaluated without fine-tuning across 32 distinct generative architectures:
  - **GAN-Set (16 models)**: ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, AttGAN, BEGAN, CramerGAN, InfoMaxGAN, MMDGAN, RelGAN, S3GAN, SNGAN, STGAN.
  - **Diffusion-Set (16 models)**: DDPM, IDDPM, ADM, LDM, PNDM, VQ-Diffusion, Stable Diffusion v1.4, SD v1.5, DALL-E mini, Glide (3 variants), LDM (2 variants), Midjourney, DALL-E 2.
- **Empirical Takeaway**: Despite training solely on 2017-era GAN objects, MLEP achieves an extraordinary 97.8% average accuracy on unseen diffusion models, proving that local entropy patterns are universal, source-invariant fingerprints of artificial image synthesis.

#### 2. LOTA Generalization Regime
- **Training Protocol**: Trained individually on 8 subset splits of the **GenImage** benchmark (e.g., training a model exclusively on Stable Diffusion v1.5 vs. ImageNet real photos).
- **Cross-Generator Evaluation**: Evaluated across all 8 generator testing splits (BigGAN, Midjourney, Wukong, SD V1.4, SD V1.5, ADM, GLIDE, VQDM).
- **Empirical Takeaway**: LOTA achieves an average accuracy of 98.9% across all cross-generator combinations, outperforming multi-step error reconstruction methods (DIRE, LaRE$^2$) by 11.9% to 25.5% while running 100 $\times$ faster.

#### 3. Proposed Dual-Cue Fusion Training Strategy
For a student project executing on consumer hardware (e.g., RTX 4050 6GB VRAM):
1. **Base Dataset**: Train the combined architecture on the lightweight 4-category ProGAN + LSUN subset (or GenImage SD v1.5 subset) to ensure fast epoch convergence ($<2$ hours per run).
2. **Modern Stress-Test**: Evaluate zero-shot generalization against a curated test set of 2025/2026-era generators (*Stable Diffusion 3, FLUX.1, Midjourney v6, and DALL-E 3*) alongside high-compression web datasets.

---

### 7. Dataset Architecture & Evaluation Benchmarks

| Dataset Name | Composition & Source | Real Images | Generated Images | Primary Role in Project |
| :--- | :--- | :--- | :--- | :--- |
| **ForenSynths** (Wang et al.) | 20 object categories from LSUN (Real) and ProGAN (Fake). | ~18,000 (4 training classes) | ~18,000 (4 training classes) | **Primary Training Set**: Small, fast to download, highly standardized baseline. |
| **GenImage** (Zhu et al.) | ImageNet (Real) vs. 8 Generators (BigGAN, SD v1.4/v1.5, Midjourney, ADM, GLIDE, Wukong, VQDM). | 1,331,167 | 1,350,000 | **Large-Scale Evaluation**: Used for cross-generator benchmark tables. |
| **DiffusionForensics / GANGen** | Diverse validation sets containing 16 GANs and 16 Diffusion architectures. | Equal pairing per domain | Equal pairing per domain | **Zero-Shot Stress Testing**: Verifies out-of-distribution generalizability. |
| **Diffusion-2026 Stress Set** *(Project Delta)* | Curated high-resolution generations from FLUX.1, SD3, and Midjourney v6 + JPEG re-compressed variants. | ~5,000 curated real frames | ~5,000 generated fakes | **Zero-Shot & Robustness Benchmark**: Tests out-of-distribution generalizability and frequency denoising resilience. |

---

### 8. Hyperparameter Configurations

An exhaustive side-by-side comparison of baseline hyperparameter settings across both papers:

| Hyperparameter | MLEP Baseline (NeurIPS 2025) | LOTA Baseline (ICCV 2025) | Unified Fusion Recommendation |
| :--- | :--- | :--- | :--- |
| **Input Image Resolution** | $224 \times 224 \times 3$ | $256 \times 256 \times 3$ | **$256 \times 256 \times 3$** (Standardizes tensor spatial dimensions across both branches). |
| **Patch Shuffling Size ($L$)** | $2 \times 2$ pixels ($L=2$) | N/A (No shuffling used) | **$L=2$** (Applied exclusively inside the MLEP feature branch). |
| **Resampling Scaling ($\mathbb{S}$)**| $\{1.0, 0.5, 0.25\}$ | N/A | **$\{1.0, 0.5, 0.25\}$** (Bilinear down/up-sampling in MLEP branch). |
| **Entropy Sliding Window**| $2 \times 2$ window, stride $1 \times 1$ | N/A | **$2 \times 2$ window, stride $1 \times 1$** |
| **Bit-Plane Selection ($k$)** | N/A | Lowest 3 planes ($k \in \{0,1,2\}$)| **$k \in \{0, 1, 2\}$** (Composed via $4x_2 + 2x_1 + x_0$). |
| **Normalization Method** | Discrete mapping $\mathbb{V}$ | Binarized Thresholding ($>0 \to 255$)| **Thresholding** for LOTA branch; **Division by $2.0$** for MLEP branch $[0, 1]$. |
| **MGPS Grid & Patch Size** | N/A | $8 \times 8$ grid of $32 \times 32$ patches | **Top-$K$ ($K=4$ or $8$) patches of size $32 \times 32$** (To preserve spatial diversity). |
| **CNN Backbone Encoder** | ResNet-50 (ImageNet init) | ResNet-50 (ImageNet init) | **Shared or Dual ResNet-18 / ResNet-50** (ResNet-18 recommended for high-throughput batch evaluation). |
| **Optimizer & Learning Rate**| Adam, $\text{lr} = 0.002$ ($2 \times 10^{-3}$) | Adam, $\text{lr} = 0.0001$ ($1 \times 10^{-4}$)| **AdamW, $\text{lr} = 5 \times 10^{-4}$**, cosine annealing decay with warmup. |
| **Batch Size & Epochs** | Batch 64, ~30–50 Epochs | Batch 64, 30 Epochs | **Batch 32 or 64** (Fits comfortably inside RTX 4050 6GB VRAM in fp16/bf16). |

---

### 9. Architectural & Empirical Weaknesses

A rigorous critique of the baseline papers reveals key technical vulnerabilities that justify the need for an advanced fusion architecture:

#### 1. MLEP Weaknesses
- **Catastrophic Vulnerability to Image Degradation**: As shown in MLEP's limitation analysis, when images undergo standard post-processing such as JPEG compression, Gaussian blurring, or social-media resizing, detection accuracy drops precipitously by **17% to 45%**. Because Shannon entropy is computed over micro $2 \times 2$ windows, high-frequency compression artifacts (such as JPEG 8x8 DCT quantization grids) artificially flood natural images with entropy, completely washing out the generative boundary distinction.
- **High Computational Overhead at Scale**: Computing sliding-window entropy across 9 channels (3 scales $\times$ 3 colors) with stride 1 creates dense loops that can become a training bottleneck if not implemented with custom CUDA kernels or highly optimized PyTorch unfold/vectorization operations.
- **Loss of Spatial Explainability**: By performing global spatial patch shuffling ($\pi$) prior to feature extraction, MLEP scrambles all coordinate geometry. Consequently, standard interpretability tools like **Grad-CAM** or saliency maps produce visually meaningless, scrambled noise patterns that cannot show a human reviewer *where* the forgery is located on the subject's face.

#### 2. LOTA Weaknesses
- **Severe Degradation Under Extreme Blurring**: In LOTA ablation experiments (Fig. 6), when Gaussian blur intensity increases to $\sigma = 2$ or $3$, the model degenerates near random guessing (~50% accuracy). Blurring destroys the least significant bit-planes by averaging adjacent pixel intensities, obliterating the LSB noise fingerprint.
- **Myopic Top-1 Patch Selection (MGPS)**: MGPS selects only a single $32 \times 32$ patch out of sixty-four candidates based strictly on gradient sharpness ($L_1$ norm). In an AI-generated portrait where the face and background are smooth but a localized forgery artifact exists in a warped hand or ear, MGPS will frequently discard the forged region and select a high-contrast natural texture (e.g., text on a shirt or tree leaves in the background), leading to false negatives.
- **Total Semantic Blindness in NBC**: The standalone Noise-Based Classifier (NBC) completely discards raw image context, classifying images solely from a bilinearly upsampled $32 \times 32$ noise patch. This prevents the model from understanding contextual anomalies (e.g., smooth skin exhibiting unnatural local quantization).

---

### 10. Reusable Components

The following modular blocks from the official reference implementations represent robust, mathematically sound primitives that should be reused directly in your project pipeline:

1. **MLEP Multi-Scale Resampling Pyramid Engine**:
   - Re-use the exact bilinear downsampling ($\text{Down}$) and upsampling ($\text{Up}$) interpolation functions for scaling factors $\mathbb{S} = \{1.0, 0.5, 0.25\}$.
2. **MLEP Vectorized Shannon Entropy Calculator**:
   - Re-use the $2 \times 2$ windowed entropy lookup algorithm. To maximize training throughput on consumer GPUs, utilize vectorized tensor unfolding (`torch.nn.functional.unfold`) mapped to the 5 discrete entropy levels $\mathbb{V}$.
3. **LOTA Bit-Plane Slicing & LSB Composition Module**:
   - Re-use the exact bitwise right-shift, masking, and weighted addition formulation: $z^c = 4x_2^c + 2x_1^c + x_0^c$. This executes at near-zero latency ($<1.0$ ms).
4. **LOTA Binarized Thresholding Normalization**:
   - Re-use the simple thresholding filter: $\tilde{z}_{i,j}^c = 255 \cdot \mathbb{I}(z_{i,j}^c > 0)$, which empirically outperforms min-max scaling by 0.2% to 2.3% across benchmarks.
5. **LOTA MGPS Directional Convolution Kernels**:
   - Re-use the exact 4 fixed $2 \times 2$ gradient convolution kernels ($g_x, g_y, g_{xy}, g_{yx}$) for measuring local image divergence and noise sparsity.
6. **Standard Backbone Architectures & Training Infrastructure**:
   - Re-use standard ImageNet-pretrained ResNet-18 and ResNet-50 implementations from `torchvision.models`, along with standard GenImage/ForenSynths dataset parsing scripts and evaluation metric calculations (Accuracy, AP, AUC).

---

### 11. Components That Must NOT Be Copied

To ensure your submission represents an original, defensible contribution rather than a simple code concatenation, the following architectural choices from the baseline papers **must be discarded or redesigned**:

1. **Standalone Baseline Classifier Heads (ResNet FC Layers)**:
   - **Do NOT copy** MLEP's standalone ResNet-50 classification head or LOTA's standalone NBC/NGC heads as your final decision makers. Re-running either model independently eliminates your novelty claim. The individual linear classification layers must be stripped away and replaced by a unified multi-modal fusion architecture.
2. **LOTA's Hard Top-1 MGPS Patch Selection**:
   - **Do NOT copy** the argmax $\text{top-}1$ patch selection ($\tilde{z}_{p^*} = \arg\max g_p$) that discards 63 out of 64 noise patches. Discarding 98.4% of the noise image destroys global spatial distribution and prevents cross-modal alignment with spatial entropy maps.
3. **Un-normalized Entropy & Noise Feature Blending**:
   - **Do NOT directly concatenate** raw MLEP entropy maps ($\mathbb{V} \in [0, 2.0]$) with LOTA thresholded noise maps ($[0, 255]$) into a single CNN stem. The 100$\times$ scale disparity will cause severe gradient explosion or dominance of the bit-plane branch during backpropagation. All feature branches must undergo layer normalization or batch normalization prior to fusion.
4. **Global Irreversible Shuffling Before Explainability**:
   - **Do NOT apply** MLEP's global spatial shuffling across the entire image if you intend to execute Project Extension 2 (Grad-CAM explainability overlay). Global shuffling destroys coordinate mapping. Shuffling must either be applied *locally within grid windows* or confined to a dedicated parallel branch while preserving an un-shuffled spatial coordinate branch for saliency heatmap generation.

---

### 12. Opportunities for Algorithmic & System Improvement

This section defines your **Compulsory Delta**—the original algorithmic contributions, architectural redesigns, and applied engineering extensions that elevate this project into an original piece of research.

```
[Raw Image x] ──► [Stage 1: Semantic ROI Extraction & Color Normalization]
                                      │
         ┌────────────────────────────┴────────────────────────────┐
         ▼                                                         ▼
[MLEP Branch: Spatial-Preserving Windowed Shuffling]       [LOTA Branch: Top-K (K=4) MGPS Noise Selection]
  ├─ Multi-Scale Resampling {1, 0.5, 0.25}                  ├─ Bit-Plane Slicing & Thresholding (k=0,1,2)
  ├─ Vectorized 2×2 Shannon Entropy                         ├─ Extract Top-4 Diverse Gradient Patches
  └─ Feature Map F_MLEP ∈ ℝ^(B × C_1 × H' × W')             └─ Feature Map F_LOTA ∈ ℝ^(B × C_2 × H' × W')
         │                                                         │
         └────────────────────────────┬────────────────────────────┘
                                      ▼
             [Novel Contribution: Cross-Modal Gating & Attention Fusion Head]
               ├─ Spatial Feature Alignment & Layer Normalization
               ├─ Cross-Attention / Dynamic Gating Weights (α_MLEP, α_LOTA)
               ├─ Exploit Complementarity: Texture Entropy vs. LSB Quantization
               └─ Fused Representation F_fused ∈ ℝ^(B × d_model)
                                      │
         ┌────────────────────────────┴────────────────────────────┐
         ▼                                                         ▼
[Binary Classifier Head (Real / Fake)]            [Grad-CAM Saliency Explainability Overlay]
                                                    └─ Un-scrambled spatial anomaly heatmap
```

#### 1. Novel Contribution: Cross-Modal Gating & Attention Fusion Head
Neither source paper combines statistical entropy with LSB quantization noise. You will design and train a novel **Cross-Modal Gating Network** that dynamically weights the two evidence branches based on input degradation:
- **Theoretical Justification**: Why are these cues complementary rather than redundant? 
  - When an AI-generated image undergoes **JPEG compression or blurring**, MLEP's $2 \times 2$ entropy differences are smoothed over or corrupted by DCT grid noise. However, LOTA's LSB bit-plane noise still retains quantization error divergence.
  - Conversely, when evaluating **high-resolution diffusion models (SD3 / FLUX)** with pristine pixel transitions, LSB noise may appear natural, but multi-scale resampling entropy (MLEP) exposes underlying decoder upsampling inconsistencies.
- **Architectural Design**: Project both branch feature maps ($F_{\text{MLEP}}$ and $F_{\text{LOTA}}$) into a shared latent dimension $d_{\text{model}}$. Pass them through a lightweight 2-layer Cross-Attention or Squeeze-and-Excitation gating module that predicts dynamic scalar weights $\alpha_{\text{MLEP}}, \alpha_{\text{LOTA}} \in [0, 1]$ where $\alpha_{\text{MLEP}} + \alpha_{\text{LOTA}} = 1$. The fused representation $F_{\text{fused}} = \alpha_{\text{MLEP}} \cdot F_{\text{MLEP}} + \alpha_{\text{LOTA}} \cdot F_{\text{LOTA}}$ feeds into the final classifier.

#### 2. Top-$K$ Diverse MGPS Noise Selection & Spatial Alignment
To overcome LOTA's myopic single-patch selection, modify MGPS to extract the **Top-$K$ ($K=4$ or $8$) spatial noise patches** across different image quadrants, or retain the full $8 \times 8$ grid of gradient divergence scores as an auxiliary spatial attention map. Aligning this $8 \times 8$ noise grid with MLEP's spatial feature maps ensures that local noise anomalies guide spatial entropy evaluation.

#### 3. Spatial-Preserving Windowed Shuffling for Grad-CAM Explainability
To fulfill Project Extension 2 (visualizing *where* anomalies concentrate), replace MLEP's global shuffling with **Local Windowed Shuffling** (e.g., shuffling $2 \times 2$ micro-patches only within bounded $16 \times 16$ macro-windows). This preserves macro-spatial coordinate integrity. When backpropagating gradients from the prediction logit to the last convolutional layer via **Grad-CAM**, you can generate clean, localized heatmaps that highlight exact forgery zones (e.g., blending seams around lips or distorted eyes) across both entropy and bit-plane branches simultaneously.

#### 4. Robustness Training & Frequency Denoising Pre-Filter
To solve MLEP's 45% accuracy collapse under JPEG compression (Project Extension 4), introduce an active **Robustness Data Augmentation Pipeline** during fusion training. Randomly apply online JPEG recompression (quality 70–100%), Gaussian blur ($\sigma \in [0.5, 2.0]$), and social-media resizing to training batches. Optionally, prepend a lightweight, trainable frequency-domain denoising pre-filter (e.g., a 1-layer Wiener filter or learnable DCT mask) that strips high-frequency compression block artifacts before entropy calculation.

#### 5. Computational Acceleration & Mixed-Precision Benchmarking
To optimize high-throughput evaluation across large academic datasets:
- **Backbone Compression**: Replace heavy dual ResNet-50 backbones (which require ~50M parameters combined) with a single shared **ResNet-18** or **MobileNetV3-Large** stem encoder, branching only at the final feature pyramid layers for entropy vs. bit-plane processing.
- **Mixed-Precision Optimization**: Integrate automatic mixed precision (`torch.cuda.amp` fp16 / bf16) to double batch throughput and half GPU memory consumption during large-scale evaluation on GenImage and ForenSynths.
- **High-Throughput Benchmarking**: Build a clean computational pipeline to benchmark inference throughput (Images/Second) and ROC-AUC scaling curves across PyTorch CUDA without requiring physical hardware demos.

---

### 13. Module Synthesis: Reusable vs. Redesigned Components Table

The following matrix provides an executive summary of exactly which modules from the reference papers should be directly reused versus those that must be redesigned to construct your original fusion architecture:

| Architectural Module | Reference Source | Status for Your Project | Detailed Technical Action & Architectural Justification |
| :--- | :--- | :--- | :--- |
| **Multi-Scale Resampling Pyramid** | MLEP (NeurIPS '25) | **REUSE DIRECTLY** | Implement exact bilinear downsampling ($\text{Down}$) and upsampling ($\text{Up}$) for scaling factors $\mathbb{S} = \{1.0, 0.5, 0.25\}$. Crucial for exposing decoder upsampling artifacts. |
| **Vectorized Shannon Entropy Calculator**| MLEP (NeurIPS '25) | **REUSE DIRECTLY** | Re-use $2 \times 2$ windowed entropy lookup algorithm ($\mathbb{V} \in \{0, 0.8, 1.0, 1.5, 2.0\}$). Implement via vectorized PyTorch tensor unfolding to maximize GPU training throughput. |
| **Bit-Plane Slicing & LSB Composition** | LOTA (ICCV '25) | **REUSE DIRECTLY** | Re-use bitwise slicing and weighted composition of the 3 least significant bit-planes ($z^c = 4x_2^c + 2x_1^c + x_0^c$). Provides zero-cost high-frequency noise extraction. |
| **Binarized Thresholding Normalization** | LOTA (ICCV '25) | **REUSE DIRECTLY** | Re-use binarized thresholding filter ($\tilde{z}_{i,j}^c = 255 \cdot \mathbb{I}(z_{i,j}^c > 0)$). Empirically superior to min-max scaling for amplifying sparse quantization noise. |
| **Directional Gradient Convolutions** | LOTA (ICCV '25) | **REUSE DIRECTLY** | Re-use the 4 fixed $2 \times 2$ gradient convolution kernels ($g_x, g_y, g_{xy}, g_{yx}$) for measuring local image divergence and high-frequency noise sparsity. |
| **Standard CNN Backbones & Loaders** | General / PyTorch | **REUSE DIRECTLY** | Re-use standard ImageNet-pretrained ResNet-18/ResNet-50 architectures, GenImage/ForenSynths dataset parsing scripts, and standard AP/AUC evaluation metrics. |
| **Standalone Baseline Classifiers** | MLEP & LOTA | **MUST BE REDESIGNED** | **Strip away completely.** Do not use individual ResNet-50 linear classifier heads. Re-running separate classifiers eliminates your novelty claim; they must feed into a unified fusion head. |
| **Top-1 MGPS Patch Selection** | LOTA (ICCV '25) | **MUST BE REDESIGNED** | **Replace hard Top-1 selection.** Discarding 63 of 64 noise patches destroys global spatial context. Upgrade to Top-$K$ ($K=4$ or $8$) diverse patches or maintain the full $8 \times 8$ noise grid as an attention mask. |
| **Global Spatial Patch Shuffling ($\pi$)**| MLEP (NeurIPS '25) | **MUST BE REDESIGNED** | **Replace with Local Windowed Shuffling.** Global shuffling destroys spatial coordinate geometry, making Grad-CAM explainability impossible. Confine shuffling to bounded $16 \times 16$ macro-windows. |
| **Un-normalized Feature Blending** | N/A (Baseline fusion)| **MUST BE REDESIGNED** | **Implement Layer/Batch Normalization.** Raw entropy $[0, 2.0]$ and thresholded bit-planes $[0, 255]$ differ by two orders of magnitude. Normalize both feature branches before joint fusion to prevent gradient explosion. |
| **Cross-Modal Fusion Head** | **New Contribution** | **ORIGINAL DESIGN** | **Design and build from scratch.** Implement an attention-gated gating network that dynamically learns when to trust texture entropy vs. LSB quantization noise based on input degradation. |
| **Grad-CAM Saliency Overlay Module**| **New Contribution** | **ORIGINAL DESIGN** | **Design and build from scratch.** Connect Grad-CAM hooks to the last convolutional layer of the un-scrambled spatial feature branch to render visual explainability heatmaps showing exact forgery zones. |
| **Robustness Augmentation Pipeline** | **New Contribution** | **ORIGINAL DESIGN** | **Design and build from scratch.** Integrate online JPEG recompression (70–100%) and Gaussian blur into the training loop to overcome MLEP's 45% accuracy collapse under social-media compression. |
| **Mixed-Precision Computational Acceleration** | **New Contribution** | **ORIGINAL DESIGN** | **Design and build from scratch.** Optimize backbone footprint (ResNet-18/MobileNetV3) and integrate PyTorch automatic mixed precision (`fp16`/`bf16`) for high-throughput batch benchmarking on static datasets without physical demos. |

---

### 14. Concrete PyTorch Implementation Architecture & Module Specifications

To move from theoretical specification to execution, this section details the concrete, production-grade PyTorch architecture required to build the **Dual-Cue AIGID Fusion Model**. The codebase is structured around modular, vectorized operations that avoid slow Python loops and maximize GPU utilization on consumer hardware (e.g., Lenovo LOQ RTX 4050 6GB VRAM).

#### 1. Project Directory Hierarchy
```text
mlep_lota_fusion/
├── configs/
│   ├── train_baseline_progan.yaml      # Pre-training configs on ForenSynths (ProGAN)
│   ├── train_fusion_genimage.yaml      # Cross-generator fine-tuning configs
│   └── eval_zeroshot_2026.yaml         # Hyperparameters for 2026 diffusion stress benchmark
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── transforms.py               # ROI cropping, resizing, online JPEG/Blur augmentations
│   │   ├── dataset.py                  # ForenSynths & GenImage PyTorch Dataset loaders
│   │   └── samplers.py                 # Balanced Real/Fake class samplers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlep_branch.py              # Vectorized Multi-Scale Resampling & Shannon Entropy
│   │   ├── lota_branch.py              # Bit-Plane Slicing, Thresholding & Top-K MGPS
│   │   ├── fusion_head.py              # Cross-Modal Gating & Attention Fusion Head
│   │   ├── backbones.py                # Shared/Dual ResNet-18 / MobileNetV3 stems
│   │   └── dual_cue_detector.py        # End-to-end unified architecture wrapper
│   ├── utils/
│   │   ├── metrics.py                  # Accuracy, Average Precision (AP), ROC-AUC calculators
│   │   ├── explainability.py           # Grad-CAM hooks & heatmap overlay generator
│   │   └── benchmark_ops.py            # Latency (ms/img) and FP16 throughput utilities
│   └── eval/
│       ├── benchmark_throughput.py     # GPU/CPU inference speed & FP16 scaling benchmarks
│       └── robustness_suite.py         # Automated JPEG, blur, and resolution degradation evaluator
├── scripts/
│   ├── train.py                        # Main training loop with mixed precision (fp16/bf16)
│   ├── evaluate_zeroshot.py            # Cross-generator evaluation script
│   └── run_benchmark_suite.sh          # Automated launcher for quantitative evaluation tables
└── requirements.txt
```

#### 2. Vectorized MLEP Feature Extractor (`mlep_branch.py`)
To prevent CUDA training bottlenecks, the $2 \times 2$ sliding-window Shannon entropy computation must be executed via tensor unfolding (`torch.nn.functional.unfold`) rather than Python loops:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorizedMLEPExtractor(nn.Module):
    """
    Multi-granularity Local Entropy Patterns (MLEP) Feature Extractor.
    Performs channel-independent local windowed shuffling, multi-scale pyramid resampling,
    and vectorized 2x2 Shannon entropy computation.
    """
    def __init__(self, scales=(1.0, 0.5, 0.25), patch_size=2, window_size=2):
        super().__init__()
        self.scales = scales
        self.patch_size = patch_size
        self.window_size = window_size
        # Discrete entropy lookup mapping for 4-pixel window: {0.0, 0.81, 1.0, 1.5, 2.0}
        
    def _apply_local_shuffling(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W = x.shape
        # Partition into 2x2 micro-patches within bounded 16x16 macro-windows to preserve
        # global spatial coordinate geometry for Grad-CAM explainability.
        # Implementation: Reshape -> Random permutation along patch dim -> Reshape back
        return x # Pseudo-code representation

    def _compute_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pyramid_features = []
        for s in self.scales:
            if s == 1.0:
                scaled = x
            else:
                down = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
                scaled = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
            pyramid_features.append(scaled)
        # Concatenate across channels: outputs B, (C * len(scales)), H, W
        return torch.cat(pyramid_features, dim=1)

    def _vectorized_shannon_entropy(self, x: torch.Tensor) -> torch.Tensor:
        B, C_tot, H, W = x.shape
        # Unfold 2x2 sliding windows with stride 1: shape (B, C_tot * 4, L), where L = (H-1)*(W-1)
        unfolded = F.unfold(x, kernel_size=self.window_size, stride=1)
        unfolded = unfolded.view(B, C_tot, 4, -1)
        
        # Vectorized frequency count across the 4 pixels in each window
        # Map discrete distributions to {0.0, 0.8113, 1.0, 1.5, 2.0} / 2.0 (normalized to [0, 1])
        # Returns entropy tensor of shape (B, C_tot, H-1, W-1)
        entropy_map = torch.zeros((B, C_tot, H - 1, W - 1), device=x.device, dtype=x.dtype)
        return entropy_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Local windowed shuffling
        x_shuffled = self._apply_local_shuffling(x)
        # 2. Multi-scale resampling pyramid (9 channels total for RGB x 3 scales)
        x_pyramid = self._compute_pyramid(x_shuffled)
        # 3. Vectorized entropy computation
        entropy_map = self._vectorized_shannon_entropy(x_pyramid)
        return entropy_map
```

#### 3. Top-K MGPS LOTA Feature Extractor (`lota_branch.py`)
To overcome LOTA's baseline vulnerability of selecting a single background patch, we generalize Maximum Gradient Patch Selection (MGPS) to extract the **Top-$K$ ($K=4$) spatially diverse patches**:

```python
class TopKLOTAExtractor(nn.Module):
    """
    LOw-biT pAtch (LOTA) Feature Extractor with Top-K Diverse MGPS.
    Extracts LSB planes (k=0,1,2), applies thresholding normalization, and selects
    the top-K gradient divergence patches across image quadrants.
    """
    def __init__(self, k_patches=4, patch_size=32, grid_size=8):
        super().__init__()
        self.k_patches = k_patches
        self.patch_size = patch_size
        self.grid_size = grid_size
        
        # Register fixed directional gradient kernels (gx, gy, gxy, gyx) as non-trainable buffers
        gx = torch.tensor([[-1.0, 1.0], [0.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gy = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gxy = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gyx = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernels', torch.cat([gx, gy, gxy, gyx], dim=0)) # 4, 1, 2, 2

    def _extract_lsb_threshold(self, x: torch.Tensor) -> torch.Tensor:
        # Convert fp32 tensor [0, 255] to uint8 bit representation
        x_int = x.to(torch.uint8)
        # Extract k=0, 1, 2 bit-planes via bitwise AND and weighted composition: z = 4*x2 + 2*x1 + x0
        z = ((x_int & 4)) + ((x_int & 2)) + ((x_int & 1))
        # Binarized thresholding normalization: map >0 to 255.0, 0 remains 0.0
        z_norm = torch.where(z > 0, torch.tensor(255.0, device=x.device), torch.tensor(0.0, device=x.device))
        return z_norm

    def _compute_mgps_scores(self, z_norm: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_norm.shape
        # Convolve each color channel independently against 4 directional gradient kernels
        z_flat = z_norm.view(B * C, 1, H, W)
        grads = F.conv2d(z_flat, self.kernels, padding=0) # shape: (B*C, 4, H-1, W-1)
        grad_l1 = grads.abs().sum(dim=1).view(B, C, H - 1, W - 1).sum(dim=1) # Sum over channels -> (B, H-1, W-1)
        
        # Partition into 8x8 grid of 32x32 patches and compute L1 score per patch
        # Returns patch divergence score matrix of shape (B, 64)
        scores = torch.zeros((B, self.grid_size * self.grid_size), device=z_norm.device)
        return scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_norm = self._extract_lsb_threshold(x)
        scores = self._compute_mgps_scores(z_norm)
        
        # Select Top-K diverse patch indices (e.g., K=4 across 4 quadrants to avoid spatial overlap)
        topk_indices = torch.topk(scores, k=self.k_patches, dim=-1).indices # (B, K)
        
        # Extract and stack the Top-K 32x32 noise patches -> Upsample/align to shared spatial resolution
        # Outputs tensor of shape (B, K * 3, H_target, W_target)
        topk_patches = z_norm # Pseudo-code alignment return
        return topk_patches
```

#### 4. Cross-Modal Gating & Attention Fusion Head (`fusion_head.py`)
This module implements our core architectural contribution: dynamically weighting spatial texture entropy against LSB quantization noise:

```python
class CrossModalGatingFusionHead(nn.Module):
    """
    Cross-Modal Attention Gating Network.
    Normalizes divergent feature scales, projects both cues into a shared latent space,
    and applies dynamic attention gating weights based on input degradation.
    """
    def __init__(self, in_channels_mlep=512, in_channels_lota=512, latent_dim=256):
        super().__init__()
        # Layer Normalization to reconcile [0, 1] entropy maps with [0, 255] thresholded bit-planes
        self.norm_mlep = nn.LayerNorm(in_channels_mlep)
        self.norm_lota = nn.LayerNorm(in_channels_lota)
        
        self.proj_mlep = nn.Sequential(nn.Linear(in_channels_mlep, latent_dim), nn.ReLU(), nn.Dropout(0.3))
        self.proj_lota = nn.Sequential(nn.Linear(in_channels_lota, latent_dim), nn.ReLU(), nn.Dropout(0.3))
        
        # Attention gating network predicting scalar weights (alpha_mlep, alpha_lota)
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        
        # Final classification classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Logit output for Real (0) vs Fake (1)
        )

    def forward(self, feat_mlep: torch.Tensor, feat_lota: torch.Tensor) -> torch.Tensor:
        # feat_mlep, feat_lota: Global average pooled feature vectors (B, C)
        feat_mlep = self.norm_mlep(feat_mlep)
        feat_lota = self.norm_lota(feat_lota)
        
        h_mlep = self.proj_mlep(feat_mlep)
        h_lota = self.proj_lota(feat_lota)
        
        # Concatenate and compute attention gating weights
        combined = torch.cat([h_mlep, h_lota], dim=-1)
        weights = self.gating_network(combined) # (B, 2)
        alpha_mlep = weights[:, 0:1]
        alpha_lota = weights[:, 1:2]
        
        # Dynamic cross-modal fusion
        h_fused = alpha_mlep * h_mlep + alpha_lota * h_lota
        logits = self.classifier(h_fused)
        return logits, weights
```

---

### 15. Step-by-Step Implementation Roadmap & Team Task Division

To ensure a seamless, professional execution of this project, the implementation workflow is organized into four chronological phases with a clear **2-Person Task Division Matrix**.

```
[Phase 1: Week 1-2] ────► [Phase 2: Week 3-4] ────► [Phase 3: Week 5] ────► [Phase 4: Week 6]
Baselines & Primitives    Dual-Cue Fusion Head    Robustness & Stress     Computational Benchmarking
```

#### Chronological Execution Timeline

##### Phase 1: Foundation, Data Pipelines & Baseline Primitives (Weeks 1–2)
- **Objective**: Establish standardized data ingestion, implement standalone MLEP and LOTA feature extractors, and reproduce baseline accuracy on ForenSynths (ProGAN).
- **Milestones**:
  1. Build `src/data/dataset.py` with automatic bounding-box ROI cropping and RGB normalization.
  2. Implement and verify vectorized Shannon entropy folding (`VectorizedMLEPExtractor`).
  3. Implement bit-plane slicing, binarized thresholding, and MGPS gradient scoring (`TopKLOTAExtractor`).
  4. Validate that standalone MLEP achieves $>95\%$ validation accuracy on ProGAN cars/cats.

##### Phase 2: Dual-Cue Fusion & Spatial Alignment (Weeks 3–4)
- **Objective**: Synthesize the two pipelines using our novel `CrossModalGatingFusionHead` and integrate Grad-CAM explainability hooks.
- **Milestones**:
  1. Assemble `DualCueAIGIDModel` connecting both feature extractors to shared ResNet-18/50 stems.
  2. Implement Layer Normalization and dynamic attention weighting ($\alpha_{\text{MLEP}}, \alpha_{\text{LOTA}}$).
  3. Train end-to-end on GenImage (SD v1.5 vs. ImageNet real) using mixed precision (`fp16`).
  4. Attach Grad-CAM backward hooks to render un-scrambled spatial saliency heatmaps on test images.

##### Phase 3: Zero-Shot Generalization & Robustness Stress-Testing (Week 5)
- **Objective**: Execute out-of-distribution stress testing across modern 2025/2026 generative models and validate compression resilience.
- **Milestones**:
  1. Evaluate zero-shot inference across 16 GANs and 16 Diffusion architectures without retraining.
  2. Introduce online JPEG compression ($Q=70..100$) and Gaussian blur ($\sigma=0.5..2.0$) into the training loop.
  3. Benchmark precision, recall, accuracy, and ROC-AUC; generate comparative markdown tables and ROC curves.

##### Phase 4: Computational Benchmarking & Mixed-Precision Optimization (Week 6)
- **Objective**: Optimize inference throughput and compile exhaustive quantitative benchmark tables across all 32 generative architectures.
- **Milestones**:
  1. Integrate PyTorch automatic mixed precision (`fp16`/`bf16`) to double batch processing throughput.
  2. Build `benchmark_throughput.py` to evaluate latency (ms/image) and memory consumption across GPU and CPU backends.
  3. Execute automated robustness stress tests under systematic JPEG compression ($Q=70..100$) and Gaussian blur ($\sigma=0.5..2.0$).
  4. Compile final academic tables and ROC curves comparing baseline vs. MGA-Net and MoE specialized architectures.

#### 2-Person Task Division Matrix

To ensure balanced workload distribution without file-collision conflicts, tasks are partitioned by architectural domain:

| Project Phase | Team Member A (Statistical Entropy & Core Infrastructure) | Team Member B (LSB Quantization, Fusion Head & Eval Benchmarks) |
| :--- | :--- | :--- |
| **Phase 1: Baselines** | • Build PyTorch dataset loaders & data augmentation scripts.<br>• Implement `VectorizedMLEPExtractor` and multi-scale pyramid.<br>• Train & verify MLEP baseline model on ForenSynths. | • Build bit-plane slicing and thresholding modules.<br>• Implement 4-directional gradient convolution kernels & Top-$K$ MGPS.<br>• Train & verify LOTA baseline on GenImage SD v1.5 subset. |
| **Phase 2: Fusion** | • Build shared/dual ResNet-18 backbone stem adapters.<br>• Implement Local Windowed Shuffling ($16 \times 16$ macro-grid).<br>• Optimize CUDA memory footprint and batch data loading. | • Design & code `CrossModalGatingFusionHead` (attention gating).<br>• Implement Layer Normalization and cross-modal loss tracking.<br>• Implement Grad-CAM explainability hooks and heatmap overlay generator. |
| **Phase 3: Robustness** | • Build online JPEG recompression and Gaussian blur transforms.<br>• Execute zero-shot benchmark suite on 32 GAN/Diffusion models.<br>• Compile quantitative benchmark tables and accuracy metrics. | • Implement automated robustness stress suite (`robustness_suite.py`).<br>• Build custom Diffusion-2026 stress test evaluation script.<br>• Conduct failure mode analysis under extreme downsampling. |
| **Phase 4: Benchmarking**| • Refactor model weights for clean computational evaluation APIs.<br>• Document repository instructions, README, and hyperparameter tables.<br>• Verify quantitative reproduction tables against reference papers. | • Build mixed-precision benchmark suite (`benchmark_throughput.py`).<br>• Implement Multi-Granularity Cross-Attention (MGA-Net) evaluation.<br>• Conduct high-throughput batch evaluation across CUDA/Metal devices. |

---

### 16. Extended Capability Matrix: What All Can Be Implemented

To provide a complete menu of engineering possibilities, this section categorizes all features that can be implemented into a structured three-tier hierarchy: **Tier 1 (Core Deliverables)**, **Tier 2 (High-Impact Research Extensions)**, and **Tier 3 (Advanced Industrial Polish)**.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: ADVANCED INDUSTRIAL POLISH                                               │
│  • Frequency-Domain Denoising Pre-Filter (Learnable DCT / Wiener Filter)         │
│  • Supervised Contrastive Pre-Training (SupCon Cross-Modal Alignment)            │
│  • Automated Adversarial Attack Stress-Testing (FGSM / PGD / CW Perturbations)   │
├──────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: HIGH-IMPACT RESEARCH EXTENSIONS                                          │
│  • Top-K (K=4) Diverse MGPS Spatial Noise Selection                              │
│  • Local Windowed Shuffling + Grad-CAM Visual Saliency Overlay                   │
│  • Online Robustness Augmentation Pipeline (JPEG Q=70..100, Blur σ=0.5..2.0)     │
│  • High-Throughput Mixed-Precision Benchmarking & Latency Scaling Analysis       │
├──────────────────────────────────────────────────────────────────────────────────┤
│ TIER 1: CORE ARCHITECTURAL DELIVERABLES (COMPULSORY BASELINE)                    │
│  • Vectorized Multi-Scale Resampling Pyramid {1.0, 0.5, 0.25}                    │
│  • Vectorized 2×2 Sliding Window Shannon Entropy Map Extraction                  │
│  • Bit-Plane Slicing (k=0,1,2) & Binarized Thresholding Normalization            │
│  • Cross-Modal Gating Network with Dynamic Attention Weighting                   │
│  • End-to-End Training & Zero-Shot Evaluation on ForenSynths and GenImage        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

#### Tier 1: Core Architectural Deliverables (Must Implement)
1. **Vectorized Multi-Scale Resampling Pyramid**: Complete implementation of bilinear down/up-sampling across $\{1.0, 0.5, 0.25\}$ scales.
2. **Vectorized Shannon Entropy Calculation**: GPU-accelerated $2 \times 2$ sliding window entropy mapping to discrete levels $\mathbb{V}$.
3. **Bit-Plane Slicing & Thresholding**: Extraction of low-order LSB bit-planes ($k=0,1,2$) with binarized $>0 \to 255$ normalization.
4. **Cross-Modal Gating & Attention Fusion**: The core unified classifier combining normalized entropy and LSB noise via dynamic scalar weights ($\alpha_{\text{MLEP}}, \alpha_{\text{LOTA}}$).
5. **Standard Benchmark Pipeline**: Automated data ingestion, training loops, and quantitative AUC/AP evaluation on ForenSynths and GenImage.

#### Tier 2: High-Impact Research Extensions (Recommended for Outstanding Grade)
1. **Top-$K$ Diverse MGPS Spatial Selection**: Replacing hard Top-1 selection with 4-quadrant diverse noise extraction to eliminate localized false negatives.
2. **Local Windowed Shuffling + Grad-CAM Saliency**: Confining patch shuffling to bounded $16 \times 16$ macro-grids, enabling clean Grad-CAM heatmaps that visually explain *where* forgeries exist.
3. **Online Compression Robustness Training**: Active training augmentation with online JPEG recompression and blurring to eliminate baseline accuracy drop-off.
4. **High-Throughput Mixed-Precision Benchmarking**: PyTorch automatic mixed precision (`fp16`/`bf16`) optimization and automated latency/throughput evaluation across GPU and CPU backends.

#### Tier 3: Advanced Industrial Polish (Optional State-of-the-Art Enhancements)
1. **Learnable Frequency-Domain Denoising Pre-Filter**:
   - *Concept*: Prepend a lightweight, trainable 2D Wiener filter or parameterized DCT frequency mask directly before the MLEP entropy stem.
   - *Impact*: Automatically strips 8x8 DCT block quantization blockiness caused by JPEG compression before calculating Shannon entropy, rendering the classifier virtually immune to social media recompression.
2. **Supervised Contrastive Cross-Modal Pre-Training ($\text{SupCon}$)**:
   - *Concept*: Before training the classification head, pre-train the MLEP and LOTA feature extractors using a Supervised Contrastive Loss ($\mathcal{L}_{\text{SupCon}}$).
   - *Impact*: Forces the latent representations of AI-generated texture entropy and AI-generated quantization noise to align tightly in embedding space, dramatically accelerating downstream linear classification convergence and improving zero-shot generalizability on unseen 2026 generators (e.g., FLUX.1 and DALL-E 3).
3. **Automated Adversarial Robustness Auditing**:
   - *Concept*: Build a testing suite that subjects the trained detector to white-box and black-box adversarial attacks (e.g., Fast Gradient Sign Method [FGSM], Projected Gradient Descent [PGD], and Carlini-Wagner perturbations).
   - *Impact*: Quantifies the exact perturbation threshold required to fool the dual-cue detector, providing academic rigor and proving that fusing statistical entropy with LSB steganalysis yields significantly higher adversarial defense boundaries than standalone CNN architectures.

---
*End of Technical Specification and Implementation Masterplan.*

