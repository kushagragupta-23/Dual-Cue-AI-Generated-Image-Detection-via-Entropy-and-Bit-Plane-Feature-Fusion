# Multi-Level Entropy Pyramid (MLEP) Architecture

## 1. Abstract

The **Multi-Level Entropy Pyramid (MLEP)** is a novel architecture for detecting AI-generated images by analyzing structural and semantic anomalies. Generative models (like Diffusion Models and GANs) synthesize images that look globally coherent to the human eye, but at a microscopic structural level, they fail to reproduce the natural entropy and randomness found in real photographs. 

MLEP exposes these structural flaws by employing **Local Windowed Patch Shuffling** and calculating **Shannon Entropy** across a multi-scale feature pyramid. A ResNet-50 backbone then encodes these entropy maps into a deep feature representation, followed by a multi-layer classifier head.

---

## 2. Mathematical Formulation of MLEP

### A. Local Windowed Patch Shuffling
Given an input image tensor X, MLEP first divides the image into non-overlapping Macro-Windows of size M x M. 

Within each Macro-Window, the pixels are further subdivided into Micro-Patches of size L x L. 
A pseudo-random permutation is applied to the Micro-Patches strictly *within* their respective Macro-Windows.

This local shuffling achieves a critical objective:
* **Global Semantics are Preserved**: Because patches are only shuffled locally, the overall scene (e.g., a face, a car) remains globally recognizable to the ResNet backbone.
* **Local Continuity is Destroyed**: The micro-structure is aggressively corrupted. Real images, possessing natural high entropy, exhibit massive statistical drops in feature confidence when shuffled. AI images, possessing low entropy (over-smoothed patches), exhibit surprisingly little change when shuffled.

### B. Multi-Scale Feature Pyramid
The shuffled tensor X_shuffled is processed through a multi-scale resampling pyramid at three distinct scales (1.0x, 0.5x, 0.25x) to capture both fine-grained texture anomalies and coarse semantic flaws:
* Level 1 (1.0x): Full resolution — captures pixel-level noise patterns
* Level 2 (0.5x): Half resolution — captures texture-level anomalies
* Level 3 (0.25x): Quarter resolution — captures semantic structure

### C. Shannon Entropy Calculation
For each spatial location (i, j) at each scale, we compute a 2x2 sliding window Shannon Entropy. The entropy H(i,j) is calculated as:

    H(i,j) = - Sum(p_d(i,j) * log2(p_d(i,j)))

Where p_d is the normalized pixel probability within the local window. The resulting 2D Entropy Map serves as a dense anomaly heatmap. The 3 scales × 3 RGB channels yield a 9-channel entropy feature tensor.

---

## 3. Classification Head

The 9-channel entropy feature tensor is passed through:

1. **BatchNorm2d**: Normalizes the entropy maps to zero-mean, unit-variance to match the ResNet backbone's expected input distribution.
2. **ResNet-50 Backbone**: Encodes the normalized entropy maps into a 2048-dimensional global average pooled feature vector.
3. **MLP Classifier**: A dropout-regularized multi-layer perceptron (Dropout → Linear(2048→512) → ReLU → Dropout → Linear(512→1)) outputs a binary logit for Real vs. AI-Generated classification.

---

## 4. Hardware Implementation & Vectorization

The MLEP architecture has been rigorously optimized for execution on modern Windows hardware.

* **Target Device**: Lenovo LOQ with NVIDIA RTX 4050
* **Vectorized Execution**: The Local Windowed Patch Shuffling is implemented natively in PyTorch utilizing advanced `einops` style reshaping and CUDA-accelerated `gather` operations. There are no Python `for` loops in the critical execution path.
* **VRAM Efficiency**: The multi-scale pyramid leverages in-place operations and gradient checkpointing where applicable to remain well within the 6GB VRAM constraint of the RTX 4050.
* **Mixed Precision**: Automatic Mixed Precision (AMP) with GradScaler maximizes throughput on the RTX 4050's Tensor Cores.
