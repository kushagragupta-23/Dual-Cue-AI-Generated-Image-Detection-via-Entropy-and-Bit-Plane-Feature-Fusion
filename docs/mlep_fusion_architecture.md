# Multi-Level Entropy Pyramid (MLEP) & Fusion Architecture

## 1. Abstract

The **Multi-Level Entropy Pyramid (MLEP)** is a novel architecture for detecting AI-generated images by analyzing structural and semantic anomalies. Generative models (like Diffusion Models and GANs) synthesize images that look globally coherent to the human eye, but at a microscopic structural level, they fail to reproduce the natural entropy and randomness found in real photographs. 

MLEP exposes these structural flaws by employing **Local Windowed Patch Shuffling** and calculating **Shannon Entropy** across a multi-scale feature pyramid. Finally, it uses a **Dynamic Cross-Modal Attention Gating Network** to fuse these global semantic features with local high-frequency noise patches, yielding state-of-the-art detection accuracy.

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
The shuffled tensor X_shuffled is processed through a Convolutional Neural Network (CNN) backbone (e.g., ResNet-18). We extract intermediate feature maps at three distinct scales to capture both fine-grained texture anomalies and coarse semantic flaws:
* Level 1: Early convolutional layers (high resolution, edge detection)
* Level 2: Intermediate layers (texture and pattern detection)
* Level 3: Deep layers (semantic structure)

### C. Shannon Entropy Calculation
For each spatial location (i, j) in the D-dimensional feature map at scale s, we compute a probability distribution over the feature channels using the Softmax function. The localized Shannon Entropy H(i,j) is calculated as:

    H(i,j) = - Sum(p_d(i,j) * log2(p_d(i,j)))

Where p_d is the normalized activation probability. The resulting 2D Entropy Map serves as a dense anomaly heatmap.

---

## 3. Dual-Cue Fusion Engine

To maximize robustness, MLEP is fused with a secondary input cue (e.g., high-frequency noise patches). This is achieved via the **Dynamic Cross-Modal Attention Gating Network**.

### Mechanism
1. **MLEP Branch**: Processes the shuffled, global image to output a deep feature vector v_mlep.
2. **Local Branch**: Processes isolated high-frequency noise patches to output a deep feature vector v_local.
3. **Cross-Attention**: A learned multi-layer perceptron (MLP) analyzes both vectors and predicts a dynamic gating weight alpha between 0 and 1.
4. **Weighted Fusion**: The final classification vector is a convex combination of both cues:
   
   v_final = alpha * v_mlep + (1 - alpha) * v_local

This gating mechanism allows the network to dynamically trust the MLEP structural anomalies when analyzing highly coherent synthetic scenes, but switch its attention to the local high-frequency noise when analyzing heavily compressed or distorted images.

---

## 4. Hardware Implementation & Vectorization

The MLEP architecture has been rigorously optimized for execution on modern Windows hardware.

* **Target Device**: Lenovo LOQ with NVIDIA RTX 4050
* **Vectorized Execution**: The Local Windowed Patch Shuffling is implemented natively in PyTorch utilizing advanced `einops` style reshaping and CUDA-accelerated `gather` operations. There are no Python `for` loops in the critical execution path.
* **VRAM Efficiency**: The multi-scale pyramid leverages in-place operations and gradient checkpointing where applicable to remain well within the 6GB VRAM constraint of the RTX 4050.
