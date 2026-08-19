# MLEP Architecture: Multi-Level Entropy Pyramid

## 1. Overview

The MLEP (Multi-Level Entropy Pyramid) architecture detects AI-generated images by analyzing their entropy characteristics. The key insight is that generative models (Stable Diffusion, GANs, etc.) tend to smooth out high-frequency noise during the denoising process. Real camera sensors capture natural photonic noise that produces higher local entropy compared to AI-generated images.

Our experiments show a small but consistent entropy gap: Real images have a mean entropy of ~1.784 while AI-generated images measure ~1.766. MLEP computes Shannon entropy across a multi-scale feature pyramid and feeds the resulting entropy maps into a ResNet-50 backbone for classification.

---

## 2. Pipeline

### Step 1: Patch Shuffling (Optional — Currently Disabled)

The MLEP paper describes a local patch shuffling step where the image is divided into micro-patches that are spatially permuted. The idea is that real images with natural noise should show larger statistical changes when shuffled compared to smooth AI images.

**Current status:** This step is implemented (`MLEPExtractor.shuffle_patches()`) but disabled in the forward pass (`use_shuffling=False`). The reason is that our backbone uses pretrained ImageNet weights which expect spatially coherent input. Enabling shuffling would require training the backbone from scratch.

### Step 2: Multi-Scale Resampling Pyramid

The input image is processed through a 3-scale resampling pyramid:
- **Scale 1.0x:** Full resolution — captures pixel-level noise patterns
- **Scale 0.5x:** Half resolution then upsampled — captures texture-level smoothing
- **Scale 0.25x:** Quarter resolution then upsampled — captures coarse structural artifacts

Each scale is bilinearly downsampled then upsampled back to the original resolution. The three scales are concatenated along the channel dimension, producing a 9-channel tensor (3 scales × 3 RGB channels).

### Step 3: Shannon Entropy Computation

For each spatial location, a 2×2 sliding window computes discrete Shannon entropy:

    H(i,j) = -Σ p(x) · log₂(p(x))

where p(x) is the empirical frequency of each pixel value within the 4-pixel window. Possible entropy values are {0.0, 0.8113, 1.0, 1.5, 2.0}. This produces a 9-channel entropy feature map of shape (B, 9, H-1, W-1).

### Step 4: Classification

The 9-channel entropy map passes through:

1. **BatchNorm2d:** Normalizes the entropy values to zero-mean, unit-variance to match the ResNet backbone's expected input distribution.
2. **ResNet-50 Backbone:** Encodes the entropy maps into a 2048-D global average pooled feature vector. The first convolutional layer is adapted from 3 to 9 input channels by tiling pretrained weights.
3. **MLP Classifier:** `Dropout(0.5) → Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→1)` outputs a binary logit (Real vs AI).

---

## 3. Training & Regularization

To combat overfitting when training on entropy patterns:

- **Optimizer**: AdamW with weight decay (L2 penalty) of 0.05.
- **Differential Learning Rates**: 
  - Base: 2e-4
  - Backbone: 1e-4 (0.5x) to preserve ImageNet pretraining.
  - Head: 1e-3 (5x) for rapid convergence of the new classifier.
- **LR Scheduler**: CosineAnnealingLR (T_max=epochs) to smoothly anneal the learning rate.
- **Dropout**: Strong spatial regularization using Dropout(0.5) and Dropout(0.3) in the MLP head.
- **Early Stopping**: Patience set to 7 epochs (monitors validation accuracy).

---

## 4. Hardware Details (Optimized)

- **GPU**: NVIDIA RTX 4050 (6GB VRAM, Ada Lovelace)
- **Automatic Mixed Precision (AMP)**: Enabled with GradScaler for FP16 training speedups and reduced VRAM.
- **TF32 Tensor Cores**: Enabled (`torch.backends.cuda.matmul.allow_tf32 = True`) for near-FP16 speed with FP32 precision.
- **cuDNN Benchmark**: Enabled for auto-tuning convolutional kernels.
- **Entropy Computation**: Fully vectorized using `F.unfold` — no Python loops over spatial dimensions.
