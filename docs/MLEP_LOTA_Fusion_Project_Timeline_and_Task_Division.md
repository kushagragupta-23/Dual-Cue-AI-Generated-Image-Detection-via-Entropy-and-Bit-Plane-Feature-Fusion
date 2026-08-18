# Executive Project Report & Engineering Timeline: Dual-Cue AI-Generated Image Detection (MLEP & LOTA Fusion)

**Project Title:** Dual-Cue AI-Generated Image Detection: Statistical Entropy (MLEP) & Local Gradient Quantization (LOTA) Fusion  
**Sprint Duration:** 3 Weeks (28 July 2026 – 18 August 2026)  
**Target Deadline:** 18 August 2026  
**Authors / Lead Engineering Team:** Kushagra Gupta (`Kushagra.G27pgai@jioinstitute.edu.in`) & Aishwarya Nevrekar (`Aishwarya.N27pgai@jioinstitute.edu.in`)  
**Institution:** Artificial Intelligence & Data Science Programme, Jio Institute, Navi Mumbai, Maharashtra, India  
**Document Status:** Final Engineering Specification & Task Allocation  

---

## 1. Executive Summary & Strategy Overview

This report establishes the definitive 3-week engineering roadmap, technical milestone schedule, and 2-person task division matrix for building a research-ready codebase for **Dual-Cue AI-Generated Image Detection (MLEP & LOTA Fusion)**. 

To ensure maximum development velocity and zero file-collision conflicts, the project strategy focuses on **one major architectural milestone per week**, with continuous modular testing and a dedicated multi-day integration sprint at the conclusion of the timeline. 

In accordance with the updated team responsibility assignment, **all tasks formerly assigned to Aishwarya (MLEP Preprocessing, Architecture I, and Architecture III) have been transferred to Kushagra**, while **all tasks formerly assigned to Kushagra (LOTA Preprocessing, Architecture II, and the Evaluation/Explainability Pipeline) have been transferred to Aishwarya**. This task exchange has been carefully analyzed against the physical hardware specifications of both team members and is demonstrated to be **technically optimal** for memory bandwidth, CUDA native kernel execution, and cross-platform compute efficiency.

---

## 2. Hardware Architecture & Task Allocation Justification

A critical engineering achievement of this project division is the direct alignment of mathematical module requirements with the specific physical hardware architectures available to each researcher.

### Team Member Hardware Specifications
* **Kushagra (Lenovo Gaming Laptop – "ELDORIA"):**
  * **CPU:** 13th Gen Intel(R) Core(TM) i5-13450HX (10 Cores / 16 Threads, High Multi-Threaded Throughput)
  * **Memory:** 16 GB System RAM
  * **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB Discrete GDDR6 VRAM, Dedicated CUDA Tensor Cores)
  * **OS & Runtime:** Windows 11 64-bit, PyTorch (`cuda` backend, Native Precision `fp16`/`bf16` via AMP GradScaler)

* **Aishwarya (Apple MacBook Air M4):**
  * **CPU/GPU Chip:** Apple Silicon M4 System-on-Chip (SoC)
  * **Memory:** High-Bandwidth Unified Memory Architecture (UMA) shared dynamically across CPU and Apple GPU cores
  * **OS & Runtime:** macOS, PyTorch (`mps` Metal Performance Shaders backend, Optimized Vectorized Linear Algebra)

### Technical Justification for the Task Swap

#### Why Kushagra is Optimal for MLEP, Architecture I (SupCon), and Architecture III (MoE + DANN):
1. **CUDA Native Support for Dynamic Routing and Custom Kernels:** Architecture III (`MoEDomainGeneralizer`) requires implementing Sparse Mixture of Experts (MoE) with Top-2 dynamic expert routing and Gradient Reversal Layers (GRL) for Domain Adversarial Training (DANN). Advanced custom backward hooks and dynamic branching in PyTorch operate natively and with maximum stability on NVIDIA CUDA primitives. Executing GRL and dynamic routing on CUDA eliminates potential Metal API fallback latency.
2. **Tensor Core Acceleration for Contrastive Multi-View Batching:** Architecture I (`LearnableFreqSupConNet`) utilizes Supervised Contrastive Learning (SupCon), which requires computing dense $N \times N$ similarity matrices across multi-view augmented batch pairs. The dedicated Tensor Cores on Kushagra's RTX 4050 accelerate these heavy matrix-multiplication operations in mixed precision (`fp16`).
3. **High CPU Thread Count for Multi-Scale Patch Shuffling:** The MLEP preprocessing pipeline requires extracting image patches across a 3-level Gaussian pyramid, applying Local Windowed Shuffling ($16 \times 16$ macro-grids), and computing vectorized Shannon entropy. The Intel i5-13450HX processor provides 16 hardware threads, enabling rapid CPU-based parallel image folding and entropy tensor generation before GPU transfer.

#### Why Aishwarya is Optimal for LOTA, Architecture II (MGA-Net), and Evaluation/Explainability:
1. **Apple Silicon Unified Memory for High-Dimensional Bit-Plane Tensors:** The LOTA preprocessing pipeline (`TopKLOTAExtractor`) extracts 8 binary bit-planes across all 3 RGB color channels, generating 24 separate spatial bit-maps per image. On discrete GPU systems with a strict 6 GB VRAM ceiling (like the RTX 4050), storing 24 bit-plane tensors across large training batches can trigger Out-Of-Memory (OOM) exceptions or force slow PCIe bus data transfers between system RAM and VRAM. Aishwarya's MacBook Air M4 leverages **Apple Unified Memory Architecture**, where system memory and GPU memory are physically unified. This allows massive bit-plane tensor expansion, 4-directional MGPS gradient convolution filtering, and Top-$K$ patch ranking to execute in-memory without PCIe bottlenecking or 6 GB VRAM constraints.
2. **Metal (MPS) Optimization for Multi-Granularity Attention:** Architecture II (`MGANet`) introduces Cross-Attention and Pyramid Gating across spatial scales. Standard transformer-style attention query-key-value ($Q, K, V$) projections and scaled dot-product attention map directly onto Apple Silicon's highly optimized matrix-multiplication accelerators in the MPS backend.
3. **High-Resolution Explainability & Rendering:** Generating Grad-CAM saliency heatmaps, attention weight overlays, ROC/AUC curves, and high-DPI publication graphs requires intensive memory interaction between model outputs and visualization libraries (Matplotlib/Seaborn). Apple Silicon's single-core CPU performance and unified memory enable instantaneous rendering and exporting of visual artifacts without GPU-to-CPU memory copy overhead.

---

## 3. Chronological 3-Week Roadmap & Weekly Milestones

```
[Week 1: 28 Jul – 3 Aug] ────► [Week 2: 4 Aug – 10 Aug] ────► [Week 3: 11 Aug – 18 Aug]
Baseline Preprocessing         Specialized Architectures     Arch III, Eval Suite &
MLEP (Kushagra) & LOTA (Aish)   Arch I (Kush) & Arch II (Aish) End-to-End Integration
```

### Project Timeline Gantt Chart (Visual Graphic)

![MLEP & LOTA Fusion Project Gantt Chart](MLEP_LOTA_Gantt_Chart.png)

*(Note: High-resolution visual chart is saved as `MLEP_LOTA_Gantt_Chart.png` and interactive HTML version as `MLEP_LOTA_Gantt_Chart_Visual.html` in your project folder).*


### Week 1 (28 July – 3 August 2026): Baseline Preprocessing Pipelines & Primitives

#### Primary Goal
Build, verify, and unit-test both standalone dual-cue feature extraction pipelines independently from scratch, establishing solid data ingestion and preprocessing primitives.

#### Kushagra's Tasks (MLEP Preprocessing Pipeline & Infrastructure Lead)
* **Read & Ingest:** Thoroughly review Section 1 ("Architecture of MLEP"), Section 3 ("Input Pipeline & Preprocessing"), and Section 4 of the Implementation Specification.
* **Project Repository Setup:** Initialize project directory structure, `pyproject.toml` / `requirements.txt`, git formatting rules (`black`, `flake8`), and foundational data loaders (`src/data/dataset.py`) with automatic bounding-box ROI cropping and RGB normalization.
* **Implement Vectorized MLEP Extractor (`src/models/mlep.py`):**
  * Image loading and standard pre-transformations.
  * Multi-scale Gaussian pyramid folding (Level 0: $1.0\times$, Level 1: $0.75\times$, Level 2: $0.5\times$).
  * Patch partitioning and Local Windowed Shuffling across a $16 \times 16$ macro-grid to disrupt spatial artifacts while preserving localized statistical distributions.
  * Vectorized Shannon entropy computation across color channels: $H(P) = -\sum p_i \log_2(p_i)$.
  * Final entropy tensor generation and formatting into standardized spatial feature maps.
* **Visualization & Verification:** Write `scripts/visualize_mlep.py` to output visual entropy heatmaps comparing authentic images against ProGAN/Stable Diffusion synthetics.
* **Testing & Docs:** Create unit test suite in `tests/test_mlep.py` verifying tensor output shapes, zero-NaN assertions, and entropy bounds; document module API in `docs/mlep_pipeline.md`.

#### Aishwarya's Tasks (LOTA Preprocessing Pipeline Lead)
* **Read & Ingest:** Thoroughly review Section 2 ("Architecture of LOTA"), Section 3 ("Input Pipeline & Preprocessing"), and Section 4 of the Implementation Specification.
* **Implement Top-K LOTA Extractor (`src/models/lota.py`):**
  * Bit-plane slicing across the 8 least significant bits (LSBs) for RGB channels ($8 \times 3 = 24$ binary planes).
  * LSB composition and binarized threshold normalization to isolate generative quantization artifacts from high-order semantic content.
  * Multi-directional Gradient Patch Scoring (MGPS) applying 4-directional convolution kernels ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) to detect high-frequency grid anomalies.
  * Top-$K$ patch ranking and extraction algorithm to isolate the most anomalous artifact regions.
* **Visualization & Verification:** Write `scripts/visualize_lota.py` to render individual bit-plane decompositions and MGPS gradient anomaly maps.
* **Testing & Docs:** Create unit test suite in `tests/test_lota.py` validating bit-plane extraction exactness, MGPS filter responses, and Top-$K$ index sorting; document module API in `docs/lota_pipeline.md`.

#### Weekend Synchronization (3–4 August)
* **Peer Code Review:** Kushagra audits Aishwarya's LOTA bit-plane logic; Aishwarya audits Kushagra's MLEP entropy folding.
* **Branch Merging:** Merge feature branches (`feature/mlep-pipeline` and `feature/lota-pipeline`) into the `develop` branch.
* **Code Style Standardization:** Ensure uniform PEP8 formatting, type hinting, and docstrings across all modules.
* **Verification:** Execute joint pipeline verification script on sample ForenSynths (ProGAN) and GenImage images to confirm zero runtime errors and correct tensor shapes.

#### Milestone 1 Deliverable (Completed by 3 August)
* ✅ Fully working, tested, and documented MLEP Preprocessing Pipeline (`VectorizedMLEPExtractor`).
* ✅ Fully working, tested, and documented LOTA Preprocessing Pipeline (`TopKLOTAExtractor`).
* ✅ Unified data ingestion and preprocessing repository structure.

---

### Week 2 (4 August – 10 August 2026): Specialized Deep Learning Architectures I & II

#### Primary Goal
Implement the core deep learning backbones, projection heads, and specialized feature alignment networks (Architecture I & Architecture II) for dual-cue feature processing.

#### Kushagra's Tasks (ResNet Stem & Architecture I: Learnable Frequency & SupCon)
* **Read & Ingest:** Review Section 14 of the specification and `Specialized_Architecture_1_Learnable_Frequency_Denoising_and_SupCon_Alignment.md`.
* **ResNet Backbone Adapters (`src/models/backbones.py`):** Implement shared/dual ResNet-18 and ResNet-50 feature extractor stems with modified input channels to accept MLEP entropy tensors and LOTA artifact maps.
* **Implement Architecture I (`src/models/arch1_supcon.py` - `LearnableFreqSupConNet`):**
  * **Learnable Frequency Filter:** Build 2D FFT/DCT domain gating layers with learnable spectral amplitude weights to dynamically suppress camera sensor noise while amplifying synthetic generator frequency discrepancies.
  * **Projection Head:** Build non-linear MLP projection heads mapping spatial features into a compact, normalized unit hypersphere embedding space.
  * **Contrastive Learning Engine:** Implement Supervised Contrastive (SupCon) Loss formulation:
    $$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$
  * **Training Infrastructure:** Create multi-view batch augmentation pipeline and training script (`scripts/train_arch1.py`) with CUDA mixed-precision (`fp16`) support.
* **Testing & Docs:** Write unit tests in `tests/test_arch1.py` verifying gradient backpropagation through FFT layers and SupCon loss stability; update architecture documentation.

#### Aishwarya's Tasks (ResNet Stem & Architecture II: MGA-Net Cross-Attention)
* **Read & Ingest:** Review Section 14 of the specification and `Specialized_Architecture_2_Multi_Granularity_Cross_Attention_and_Pyramid_Gating.md`.
* **Implement Architecture II (`src/models/arch2_mganet.py` - `MGANet`):**
  * **Multi-Granularity Cross-Attention (MGA):** Build bidirectional cross-attention modules where MLEP entropy features act as Queries ($Q$) to attend over LOTA bit-plane Keys/Values ($K, V$), and vice versa, fusing statistical and structural cues.
  * **Pyramid Gating Mechanism:** Implement spatial gating across multi-scale feature hierarchies to dynamically modulate feature importance based on artifact granularity.
  * **Residual Refinement Layers:** Integrate feed-forward residual bottleneck blocks to refine fused token representations before classification.
  * **Training Infrastructure:** Create standalone training script (`scripts/train_arch2.py`) optimized for Apple Silicon MPS backend execution.
* **Testing & Docs:** Write unit tests in `tests/test_arch2.py` verifying cross-attention tensor matrix multiplications, attention map normalization ($\sum \text{attn} = 1.0$), and forward pass latency; update architecture documentation.

#### Weekend Synchronization (10–11 August)
* **Joint Architecture Merge:** Merge `feature/arch1-supcon` and `feature/arch2-mganet` into `develop`.
* **Tensor Dimension & Interface Verification:** Ensure clean interface compatibility between preprocessing outputs, ResNet stems, and both specialized architecture heads.
* **Forward Pass & Gradient Flow Testing:** Execute end-to-end forward and backward pass stress tests across both architectures on dummy batch tensors.
* **Bug Resolution:** Identify and resolve any tensor shape mismatches, precision incompatibilities, or device allocation bugs (`cuda` vs `mps`).

#### Milestone 2 Deliverable (Completed by 10 August)
* ✅ Working ResNet-18/50 backbone stem adapters.
* ✅ Complete Architecture I (`LearnableFreqSupConNet`) with Learnable Frequency Filter and SupCon Loss.
* ✅ Complete Architecture II (`MGANet`) with Multi-Granularity Cross-Attention and Pyramid Gating.
* ✅ Verified forward and backward gradient propagation across both networks.

---

### Week 3 (11 August – 18 August 2026): Advanced Generalization, Evaluation Pipeline & Final Integration

#### Primary Goal
Construct Architecture III (MoE & Domain Generalization), build the comprehensive evaluation/explainability suite, and execute a 3-day intensive integration and benchmarking sprint to finalize the research codebase.

#### Kushagra's Tasks (11–14 Aug: Architecture III: MoE & DANN Generalization)
* **Read & Ingest:** Review Section 14 of the specification and `Specialized_Architecture_3_Mixture_of_Experts_MoE_and_Domain_Adversarial_Generalization.md`.
* **Implement Architecture III (`src/models/arch3_moe.py` - `MoEDomainGeneralizer`):**
  * **Sparse Mixture of Experts (MoE):** Build a multi-expert routing architecture featuring 4 specialized feed-forward expert networks and a Top-2 noisy gating router to specialize in different generator families (e.g., GAN vs. Diffusion vs. Autoregressive).
  * **Load-Balancing Auxiliary Loss:** Implement importance and load-balancing auxiliary loss terms ($\mathcal{L}_{\text{aux}}$) to prevent routing collapse and ensure equal expert utilization.
  * **Gradient Reversal Layer (GRL):** Build custom autograd GRL operator ($R_\lambda(x) = x$ in forward pass; $\frac{d R_\lambda}{dx} = -\lambda \mathbf{I}$ in backward pass).
  * **Domain Discriminator Network:** Construct adversarial domain classifier head trained to predict source generator architecture, forcing the feature encoder to learn domain-invariant, universal artifact representations.
* **Testing & Docs:** Write unit tests in `tests/test_arch3.py` verifying Top-2 expert routing distribution, GRL negative gradient scaling, and auxiliary loss computation; update documentation.

#### Aishwarya's Tasks (11–14 Aug: Comprehensive Evaluation & Explainability Pipeline)
* **Read & Ingest:** Review Sections 7, 8, and 15 of the specification.
* **Implement Evaluation & Benchmarking Engine (`src/eval/metrics.py` & `src/eval/evaluator.py`):**
  * Automated computation of academic classification metrics: Overall Accuracy, Precision, Recall, F1-Score, Receiver Operating Characteristic (ROC) curves, and Area Under the Curve (ROC-AUC).
  * Confusion Matrix generator with normalized heatmaps and breakdown by generator architecture (ProGAN, StyleGAN, Stable Diffusion v1.5/v2.1, Midjourney, ADM, etc.).
* **Implement Explainability & Visual Diagnostics (`src/eval/explainability.py`):**
  * **Grad-CAM Integration:** Implement backward hooks to extract gradients from the cross-modal fusion layer and render un-scrambled spatial saliency heatmaps overlaid on original RGB images.
  * **Attention Weight Visualizer:** Build extraction tools to extract and plot MGA cross-attention maps and MoE expert routing gate distributions.
* **Publication Graphing Suite (`scripts/generate_figures.py`):**
  * Build automated Matplotlib/Seaborn plotting scripts to generate publication-ready high-DPI figures: comparative ROC curves, robustness under JPEG compression ($Q=70..100$), and blur stress tests ($\sigma=0.5..2.0$).
* **Testing & Docs:** Create unit tests in `tests/test_eval.py` verifying metric calculation exactness against synthetic ground-truth arrays; document evaluation CLI.

---

### Intensive Joint Integration & Refinement Sprint (15–17 August)

From August 15 to August 17, Aishwarya and Kushagra will work synchronously to unite all modular components into the unified dual-cue detection engine.

#### 15 August: Master Pipeline Assembly & Data Integration
* **End-to-End Model Assembly:** Build `src/models/fusion_model.py` (`DualCueAIGIDModel`), combining MLEP Preprocessing, LOTA Preprocessing, ResNet stems, Architecture I, Architecture II, and Architecture III into a unified, configurable PyTorch module.
* **Multi-Dataset Loading Integration:** Integrate unified dataset loaders across ForenSynths (ProGAN cars/cats/horses) and GenImage (SD v1.5 vs. ImageNet real).
* **Forward Pass Verification:** Execute full forward passes of raw RGB image batches through the unified model, verifying zero bottleneck latency and proper feature gating.

#### 16 August: Full Training, Validation, & Cross-Platform Debugging
* **End-to-End Training Loop Execution:** Launch full training and validation runs using `scripts/train_end_to_end.py`.
* **Cross-Platform Compute Optimization:**
  * Test and optimize CUDA execution on Kushagra's RTX 4050 using PyTorch Automatic Mixed Precision (`fp16` / `GradScaler`), tuning batch size to maximize VRAM occupancy without OOM.
  * Test and optimize Metal MPS execution on Aishwarya's MacBook Air M4, ensuring clean fallback for operations not natively vectorized in Metal.
* **Hyperparameter & Loss Tuning:** Verify convergence stability across total loss formulation:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{SupCon}} + \lambda_2 \mathcal{L}_{\text{aux}} + \lambda_3 \mathcal{L}_{\text{domain}}$$
* **Debugging:** Systematically identify and eliminate any numerical underflow/overflow, NaN losses, or memory leak anomalies.

#### 17 August: Benchmarking, Explainability Rendering & Documentation
* **Execution of Evaluation Suite:** Run Aishwarya's evaluation engine across test sets; generate automated Markdown benchmark tables comparing standalone baselines vs. MGA-Net and MoE models.
* **Explainability Artifact Generation:** Export Grad-CAM saliency heatmaps and cross-attention weight overlays across authentic vs. AI-generated sample pairs.
* **Robustness Stress-Testing:** Run automated scripts evaluating model degradation under online JPEG compression and Gaussian blur.
* **Codebase Cleanup & Documentation Polish:** Finalize master `README.md`, inline code comments, docstrings, and update architecture system diagrams.

---

### Final Delivery Day (18 August 2026): Verification Checklist & Sign-Off

On the final target date, both team members will execute a complete checklist verification to ensure the repository is 100% research-ready and error-free:

1. **Full Forward Pass Check:** Verify that passing an arbitrary directory of RGB images produces clean anomaly prediction scores without warnings or crashes.
2. **Full Backward Pass Check:** Verify clean gradient backpropagation through all sub-modules (MLEP, LOTA, FFT filters, Attention gates, MoE routers, GRL) without gradient explosion or vanishing.
3. **Training Launch Verification:** Verify that `python scripts/train_end_to_end.py --config configs/default.yaml` initializes cleanly, logs metrics to TensorBoard/W&B, and saves checkpoint weights correctly on both Windows (CUDA) and macOS (MPS).
4. **Results Export Verification:** Verify that evaluation scripts generate CSV/JSON metrics reports and high-resolution PDF/PNG publication graphs seamlessly.
5. **Documentation & Diagram Sign-Off:** Confirm that system architecture diagrams, module documentation, and README setup instructions are complete and accurate.
6. **Demo & Presentation Readiness:** Finalize slide deck and live presentation demo script showcasing real-time artifact detection and Grad-CAM visualization.

---

## 4. Responsibility Matrix, Task Ownership & Timeline Schedule

| Project Module / Task Area | Timeline & Milestone Schedule | Primary Lead / Responsible | Secondary / Review & Audit | Hardware Compute Allocation |
| :--- | :--- | :--- | :--- | :--- |
| **Project Repository Setup & CI/CD** | **Week 1** (28 Jul – 30 Jul) | **Kushagra** & **Aishwarya** (Joint) | Mutual Review | Cross-Platform (Windows/macOS) |
| **Data Ingestion & ROI Cropping (`src/data/`)** | **Week 1** (28 Jul – 31 Jul) | **Kushagra** | Aishwarya | CPU Multi-Threading (Intel i5 / M4) |
| **MLEP Preprocessing Pipeline (`src/models/mlep.py`)** | **Week 1** (30 Jul – 03 Aug) | **Kushagra** | Aishwarya | Intel i5-13450HX CPU & RTX 4050 CUDA |
| **LOTA Preprocessing Pipeline (`src/models/lota.py`)** | **Week 1** (30 Jul – 03 Aug) | **Aishwarya** | Kushagra | Apple Silicon M4 Unified Memory & MPS |
| **Milestone 1 Review & Baseline Merge** | **Week 1 Weekend** (03 Aug – 04 Aug) | **Kushagra** & **Aishwarya** (Joint) | **Milestone 1 Sign-Off** | Cross-Platform Code Verification |
| **ResNet Stems & Adapters (`src/models/backbones.py`)** | **Week 2** (04 Aug – 07 Aug) | **Kushagra** | Aishwarya | CUDA / MPS Agnostic |
| **Architecture I: SupCon & Freq Filter (`arch1_supcon.py`)**| **Week 2** (06 Aug – 10 Aug) | **Kushagra** | Aishwarya | NVIDIA RTX 4050 Tensor Cores (fp16 AMP) |
| **Architecture II: MGA-Net Attention (`arch2_mganet.py`)** | **Week 2** (05 Aug – 10 Aug) | **Aishwarya** | Kushagra | Apple Silicon M4 Metal MPS Backend |
| **Milestone 2 Architecture Integration & Test** | **Week 2 Weekend** (10 Aug – 11 Aug) | **Kushagra** & **Aishwarya** (Joint) | **Milestone 2 Sign-Off** | Joint Forward/Backward Gradient Testing |
| **Architecture III: MoE & DANN (`arch3_moe.py`)** | **Week 3** (11 Aug – 15 Aug) | **Kushagra** | Aishwarya | NVIDIA RTX 4050 CUDA (Dynamic Routing & GRL) |
| **Evaluation Engine & Metrics (`src/eval/`)** | **Week 3** (11 Aug – 15 Aug) | **Aishwarya** | Kushagra | Apple Silicon M4 Unified Memory |
| **Explainability (Grad-CAM / Attention Overlays)** | **Week 3** (12 Aug – 15 Aug) | **Aishwarya** | Kushagra | Apple Silicon M4 High-Res Rendering |
| **Publication Figure Generation (`scripts/generate_...`)** | **Week 3** (12 Aug – 15 Aug) | **Aishwarya** | Kushagra | macOS Matplotlib / Seaborn Engine |
| **End-to-End Model Integration (`fusion_model.py`)** | **Integration Sprint** (15 Aug – 16 Aug) | **Kushagra** & **Aishwarya** (Joint) | Mutual Review | Both (CUDA & MPS Cross-Validation) |
| **Full Training, Debugging & Performance Tuning** | **Integration Sprint** (16 Aug – 17 Aug) | **Kushagra** & **Aishwarya** (Joint) | Mutual Review | Both (CUDA & MPS Cross-Validation) |
| **Final Documentation, Benchmarking & README** | **Integration Sprint** (17 Aug – 18 Aug) | **Kushagra** & **Aishwarya** (Joint) | Mutual Review | Both |
| **Final Sign-Off & Project Delivery** | **Target Deadline** (18 Aug 2026) | **Kushagra** & **Aishwarya** (Joint) | **Final Delivery Sign-Off** | Full Research-Ready Codebase Delivery |

---

## 5. Technical Codebase File Structure & Ownership Mapping

```
DL_AND_CV_PROJECT/
├── configs/
│   ├── default.yaml                  # [Joint] Master training configuration
│   └── sweep.yaml                    # [Joint] Hyperparameter sweep configuration
├── docs/
│   ├── mlep_pipeline.md              # [Kushagra] MLEP mathematical & module docs
│   ├── lota_pipeline.md              # [Aishwarya] LOTA bit-plane & MGPS docs
│   └── architectures.md              # [Joint] Arch I, II, and III network docs
├── scripts/
│   ├── train_arch1.py                # [Kushagra] Standalone Arch I training script
│   ├── train_arch2.py                # [Aishwarya] Standalone Arch II training script
│   ├── train_end_to_end.py           # [Joint] Master end-to-end fusion training
│   ├── evaluate_model.py             # [Aishwarya] Master evaluation & benchmarking CLI
│   ├── visualize_mlep.py             # [Kushagra] Entropy heatmap visualizer
│   ├── visualize_lota.py             # [Aishwarya] Bit-plane & gradient map visualizer
│   └── generate_figures.py           # [Aishwarya] Publication figure generator
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # [Kushagra] Dataset loader, ROI crop, normalization
│   │   └── augmentations.py          # [Kushagra] Robustness transforms (JPEG, Blur)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlep.py                   # [Kushagra] VectorizedMLEPExtractor & Pyramid
│   │   ├── lota.py                   # [Aishwarya] TopKLOTAExtractor & MGPS filter
│   │   ├── backbones.py              # [Kushagra] ResNet-18/50 stems & channel adapters
│   │   ├── arch1_supcon.py           # [Kushagra] LearnableFreqSupConNet
│   │   ├── arch2_mganet.py           # [Aishwarya] MGANet (Cross-Attention & Pyramid)
│   │   ├── arch3_moe.py              # [Kushagra] MoEDomainGeneralizer (MoE & GRL)
│   │   └── fusion_model.py           # [Joint] DualCueAIGIDModel master assembly
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py                # [Aishwarya] Accuracy, F1, ROC, AUC, Confusion Matrix
│       ├── evaluator.py              # [Aishwarya] Evaluation loop & table compiler
│       └── explainability.py         # [Aishwarya] Grad-CAM & attention weight visualizer
├── tests/
│   ├── test_mlep.py                  # [Kushagra] MLEP unit tests
│   ├── test_lota.py                  # [Aishwarya] LOTA unit tests
│   ├── test_arch1.py                 # [Kushagra] Arch I unit & backward tests
│   ├── test_arch2.py                 # [Aishwarya] Arch II unit & attention tests
│   ├── test_arch3.py                 # [Kushagra] Arch III MoE & GRL tests
│   └── test_eval.py                  # [Aishwarya] Evaluation metric assertions
├── .gitignore
├── pyproject.toml
├── README.md                         # [Joint] Master setup and execution documentation
└── requirements.txt                  # [Joint] Cross-platform dependency definitions
```

---

## 6. Cross-Platform Engineering Standards & Risk Mitigation

To prevent cross-platform execution bugs between Kushagra's Windows/NVIDIA CUDA environment and Aishwarya's macOS/Apple Silicon Metal environment, all code developed in this project must adhere to three mandatory engineering rules:

### Rule 1: Dynamic Device Allocation
Never hardcode `'cuda'`, `'mps'`, or `'cpu'` within module logic. All model initialization and tensor allocations must dynamically resolve the target compute device:
```python
import torch

def get_compute_device() -> torch.device:
    """Dynamically select optimal available compute backend."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

### Rule 2: Precision Agnostic Execution & Fallbacks
NVIDIA RTX 4050 supports CUDA Automatic Mixed Precision (AMP) with `fp16` and `bf16` via `torch.cuda.amp.GradScaler`. Apple Silicon MPS supports `fp16` and `fp32`, but certain complex linear algebra operators may lack native `bf16` support or require fallback to standard 32-bit float.
* **Standard Guideline:** Training scripts must accept a `--precision` flag (`fp32`, `fp16`, `bf16`). If `bf16` or `fp16` is requested on a backend where specific complex ops fail, the training loop must gracefully fallback to `fp32` without crashing the integration pipeline.

### Rule 3: Operating System File Path Compatibility
Windows file systems use backslashes (`\`), whereas macOS uses forward slashes (`/`). Hardcoding string paths will break dataset loading during peer code reviews.
* **Standard Guideline:** All file system read/write operations, dataset indexing, and checkpoint saving must be implemented using Python's standard `pathlib.Path` module:
```python
from pathlib import Path

# Correct cross-platform path resolution
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "forensynths"
```

---

## 7. Summary of Guaranteed Deliverables by 18 August 2026

By the end of the timeline on 18 August 2026, the team guarantees the delivery of a fully integrated, research-ready codebase featuring:
1. ✅ **Complete MLEP Implementation:** Fully functional multi-scale entropy folding pipeline (`VectorizedMLEPExtractor`) with visual diagnostic scripts.
2. ✅ **Complete LOTA Implementation:** Fully functional 24 bit-plane slicing and 4-directional MGPS gradient anomaly detection pipeline (`TopKLOTAExtractor`).
3. ✅ **Architecture I (Frequency + SupCon):** Operational `LearnableFreqSupConNet` featuring 2D DCT/FFT learnable gating and Supervised Contrastive alignment.
4. ✅ **Architecture II (MGA-Net):** Operational `MGANet` featuring Multi-Granularity Cross-Attention and Pyramid Gating across statistical and structural feature tokens.
5. ✅ **Architecture III (MoE + DANN):** Operational `MoEDomainGeneralizer` featuring Top-2 routed Mixture of Experts and Gradient Reversal Layer domain adversarial generalization.
6. ✅ **Unified End-to-End Training Pipeline:** Configurable master model (`DualCueAIGIDModel`) and training loop supporting ForenSynths and GenImage datasets across CUDA and Metal backends.
7. ✅ **Evaluation & Explainability Suite:** Comprehensive academic metrics engine (ROC-AUC, Precision, Recall, F1, Confusion Matrices) and Grad-CAM spatial heatmap visualizer.
8. ✅ **Academic Documentation & Diagrams:** Master `README.md`, mathematical API documentation, and publication-ready graphs ready for research submission and presentation defense.

---
*Report generated and verified against project implementation specifications on 28 July 2026.*
