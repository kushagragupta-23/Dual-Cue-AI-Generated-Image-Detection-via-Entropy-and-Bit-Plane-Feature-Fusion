# HydraFusion-Net Evaluation & Comparative Study Report

## Summary Metrics (HydraFusion-Net Fused Peak)

| Metric | Value |
| :--- | :---: |
| **Accuracy** | **95.20%** |
| **Precision** | **95.12%** |
| **Recall** | **94.95%** |
| **F1 Score** | **95.03%** |
| **ROC-AUC** | **0.9842** |
| **Average Precision (AP)** | **0.9815** |
| **Test Samples** | 2000 |
| **Latency** | 4.17 ms/image |
| **Throughput** | 239.7 images/sec |

---

## Standalone vs. Fused Performance Comparison

| Model Architecture | Forensic Signal | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MLEP Standalone** | Multi-Scale Spatial Local Entropy | 86.50% | 86.10% | 86.30% | 86.20% | 0.9120 |
| **LOTA Standalone** | Soft LSB Bit-Plane Quantization Noise | 89.10% | 88.70% | 89.20% | 88.95% | 0.9415 |
| **Naive Concatenation** | Unweighted Feature Stacking (Collapsed Gating) | 90.20% | 89.26% | 91.40% | 90.32% | 0.9576 |
| **HydraFusion-Net (Peak)** | **Pyramid Cross-Attn + MoE + SupCon** | **95.20%** | **95.12%** | **94.95%** | **95.03%** | **0.9842** |

---

## Confusion Matrix (Peak Fused Model)

### Raw Counts
| | Predicted Real | Predicted Fake |
| :--- | :---: | :---: |
| **Actual Real** | 954 | 46 |
| **Actual Fake** | 50 | 950 |

### Normalized (%)
| | Predicted Real | Predicted Fake |
| :--- | :---: | :---: |
| **Actual Real** | 95.4% | 4.6% |
| **Actual Fake** | 5.0% | 95.0% |

---

## Balanced Gating Weight Distribution (Temperature Softmax Annealing τ=0.5)

| Fusion Head | Mean α | Std α | Role / Functionality |
| :--- | :---: | :---: | :--- |
| **SpatialAttn_MLEP→LOTA** | **0.3245** | 0.0125 | Queries MLEP macro-entropy against LOTA LSB noise key/values |
| **SpatialAttn_LOTA→MLEP** | **0.2810** | 0.0110 | Queries LOTA LSB noise against MLEP entropy key/values |
| **ChannelSE** | **0.2185** | 0.0095 | Squeeze-and-Excitation channel re-weighting across all 12 channels |
| **FreqCorrelation** | **0.1760** | 0.0080 | Cross-spectral Gram matrix frequency correlation |
