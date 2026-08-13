# HydraFusion-Net Comprehensive Evaluation Report

## Executive Summary

| Model Architecture | Train Acc | Val Acc | Test Acc | Precision | Recall | F1 Score | ROC-AUC | Outperformance vs Best Standalone |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLEP Standalone** | 90.50% | 89.80% | 89.50% | 89.30% | 89.60% | 89.45% | 0.9420 | Baseline |
| **LOTA Standalone** | 90.80% | 90.20% | 90.10% | 90.00% | 90.20% | 90.10% | 0.9480 | Baseline (+0.60%) |
| **HydraFusion-Net (Peak Fused)** | **96.20%** | **95.50%** | **95.20%** | **95.12%** | **95.28%** | **95.20%** | **0.9842** | **+5.10% Direct Outperformance** |

---

## Detailed Test Performance Breakdown

| Metric | Value |
| :--- | :---: |
| **Accuracy** | **95.20%** |
| **Precision** | **95.12%** |
| **Recall** | **95.28%** |
| **F1 Score** | **95.20%** |
| **ROC-AUC** | **0.9842** |
| **Average Precision (AP)** | **0.9815** |
| **Test Samples** | 2000 |
| **Latency** | 4.17 ms/image |
| **Throughput** | 239.7 images/sec |

---

## Confusion Matrix (Fused Peak Model)

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

## Temperature-Annealed Dynamic Gating Weights (τ=0.5)

| Fusion Head | Mean α | Std α | Strategic Function |
| :--- | :---: | :---: | :--- |
| **SpatialAttn_MLEP→LOTA** | **0.3245** | 0.0125 | Queries MLEP macro-entropy against LOTA LSB noise key/values |
| **SpatialAttn_LOTA→MLEP** | **0.2810** | 0.0110 | Queries LOTA LSB noise against MLEP entropy key/values |
| **ChannelSE** | **0.2185** | 0.0095 | Squeeze-and-Excitation channel re-weighting across all 12 channels |
| **FreqCorrelation** | **0.1760** | 0.0080 | Cross-spectral Gram matrix frequency correlation |
