# HydraFusion-Net — Zero-Shot Evaluation Report

> **Note**: These results are from **zero-shot evaluation** using `evaluate_zeroshot.py` on a pre-trained checkpoint. For the best trained model results (95.20% accuracy), see `metrics.json`.

## Summary Metrics

| Metric | Value |
| :--- | :---: |
| **Accuracy** | 90.20% |
| **Precision** | 89.26% |
| **Recall** | 91.40% |
| **F1 Score** | 90.32% |
| **ROC-AUC** | 0.9576 |
| **Average Precision (AP)** | 0.9466 |
| **Test Samples** | 2000 |
| **Latency** | 5.68 ms/image |
| **Throughput** | 176.0 images/sec |

## Confusion Matrix

### Raw Counts
| | Predicted Real | Predicted Fake |
| :--- | :---: | :---: |
| **Actual Real** | 890 | 110 |
| **Actual Fake** | 86 | 914 |

### Normalized (%)
| | Predicted Real | Predicted Fake |
| :--- | :---: | :---: |
| **Actual Real** | 89.0% | 11.0% |
| **Actual Fake** | 8.6% | 91.4% |

## Gating Weight Distribution

| Fusion Head | Mean α | Std α |
| :--- | :---: | :---: |
| SpatialAttn_MLEP→LOTA | 0.9990 | 0.0008 |
| SpatialAttn_LOTA→MLEP | 0.0000 | 0.0000 |
| ChannelSE | 0.0001 | 0.0001 |
| FreqCorrelation | 0.0009 | 0.0007 |

> **Observation**: Gating weights show collapse to Head 1 (SpatialAttn MLEP→LOTA) in this evaluation run. The trained model with temperature-annealed routing achieves balanced weights `[0.3245, 0.2810, 0.2185, 0.1760]` as reported in `metrics.json`.
