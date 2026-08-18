# Robustness Evaluation Results

> Systematic degradation analysis of HydraFusion-Net under JPEG recompression and Gaussian blur perturbations. Evaluated on 2,000 test images from `dataset10000`.

### JPEG Recompression Robustness

| Condition | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC | AP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| JPEG Q=100 (No compression) | 90.20 | 89.26 | 91.40 | 90.32 | 0.9576 | 0.9466 |
| JPEG Q=90 | 90.80 | 89.92 | 91.90 | 90.90 | 0.9590 | 0.9477 |
| JPEG Q=80 | 90.55 | 89.72 | 91.60 | 90.65 | 0.9584 | 0.9451 |
| JPEG Q=70 | 90.60 | 89.57 | 91.90 | 90.72 | 0.9575 | 0.9455 |

### Gaussian Blur Robustness

| Condition | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC | AP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Blur σ=0.5 | 90.40 | 88.33 | 93.10 | 90.65 | 0.9559 | 0.9385 |
| Blur σ=1.0 | 89.00 | 84.09 | 96.20 | 89.74 | 0.9454 | 0.9140 |
| Blur σ=2.0 | 79.55 | 71.49 | 98.30 | 82.78 | 0.8826 | 0.8179 |

### Combined Degradation

| Condition | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC | AP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Combined (Q=80, σ=1.0) | 88.70 | 83.65 | 96.20 | 89.49 | 0.9465 | 0.9148 |

### Key Observations

1. **JPEG Resilience**: Performance remains stable (±0.6% accuracy) across Q∈{70,80,90,100}, demonstrating the frequency prefilter successfully decouples entropy computation from JPEG blockiness artifacts.
2. **Blur Sensitivity**: Performance degrades significantly at σ≥2.0 (−10.65% accuracy), as heavy Gaussian blur destroys the fine-grained LSB noise patterns that LOTA depends on.
3. **Combined Degradation**: The model retains 88.70% accuracy under simultaneous JPEG Q=80 + Blur σ=1.0, a realistic social media sharing scenario.