"""
Evaluation Suite for Dual-Cue AI-Generated Image Detection.

Exports:
    - metrics: Academic metric calculators (Accuracy, AP, ROC-AUC, F1)
    - evaluator: Cross-generator evaluation loop & markdown compiler
    - explainability: Grad-CAM hooks and attention overlay generator
    - robustness_suite: JPEG compression and blur degradation evaluator
    - benchmark_throughput: GPU/Metal throughput scaling benchmarks
"""
from src.eval.metrics import *
from src.eval.evaluator import *
