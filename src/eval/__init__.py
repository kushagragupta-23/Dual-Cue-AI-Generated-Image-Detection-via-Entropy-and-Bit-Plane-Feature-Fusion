"""
HydraFusion-Net Evaluation Package.

Provides:
  - metrics: Academic metric calculators (Accuracy, AP, ROC-AUC, F1, Confusion Matrices)
  - evaluator: Automated evaluation loop & report compiler
  - explainability: Grad-CAM saliency hooks & attention overlay generator
  - robustness_suite: JPEG compression & blur degradation evaluator
"""

from src.eval.metrics import ForensicMetricsCalculator, MetricsResult, format_metrics_table
from src.eval.evaluator import HydraFusionEvaluator
from src.eval.explainability import GradCAM, DualBranchExplainer
from src.eval.robustness_suite import RobustnessSuite
