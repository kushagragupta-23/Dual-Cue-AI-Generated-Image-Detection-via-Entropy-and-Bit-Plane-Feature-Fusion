"""
Fusion Model: DualCueAIGIDModel — Master End-to-End Assembly.

Integrates all specialized architectures into a single configurable module.
"""

from src.models.fusion.modules import CrossModalGatingFusionHead
from src.models.fusion.model import DualCueAIGIDModel

__all__ = [
    "CrossModalGatingFusionHead",
    "DualCueAIGIDModel",
]
