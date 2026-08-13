"""
Master Dual-Cue Fusion Model Assembly — Roadmap Alias.

Exposes HydraFusionNet as DualCueAIGIDModel and HydraFusionNet.
"""

from src.models.hydrafusion_net import HydraFusionNet
from src.models.hydrafusion_net import HydraFusionNet as DualCueAIGIDModel

__all__ = ["HydraFusionNet", "DualCueAIGIDModel"]
