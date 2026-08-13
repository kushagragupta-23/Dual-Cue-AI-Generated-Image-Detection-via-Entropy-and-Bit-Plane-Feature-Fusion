"""
Architecture III: Adaptive Gating Router & Domain Adversarial Head — Roadmap Alias.

Exposes AdaptiveGatingRouter, GradientReversalLayer, and DomainAdversarialHead.
"""

from src.models.gating_router import AdaptiveGatingRouter, SparseMoEForensicModule, ExpertModule
from src.models.domain_adversarial import GradientReversalLayer, DomainAdversarialHead

__all__ = [
    "AdaptiveGatingRouter",
    "SparseMoEForensicModule",
    "ExpertModule",
    "GradientReversalLayer",
    "DomainAdversarialHead",
]
