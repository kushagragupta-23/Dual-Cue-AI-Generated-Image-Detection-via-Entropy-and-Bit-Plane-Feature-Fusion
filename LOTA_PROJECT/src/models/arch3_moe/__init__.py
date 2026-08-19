"""
Architecture III: Sparse Mixture-of-Experts (MoE) & Domain-Adversarial
Generalization (DANN).

Standalone model with Top-2 dynamic expert routing and adversarial domain unlearning.
"""

from src.models.arch3_moe.modules import (
    ExpertModule,
    SparseMoEForensicModule,
    GradientReversalFunction,
    GradientReversalLayer,
    DomainDiscriminator,
)
from src.models.arch3_moe.model import (
    DomainAdversarialMoEDetector,
    MoEStandaloneDualCueDetector,
)

__all__ = [
    "ExpertModule",
    "SparseMoEForensicModule",
    "GradientReversalFunction",
    "GradientReversalLayer",
    "DomainDiscriminator",
    "DomainAdversarialMoEDetector",
    "MoEStandaloneDualCueDetector",
]
