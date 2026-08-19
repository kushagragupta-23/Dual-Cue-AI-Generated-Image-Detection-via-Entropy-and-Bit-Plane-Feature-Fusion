"""
LOTA & MLEP Fusion Model Architecture Package.

Master registry that re-exports all model components from their new sub-package
locations. This provides full backward compatibility — existing code that does
`from src.models import TopKLOTAExtractor` will continue to work.

Sub-packages:
    src.models.arch1_supcon/   — Architecture I: SupCon Pre-Training
    src.models.arch2_mganet/   — Architecture II: MGA-Net Cross-Attention
    src.models.arch3_moe/      — Architecture III: MoE + DANN
    src.models.fusion/         — Master Fusion Model
    src.models.legacy/         — Phase 1 models (DualCueClassifier, LOTAClassifier)

Shared components:
    src.shared.extractors/     — LOTA and MLEP feature extractors
    src.shared.backbones/      — Channel-adapted ResNet stems
    src.shared.losses/         — Shared loss functions (SupCon)
"""

# ── Shared: Feature Extractors ──
from src.shared.extractors import TopKLOTAExtractor, VectorizedMLEPExtractor

# ── Shared: Backbones ──
from src.shared.backbones import ChannelAdaptedResNet, DualStemBackbone

# ── Shared: Losses ──
from src.shared.losses import DualCueSupConLoss

# ── Architecture I: SupCon ──
from src.models.arch1_supcon.modules import (
    LearnableFrequencyPreFilter,
    ProjectionHead,
)
from src.models.arch1_supcon.model import LearnableFreqSupConNet

# ── Architecture II: MGA-Net ──
from src.models.arch2_mganet.modules import PyramidCrossAttentionModule
from src.models.arch2_mganet.model import MGANetDualCueDetector

# ── Architecture III: MoE + DANN ──
from src.models.arch3_moe.modules import (
    ExpertModule,
    SparseMoEForensicModule,
    GradientReversalLayer,
    DomainDiscriminator,
)
from src.models.arch3_moe.model import (
    DomainAdversarialMoEDetector,
    MoEStandaloneDualCueDetector,
)

# ── Fusion Model ──
from src.models.fusion.modules import CrossModalGatingFusionHead
from src.models.fusion.model import DualCueAIGIDModel

# ── Legacy Models ──
from src.models.legacy import LOTAClassifier, DualCueClassifier


__all__ = [
    # Shared Extractors
    "TopKLOTAExtractor",
    "VectorizedMLEPExtractor",
    # Shared Backbones
    "ChannelAdaptedResNet",
    "DualStemBackbone",
    # Shared Losses
    "DualCueSupConLoss",
    # Architecture I — SupCon
    "LearnableFrequencyPreFilter",
    "ProjectionHead",
    "LearnableFreqSupConNet",
    # Architecture II — MGA-Net
    "PyramidCrossAttentionModule",
    "MGANetDualCueDetector",
    # Architecture III — MoE + DANN
    "ExpertModule",
    "SparseMoEForensicModule",
    "GradientReversalLayer",
    "DomainDiscriminator",
    "DomainAdversarialMoEDetector",
    "MoEStandaloneDualCueDetector",
    # Fusion
    "CrossModalGatingFusionHead",
    "DualCueAIGIDModel",
    # Legacy
    "LOTAClassifier",
    "DualCueClassifier",
]
