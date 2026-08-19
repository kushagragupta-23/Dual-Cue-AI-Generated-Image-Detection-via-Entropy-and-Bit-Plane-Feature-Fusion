"""
Architecture I: Learnable Frequency-Domain Denoising & Supervised Contrastive
Cross-Modal Alignment (SupCon).

Standalone model for contrastive pre-training of MLEP and LOTA representations.
"""

from src.models.arch1_supcon.modules import (
    LearnableFrequencyPreFilter,
    ProjectionHead,
)
from src.models.arch1_supcon.model import LearnableFreqSupConNet
from src.shared.losses.supcon_loss import DualCueSupConLoss

__all__ = [
    "LearnableFrequencyPreFilter",
    "ProjectionHead",
    "LearnableFreqSupConNet",
    "DualCueSupConLoss",
]
