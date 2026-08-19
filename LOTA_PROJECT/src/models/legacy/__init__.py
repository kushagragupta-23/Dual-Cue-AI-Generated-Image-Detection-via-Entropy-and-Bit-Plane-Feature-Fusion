"""
Legacy Models: Phase 1 architectures (DualCueClassifier, LOTAClassifier).

These are the original pre-fusion models kept for backward compatibility
and baseline comparison.
"""

from src.models.legacy.classifier import LOTAClassifier
from src.models.legacy.dual_cue import DualCueClassifier

__all__ = ["LOTAClassifier", "DualCueClassifier"]
