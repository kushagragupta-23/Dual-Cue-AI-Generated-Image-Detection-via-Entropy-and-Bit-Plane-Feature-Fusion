"""
LOTA Extractor Module — Roadmap Alias.

Exposes TopKLOTAExtractor and LOTAExtractor.
"""

from src.models.lota_extractor import TopKLOTAExtractor
from src.models.lota_extractor import TopKLOTAExtractor as LOTAExtractor

__all__ = ["TopKLOTAExtractor", "LOTAExtractor"]
