"""
Backbone Adapters: Channel-adapted ResNet stems for MLEP and LOTA branches.
"""

from src.shared.backbones.resnet_adapter import ChannelAdaptedResNet, DualStemBackbone

__all__ = ["ChannelAdaptedResNet", "DualStemBackbone"]
