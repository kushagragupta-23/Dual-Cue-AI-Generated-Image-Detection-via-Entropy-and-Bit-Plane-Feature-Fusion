"""
Transforms for MLEP standalone execution.
Re-uses centralized transforms from HydraFusion.
"""

import sys
import importlib.util
from pathlib import Path

hydrafusion_transforms_path = Path(__file__).resolve().parent.parent.parent.parent / "HydraFusion" / "src" / "data" / "transforms.py"

if not hydrafusion_transforms_path.exists():
    raise FileNotFoundError(f"Centralized transforms file not found at {hydrafusion_transforms_path}")

spec = importlib.util.spec_from_file_location("hydrafusion_transforms", hydrafusion_transforms_path)
hf_transforms_mod = importlib.util.module_from_spec(spec)
sys.modules["hydrafusion_transforms"] = hf_transforms_mod
spec.loader.exec_module(hf_transforms_mod)

MLEPPreprocessingTransform = getattr(hf_transforms_mod, 'MLEPPreprocessingTransform', None)

__all__ = ["MLEPPreprocessingTransform"]
