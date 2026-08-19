"""
Multi-granularity Local Entropy Pattern (MLEP) Preprocessing Core.
Re-exports centralized MLEPExtractor from HydraFusion/src/models/mlep_extractor.py.
"""

import sys
import importlib.util
from pathlib import Path

hydrafusion_mlep_path = Path(__file__).resolve().parent.parent.parent.parent / "HydraFusion" / "src" / "models" / "mlep_extractor.py"

if not hydrafusion_mlep_path.exists():
    raise FileNotFoundError(f"Centralized mlep_extractor file not found at {hydrafusion_mlep_path}")

spec = importlib.util.spec_from_file_location("hydrafusion_mlep_extractor", hydrafusion_mlep_path)
hf_mlep_mod = importlib.util.module_from_spec(spec)
sys.modules["hydrafusion_mlep_extractor"] = hf_mlep_mod
spec.loader.exec_module(hf_mlep_mod)

MLEPExtractor = hf_mlep_mod.MLEPExtractor

__all__ = ["MLEPExtractor"]
