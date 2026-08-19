"""
Logger Utility for MLEP.
Re-uses get_logger from HydraFusion/src/utils/logger.py.
"""

import sys
import importlib.util
from pathlib import Path

hydrafusion_logger_path = Path(__file__).resolve().parent.parent.parent.parent / "HydraFusion" / "src" / "utils" / "logger.py"

if not hydrafusion_logger_path.exists():
    raise FileNotFoundError(f"Centralized logger file not found at {hydrafusion_logger_path}")

spec = importlib.util.spec_from_file_location("hydrafusion_logger", hydrafusion_logger_path)
hf_logger_mod = importlib.util.module_from_spec(spec)
sys.modules["hydrafusion_logger"] = hf_logger_mod
spec.loader.exec_module(hf_logger_mod)

get_logger = hf_logger_mod.get_logger

__all__ = ["get_logger"]
