"""
Device Selector Utility for MLEP.
Re-uses get_device from HydraFusion/src/utils/device.py.
"""

import sys
import importlib.util
from pathlib import Path

hydrafusion_device_path = Path(__file__).resolve().parent.parent.parent.parent / "HydraFusion" / "src" / "utils" / "device.py"

if not hydrafusion_device_path.exists():
    raise FileNotFoundError(f"Centralized device file not found at {hydrafusion_device_path}")

spec = importlib.util.spec_from_file_location("hydrafusion_device", hydrafusion_device_path)
hf_device_mod = importlib.util.module_from_spec(spec)
sys.modules["hydrafusion_device"] = hf_device_mod
spec.loader.exec_module(hf_device_mod)

get_device = hf_device_mod.get_device

__all__ = ["get_device"]
