"""
AI-Generated Image Detection (MLEP) package.
Re-uses centralized core modules from HydraFusion.
"""

import sys
from pathlib import Path

# Ensure HydraFusion root is in sys.path so shared modules are re-used
hydrafusion_root = Path(__file__).resolve().parent.parent.parent / "HydraFusion"
if hydrafusion_root.exists() and str(hydrafusion_root) not in sys.path:
    sys.path.insert(0, str(hydrafusion_root))
