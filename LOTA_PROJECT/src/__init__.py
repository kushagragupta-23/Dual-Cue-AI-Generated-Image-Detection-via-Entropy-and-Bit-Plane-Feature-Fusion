"""
Dual-Cue AI-Generated Image Detection (MLEP & LOTA Fusion) package.
Self-contained — no external project dependencies required.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for clean imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
