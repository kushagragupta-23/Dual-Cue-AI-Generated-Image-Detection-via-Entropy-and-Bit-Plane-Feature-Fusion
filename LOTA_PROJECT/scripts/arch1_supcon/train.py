#!/usr/bin/env python3
"""
Entry point: Architecture I — SupCon Contrastive Pre-Training.

Usage:
    python scripts/arch1_supcon/train.py --backbone resnet50 --epochs 50
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.models.arch1_supcon.trainer import main

if __name__ == "__main__":
    main()
