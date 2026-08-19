#!/usr/bin/env python3
"""
Entry point: Architecture II — MGA-Net Cross-Attention Training.

Usage:
    python scripts/arch2_mganet/train.py --backbone resnet50 --epochs 30
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.models.arch2_mganet.trainer import main

if __name__ == "__main__":
    main()
