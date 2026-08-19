#!/usr/bin/env python3
"""
Entry point: Architecture III — MoE + DANN Training.

Usage:
    python scripts/arch3_moe/train.py --backbone resnet50 --epochs 30
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.models.arch3_moe.trainer import main

if __name__ == "__main__":
    main()
