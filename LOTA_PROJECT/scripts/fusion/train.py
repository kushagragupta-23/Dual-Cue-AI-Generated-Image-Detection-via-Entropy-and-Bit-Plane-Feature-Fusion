#!/usr/bin/env python3
"""
Entry point: Fusion Model — 2-Stage Training (SupCon → Gated Fine-Tuning).

Usage:
    python scripts/fusion/train.py --backbone resnet50 --epochs-stage1 50 --epochs-stage2 30
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.models.fusion.trainer import main

if __name__ == "__main__":
    main()
