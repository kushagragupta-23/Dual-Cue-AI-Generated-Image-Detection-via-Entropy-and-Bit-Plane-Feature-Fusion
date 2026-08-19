"""
Standard Logger Utility for MLEP.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

def get_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = "outputs/logs",
    level: str = "INFO",
    log_filename: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_dir:
            try:
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                fname = log_filename or f"{name}.log"
                fh = logging.FileHandler(log_path / fname)
                fh.setLevel(log_level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception:
                pass
    return logger

__all__ = ["get_logger"]
