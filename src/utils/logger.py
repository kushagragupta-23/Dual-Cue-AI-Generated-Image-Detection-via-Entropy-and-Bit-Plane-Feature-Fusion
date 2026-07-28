"""
Production logging utilities for tracking dataset metrics, LOTA extraction, and throughput.
"""

import logging
from pathlib import Path
import sys
from typing import Optional, Union


def get_logger(
    name: str = "lota_pipeline",
    log_dir: Optional[Union[str, Path]] = None,
    level: str = "INFO",
    log_filename: str = "execution.log",
) -> logging.Logger:
    """
    Initialize and return a configured logger with console and optional file output.

    Args:
        name: Name identifier for the logger.
        log_dir: Optional directory to save log files. If provided, writes to file.
        level: Logging severity level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_filename: Name of the log file within log_dir.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Resolve numeric logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    # Log formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_filename, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid double printing
    logger.propagate = False

    return logger
