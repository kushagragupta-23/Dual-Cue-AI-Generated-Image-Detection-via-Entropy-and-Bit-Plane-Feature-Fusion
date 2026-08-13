import logging
import sys
from pathlib import Path

def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """Configure and return a standard logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path / f"{name}.log")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
