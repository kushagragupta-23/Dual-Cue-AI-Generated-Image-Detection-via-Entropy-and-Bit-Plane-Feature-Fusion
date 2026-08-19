"""
Configuration utilities and typed dataclasses for the MLEP preprocessing pipeline.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class DatasetConfig:
    """Configuration for data loading, transformations, and augmentation hooks."""
    data_dir: str = "dataset10000"
    image_size: int = 256
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    # Online robustness augmentation hooks
    enable_augmentations: bool = False
    jpeg_quality_min: int = 70
    jpeg_quality_max: int = 100
    blur_sigma_min: float = 0.5
    blur_sigma_max: float = 2.0





@dataclass
class MLEPConfig:
    """Configuration for Multi-granularity Local Entropy Pattern (MLEP) extraction."""
    patch_size: int = 2
    scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    window_size: int = 2
    seed: int = 42


@dataclass
class LoggingConfig:
    """Configuration for logging and experiment output tracking."""
    log_dir: str = "logs"
    level: str = "INFO"
    save_file: bool = True


@dataclass
class ProjectConfig:
    """Master project configuration bundling dataset, MLEP, and logging settings."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    mlep: MLEPConfig = field(default_factory=MLEPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration dataclass to dictionary."""
        return asdict(self)


def _dict_to_config(data: Dict[str, Any]) -> ProjectConfig:
    """Helper to convert nested dictionary into ProjectConfig dataclass."""
    dataset_cfg = DatasetConfig(**data.get("dataset", {})) if "dataset" in data else DatasetConfig()
    mlep_cfg = MLEPConfig(**data.get("mlep", {})) if "mlep" in data else MLEPConfig()
    logging_cfg = LoggingConfig(**data.get("logging", {})) if "logging" in data else LoggingConfig()
    return ProjectConfig(dataset=dataset_cfg, mlep=mlep_cfg, logging=logging_cfg)


def load_config(config_path: Union[str, Path]) -> ProjectConfig:
    """
    Load project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ProjectConfig: Loaded and typed project configuration object.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return _dict_to_config(data)


def save_config(config: ProjectConfig, save_path: Union[str, Path]) -> None:
    """
    Save project configuration to a YAML file.

    Args:
        config: ProjectConfig instance to save.
        save_path: Destination path for the YAML file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
