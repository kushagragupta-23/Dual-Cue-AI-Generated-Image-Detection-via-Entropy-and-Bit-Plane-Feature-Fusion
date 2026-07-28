"""
Unit tests for configuration loading and logger utilities.
"""

import logging
from pathlib import Path
import pytest
import yaml
from src.utils.config import (
    DatasetConfig,
    LOTAConfig,
    LoggingConfig,
    ProjectConfig,
    load_config,
    save_config,
)
from src.utils.logger import get_logger


def test_default_project_config():
    """Verify default initializations of ProjectConfig dataclasses."""
    cfg = ProjectConfig()
    assert cfg.dataset.image_size == 256
    assert cfg.dataset.batch_size == 32
    assert cfg.lota.bit_planes == [0, 1, 2]
    assert cfg.lota.k_patches == 4
    assert cfg.lota.grid_size == 8
    assert cfg.logging.level == "INFO"


def test_save_and_load_config(tmp_path: Path):
    """Verify round-trip YAML serialization and deserialization of ProjectConfig."""
    cfg = ProjectConfig()
    cfg.dataset.image_size = 512
    cfg.lota.k_patches = 8
    
    config_file = tmp_path / "test_config.yaml"
    save_config(cfg, config_file)
    
    assert config_file.exists()
    
    loaded_cfg = load_config(config_file)
    assert isinstance(loaded_cfg, ProjectConfig)
    assert loaded_cfg.dataset.image_size == 512
    assert loaded_cfg.lota.k_patches == 8


def test_load_nonexistent_config():
    """Assert FileNotFoundError is raised when config path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_path_to_config.yaml")


def test_logger_initialization(tmp_path: Path):
    """Verify get_logger configures handlers correctly and writes to disk."""
    log_dir = tmp_path / "test_logs"
    logger = get_logger(
        name="test_logger",
        log_dir=log_dir,
        level="DEBUG",
        log_filename="test.log",
    )
    
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) >= 2  # StreamHandler and FileHandler
    
    logger.debug("Test debug message")
    logger.info("Test info message")
    
    log_file = log_dir / "test.log"
    assert log_file.exists()
    
    content = log_file.read_text(encoding="utf-8")
    assert "Test debug message" in content
    assert "Test info message" in content


def test_logger_no_duplicate_handlers(tmp_path: Path):
    """Verify calling get_logger repeatedly with same name does not duplicate handlers."""
    logger1 = get_logger(name="dup_test_logger", log_dir=tmp_path)
    handler_count = len(logger1.handlers)
    
    logger2 = get_logger(name="dup_test_logger", log_dir=tmp_path)
    assert len(logger2.handlers) == handler_count
    assert logger1 is logger2
