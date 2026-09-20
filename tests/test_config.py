"""Unit tests for configuration loading and validation (P0-1)."""

from pathlib import Path
import pytest

from src.pipeline.config import load_config, PipelineConfig, setup_logging


def test_load_default_config() -> None:
    """Test loading default config/config.yaml and verifying required values."""
    config = load_config()

    # Ingestion checks (DECISIONS.md #1)
    assert config.ingestion.window_size == 200
    assert config.ingestion.overlap_ratio == 0.5
    assert config.ingestion.default_sample_rate == 360.0

    # DSP checks (DECISIONS.md #2)
    assert config.dsp.filter_order == 4
    assert config.dsp.low_cutoff == 0.5
    assert config.dsp.high_cutoff == 45.0

    # Scheduler checks (DECISIONS.md #13, #15)
    assert config.scheduler.threshold == 0.35
    assert config.scheduler.default_state == "SLEEP"
    assert config.scheduler.active_state == "ACTIVE"

    # Model & KD checks (DECISIONS.md #6)
    assert config.model.temperature == 3.0
    assert config.model.alpha == 0.7
    assert config.model.random_seed == 42
    assert config.model.num_classes == 2

    # Resource budgets (PRD NFR-1, NFR-2, NFR-3)
    assert config.budgets.sram_limit_kb == 256.0
    assert config.budgets.flash_limit_mb == 1.0
    assert config.budgets.latency_limit_ms == 50.0

    # Telemetry
    assert config.telemetry.encryption_algorithm == "AES-GCM"
    assert config.telemetry.key_size_bits == 256


def test_config_validation_failures() -> None:
    """Test that invalid configuration options raise appropriate ValueErrors."""
    base_raw = {
        "ingestion": {"window_size": 200, "overlap_ratio": 0.5, "default_sample_rate": 360.0},
        "dsp": {"filter_order": 4, "low_cutoff": 0.5, "high_cutoff": 45.0},
        "scheduler": {"threshold": 0.85, "default_state": "SLEEP", "active_state": "ACTIVE"},
        "model": {"temperature": 3.0, "alpha": 0.7, "random_seed": 42, "num_classes": 2},
        "budgets": {"sram_limit_kb": 256.0, "flash_limit_mb": 1.0, "latency_limit_ms": 50.0},
        "telemetry": {"encryption_algorithm": "AES-GCM", "key_size_bits": 256},
    }

    # Invalid window size
    bad_ingestion = dict(base_raw)
    bad_ingestion["ingestion"] = {"window_size": -10, "overlap_ratio": 0.5, "default_sample_rate": 360.0}
    with pytest.raises(ValueError, match="window_size must be positive"):
        PipelineConfig(bad_ingestion)

    # Low cutoff >= high cutoff
    bad_dsp = dict(base_raw)
    bad_dsp["dsp"] = {"filter_order": 4, "low_cutoff": 50.0, "high_cutoff": 45.0}
    with pytest.raises(ValueError, match="low_cutoff must be strictly less than high_cutoff"):
        PipelineConfig(bad_dsp)

    # SRAM ceiling exceeded
    bad_sram = dict(base_raw)
    bad_sram["budgets"] = {"sram_limit_kb": 512.0, "flash_limit_mb": 1.0, "latency_limit_ms": 50.0}
    with pytest.raises(ValueError, match="sram_limit_kb exceeds hard ceiling"):
        PipelineConfig(bad_sram)


def test_logging_setup() -> None:
    """Test logging setup initializes without errors."""
    logger = setup_logging()
    assert logger is not None
    assert logger.name == "edgeai_ecg"
