"""Configuration loader and management module for EdgeAI ECG Monitoring pipeline.

Loads, validates, and provides strongly-typed configuration settings from config.yaml.
Governed by PRD.md and AGENTS.md rules.
"""

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, Dict, Optional
import yaml


@dataclass(frozen=True)
class IngestionConfig:
    """Ingestion configuration parameters."""

    window_size: int
    overlap_ratio: float
    default_sample_rate: float  # Hz


@dataclass(frozen=True)
class DSPConfig:
    """DSP filter configuration parameters."""

    filter_order: int
    low_cutoff: float  # Hz
    high_cutoff: float  # Hz


@dataclass(frozen=True)
class SchedulerConfig:
    """State scheduler configuration parameters."""

    threshold: float
    default_state: str
    active_state: str


@dataclass(frozen=True)
class ModelConfig:
    """Machine learning model and distillation configuration parameters."""

    temperature: float
    alpha: float
    random_seed: int
    num_classes: int


@dataclass(frozen=True)
class BudgetConfig:
    """Resource budget constraints (simulated limits)."""

    sram_limit_kb: float  # KB (256 KB max per NFR-1)
    flash_limit_mb: float  # MB (1.0 MB max per NFR-2)
    latency_limit_ms: float  # ms (50.0 ms max per NFR-3)


@dataclass(frozen=True)
class TelemetryConfig:
    """Secure telemetry configuration parameters."""

    encryption_algorithm: str
    key_size_bits: int


@dataclass(frozen=True)
class PathConfig:
    """Directory paths for data, reports, and models."""

    data_dir: Path
    reports_dir: Path
    models_dir: Path


class PipelineConfig:
    """Main configuration class that parses config.yaml and validates settings."""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """Initialize PipelineConfig from dictionary.

        Args:
            config_dict: Dictionary containing parsed configuration values.
        """
        self._raw_config = config_dict
        self._validate_and_build()

    def _validate_and_build(self) -> None:
        """Validate all required fields and construct dataclass instances."""
        try:
            ingestion = self._raw_config["ingestion"]
            self.ingestion = IngestionConfig(
                window_size=int(ingestion["window_size"]),
                overlap_ratio=float(ingestion["overlap_ratio"]),
                default_sample_rate=float(ingestion["default_sample_rate"]),
            )

            dsp = self._raw_config["dsp"]
            self.dsp = DSPConfig(
                filter_order=int(dsp["filter_order"]),
                low_cutoff=float(dsp["low_cutoff"]),
                high_cutoff=float(dsp["high_cutoff"]),
            )

            scheduler = self._raw_config["scheduler"]
            self.scheduler = SchedulerConfig(
                threshold=float(scheduler["threshold"]),
                default_state=str(scheduler["default_state"]),
                active_state=str(scheduler["active_state"]),
            )

            model = self._raw_config["model"]
            self.model = ModelConfig(
                temperature=float(model["temperature"]),
                alpha=float(model["alpha"]),
                random_seed=int(model["random_seed"]),
                num_classes=int(model["num_classes"]),
            )

            budgets = self._raw_config["budgets"]
            self.budgets = BudgetConfig(
                sram_limit_kb=float(budgets["sram_limit_kb"]),
                flash_limit_mb=float(budgets["flash_limit_mb"]),
                latency_limit_ms=float(budgets["latency_limit_ms"]),
            )

            telemetry = self._raw_config["telemetry"]
            self.telemetry = TelemetryConfig(
                encryption_algorithm=str(telemetry["encryption_algorithm"]),
                key_size_bits=int(telemetry["key_size_bits"]),
            )

            paths = self._raw_config.get("paths", {})
            self.paths = PathConfig(
                data_dir=Path(paths.get("data_dir", "data")),
                reports_dir=Path(paths.get("reports_dir", "reports")),
                models_dir=Path(paths.get("models_dir", "models")),
            )

            self._validate_constraints()

        except KeyError as err:
            raise ValueError(f"Missing required configuration key: {err}") from err

    def _validate_constraints(self) -> None:
        """Validate logical boundary constraints on configuration parameters."""
        if self.ingestion.window_size <= 0:
            raise ValueError("window_size must be positive integer")
        if not (0.0 <= self.ingestion.overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be between 0.0 and 1.0")
        if self.dsp.low_cutoff >= self.dsp.high_cutoff:
            raise ValueError("dsp low_cutoff must be strictly less than high_cutoff")
        if not (0.0 <= self.scheduler.threshold <= 1.0):
            raise ValueError("scheduler threshold must be between 0.0 and 1.0")
        if self.budgets.sram_limit_kb > 256.0:
            raise ValueError("sram_limit_kb exceeds hard ceiling of 256 KB")
        if self.budgets.flash_limit_mb > 1.0:
            raise ValueError("flash_limit_mb exceeds hard ceiling of 1 MB")
        if self.budgets.latency_limit_ms > 50.0:
            raise ValueError("latency_limit_ms exceeds budget of 50 ms")


def load_config(config_path: Optional[str | Path] = None) -> PipelineConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to YAML config file. Defaults to config/config.yaml.

    Returns:
        PipelineConfig instance.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        ValueError: If YAML parsing or validation fails.
    """
    if config_path is None:
        # Default path relative to project root
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Invalid YAML structure in {config_path}")

    return PipelineConfig(raw_data)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging for the pipeline.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger("edgeai_ecg")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
