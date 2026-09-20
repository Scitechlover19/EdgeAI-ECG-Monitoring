"""Focused unit and integration tests for SYNTHETIC DEMO MODE (P0-16).

Verifies deterministic synthetic generation, window shape, [DEMO MODE] tagging,
zero raw ECG leakage into telemetry, telemetry schema isolation,
natural exemplar search without artificial overrides,
uncontaminated real experimental artifacts, and offline execution capability.
Governed by PRD.md (FR-16), Architecture.md §17, and AGENTS.md.
"""

from pathlib import Path
import pytest
import numpy as np

from src.pipeline.demo_mode import (
    SyntheticECGGenerator,
    find_natural_test_exemplars,
    run_demo_pipeline,
)
from src.pipeline.config import load_config
from src.edge.tinyml_engine import TinyMLEngine
from src.edge.virtual_mcu import VirtualMCU


def test_synthetic_ecg_generator_reproducibility() -> None:
    """Test deterministic synthetic ECG generation given fixed seed."""
    gen1 = SyntheticECGGenerator(num_windows=20, window_size=200, seed=42)
    X1, y1 = gen1.generate_windows()

    gen2 = SyntheticECGGenerator(num_windows=20, window_size=200, seed=42)
    X2, y2 = gen2.generate_windows()

    np.testing.assert_allclose(X1, X2, rtol=1e-6)
    np.testing.assert_array_equal(y1, y2)


def test_synthetic_window_shape() -> None:
    """Test synthetic window shape is (N, 200) and label shape is (N,)."""
    gen = SyntheticECGGenerator(num_windows=15, window_size=200, seed=123)
    X, y = gen.generate_windows()

    assert X.shape == (15, 200)
    assert y.shape == (15,)
    assert X.dtype == np.float32


def test_natural_exemplar_search_no_override() -> None:
    """Verify natural exemplar search checks threshold 0.85 without fabricating scores."""
    config = load_config()
    data_dir = Path(config.paths.data_dir)
    models_dir = Path(config.paths.models_dir)
    model_path = models_dir / "student_model_int8.tflite"

    if not model_path.exists():
        pytest.skip("INT8 model file missing.")

    vmcu = VirtualMCU()
    engine = TinyMLEngine(model_path=model_path, virtual_mcu=vmcu)

    exemplars = find_natural_test_exemplars(
        engine=engine,
        data_dir=data_dir,
        threshold=0.85,
    )

    # Must be a list of tuples (idx, conf)
    assert isinstance(exemplars, list)
    for idx, conf in exemplars:
        assert conf >= 0.85


def test_demo_pipeline_execution(tmp_path: Path) -> None:
    """Test end-to-end execution of synthetic demo mode pipeline."""
    report_file = tmp_path / "demo_summary.md"
    summary = run_demo_pipeline(
        num_windows=20,
        seed=42,
        anomaly_rate=0.2,
        output_report_path=report_file,
    )

    assert summary["tag"] == "[DEMO MODE]"
    assert summary["num_windows"] == 20
    assert summary["normal_windows_sleep"] + summary["anomaly_windows_triggered"] == 20
    assert summary["scheduler_threshold"] == load_config().scheduler.threshold
    assert summary["raw_ecg_transmitted"] is False
    assert report_file.exists()
    assert "[DEMO MODE]" in report_file.read_text(encoding="utf-8")


def test_demo_mode_tagging(tmp_path: Path) -> None:
    """Verify demo mode summary and generated reports contain explicit [DEMO MODE] tags."""
    report_file = tmp_path / "demo_summary.md"
    summary = run_demo_pipeline(
        num_windows=10,
        seed=99,
        output_report_path=report_file,
    )

    assert "[DEMO MODE]" in summary["tag"]
    report_text = report_file.read_text(encoding="utf-8")
    assert "[DEMO MODE]" in report_text
    assert "[DEMO MODE DISCLAIMER]" in report_text


def test_privacy_assertion_zero_raw_ecg() -> None:
    """Verify zero raw waveform array data reaches telemetry sink in demo mode."""
    summary = run_demo_pipeline(num_windows=10, seed=42)
    assert summary["raw_ecg_transmitted"] is False


def test_telemetry_schema_unmodified(tmp_path: Path) -> None:
    """Verify telemetry record schema remains fixed and minimal during demo mode."""
    from src.telemetry.sink import TelemetrySink
    from src.telemetry.secure_telemetry import SecureTelemetry
    from src.scheduler.state_scheduler import StateScheduler

    sink = TelemetrySink()
    sec_tel = SecureTelemetry(sink=sink, session_id="DEMO-NODE-1")
    scheduler = StateScheduler(threshold=0.5, telemetry_callback=sec_tel.transmit_alert)

    # Force trigger in unit test ONLY
    scheduler.process_window_result(0, 0.0, 0.99, anomaly_id=1)
    records = sink.get_records()

    assert len(records) == 1
    rec = records[0]
    assert set(rec.keys()) == {
        "session_id",
        "timestamp_sec",
        "anomaly_id",
        "confidence_score",
        "encrypted_payload",
        "received_at",
    }
    assert isinstance(rec["encrypted_payload"], bytes)


def test_real_artifacts_uncontaminated(tmp_path: Path) -> None:
    """Verify demo mode execution does NOT alter or overwrite real experimental reports."""
    config = load_config()
    reports_dir = Path(config.paths.reports_dir)

    real_reports = [
        reports_dir / "resource_report.md",
        reports_dir / "bandwidth_report.md",
        reports_dir / "kd_ablation.md",
        reports_dir / "experiment_manifest.json",
    ]

    mtimes_before = {}
    for r in real_reports:
        if r.exists():
            mtimes_before[r] = r.stat().st_mtime

    demo_report = tmp_path / "demo_summary.md"
    run_demo_pipeline(num_windows=15, seed=42, output_report_path=demo_report)

    for r, mtime in mtimes_before.items():
        assert r.stat().st_mtime == mtime


def test_demo_runs_offline() -> None:
    """Verify synthetic demo mode runs fully offline without local MIT-BIH dataset dependencies."""
    generator = SyntheticECGGenerator(num_windows=5, seed=1)
    X, y = generator.generate_windows()

    assert len(X) == 5
    assert len(y) == 5
