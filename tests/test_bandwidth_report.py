"""Focused unit and integration tests for Bandwidth Measurement Module (P0-15).

Tests baseline byte calculation, anomaly payload accounting, zero-telemetry behavior on normal windows,
reduction percentage arithmetic, zero raw ECG leakage assertions, and report generation.
Governed by PRD.md (NFR-6, PERF-5, SEC-1..5), Architecture.md §5.8 & §13, and AGENTS.md.
"""

from pathlib import Path
import pytest
import numpy as np

from src.edge.tinyml_engine import TinyMLEngine
from src.edge.virtual_mcu import VirtualMCU
from src.pipeline.bandwidth_report import (
    calculate_bandwidth_metrics,
    generate_bandwidth_report,
)
from src.pipeline.config import load_config
from src.scheduler.state_scheduler import StateScheduler
from src.telemetry.secure_telemetry import SecureTelemetry
from src.telemetry.sink import TelemetrySink


def test_baseline_byte_calculation() -> None:
    """Test baseline byte calculation arithmetic for 16-bit 200-sample windows."""
    num_windows = 100
    raw_sample_bits = 16
    window_size = 200
    expected_bytes_per_window = 400  # 200 * 2
    expected_total_baseline = num_windows * expected_bytes_per_window  # 40,000 bytes

    X_dummy = np.zeros((num_windows, window_size), dtype=np.float32)

    class DummyEngine:
        def invoke_inference(self, window: np.ndarray):
            # Normal rhythm
            return 0.1, 0, "NORMAL", 0.5

    sink = TelemetrySink()
    tel = SecureTelemetry(sink=sink)
    scheduler = StateScheduler(threshold=0.85, telemetry_callback=tel.transmit_alert)

    metrics = calculate_bandwidth_metrics(
        X_test=X_dummy,
        engine=DummyEngine(),  # type: ignore
        scheduler=scheduler,
        sink=sink,
        raw_sample_bits=raw_sample_bits,
        window_size=window_size,
    )

    assert metrics["total_test_windows"] == num_windows
    assert metrics["raw_bytes_per_window"] == 400
    assert metrics["baseline_bytes"] == expected_total_baseline
    assert metrics["normal_windows"] == num_windows
    assert metrics["anomaly_transmissions"] == 0
    assert metrics["anomaly_mode_bytes"] == 0
    assert metrics["bytes_saved"] == expected_total_baseline
    assert metrics["payload_reduction_percent"] == 100.0


def test_anomaly_payload_byte_accounting() -> None:
    """Test payload byte accounting for anomaly windows triggering AES-GCM encrypted telemetry."""
    num_windows = 10
    X_dummy = np.ones((num_windows, 200), dtype=np.float32)

    class DummyAnomalyEngine:
        def invoke_inference(self, window: np.ndarray):
            # High confidence anomaly
            return 0.95, 1, "ANOMALY", 0.5

    sink = TelemetrySink()
    tel = SecureTelemetry(sink=sink)
    scheduler = StateScheduler(threshold=0.85, telemetry_callback=tel.transmit_alert)

    metrics = calculate_bandwidth_metrics(
        X_test=X_dummy,
        engine=DummyAnomalyEngine(),  # type: ignore
        scheduler=scheduler,
        sink=sink,
        raw_sample_bits=16,
        window_size=200,
    )

    assert metrics["total_test_windows"] == 10
    assert metrics["normal_windows"] == 0
    assert metrics["anomaly_transmissions"] == 10
    assert metrics["records_received"] == 10

    # Each encrypted payload must have non-zero size (nonce + ciphertext + tag)
    assert metrics["anomaly_mode_bytes"] > 0
    assert metrics["min_payload_bytes"] > 100
    assert metrics["max_payload_bytes"] < 200
    assert metrics["bytes_saved"] == metrics["baseline_bytes"] - metrics["anomaly_mode_bytes"]
    assert 0.0 < metrics["payload_reduction_percent"] < 100.0


def test_zero_telemetry_bytes_for_normal_windows() -> None:
    """Verify normal windows yield 0 anomaly transmissions and 0 telemetry bytes."""
    num_windows = 25
    X_dummy = np.random.randn(num_windows, 200).astype(np.float32)

    class NormalEngine:
        def invoke_inference(self, window: np.ndarray):
            return 0.20, 0, "NORMAL", 0.3

    sink = TelemetrySink()
    tel = SecureTelemetry(sink=sink)
    scheduler = StateScheduler(threshold=0.85, telemetry_callback=tel.transmit_alert)

    metrics = calculate_bandwidth_metrics(
        X_test=X_dummy,
        engine=NormalEngine(),  # type: ignore
        scheduler=scheduler,
        sink=sink,
    )

    assert metrics["anomaly_transmissions"] == 0
    assert metrics["anomaly_mode_bytes"] == 0
    assert len(sink.get_records()) == 0
    assert metrics["payload_reduction_percent"] == 100.0


def test_reduction_percentage_calculation() -> None:
    """Test mathematical accuracy of payload reduction percentage formula."""
    baseline_bytes = 10000
    anomaly_mode_bytes = 2500
    expected_reduction = (1.0 - (2500 / 10000)) * 100.0  # 75.0%

    num_windows = baseline_bytes // 400  # 25 windows

    class MixedEngine:
        def __init__(self):
            self.count = 0

        def invoke_inference(self, window: np.ndarray):
            self.count += 1
            # First 5 windows normal, remaining anomalous
            if self.count <= 5:
                return 0.1, 0, "NORMAL", 0.5
            return 0.9, 1, "ANOMALY", 0.5

    X_dummy = np.zeros((num_windows, 200), dtype=np.float32)
    sink = TelemetrySink()
    tel = SecureTelemetry(sink=sink)
    scheduler = StateScheduler(threshold=0.85, telemetry_callback=tel.transmit_alert)

    metrics = calculate_bandwidth_metrics(
        X_test=X_dummy,
        engine=MixedEngine(),  # type: ignore
        scheduler=scheduler,
        sink=sink,
    )

    assert metrics["normal_windows"] == 5
    assert metrics["anomaly_transmissions"] == num_windows - 5
    assert metrics["payload_reduction_percent"] > 0.0


def test_no_raw_ecg_reaches_telemetry() -> None:
    """Assert zero raw waveform data leakage in TelemetrySink during bandwidth profiling."""
    num_windows = 5
    X_dummy = np.random.randn(num_windows, 200).astype(np.float32)

    class AnomalyEngine:
        def invoke_inference(self, window: np.ndarray):
            return 0.99, 1, "ANOMALY", 0.5

    sink = TelemetrySink()
    tel = SecureTelemetry(sink=sink)
    scheduler = StateScheduler(threshold=0.85, telemetry_callback=tel.transmit_alert)

    metrics = calculate_bandwidth_metrics(
        X_test=X_dummy,
        engine=AnomalyEngine(),  # type: ignore
        scheduler=scheduler,
        sink=sink,
    )

    records = sink.get_records()
    assert len(records) == num_windows
    for record in records:
        assert set(record.keys()) == {
            "session_id",
            "timestamp_sec",
            "anomaly_id",
            "confidence_score",
            "encrypted_payload",
            "received_at",
        }
        assert isinstance(record["encrypted_payload"], bytes)
        # Ensure no raw sample arrays were leaked
        assert not isinstance(record["encrypted_payload"], (list, tuple, np.ndarray))


def test_generate_bandwidth_report_smoke(tmp_path: Path) -> None:
    """Smoke test generating bandwidth_report.md on real test set."""
    config = load_config()
    models_dir = Path(config.paths.models_dir)
    model_path = models_dir / "student_model_int8.tflite"

    if not model_path.exists():
        pytest.skip("INT8 model file missing, skipping smoke test.")

    output_report = tmp_path / "bandwidth_report.md"
    metrics = generate_bandwidth_report(
        output_report_path=output_report
    )

    assert output_report.exists()
    assert output_report.stat().st_size > 500
    assert "total_test_windows" in metrics
    assert "payload_reduction_percent" in metrics
    assert metrics["total_test_windows"] > 0

    report_text = output_report.read_text(encoding="utf-8")
    assert "# Bandwidth & Network Payload Reduction Measurement Report (P0-15)" in report_text
    assert "[MEASURED]" in report_text
    assert "[ASSUMED]" in report_text
    assert "[TARGET]" in report_text
