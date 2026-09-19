"""Integration tests for end-to-end PipelineController vertical slice (P0-13)."""

import numpy as np
import pytest

from src.ingestion.data_ingestion import DataIngestion
from src.pipeline.controller import PipelineController
from src.telemetry.sink import TelemetrySink


def test_vertical_slice_pipeline_integration() -> None:
    """Test full end-to-end vertical slice (Ingestion -> DSP -> TinyML -> Scheduler -> Telemetry -> Sink)."""
    # Construct synthetic ECG stream: 1000 samples = 9 windows at 200 size, 50% overlap (step 100)
    # Windows 0..2: Normal sinewave
    # Window 3 & 4 (samples 300..500): Severe impulse anomaly spike
    # Windows 5..8: Normal sinewave
    fs = 360.0
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000), dtype=np.float32) * 0.5
    # Inject sharp impulse arrhythmia anomaly spike in samples 450..455
    signal[450:455] = 50.0

    ingestion = DataIngestion(signal, window_size=200, overlap_ratio=0.5, sample_rate=fs)
    sink = TelemetrySink()

    controller = PipelineController(ingestion=ingestion, sink=sink)
    summary = controller.run()

    assert summary["status_label"] == "[SIMULATED]"
    assert summary["total_windows"] == 9
    assert summary["anomaly_triggers"] >= 1
    assert summary["telemetry_records_received"] == summary["anomaly_triggers"]
    assert summary["latency_budget_met"] is True

    # Assert Sink records contain NO raw ECG
    records = sink.get_records()
    assert len(records) == summary["anomaly_triggers"]
    for rec in records:
        assert "raw_ecg" not in rec
        assert "raw_waveform" not in rec
        assert rec["confidence_score"] >= 0.85
        assert len(rec["encrypted_payload"]) > 12  # AES-GCM nonce + ciphertext


def test_normal_only_stream_zero_alerts() -> None:
    """Test that a pure normal baseline stream produces ZERO telemetry alerts."""
    fs = 360.0
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000), dtype=np.float32) * 0.5

    ingestion = DataIngestion(signal, window_size=200, overlap_ratio=0.5, sample_rate=fs)
    sink = TelemetrySink()

    controller = PipelineController(ingestion=ingestion, sink=sink)
    summary = controller.run()

    assert summary["total_windows"] == 9
    assert summary["anomaly_triggers"] == 0
    assert summary["telemetry_records_received"] == 0
    assert len(sink.get_records()) == 0
