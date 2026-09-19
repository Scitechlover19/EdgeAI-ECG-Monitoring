"""Unit tests for Resource Profiling Report module (P0-14).

Verifies latency benchmarking, Flash size measurement, VirtualMCU SRAM estimation,
NFR budget compliance evaluation, and resource_report.md generation.
Governed by AGENTS.md §3 and DECISIONS.md #10/#11/#12.
"""

from pathlib import Path
import numpy as np
import pytest

from src.pipeline.profile_report import (
    generate_profile_report,
    profile_inference_latency,
)


def test_profile_inference_latency():
    """Test high-precision latency benchmarking across test windows."""
    # Use existing INT8 tflite model if available
    model_path = Path("models/student_model_int8.tflite")
    if not model_path.exists():
        pytest.skip("models/student_model_int8.tflite not found")

    from src.edge.tinyml_engine import TinyMLEngine

    engine = TinyMLEngine(model_path=model_path)
    dummy_windows = np.random.randn(20, 200).astype(np.float32)

    stats = profile_inference_latency(engine, dummy_windows, num_windows=10)

    assert "mean_ms" in stats
    assert "median_ms" in stats
    assert "p95_ms" in stats
    assert "min_ms" in stats
    assert "max_ms" in stats

    assert stats["mean_ms"] >= 0.0
    assert stats["min_ms"] <= stats["median_ms"] <= stats["max_ms"]
    assert stats["p95_ms"] <= stats["max_ms"]


def test_generate_profile_report_execution(tmp_path: Path):
    """Test end-to-end report generation and NFR compliance verification."""
    model_path = Path("models/student_model_int8.tflite")
    if not model_path.exists():
        pytest.skip("models/student_model_int8.tflite not found")

    out_report = tmp_path / "resource_report.md"

    results = generate_profile_report(
        tflite_model_path=model_path,
        num_benchmark_windows=20,
        output_report_path=out_report,
    )

    assert out_report.exists()
    assert results["flash_kb"] > 0
    assert results["sram_peak_kb"] > 0
    assert results["sram_peak_kb"] <= 256.0
    assert results["flash_kb"] <= 1024.0

    content = out_report.read_text(encoding="utf-8")
    assert "Resource Profiling" in content
    assert "Mean Latency" in content
    assert "[MEASURED]" in content
    assert "[ESTIMATED]" in content
    assert "[SIMULATED]" in content
    assert "NFR-1" in content
    assert "NFR-2" in content
    assert "NFR-3" in content
    # Assert process RSS is never used as SRAM proxy
    assert "psutil" not in content
