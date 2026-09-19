"""Unit tests for VirtualMCU simulated hardware harness (P0-9)."""

import time
import pytest

from src.edge.virtual_mcu import VirtualMCU, VirtualMCUOverflowError


def test_tensor_arena_valid_allocation() -> None:
    """Test valid tensor arena memory calculation within 256 KB SRAM limit."""
    mcu = VirtualMCU(sram_limit_kb=256.0)

    # Simulated model size: 50 KB = 51200 bytes
    model_bytes = 51200
    report = mcu.calculate_tensor_arena(
        model_size_bytes=model_bytes,
        input_shape=(1, 200, 1),
        output_shape=(1, 2),
        working_buffer_bytes=4096,
        dtype_size_bytes=1,  # INT8
    )

    assert report["status_label"] == "[SIMULATED]"
    assert report["model_memory_bytes"] == 51200
    assert report["input_tensor_bytes"] == 200
    assert report["output_tensor_bytes"] == 2
    assert report["estimated_tensor_arena_bytes"] == 200 + 2 + 4096
    assert report["total_estimated_sram_bytes"] == 51200 + 4298
    assert report["sram_limit_kb"] == 256.0
    assert report["headroom_kb"] > 0
    assert not report["exceeds_limit"]


def test_sram_overflow_raises_exception() -> None:
    """Test that SRAM allocation exceeding 256 KB raises VirtualMCUOverflowError."""
    mcu = VirtualMCU(sram_limit_kb=256.0)

    # 300 KB model bytes > 256 KB
    huge_model_bytes = 300 * 1024
    with pytest.raises(VirtualMCUOverflowError, match="\\[SIMULATED\\] SRAM limit exceeded"):
        mcu.calculate_tensor_arena(
            model_size_bytes=huge_model_bytes,
            input_shape=(1, 200, 1),
            output_shape=(1, 2),
        )


def test_flash_footprint_check() -> None:
    """Test Flash footprint calculation against 1 MB limit."""
    mcu = VirtualMCU(flash_limit_mb=1.0)

    # 500 KB model
    model_bytes = 500 * 1024
    report = mcu.check_flash_footprint(model_bytes)

    assert report["status_label"] == "[SIMULATED]"
    assert report["total_flash_mb"] == pytest.approx(0.4883, abs=0.001)
    assert report["headroom_mb"] > 0
    assert not report["exceeds_limit"]


def test_flash_overflow_raises_exception() -> None:
    """Test that Flash footprint > 1 MB raises VirtualMCUOverflowError."""
    mcu = VirtualMCU(flash_limit_mb=1.0)

    # 1.5 MB model
    huge_flash_bytes = int(1.5 * 1024 * 1024)
    with pytest.raises(VirtualMCUOverflowError, match="\\[SIMULATED\\] Flash limit exceeded"):
        mcu.check_flash_footprint(huge_flash_bytes)


def test_profile_execution_timing() -> None:
    """Test latency harness timing measurement."""
    mcu = VirtualMCU()

    def dummy_func(duration: float) -> str:
        time.sleep(duration)
        return "done"

    result, latency_ms = mcu.profile_execution(dummy_func, 0.01)
    assert result == "done"
    assert latency_ms >= 8.0  # ~10 ms with slight timing tolerance
