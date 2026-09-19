"""Unit tests for StateScheduler module (P0-11)."""

import pytest

from src.scheduler.state_scheduler import StateScheduler, StateSchedulerError


def test_state_scheduler_normal_window_remains_sleep() -> None:
    """Test that confidence score below threshold keeps state SLEEP and triggers 0 telemetry calls."""
    telemetry_calls = []

    def mock_telemetry(w_idx: int, t_sec: float, score: float, anomaly_id: int):
        telemetry_calls.append((w_idx, t_sec, score, anomaly_id))

    scheduler = StateScheduler(threshold=0.85, telemetry_callback=mock_telemetry)
    assert scheduler.current_state == "SLEEP"

    state, triggered, response = scheduler.process_window_result(
        window_index=0, timestamp_sec=0.0, confidence_score=0.30
    )

    assert state == "SLEEP"
    assert not triggered
    assert response is None
    assert len(telemetry_calls) == 0


def test_state_scheduler_anomaly_window_triggers_active() -> None:
    """Test that confidence score at/above threshold transitions to ACTIVE and invokes telemetry exactly once."""
    telemetry_calls = []

    def mock_telemetry(w_idx: int, t_sec: float, score: float, anomaly_id: int):
        telemetry_calls.append((w_idx, t_sec, score, anomaly_id))
        return "ACK_SENT"

    scheduler = StateScheduler(threshold=0.85, telemetry_callback=mock_telemetry)

    state, triggered, response = scheduler.process_window_result(
        window_index=5, timestamp_sec=2.5, confidence_score=0.92, anomaly_id=1
    )

    # After transmission, state resets to SLEEP
    assert state == "SLEEP"
    assert triggered
    assert response == "ACK_SENT"
    assert len(telemetry_calls) == 1
    assert telemetry_calls[0] == (5, 2.5, 0.92, 1)


def test_threshold_exact_boundary() -> None:
    """Test that confidence score exactly equal to threshold (0.85) triggers telemetry."""
    telemetry_calls = []
    scheduler = StateScheduler(threshold=0.85, telemetry_callback=lambda *args: telemetry_calls.append(args))

    # Exactly 0.85 -> trigger
    _, triggered, _ = scheduler.process_window_result(0, 0.0, 0.85)
    assert triggered
    assert len(telemetry_calls) == 1

    # Just below 0.8499 -> no trigger
    _, triggered_below, _ = scheduler.process_window_result(1, 0.5, 0.8499)
    assert not triggered_below
    assert len(telemetry_calls) == 1


def test_transition_history_logging() -> None:
    """Test transition history logging and summary calculations."""
    scheduler = StateScheduler(threshold=0.85)

    scheduler.process_window_result(0, 0.0, 0.10)
    scheduler.process_window_result(1, 0.5, 0.95)
    scheduler.process_window_result(2, 1.0, 0.20)

    summary = scheduler.get_summary()
    assert summary["total_windows_evaluated"] == 3
    assert summary["anomaly_triggers"] == 1
    assert summary["radio_active_ratio"] == pytest.approx(1 / 3, abs=0.01)


def test_invalid_scores_and_thresholds() -> None:
    """Test error handling for bad parameters."""
    with pytest.raises(StateSchedulerError, match="Invalid threshold"):
        StateScheduler(threshold=1.5)

    scheduler = StateScheduler(threshold=0.85)
    with pytest.raises(StateSchedulerError, match="Confidence score .* out of range"):
        scheduler.process_window_result(0, 0.0, 1.5)
