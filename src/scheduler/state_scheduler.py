"""StateScheduler module for anomaly-driven power management and radio control.

Controls SLEEP/ACTIVE state transitions based on TinyMLEngine confidence scores.
Ensures radio state remains SLEEP (OFF) during normal baseline rhythms and transitions to ACTIVE (ON)
only when anomaly confidence breaches the configured threshold (default 0.85).
Governed by PRD.md (FR-11, FR-12, SEC-4), Architecture.md §5.5 & §6, AGENTS.md Rule 6,
and DECISIONS.md #13.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


class StateSchedulerError(Exception):
    """Custom exception raised for StateScheduler failures."""
    pass


@dataclass(frozen=True)
class StateTransitionRecord:
    """Record of a single state transition event for audit profiling."""

    window_index: int
    timestamp_sec: float
    from_state: str
    to_state: str
    confidence_score: float
    telemetry_triggered: bool


class StateScheduler:
    """State machine scheduler enforcing anomaly-driven radio wake/sleep control."""

    STATE_SLEEP = "SLEEP"
    STATE_ACTIVE = "ACTIVE"

    def __init__(
        self,
        threshold: float = 0.85,
        telemetry_callback: Optional[Callable[[int, float, float, int], Any]] = None,
    ) -> None:
        """Initialize StateScheduler.

        Args:
            threshold: Anomaly confidence threshold in [0.0, 1.0] (default 0.85).
            telemetry_callback: Optional callback to SecureTelemetry.transmit_alert on ACTIVE state.

        Raises:
            StateSchedulerError: If threshold is out of range [0.0, 1.0].
        """
        if not (0.0 <= threshold <= 1.0):
            raise StateSchedulerError(f"Invalid threshold {threshold}. Must be in range [0.0, 1.0].")

        self.threshold: float = float(threshold)
        self.current_state: str = self.STATE_SLEEP
        self.telemetry_callback = telemetry_callback
        self.transition_history: List[StateTransitionRecord] = []

    @property
    def is_radio_active(self) -> bool:
        """Return True if radio is currently in ACTIVE (ON) state."""
        return self.current_state == self.STATE_ACTIVE

    def process_window_result(
        self,
        window_index: int,
        timestamp_sec: float,
        confidence_score: float,
        anomaly_id: int = 1,
    ) -> Tuple[str, bool, Optional[Any]]:
        """Evaluate inference output and determine state transition and telemetry trigger.

        Args:
            window_index: Index of the current ECG window.
            timestamp_sec: Timestamp of the window start in seconds.
            confidence_score: Anomaly confidence score from TinyMLEngine (0.0 to 1.0).
            anomaly_id: Classification code of the anomaly (default 1).

        Returns:
            Tuple containing:
            - resulting_state (str): "SLEEP" (returns to SLEEP after transmission if ACTIVE).
            - telemetry_triggered (bool): True if telemetry alert was invoked, False otherwise.
            - telemetry_response (Optional[Any]): Result of telemetry callback if triggered.
        """
        if not (0.0 <= confidence_score <= 1.0):
            raise StateSchedulerError(f"Confidence score {confidence_score} out of range [0.0, 1.0].")

        telemetry_response = None
        telemetry_triggered = False

        if confidence_score >= self.threshold:
            # Transition SLEEP -> ACTIVE
            from_state = self.current_state
            self.current_state = self.STATE_ACTIVE
            telemetry_triggered = True

            # Record transition to ACTIVE
            self.transition_history.append(
                StateTransitionRecord(
                    window_index=window_index,
                    timestamp_sec=timestamp_sec,
                    from_state=from_state,
                    to_state=self.STATE_ACTIVE,
                    confidence_score=confidence_score,
                    telemetry_triggered=True,
                )
            )

            # Invoke telemetry if callback provided
            if self.telemetry_callback is not None:
                telemetry_response = self.telemetry_callback(
                    window_index, timestamp_sec, confidence_score, anomaly_id
                )

            # Instantly reset state ACTIVE -> SLEEP per Fig 10.3 sequence contract
            self.current_state = self.STATE_SLEEP
            self.transition_history.append(
                StateTransitionRecord(
                    window_index=window_index,
                    timestamp_sec=timestamp_sec,
                    from_state=self.STATE_ACTIVE,
                    to_state=self.STATE_SLEEP,
                    confidence_score=confidence_score,
                    telemetry_triggered=False,
                )
            )
        else:
            # Remain in SLEEP state (radio OFF)
            self.transition_history.append(
                StateTransitionRecord(
                    window_index=window_index,
                    timestamp_sec=timestamp_sec,
                    from_state=self.STATE_SLEEP,
                    to_state=self.STATE_SLEEP,
                    confidence_score=confidence_score,
                    telemetry_triggered=False,
                )
            )

        return self.current_state, telemetry_triggered, telemetry_response

    def get_summary(self) -> Dict[str, Any]:
        """Return profiling summary of all processed window transitions.

        Returns:
            Dict containing total windows, anomaly triggers, radio ON count, and active ratio.
        """
        total_evaluations = len([r for r in self.transition_history if r.from_state == self.STATE_SLEEP])
        anomaly_triggers = len([r for r in self.transition_history if r.telemetry_triggered])
        active_ratio = (anomaly_triggers / total_evaluations) if total_evaluations > 0 else 0.0

        return {
            "total_windows_evaluated": total_evaluations,
            "anomaly_triggers": anomaly_triggers,
            "radio_active_ratio": round(active_ratio, 4),
            "threshold": self.threshold,
        }
