"""TelemetrySink module for simulated hospital endpoint storage.

Stores encrypted anomaly telemetry payloads and metadata in memory.
Governed by PRD.md (SEC-1, SEC-2, SEC-5), Architecture.md §12, and DECISIONS.md #9.

STRICT SECURITY RULE (AGENTS.md §1.5):
Under NO condition may raw ECG array data be accepted, passed, or stored in TelemetrySink.
"""

from typing import Any, Dict, List


class TelemetrySinkError(Exception):
    """Custom exception raised for invalid telemetry sink interactions."""
    pass


class TelemetrySink:
    """In-memory telemetry sink modeling a secure remote clinical receiver."""

    def __init__(self) -> None:
        """Initialize empty in-memory telemetry storage."""
        self._stored_records: List[Dict[str, Any]] = []

    def receive_alert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Receive and record encrypted anomaly alert payload.

        Args:
            payload: Dict containing metadata and encrypted payload.

        Returns:
            Dict containing acknowledgement status.

        Raises:
            TelemetrySinkError: If raw ECG data is detected or schema is violated.
        """
        # Security Assertion: Check for raw waveform data injection
        for key, val in payload.items():
            if isinstance(val, (list, tuple)) and len(val) > 5:
                raise TelemetrySinkError(
                    f"SECURITY VIOLATION! Raw array field '{key}' detected in telemetry payload."
                )

        required_keys = {"session_id", "timestamp_sec", "anomaly_id", "confidence_score", "encrypted_payload"}
        if not required_keys.issubset(payload.keys()):
            missing = required_keys - set(payload.keys())
            raise TelemetrySinkError(f"Payload schema incomplete. Missing fields: {missing}")

        record = {
            "session_id": str(payload["session_id"]),
            "timestamp_sec": float(payload["timestamp_sec"]),
            "anomaly_id": int(payload["anomaly_id"]),
            "confidence_score": float(payload["confidence_score"]),
            "encrypted_payload": bytes(payload["encrypted_payload"]),
            "received_at": float(payload.get("received_at", 0.0)),
        }

        self._stored_records.append(record)
        return {
            "status": "SUCCESS",
            "ack": True,
            "recorded_count": len(self._stored_records),
        }

    def get_records(self) -> List[Dict[str, Any]]:
        """Return copies of stored telemetry records."""
        return [dict(r) for r in self._stored_records]

    def clear(self) -> None:
        """Clear stored records."""
        self._stored_records.clear()
