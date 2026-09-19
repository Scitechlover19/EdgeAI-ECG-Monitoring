"""Unit tests for SecureTelemetry and TelemetrySink modules (P0-12)."""

import pytest
import numpy as np

from src.telemetry.secure_telemetry import SecureTelemetry, SecureTelemetryError
from src.telemetry.sink import TelemetrySink, TelemetrySinkError


def test_metadata_schema_formatting() -> None:
    """Test format_metadata produces exact 4-field metadata dictionary."""
    telemetry = SecureTelemetry(session_id="NODE-TEST-1")
    metadata = telemetry.format_metadata(timestamp_sec=12.5, anomaly_id=1, confidence_score=0.92)

    assert metadata == {
        "session_id": "NODE-TEST-1",
        "timestamp_sec": 12.5,
        "anomaly_id": 1,
        "confidence_score": 0.92,
    }


def test_raw_ecg_rejection_in_telemetry() -> None:
    """Test that passing a raw ECG sample array into telemetry extra_data raises SecureTelemetryError."""
    telemetry = SecureTelemetry()
    raw_samples = list(np.random.randn(200))

    with pytest.raises(SecureTelemetryError, match="SECURITY VIOLATION! Array field .* rejected"):
        telemetry.format_metadata(
            timestamp_sec=0.0,
            anomaly_id=1,
            confidence_score=0.90,
            extra_data={"raw_ecg": raw_samples},
        )


def test_aes_gcm_encryption_roundtrip() -> None:
    """Test AES-GCM encryption and decryption roundtrip."""
    telemetry = SecureTelemetry()
    original_metadata = {
        "session_id": "NODE-01",
        "timestamp_sec": 42.0,
        "anomaly_id": 2,
        "confidence_score": 0.98,
    }

    encrypted_bytes = telemetry.encrypt_payload(original_metadata)
    assert isinstance(encrypted_bytes, bytes)
    assert len(encrypted_bytes) > 12  # Nonce (12 bytes) + Ciphertext

    decrypted_metadata = telemetry.decrypt_payload(encrypted_bytes)
    assert decrypted_metadata == original_metadata


def test_aes_gcm_tampered_ciphertext_fails() -> None:
    """Test that tampered/corrupted ciphertext fails AES-GCM authentication tag check."""
    telemetry = SecureTelemetry()
    original_metadata = {"session_id": "NODE-01", "timestamp_sec": 1.0, "anomaly_id": 1, "confidence_score": 0.90}
    encrypted_bytes = bytearray(telemetry.encrypt_payload(original_metadata))

    # Tamper with one byte in ciphertext
    encrypted_bytes[-1] ^= 0xFF

    with pytest.raises(SecureTelemetryError, match="AES-GCM decryption failed"):
        telemetry.decrypt_payload(bytes(encrypted_bytes))


def test_telemetry_sink_receive_and_record() -> None:
    """Test full telemetry alert transmission into TelemetrySink."""
    sink = TelemetrySink()
    telemetry = SecureTelemetry(sink=sink, session_id="TEST-NODE")

    response = telemetry.transmit_alert(window_index=3, timestamp_sec=1.5, confidence_score=0.88, anomaly_id=1)

    assert response["status"] == "SUCCESS"
    assert response["ack"] is True
    assert response["recorded_count"] == 1

    records = sink.get_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["session_id"] == "TEST-NODE"
    assert rec["timestamp_sec"] == 1.5
    assert rec["anomaly_id"] == 1
    assert rec["confidence_score"] == 0.88
    assert "raw_ecg" not in rec


def test_sink_raw_ecg_rejection() -> None:
    """Test that TelemetrySink directly rejects payload containing raw sample array."""
    sink = TelemetrySink()
    bad_payload = {
        "session_id": "NODE-01",
        "timestamp_sec": 0.0,
        "anomaly_id": 1,
        "confidence_score": 0.90,
        "encrypted_payload": b"123",
        "raw_waveform": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],  # Length > 5 array
    }

    with pytest.raises(TelemetrySinkError, match="SECURITY VIOLATION! Raw array field .* detected"):
        sink.receive_alert(bad_payload)
