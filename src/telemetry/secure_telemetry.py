"""SecureTelemetry module for privacy-preserving encrypted metadata transmission.

Formats minimal metadata payloads (timestamp, anomaly ID, confidence score),
encrypts them using AES-GCM authenticated symmetric cipher, and transmits them to TelemetrySink.
Governed by PRD.md (FR-13, FR-14, SEC-1..5), Architecture.md §5.6 & §11, AGENTS.md Rules 5, 7,
and DECISIONS.md #8, #9.

STRICT SECURITY RULE (AGENTS.md §1.5):
Raw ECG data MUST NEVER be passed into format_metadata or transmitted.
Zero Data Leakage principle.
"""

import json
import os
import time
from typing import Any, Dict, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.telemetry.sink import TelemetrySink, TelemetrySinkError


class SecureTelemetryError(Exception):
    """Custom exception raised for SecureTelemetry errors or security contract breaches."""
    pass


class SecureTelemetry:
    """Privacy-preserving metadata telemetry formatter and AES-GCM encryptor."""

    def __init__(
        self,
        sink: Optional[TelemetrySink] = None,
        key: Optional[bytes] = None,
        session_id: str = "ECG-NODE-001",
    ) -> None:
        """Initialize SecureTelemetry.

        Args:
            sink: TelemetrySink instance for alert transmission.
            key: 256-bit (32 bytes) secret key for AES-GCM encryption. If None, generated randomly.
            session_id: Unique node/session identifier string.
        """
        self.sink = sink if sink is not None else TelemetrySink()
        self.session_id = session_id

        if key is None:
            self.key = AESGCM.generate_key(bit_length=256)
        else:
            if len(key) not in (16, 24, 32):
                raise SecureTelemetryError(f"AES key length must be 16, 24, or 32 bytes, got {len(key)} bytes.")
            self.key = key

        self.aesgcm = AESGCM(self.key)

    def format_metadata(
        self,
        timestamp_sec: float,
        anomaly_id: int,
        confidence_score: float,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format minimal metadata dictionary strictly matching security schema.

        Args:
            timestamp_sec: Event timestamp in seconds.
            anomaly_id: Classification code of detected arrhythmia anomaly.
            confidence_score: Classification confidence score [0.0, 1.0].
            extra_data: Additional optional metadata dict (MUST NOT contain waveform arrays).

        Returns:
            Dict containing minimal metadata schema fields.

        Raises:
            SecureTelemetryError: If raw array or invalid field is detected.
        """
        if extra_data:
            for k, v in extra_data.items():
                if isinstance(v, (list, tuple)) and len(v) > 5:
                    raise SecureTelemetryError(
                        f"SECURITY VIOLATION! Array field '{k}' of length {len(v)} rejected. Raw ECG transmit forbidden!"
                    )

        payload = {
            "session_id": self.session_id,
            "timestamp_sec": float(timestamp_sec),
            "anomaly_id": int(anomaly_id),
            "confidence_score": float(confidence_score),
        }
        return payload

    def encrypt_payload(self, metadata: Dict[str, Any]) -> bytes:
        """Encrypt metadata payload dictionary using AES-GCM authentication.

        Format of output bytes: 12-byte random Nonce + AES-GCM Ciphertext (includes tag).

        Args:
            metadata: Dict containing formatted metadata fields.

        Returns:
            Encrypted byte array containing Nonce + Ciphertext.

        Raises:
            SecureTelemetryError: If encryption fails.
        """
        try:
            plaintext_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
            nonce = os.urandom(12)  # 96-bit standard AES-GCM nonce
            ciphertext = self.aesgcm.encrypt(nonce, plaintext_bytes, associated_data=None)
            return nonce + ciphertext
        except Exception as err:
            raise SecureTelemetryError(f"AES-GCM encryption failed: {err}") from err

    def decrypt_payload(self, encrypted_bytes: bytes) -> Dict[str, Any]:
        """Decrypt AES-GCM encrypted payload bytes back to metadata dictionary.

        Args:
            encrypted_bytes: Encrypted byte payload (12-byte nonce + ciphertext).

        Returns:
            Decrypted metadata dictionary.

        Raises:
            SecureTelemetryError: If decryption or authentication tag check fails.
        """
        if len(encrypted_bytes) < 13:
            raise SecureTelemetryError("Encrypted payload too short.")

        try:
            nonce = encrypted_bytes[:12]
            ciphertext = encrypted_bytes[12:]
            plaintext_bytes = self.aesgcm.decrypt(nonce, ciphertext, associated_data=None)
            metadata = json.loads(plaintext_bytes.decode("utf-8"))
            return metadata
        except Exception as err:
            raise SecureTelemetryError(f"AES-GCM decryption failed (tag mismatch or corrupt): {err}") from err

    def transmit_alert(
        self,
        window_index: int,
        timestamp_sec: float,
        confidence_score: float,
        anomaly_id: int = 1,
    ) -> Dict[str, Any]:
        """Format, encrypt, and transmit anomaly alert to TelemetrySink.

        Args:
            window_index: Index of triggering window.
            timestamp_sec: Start time of window in seconds.
            confidence_score: Anomaly classification confidence.
            anomaly_id: Classification code (default 1).

        Returns:
            Dict containing sink response status.

        Raises:
            SecureTelemetryError: If transmission or formatting fails.
        """
        metadata = self.format_metadata(timestamp_sec, anomaly_id, confidence_score)
        encrypted_bytes = self.encrypt_payload(metadata)

        sink_payload = {
            "session_id": self.session_id,
            "timestamp_sec": timestamp_sec,
            "anomaly_id": anomaly_id,
            "confidence_score": confidence_score,
            "encrypted_payload": encrypted_bytes,
            "received_at": time.time(),
        }

        try:
            response = self.sink.receive_alert(sink_payload)
            return response
        except TelemetrySinkError as err:
            raise SecureTelemetryError(f"Transmission to TelemetrySink failed: {err}") from err
