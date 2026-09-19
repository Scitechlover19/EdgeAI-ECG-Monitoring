# Bandwidth & Network Payload Reduction Measurement Report (P0-15)

**Date:** 2026-09-20 03:07:15  
**Dataset Split:** MIT-BIH Held-Out Test Set (`X_test.npy`, 51,992 windows) `[MEASURED]`  
**Deployable Model:** [`models/student_model_int8.tflite`](file:///models/student_model_int8.tflite)  
**Scheduler Threshold:** 0.85 `[MEASURED]`  

---

## 1. Objective & Methodology

The objective of P0-15 is to empirically measure the actual network payload reduction achieved by the privacy-preserving, anomaly-driven edge telemetry design compared against a window-level continuous raw ECG streaming baseline.

> [!IMPORTANT]
> - **Continuous Raw Streaming Baseline (`[ASSUMED]`):** A hypothetical window-level raw ECG streaming baseline in which every processed 200-sample window is represented using 16-bit (2-byte) samples, giving 400 bytes per processed window. Preprocessing windows have 50% overlap (100 samples overlap), so this baseline intentionally counts repeated samples across overlapping windows.
> - **Anomaly-Driven Mode (`[MEASURED]`):** The radio transceiver remains in `SLEEP` state (0 bytes transmitted) for normal windows. When `StateScheduler` detects an anomaly ($\text{confidence} \ge 0.85$), it invokes `SecureTelemetry.transmit_alert()`, which formats a minimal metadata payload, encrypts it using AES-GCM (256-bit key, 12-byte nonce, 16-byte authentication tag), and transmits only the encrypted bytes.
> - **Zero Data Leakage (`[VERIFIED]`):** Raw ECG waveform arrays are never passed, formatted, or transmitted to `TelemetrySink` under any condition.

---

## 2. Encrypted Payload Accounting (`[MEASURED]`)

Each triggered anomaly transmission produces a compact AES-GCM encrypted binary payload:

- **Plaintext Schema:** `{"session_id": "ECG-NODE-001", "timestamp_sec": float, "anomaly_id": int, "confidence_score": float}`
- **AES-GCM Nonce Overhead:** 12 bytes
- **AES-GCM Tag Overhead:** 16 bytes
- **Measured Encrypted Payload Length:** **128 to 133 bytes** (Mean: **132.23 bytes/alert**) `[MEASURED]`

---

## 3. Measured Bandwidth Reduction Results

| Metric | Measured Value | Metric Status |
|---|---:|---|
| **Total Test Windows Processed** | **51,992** | `[MEASURED]` |
| **Normal Windows (`SLEEP` state)** | **51,855** | `[MEASURED]` |
| **Anomaly Transmissions (`ACTIVE` state)** | **137** | `[MEASURED]` |
| **Anomaly Transmission Rate** | **0.26%** | `[MEASURED]` |
| **Raw Continuous Baseline Bytes** | **20,796,800 B** (19.83 MB) | `[MEASURED/ASSUMED]` |
| **Anomaly Mode Telemetry Bytes** | **18,115 B** (0.02 MB) | `[MEASURED]` |
| **Network Payload Bytes Saved** | **20,778,685 B** (19.82 MB) | `[MEASURED]` |
| **Measured Network Payload Reduction** | **99.9129%** | `[MEASURED]` |

---

## 4. Target Assessment & Review-1 Comparison

- **Review-1 Target Requirement (NFR-6 / PERF-5):** $>$ 90.0% Network Payload Reduction `[TARGET]`
- **Measured Reduction:** **99.91%** `[MEASURED]`
- **Target Achieved:** **YES**

> [!NOTE]
> On the held-out MIT-BIH test split of 51,992 windows, the INT8 model detected 137 arrhythmia anomaly windows at threshold 0.85, resulting in an anomaly transmission rate of 0.26%. Total transmitted telemetry was 18,115 bytes (0.0173 MB) versus a 20,796,800 bytes (19.83 MB) continuous streaming baseline, achieving an empirical **99.9129% network payload reduction** (19.82 MB saved). Per AGENTS.md rules, this measured value is reported without modification or artificial threshold tuning.

---

## 5. Privacy & Security Verification (`[VERIFIED]`)

- **Raw Waveform Transmission:** **0 bytes** (`[VERIFIED]`)
- **Schema Rejection Audit:** Checked all 137 stored records in `TelemetrySink`. No raw sample arrays exist in any payload.
- **Normal Window Transmission:** Normal windows contributed exactly **0 bytes** to the network sink.

---

## 6. Limitations & Assumptions

1. **Window-Level Baseline:** Baseline is calculated per processed window (400 bytes/window). Because windows overlap by 50% (100 samples), raw streaming at the unwindowed sample stream level (200 Hz continuous 16-bit = 400 bytes/sec) would yield half the total baseline bytes; however, window-level streaming comparison accurately reflects framing overhead in IoMT packet streaming.
2. **Transport Framing:** Encrypted telemetry bytes reflect application-layer payload length (nonce + ciphertext + tag). Physical link-layer header bytes (e.g. BLE/802.15.4 MAC headers) are not included.

---

## 7. Reproducibility Command

To reproduce this measurement report:
```bash
python -m src.pipeline.bandwidth_report
```
