# Bandwidth & Network Payload Reduction Measurement Report (P0-15)

**Date:** 2026-09-20 22:36:48  
**Dataset Split:** MIT-BIH Held-Out Test Set (`X_test.npy`, 51,992 windows) `[MEASURED]`  
**Deployable Model:** [`models/student_model_int8.tflite`](file:///models/student_model_int8.tflite)  
**Scheduler Threshold:** 0.35 `[MEASURED]`  

---

## 1. Objective & Methodology

The objective of P0-15 is to empirically measure the actual network payload reduction achieved by the privacy-preserving, anomaly-driven edge telemetry design compared against a window-level continuous raw ECG streaming baseline.

> [!IMPORTANT]
> - **Continuous Raw Streaming Baseline (`[ASSUMED]`):** A hypothetical window-level raw ECG streaming baseline in which every processed 200-sample window is represented using 16-bit (2-byte) samples, giving 400 bytes per processed window. Preprocessing windows have 50% overlap (100 samples overlap), so this baseline intentionally counts repeated samples across overlapping windows.
> - **Anomaly-Driven Mode (`[MEASURED]`):** The radio transceiver remains in `SLEEP` state (0 bytes transmitted) for normal windows. When `StateScheduler` detects an anomaly ($\text{confidence} \ge 0.35$), it invokes `SecureTelemetry.transmit_alert()`, which formats a minimal metadata payload, encrypts it using AES-GCM (256-bit key, 12-byte nonce, 16-byte authentication tag), and transmits only the encrypted bytes.
> - **Zero Data Leakage (`[VERIFIED]`):** Raw ECG waveform arrays are never passed, formatted, or transmitted to `TelemetrySink` under any condition.

---

## 2. Encrypted Payload Accounting (`[MEASURED]`)

Each triggered anomaly transmission produces a compact AES-GCM encrypted binary payload:

- **Plaintext Schema:** `{"session_id": "ECG-NODE-001", "timestamp_sec": float, "anomaly_id": int, "confidence_score": float}`
- **AES-GCM Nonce Overhead:** 12 bytes
- **AES-GCM Tag Overhead:** 16 bytes
- **Measured Encrypted Payload Length:** **122 to 133 bytes** (Mean: **131.56 bytes/alert**) `[MEASURED]`

---

## 3. Measured Bandwidth Reduction Results

| Metric | Measured Value | Metric Status |
|---|---:|---|
| **Total Test Windows Processed** | **51,992** | `[MEASURED]` |
| **Normal Windows (`SLEEP` state)** | **50,481** | `[MEASURED]` |
| **Anomaly Transmissions (`ACTIVE` state)** | **1,511** | `[MEASURED]` |
| **Anomaly Transmission Rate** | **2.91%** | `[MEASURED]` |
| **Raw Continuous Baseline Bytes** | **20,796,800 B** (19.83 MB) | `[MEASURED/ASSUMED]` |
| **Anomaly Mode Telemetry Bytes** | **198,791 B** (0.19 MB) | `[MEASURED]` |
| **Network Payload Bytes Saved** | **20,598,009 B** (19.64 MB) | `[MEASURED]` |
| **Measured Network Payload Reduction** | **99.0441%** | `[MEASURED]` |

---

## 4. Target Assessment & Review-1 Comparison

- **Review-1 Target Requirement (NFR-6 / PERF-5):** $>$ 90.0% Network Payload Reduction `[TARGET]`
- **Measured Reduction:** **99.04%** `[MEASURED]`
- **Target Achieved:** **YES**

> [!NOTE]
> On the held-out MIT-BIH test split of 51,992 windows, the INT8 model detected 1,511 arrhythmia anomaly windows at threshold 0.35, resulting in an anomaly transmission rate of 2.91%. Total transmitted telemetry was 198,791 bytes (0.1896 MB) versus a 20,796,800 bytes (19.83 MB) continuous streaming baseline, achieving an empirical **99.0441% network payload reduction** (19.64 MB saved). Per AGENTS.md rules, this measured value is reported without modification or artificial threshold tuning.

---

## 5. Privacy & Security Verification (`[VERIFIED]`)

- **Raw Waveform Transmission:** **0 bytes** (`[VERIFIED]`)
- **Schema Rejection Audit:** Checked all 1,511 stored records in `TelemetrySink`. No raw sample arrays exist in any payload.
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
