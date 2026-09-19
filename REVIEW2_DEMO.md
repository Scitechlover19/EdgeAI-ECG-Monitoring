# REVIEW2_DEMO.md — Review-2 Presentation & Demonstration Walkthrough

**Project:** Resource-Constrained Edge-AI Pipeline for Real-Time Privacy-Preserving Patient Monitoring  
**Candidate / Author:** Nancy Singh (22MIS0027), SWE3004  
**Date:** 2026-09-19  
**Repository:** [`EdgeAI-ECG-Monitoring`](file:///d:/Project/EdgeAI-ECG-Monitoring/)  

---

## 1. Problem Statement

Continuous remote patient monitoring in IoMT (Internet of Medical Things) faces three fundamental engineering challenges documented in Review-1:

1. **Privacy/Security Risk:** Continuous streaming of raw, unencrypted biological ECG signals over public RF networks creates data-sovereignty and eavesdropping exposure.
2. **Energy Inefficiency:** Continuous RF transceiver operation is the single largest power drain on wearable devices, severely limiting battery life.
3. **Resource Mismatch:** Heavy deep learning models exceed the SRAM ($\le 256\text{ KB}$) and Flash ($< 1\text{ MB}$) constraints of ultra-low-power microcontrollers.

---

## 2. High-Level System Architecture

A 5-stage software pipeline executed in a simulated micro-architecture environment (`VirtualMCU`):

```
MIT-BIH ECG Ingestion (200-pt windows, 50% overlap)
            ↓
DSP Preprocessing (4th-order Butterworth 0.5–45 Hz + baseline correction + Z-score)
            ↓
TinyML Edge Inference Engine (INT8 quantized 1D-CNN distilled from Teacher)
            ↓
Anomaly-Driven State Scheduler (SLEEP radio default / ACTIVE on anomaly >= 0.85)
            ↓
Secure Telemetry Sink (AES-GCM 256-bit encrypted metadata — ZERO raw ECG)
```

---

## 3. Real MIT-BIH Dataset & Patient-Independent Splitting

- **Dataset:** PhysioNet MIT-BIH Arrhythmia Database (48 records, 311,952 total windows).
- **Split Strategy:** Strict patient-independent split (33 train, 7 val, 8 test records). No record appears in more than one split.
- **Held-Out Test Set:** 51,992 test windows across 8 test records (`107`, `115`, `119`, `122`, `207`, `220`, `228`, `234`).
- **Reproducibility Manifest:** [`reports/experiment_manifest.json`](file:///d:/Project/EdgeAI-ECG-Monitoring/reports/experiment_manifest.json).

---

## 4. Model Compression via Knowledge Distillation

- **Teacher 1D-CNN:** 120,674 parameters ($472.0\text{ KB}$, 90.96% test accuracy).
- **Student 1D-CNN:** 1,538 parameters ($7.0\text{ KB}$ Float32).
- **Compression Ratio:** **78.46$\times$ parameter reduction** vs. Teacher.
- **KD Hyperparameters:** Soft-target distillation with Temperature $T=3.0$, $\alpha=0.7$.
- **Ablation Report:** [`reports/kd_ablation.md`](file:///d:/Project/EdgeAI-ECG-Monitoring/reports/kd_ablation.md).

---

## 5. Post-Training Quantization (INT8 PTQ)

- **Quantization:** TensorFlow Lite full-integer INT8 Post-Training Quantization using `X_train.npy` calibration.
- **Deployable Flatbuffer:** [`models/student_model_int8.tflite`](file:///d:/Project/EdgeAI-ECG-Monitoring/models/student_model_int8.tflite) (**10.96 KB** file size).
- **Accuracy Delta:** Float32 Student Accuracy = 86.87%, INT8 Student Accuracy = 86.60% ($\Delta\text{Accuracy} = -0.26\%$).
- **Quantization Report:** [`reports/quantization_report.md`](file:///d:/Project/EdgeAI-ECG-Monitoring/reports/quantization_report.md).

---

## 6. Resource Profiling & Budget Compliance (`[MEASURED]` / `[ESTIMATED]`)

All metrics evaluated against Review-1 Non-Functional Requirements (NFRs):

| Requirement ID | Metric | Budget Limit | Measured / Estimated Value | Status |
|---|---|---|---|---|
| **NFR-1** | Peak Simulated SRAM | $\le 256.0\text{ KB}$ | **15.16 KB** `[ESTIMATED]` | **PASS** |
| **NFR-2** | Model Flash Memory | $< 1.0\text{ MB}$ ($1,024\text{ KB}$) | **10.96 KB** `[MEASURED]` | **PASS** |
| **NFR-3** | Per-Window Latency | $< 50.0\text{ ms}$ | **0.0344 ms** `[MEASURED]` | **PASS** |

Report artifact: [`reports/resource_report.md`](file:///d:/Project/EdgeAI-ECG-Monitoring/reports/resource_report.md).

---

## 7. Anomaly-Driven State Scheduler

- **States:** `SLEEP` (Radio OFF) and `ACTIVE` (Radio ON).
- **Threshold:** Anomaly confidence threshold $\ge 0.85$ (configurable).
- **Behavior:** The radio stack remains in `SLEEP` during normal rhythms. Upon detecting an anomaly window ($\text{confidence} \ge 0.85$), it transitions to `ACTIVE`, invokes secure telemetry transmission, and immediately resets to `SLEEP`.

---

## 8. Privacy-Preserving Secure Telemetry

- **Schema Restriction:** Metadata contains ONLY `session_id`, `timestamp_sec`, `anomaly_id`, `confidence_score`.
- **Encryption:** AES-GCM 256-bit symmetric cipher (12-byte nonce, 16-byte authentication tag).
- **Zero Data Leakage:** Structural schema assertion rejects any payload containing raw waveform sample arrays.

---

## 9. Network Payload & Bandwidth Reduction (`[MEASURED]`)

Evaluated over the full held-out test split (51,992 windows):

| Metric | Measured Value | Status |
|---|---:|---|
| **Total Test Windows Processed** | **51,992** | `[MEASURED]` |
| **Normal Windows (`SLEEP` state)** | **51,855** | `[MEASURED]` |
| **Anomaly Transmissions (`ACTIVE` state)** | **137** | `[MEASURED]` |
| **Raw Continuous Baseline Bytes** | **20,796,800 B** (19.83 MB) | `[MEASURED/ASSUMED]` |
| **Anomaly Mode Telemetry Bytes** | **18,115 B** (0.02 MB) | `[MEASURED]` |
| **Network Payload Bytes Saved** | **20,778,685 B** (19.82 MB) | `[MEASURED]` |
| **Measured Network Payload Reduction** | **99.9129%** | `[MEASURED]` |
| **Raw Waveform Bytes Transmitted** | **0 B** | `[VERIFIED]` |

Report artifact: [`reports/bandwidth_report.md`](file:///d:/Project/EdgeAI-ECG-Monitoring/reports/bandwidth_report.md).

---

## 10. Demonstration Modes & Scientific Integrity (`[DEMO MODE]`)

The software supports an isolated demonstration harness:

```bash
python -m src.pipeline.demo_mode
```

> [!WARNING]
> **`[DEMO MODE]` Disclaimer:** Synthetic signals generated in demo mode are artificial waveforms for software verification. They are NOT clinical ECG, medically valid data, diagnostic signals, or real patient records. Demo output is isolated under `reports/demo/` and never contaminates real MIT-BIH experimental reports.

### Demonstration Mode Sub-Paths:

1. **Synthetic Normal Path (`[DEMO MODE]`):**
   - Evaluates synthetic PQRST template waveforms through DSP $\rightarrow$ INT8 TinyMLEngine $\rightarrow$ StateScheduler.
   - Evaluates to `SLEEP` state (0 bytes transmitted), demonstrating radio suppression during normal rhythms.

2. **MIT-BIH Natural Exemplar Search (`[DEMO MODE — MIT-BIH EXEMPLAR]`):**
   - Searches local held-out test windows for natural confidences $\ge 0.85$.
   - **Factual Audit Finding:** Across the held-out test split, 137 windows naturally exceed threshold $0.85$ (max measured test confidence under INT8 model is `0.9766`).
   - **Scientific Integrity Assertion:** Per project rules, zero artificial overrides, threshold alterations, or fabricated scores were applied. All reported numbers directly trace to measured outputs in `reports/bandwidth_report.md`.

---

## 11. System Limitations & Assumptions

1. **Host-PC Simulation:** Latency and memory metrics are measured on a host x86_64 PC development environment (`VirtualMCU`), representing simulated MCU estimates rather than physical silicon measurements.
2. **Window-Level Baseline:** Baseline calculation assumes 400 bytes/window (16-bit, 200 samples) with 50% window overlap.

---

## 12. Reproduction Commands

```bash
# 1. Run full test suite regression
pytest

# 2. Run synthetic demo CLI
python -m src.pipeline.demo_mode

# 3. Generate resource profiling report
python -m src.pipeline.profile_report

# 4. Generate bandwidth reduction report
python -m src.pipeline.bandwidth_report
```
