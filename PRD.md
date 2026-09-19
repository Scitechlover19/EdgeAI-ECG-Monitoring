# PRD.md — Resource-Constrained Edge-AI Pipeline for Real-Time Privacy-Preserving Patient Monitoring

**Source of truth:** VIT Review-1 document, 22MIS0027 (Nancy Singh), SWE3004, dated 12.8.2026.
**Status:** Draft for Review-2 planning. Derived strictly from the Review-1 specification — no requirement added that contradicts it.

---

## 1. Problem Statement

Continuous remote patient monitoring (IoMT) architectures today stream raw physiological
signals (ECG) continuously to centralized cloud servers for inference. This creates three
compounding engineering failures documented in Review-1:

1. **Privacy/security exposure** — raw, personally identifiable biological waveforms
   traverse the network in transit, creating interception and data-sovereignty risk.
2. **Energy inefficiency** — the RF transceiver is the most power-hungry component on a
   wearable; continuous uplink drains battery and makes long-term monitoring infeasible.
3. **Computational mismatch** — standard deep learning models exceed the SRAM/FLOP budget
   of ultra-low-power microcontrollers (often < 256 KB SRAM), blocking direct edge deployment.

There is a need for a decentralized, hardware-aware pipeline that filters, classifies, and
acts on physiological data entirely at the edge, transmitting nothing except a minimal,
encrypted anomaly signal.

## 2. Objectives (from Review-1 §3)

1. **Signal Preprocessing** — implement DSP to remove baseline wander and high-frequency
   muscle artifact from MIT-BIH-derived ECG streams before inference.
2. **Model Compression & Optimization** — Knowledge Distillation from a heavy Teacher model
   into a lightweight 1D-CNN Student, followed by INT8 Post-Training Quantization (PTQ) to a
   sub-megabyte footprint.
3. **Resource-Constrained Simulation** — validate inference latency, SRAM utilization, and
   Flash footprint in a simulated micro-architecture environment.
4. **Privacy-Preserving Protocol** — an anomaly-driven state scheduler that keeps the radio
   off during normal rhythms and transmits only encrypted metadata on anomaly detection.

## 3. Proposed Solution (from Review-1 §5, §9)

A five-stage software pipeline, entirely simulated on standard PC hardware, no physical
fabrication:

```
MIT-BIH ECG → Data Ingestion (sliding windows)
           → DSP Preprocessing (bandpass + baseline correction)
           → TinyML Inference Engine (INT8 quantized 1D-CNN, distilled from Teacher)
           → Anomaly-Driven State Scheduler (radio OFF/ON)
           → Secure Metadata Telemetry (encrypted, on anomaly only)
```

Model path: Teacher model (heavy, feature-extraction) → Knowledge Distillation → Student
1D-CNN → INT8 Post-Training Quantization → `.tflite` flatbuffer → simulated MCU execution.

## 4. Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | System MUST read/parse MIT-BIH Arrhythmia Database records (`.csv`/`.mat`/PhysioNet format) and simulate a live ADC feed. | MUST |
| FR-2 | System MUST segment the continuous stream into sliding windows (Review-1 default: 200 points, overlapping) representing individual heartbeat/cardiac cycles. | MUST |
| FR-3 | System MUST apply a deterministic bandpass DSP filter (0.5–45 Hz per Review-1 §7 Phase 1) to every window prior to inference. | MUST |
| FR-4 | System MUST perform baseline wander correction and amplitude/noise normalization as part of DSP pre-processing. | MUST |
| FR-5 | System MUST train a Teacher model for feature extraction on a centralized workstation (not edge-deployed). | MUST |
| FR-6 | System MUST train a lightweight Student 1D-CNN via Knowledge Distillation from the Teacher (matching true labels + Teacher's output distribution). | MUST |
| FR-7 | System MUST apply INT8 Post-Training Quantization to the Student model and export a `.tflite` flatbuffer. | MUST |
| FR-8 | System MUST convert the `.tflite` model into a static C++ byte array (`.h` header) suitable for embedding in a simulated MCU binary. | SHOULD |
| FR-9 | System MUST execute on-device inference using only the quantized model within the simulated MCU environment. | MUST |
| FR-10 | System MUST classify each window's cardiac rhythm and produce a confidence score. | MUST |
| FR-11 | System MUST maintain the radio/network stack in a suppressed ("deep sleep") state whenever classification confidence indicates a normal baseline. | MUST |
| FR-12 | System MUST trigger the telemetry module only when the anomaly confidence threshold (Review-1 default: 0.85) is breached. | MUST |
| FR-13 | On anomaly, system MUST generate a lightweight metadata payload containing only timestamp, anomaly classification code, and confidence score — never the raw waveform. | MUST |
| FR-14 | System MUST encrypt the metadata payload before transmission. | MUST |
| FR-15 | System MUST log/report per-window: SRAM usage, inference latency, and radio state, for later profiling. | MUST |
| FR-16 | System SHOULD provide a virtual execution harness that runs the full pipeline on standard PC hardware (Windows/Linux), Python + C++ toolchain, without physical ICs. | MUST |
| FR-17 | System MAY expose a CLI or notebook interface for replaying a MIT-BIH record end-to-end for demonstration. | MAY |
| FR-18 | System MAY visualize SRAM/Flash/latency metrics as charts for the Review-2 report. | MAY |

## 5. Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-1 | Peak simulated SRAM utilization of the compiled inference engine + model MUST NOT exceed 256 KB. | MUST |
| NFR-2 | Total compiled Flash footprint MUST remain below 1 MB. | MUST |
| NFR-3 | Per-window latency (DSP filter + inference) MUST be under 50 ms. | MUST |
| NFR-4 | Under no system state or error condition may raw, unencrypted physiological data be transmitted over any network interface. | MUST |
| NFR-5 | The simulation environment MUST run on commodity PC hardware without requiring physical microcontrollers. | MUST |
| NFR-6 | Network payload reduction versus a continuous-streaming baseline SHOULD exceed 90% (Review-1 §5.5 claim to be empirically validated, not assumed). | SHOULD |
| NFR-7 | The pipeline SHOULD be deterministic/reproducible given a fixed random seed and dataset split. | SHOULD |

## 6. Security Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| SEC-1 | Raw ECG data MUST NEVER cross the simulated network boundary, under any code path, including error/exception paths. | MUST |
| SEC-2 | Telemetry payload MUST be encrypted before the `transmit_alert` call; plaintext payload must never be logged to a network-facing sink. | MUST |
| SEC-3 | Encryption primitives used MUST be justified against the SRAM budget (Review-1 literature review flags heavy cryptographic keys as a cause of buffer overflow on constrained MCUs — Al-Sharhan et al. 2022). Lightweight/stream ciphers or pre-shared symmetric keys should be favored over public-key schemes for the simulated MCU path. | MUST |
| SEC-4 | The State Scheduler MUST be the sole authority permitting a transmit call; no other module may invoke `SecureTelemetry.transmit_alert` directly. | MUST |
| SEC-5 | Metadata payload schema MUST be restricted to: timestamp, anomaly classification code, confidence score (per Review-1 use-case diagram annotation: "Payload contains only Timestamp, Risk Score, Anomaly ID — raw ECG never transmitted — Zero Data Leakage principle"). | MUST |

## 7. Performance Requirements

| ID | Metric | Target | Measured By |
|----|--------|--------|-------------|
| PERF-1 | Per-window inference latency | < 50 ms | Wall-clock benchmark harness, simulated MCU clock model if used |
| PERF-2 | Model size (`.tflite`) | Sub-megabyte | File size of exported flatbuffer |
| PERF-3 | Peak SRAM (tensor arena + working memory) | ≤ 256 KB | TFLite Micro interpreter arena report / memory profiler |
| PERF-4 | Flash footprint (model + inference engine binary) | < 1 MB | Compiled binary size |
| PERF-5 | Network payload reduction vs. continuous streaming | > 90% (target, to be validated) | Bytes transmitted (anomaly-only) vs. bytes that would be transmitted continuously over the same test duration |

## 8. Resource Constraints (hard ceilings, Review-1 §8.2)

- SRAM ≤ 256 KB
- Flash < 1 MB
- Inference latency < 50 ms/window
- Radio OFF by default; ON only on anomaly
- Zero raw-data transmission under any condition

## 9. Dataset Requirements

- MUST use the **MIT-BIH Arrhythmia Database** (PhysioNet) exclusively for training,
  validation, and testing (Review-1 §4).
- Data ingestion MUST support the database's native record format and produce fixed-size
  sliding windows (default 200 samples, overlapping) as the unit of inference.
- No other clinical dataset may be substituted without an explicit scope change, since
  Review-1 defines this as the sole data source.

## 10. Scope (Review-1 §4)

In scope:
- Algorithmic development: DSP filters + compressed 1D-CNN via Knowledge Distillation + INT8 PTQ.
- Full software-simulated pipeline execution (ingestion → inference → scheduling → telemetry).
- Performance profiling: Flash, SRAM, latency, bandwidth reduction.
- Exclusive use of MIT-BIH Arrhythmia Database.

## 11. Out of Scope (Review-1 §4, explicit exclusions)

- Manufacturing or fabrication of custom physical PCB hardware.
- Clinical deployment as a diagnostic medical device.
- Long-term human trials or any use with real patient data.
- Any claim of clinical validity or regulatory (e.g., FDA/CE) compliance.
- Treating simulated-PC resource numbers as literal proof of behavior on real MCU silicon
  (simulation ≠ hardware validation).

## 12. Review-2 Acceptance Criteria

The project is considered ready for Review-2 when:

1. An end-to-end run exists: a MIT-BIH record can be ingested, filtered, classified, and — on
   anomaly — produces an encrypted metadata payload, with the radio remaining off for normal
   windows.
2. A Teacher model and a distilled, quantized (INT8) Student model both exist with recorded
   accuracy/F1 on a held-out MIT-BIH split.
3. The `.tflite` Student model size, estimated tensor-arena SRAM, and measured per-window
   latency are reported against the NFR-1/NFR-3/PERF-2 targets, with pass/fail stated
   explicitly (not assumed).
4. The anomaly-driven scheduler is demonstrated to suppress transmission during a normal-only
   segment and to trigger exactly once per genuine anomaly window in a test run.
5. Automated tests cover DSPFilter, TinyMLEngine, StateScheduler, and SecureTelemetry at
   minimum (see Architecture.md §16).
6. All metrics reported are labeled as **simulated/PC-estimated**, not measured on physical
   silicon (see AGENTS.md rule on PC RAM vs MCU SRAM).

## 13. Measurable Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Classification accuracy (Student, INT8) on MIT-BIH held-out test split | Reported; SHOULD not degrade more than a defined tolerance (e.g., 2–3 pts) vs. Float32 Student — exact tolerance to be fixed during Phase 2 experiments, not asserted now |
| Estimated peak SRAM (tensor arena) | ≤ 256 KB |
| `.tflite` file size | < 1 MB (target: sub-MB per objective) |
| Mean per-window latency (DSP + inference) | < 50 ms |
| False "radio ON" triggers on a normal-only test segment | 0 |
| Missed anomaly triggers on a labeled-anomalous test segment | Reported explicitly; not silently omitted |
| Raw ECG bytes observed on the simulated network interface | 0, always |
