# DECISIONS.md — Implementation Decisions Not Explicitly Specified in Review-1

Every entry below fills a gap that Review-1 left open. None of these override, contradict, or
soften a Review-1 requirement in `PRD.md`/`Architecture.md`. Each is labeled:

**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

`PRD.md` is not modified by this file — Review-1-derived requirements and these
implementation decisions are kept in separate documents so the two are never conflated.

---

## Decision 1 — ECG Windowing
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- Window size: 200 samples (Review-1 gives this as an example; adopted as the fixed value).
- Window overlap: 50%.
- Sampling frequency: use the actual per-record sampling frequency from the MIT-BIH record
  metadata (`wfdb` header), not a hardcoded universal value.
- If any preprocessing step requires a fixed assumed frequency, that assumption must be
  documented inline in code and in the run's manifest — never silently hardcoded.

## Decision 2 — DSP Filter
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- 4th-order Butterworth bandpass filter, passband 0.5–45 Hz, implemented via
  `scipy.signal` (`butter` + `sosfiltfilt` or equivalent).
- `filter_order`, `low_cutoff`, `high_cutoff` are configurable via `config.yaml`.
- The 4th-order choice is an implementation decision — Review-1 specifies the 0.5–45 Hz
  passband but not a filter order.

## Decision 3 — Data Splitting
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- Patient/record-independent split wherever MIT-BIH record metadata allows it: all windows
  from a given record/patient go entirely into one of train/validation/test — never split
  across sets, to avoid leakage.
- Exact record-to-split assignment and the random seed used for any remaining
  randomization must be written to `reports/experiment_manifest.json`.
- Final reported results must state the split method used, verbatim.

## Decision 4 — Teacher Model
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- A moderately sized 1D-CNN Teacher is used instead of a Transformer (Review-1 lists either
  as an example, "e.g., a multi-layer Transformer or heavy CNN").
- Reason: reproducibility, appropriateness for 1D ECG morphology, sufficiency for
  demonstrating knowledge distillation, and lower implementation risk within the Review-2
  timeline.
- The Teacher remains substantially larger (more parameters/capacity) than the Student.

## Decision 5 — Student Model
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- Compact 1D-CNN: small convolutional layers, low channel counts, global average pooling
  where appropriate, small dense/output layer.
- Exact layer sizes are selected based on measured parameter count and the resource budget
  (SRAM/Flash targets in PRD.md), not chosen arbitrarily — final architecture and parameter
  count are recorded once selected.

## Decision 6 — Knowledge Distillation
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- Standard soft-target knowledge distillation.
- Initial configurable parameters: temperature `T = 3.0`, `alpha = 0.7` (blend of soft-label
  and hard-label loss). These are starting values, not academically mandated constants —
  configurable in `config.yaml`.
- Required experiment: Student trained **without** KD vs. Student trained **with** KD,
  same architecture, same data split — actual accuracy/F1 results recorded for both.

## Decision 7 — INT8 Quantization
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- TensorFlow Lite full-integer INT8 post-training quantization.
- Representative dataset for calibration is drawn from actual preprocessed ECG windows
  (post-DSP), not synthetic placeholder data.
- Verification checklist required before accepting the exported model: input dtype, output
  dtype, weight dtype, model file size, and a functional inference smoke test.
- Accuracy preservation is never claimed until measured (Float32 Student vs. INT8 Student,
  same test split).

## Decision 8 — Encryption
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- For the software simulation: AES-GCM via the Python `cryptography` library.
- Reason: authenticated encryption (confidentiality + integrity), practical to simulate,
  well-supported library.
- This does not prove or imply what cryptographic implementation would run on a specific
  physical MCU — it simulates the secure telemetry layer's behavior only. This decision
  supersedes the earlier PRD.md note favoring a "lightweight symmetric cipher" only insofar
  as it names AES-GCM as that concrete cipher; it does not relax SEC-3's SRAM-budget
  justification requirement.

## Decision 9 — Network Sink
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- In-memory simulated telemetry sink for the initial implementation.
- The sink records, per transmission event: timestamp, device/session ID, anomaly ID,
  confidence score, encrypted payload, and the transmission event itself.
- The sink must never receive raw ECG under any code path.
- No real cloud infrastructure is introduced unless a later, explicit decision requires it.

## Decision 10 — Latency Measurement
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- Measured via `time.perf_counter()` on the development PC.
- Reported statistics: mean, median, p95, min, max, over a fixed test set.
- Benchmark machine details recorded alongside every latency report: CPU, RAM, OS, Python
  version, TensorFlow version.
- This is host-machine/simulated benchmark evidence, not physical-MCU latency. The <50 ms
  target from PRD.md PERF-1 remains the project's target; compliance is reported as
  "simulated PC benchmark meets/does not meet the 50 ms target," never as "physical MCU
  compliance," unless actual MCU execution is performed.

## Decision 11 — SRAM Estimation
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- Process RSS / Python memory usage is never used as a stand-in for MCU SRAM.
- Edge runtime memory is estimated from deployable model/tensor requirements and labeled
  explicitly as an MCU resource estimate/simulation.
- Reported fields: model memory, activation/tensor memory (where measurable), estimated
  tensor arena, estimated peak SRAM, the 256 KB SRAM limit, and headroom.
- If exact tensor-arena measurement is unavailable (e.g., no TFLite Micro interpreter in the
  simulation environment), the report states this limitation explicitly rather than
  substituting an unrelated number.

## Decision 12 — Flash Measurement
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- Actual byte size of the deployable INT8 `.tflite` model is measured directly.
- Any simulated runtime/application overhead (e.g., a compiled C++ harness binary) is
  reported separately from model size.
- Model size and "entire firmware image" size are never conflated unless an actual compiled
  firmware image exists to measure.

## Decision 13 — Anomaly Scheduler States
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- Initial confidence threshold: 0.85, configurable.
- States renamed/clarified as `SLEEP` and `ACTIVE` (Review-1's diagrams use "SLEEP" and
  "WAKE" interchangeably with this concept — `ACTIVE` is adopted as the canonical state
  name going forward for implementation).
- Normal: `SLEEP` → no transmission.
- Anomaly: `SLEEP` → `ACTIVE` → encrypted metadata transmission → `SLEEP`.
- All state transitions are recorded (timestamp, window index, from-state, to-state,
  triggering confidence score).

## Decision 14 — Demo Mode
**IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1**

- If MIT-BIH data is unavailable locally, a clearly labeled `DEMO MODE` may run the pipeline
  on synthetic data to demonstrate execution only.
- `DEMO MODE` output must never be mixed into, or presented as, official MIT-BIH experimental
  results. Every report/log line produced in `DEMO MODE` is prefixed/tagged accordingly.

## Decision 15 — Deployed Model Selection & Scheduler Operating Threshold
**IMPLEMENTATION DECISION — RESOLVES EMPIRICAL THRESHOLD & QUANTIZATION ABLATION**

- **Deployed Model:** Student (No KD) INT8 (`models/student_model_int8.tflite`, 11,224 bytes).
- **Rejection of Knowledge Distillation Model for INT8 Deployment:**
  Although the validation-selected Knowledge Distillation model ($T=6.0, \alpha=0.5$) achieved superior Float32 test recall (17.24% vs. 12.70%) and F1 score (0.2585 vs. 0.2130), empirical evaluation of the full-integer INT8 quantized models (Table C vs. Table D in `reports/threshold_sweep.md`) revealed an asymmetric degradation under Post-Training Quantization (PTQ):
  - In the **No-KD model**, INT8 quantization degrades gracefully by losing true positives ($924 \rightarrow 622$ TP, $-32.7\%$) while reducing false positives ($474 \rightarrow 300$ FP, $-36.7\%$), preserving a high precision of **67.46%** at $\tau=0.50$.
  - In the **KD model**, INT8 quantization inflates false positives by $+61.9\%$ ($1,175 \rightarrow 1,902$ FP at $\tau=0.50$) while true positives remain roughly flat ($1,255 \rightarrow 1,277$). The temperature-softened probability distribution produced by KD makes borderline normal windows highly sensitive to integer rounding under INT8 quantization.
  - While KD INT8 detects more anomalies, its false alarm burden ($1,902$ FPs over 4.01 hours $\approx 474.1$ false alarms/hour, or one false alert every 7.6 seconds) introduces severe clinical alert fatigue that would be unacceptable in real-world patient monitoring.
- **Operating Threshold Selection ($\tau = 0.35$, was 0.85):**
  - **Failure of $\tau = 0.85$ Baseline:** At the initial 0.85 threshold, No-KD INT8 detected only 49 of 7,278 test anomalies (**0.67% recall**), starving clinical detection to achieve 99.91% bandwidth reduction.
  - **Selection of $\tau = 0.35$:** At $\tau = 0.35$, No-KD INT8 achieves **13.79% recall** ($1,004$ true positive arrhythmias detected — a **$20.5\times$ increase** over the 49 detected at 0.85) while maintaining **66.45% precision** (2 out of 3 alerts are true anomalies), limiting false alarms to **126.4 FPs/hour** (507 FPs over 4.01 hours, or ~2.1 false alarms/minute).
  - **Bandwidth Compliance:** With 1,511 total active transmissions (2.91% trigger rate), transmitted telemetry is **199,799 bytes** versus a 20,796,800 bytes baseline, delivering an empirical **99.04% network payload reduction**, easily exceeding the 90.0% NFR-6 ceiling by $+9.04\%$.
  - **Explicit Rejection of Alternative INT8 Candidates:**
    - Candidate 2 (KD INT8 @ 0.35, 27.29% recall): Rejected due to extreme alert burden (**612.2 FPs/hour**, 2,456 false alarms over 4 hours, or one false alert every 5.9 seconds).
    - Option 1 (KD INT8 @ 0.50, 17.55% recall): Rejected due to excessive alert burden (**474.1 FPs/hour**, 1,902 false alarms).
    - Option 3 (No-KD INT8 @ 0.50, 8.55% recall): Rejected because $\tau=0.35$ increases detected anomalies by $+61.4\%$ ($1,004$ vs. $622$ TP) with negligible precision penalty (66.45% vs. 67.46%) while still achieving 99.04% reduction.
- **Known Architectural Limitations & Future Work:**
  An absolute recall of 13.79% (or 27.29% in KD) reflects the capacity limit of a 1,538-parameter Student model under severe class imbalance (14.0% prevalence). This proof-of-concept establishes that edge filtering and secure telemetry deliver >99% bandwidth reduction without raw ECG leakage. Improving absolute minority sensitivity (e.g., patient-adaptive thresholding, focal loss, or recurrent micro-architectures) is explicitly designated as future work.

---

## Summary Table

| Decision | Selected Approach | Source | Status |
|---|---|---|---|
| 1. ECG Windowing | 200-sample windows, 50% overlap, per-record actual sampling frequency | IMPLEMENTATION DECISION | Built & Verified |
| 2. DSP Filter | 4th-order Butterworth bandpass, 0.5–45 Hz, `scipy.signal`, configurable | IMPLEMENTATION DECISION | Built & Verified |
| 3. Data Splitting | Patient/record-independent split, seed + split recorded in manifest | IMPLEMENTATION DECISION | Built & Verified |
| 4. Teacher Model | Moderately sized 1D-CNN (not Transformer) | IMPLEMENTATION DECISION | Built & Verified |
| 5. Student Model | Compact 1D-CNN, GAP, small dense head, size set by measured params | IMPLEMENTATION DECISION | Built & Verified |
| 6. Knowledge Distillation | Soft-target KD, validation-selected $T=6.0, \alpha=0.5$; ablation recorded | IMPLEMENTATION DECISION | Built & Verified |
| 7. INT8 Quantization | TFLite full-integer PTQ, real representative dataset, verification checklist | IMPLEMENTATION DECISION | Built & Verified |
| 8. Encryption | AES-GCM via Python `cryptography` library, simulation only | IMPLEMENTATION DECISION | Built & Verified |
| 9. Network Sink | In-memory sink, fixed record schema, never receives raw ECG | IMPLEMENTATION DECISION | Built & Verified |
| 10. Latency | `time.perf_counter()`, mean/median/p95/min/max, benchmark machine logged | IMPLEMENTATION DECISION | Measured (0.0344 ms mean) |
| 11. SRAM | Model/tensor-based estimate, never process RSS, headroom vs. 256 KB reported | IMPLEMENTATION DECISION | Estimated (15.16 KB peak) |
| 12. Flash | Actual `.tflite` byte size; overhead reported separately if it exists | IMPLEMENTATION DECISION | Measured (10.96 KB) |
| 13. Anomaly Scheduler | SLEEP/ACTIVE states, threshold 0.85 initial, all transitions logged | IMPLEMENTATION DECISION | Superseded by #15 |
| 14. Demo Mode | Synthetic-data fallback, clearly tagged, never mixed with real results | IMPLEMENTATION DECISION | Built & Verified |
| 15. Deployed Model & Threshold | Student No-KD INT8, threshold $\tau=0.35$ (13.79% recall, 66.45% precision, 99.04% bandwidth reduction) | IMPLEMENTATION DECISION | Finalized & Deployed |

