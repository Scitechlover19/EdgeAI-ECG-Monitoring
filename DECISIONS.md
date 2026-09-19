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

---

## Summary Table

| Decision | Selected Approach | Source | Status |
|---|---|---|---|
| 1. ECG Windowing | 200-sample windows, 50% overlap, per-record actual sampling frequency | IMPLEMENTATION DECISION | Not yet built |
| 2. DSP Filter | 4th-order Butterworth bandpass, 0.5–45 Hz, `scipy.signal`, configurable | IMPLEMENTATION DECISION | Not yet built |
| 3. Data Splitting | Patient/record-independent split, seed + split recorded in manifest | IMPLEMENTATION DECISION | Not yet built |
| 4. Teacher Model | Moderately sized 1D-CNN (not Transformer) | IMPLEMENTATION DECISION | Not yet built |
| 5. Student Model | Compact 1D-CNN, GAP, small dense head, size set by measured params | IMPLEMENTATION DECISION | Not yet built |
| 6. Knowledge Distillation | Soft-target KD, T=3.0, alpha=0.7 (configurable); KD-vs-no-KD experiment required | IMPLEMENTATION DECISION | Not yet built |
| 7. INT8 Quantization | TFLite full-integer PTQ, real representative dataset, verification checklist | IMPLEMENTATION DECISION | Not yet built |
| 8. Encryption | AES-GCM via Python `cryptography` library, simulation only | IMPLEMENTATION DECISION | Not yet built |
| 9. Network Sink | In-memory sink, fixed record schema, never receives raw ECG | IMPLEMENTATION DECISION | Not yet built |
| 10. Latency | `time.perf_counter()`, mean/median/p95/min/max, benchmark machine logged | IMPLEMENTATION DECISION | Not yet built — will be MEASURED LATER |
| 11. SRAM | Model/tensor-based estimate, never process RSS, headroom vs. 256 KB reported | IMPLEMENTATION DECISION | Not yet built — will be MEASURED LATER |
| 12. Flash | Actual `.tflite` byte size; overhead reported separately if it exists | IMPLEMENTATION DECISION | Not yet built — will be MEASURED LATER |
| 13. Anomaly Scheduler | SLEEP/ACTIVE states, threshold 0.85 configurable, all transitions logged | IMPLEMENTATION DECISION | Not yet built |
| 14. Demo Mode | Synthetic-data fallback, clearly tagged, never mixed with real results | IMPLEMENTATION DECISION | Not yet built |
