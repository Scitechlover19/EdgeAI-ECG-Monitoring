# Architecture.md — Resource-Constrained Edge-AI Pipeline

**Derived from:** Review-1 §5 (Proposed System), §9 (System Architecture), §10 (UML Diagrams).
This document does not alter the Review-1 architecture; it formalizes it for implementation.

---

## 1. System Context

Actors (from Review-1 Use Case Diagram, Fig 10.1):
- **Patient / Wearable Node** — source of the ECG stream (simulated from MIT-BIH in this
  project; no physical wearable is built).
- **ML Engineer** — trains/distills/quantizes models offline, deploys the `.tflite` binary.
- **Hospital System / Clinical Staff** — receiving end of secure alerts (simulated sink; no
  real hospital integration is built).

The system boundary is entirely software: a simulated MCU process on commodity PC hardware.
No physical IC, radio, or sensor exists. All "hardware" behavior (SRAM ceiling, radio
on/off, transmission) is modeled in software.

## 2. High-Level Architecture

```
MIT-BIH Arrhythmia DB (PhysioNet)
        │
        ▼
┌───────────────────┐
│  DataIngestion     │  sliding-window segmentation (200 pts, overlap)
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  DSPFilter         │  bandpass (0.5–45 Hz) + baseline wander correction
└─────────┬──────────┘        + amplitude/noise normalization
          ▼
┌───────────────────────────────────────────┐
│  Offline (once, not on simulated MCU path) │
│  Teacher Model → Knowledge Distillation →  │
│  Student 1D-CNN → INT8 PTQ → .tflite       │
└─────────┬───────────────────────────────────┘
          ▼  (deploy .tflite to edge path)
┌───────────────────┐
│  TinyMLEngine      │  loads quantized model, runs inference on VirtualMCU
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  StateScheduler    │  radio OFF (normal) / radio ON (anomaly ≥ threshold)
└─────────┬──────────┘
          ▼ (only on anomaly)
┌───────────────────┐
│  SecureTelemetry   │  format metadata → encrypt → transmit
└────────────────────┘

All of the above execute inside:
┌───────────────────┐
│  VirtualMCU        │  enforces SRAM ceiling, models latency, hosts the loop
└───────────────────┘
orchestrated by:
┌───────────────────┐
│  PipelineController│  wires ingestion → DSP → inference → scheduler → telemetry
└───────────────────┘
```

This matches Review-1 Fig. 9.1 exactly: Data Ingestion → DSP Pre-Processing → (offline) Model
Compression → Edge Inference Engine → Anomaly State Scheduler → Secure Telemetry.

## 3. Component Architecture

Seven modules, matching Review-1 Fig. 10.2 (Class Diagram) plus two orchestration/simulation
modules needed to make the pipeline executable:

| Module | Review-1 origin | Role |
|--------|-----------------|------|
| `DataIngestion` | Fig 10.2 class `DataIngestion` | Reads MIT-BIH windows, normalizes |
| `DSPFilter` | Fig 10.2 class `DSP_Filter` | Bandpass + baseline correction |
| `TinyMLEngine` | Fig 10.2 class `TinyML_Engine` | Loads `.tflite`, runs inference |
| `StateScheduler` | Fig 10.2 class `StateScheduler` | Radio on/off decision |
| `SecureTelemetry` | Fig 10.2 class `SecureTelemetry` | Metadata format, encrypt, transmit |
| `VirtualMCU` | Implied by §7 Phase 4 + NFR-5 | Hosts the simulated constrained runtime |
| `PipelineController` | Implied by Sequence Diagram `MainController` | Orchestrates the loop |

## 4. Data Flow

1. `PipelineController` requests the next window from `DataIngestion`.
2. `DataIngestion` returns a raw 200-point window (`raw_window: float[]`).
3. `PipelineController` passes it to `DSPFilter.apply_bandpass()` then
   `remove_baseline_wander()`, producing `clean_window: float[]`.
4. `PipelineController` calls `TinyMLEngine.invoke_inference(clean_window)`, receiving a
   `confidence_score: float` and label.
5. `PipelineController` calls `StateScheduler.check_threshold(confidence_score)`.
   - If below threshold: `trigger_sleep_mode()` — radio OFF, zero bytes transmitted.
   - If at/above threshold: `trigger_wake_mode()` — radio ON, then
     `SecureTelemetry.format_metadata()` → `encrypt_payload()` → `transmit_alert()`.
6. State returns to SLEEP; loop continues with the next window.

This is exactly the Review-1 Sequence Diagram (Fig 10.3): the `alt` block with the two
mutually exclusive branches (`Risk < Threshold` vs `Risk >= Threshold`) is the scheduler's
core contract — **exactly one branch fires per window**.

## 5. Module Responsibilities, Interfaces, Failure Modes, Tests

### 5.1 `DataIngestion`
- **Input:** MIT-BIH record path/ID; window size (default 200); overlap.
- **Output:** `float[]` raw window per call; end-of-stream signal.
- **Responsibilities:** parse MIT-BIH format, produce sliding windows, normalize amplitude at ingestion boundary if required by the model contract.
- **Dependencies:** PhysioNet MIT-BIH reader (e.g., `wfdb`), filesystem.
- **Failure modes:** corrupt/missing record; window shorter than 200 at record end; unsupported sample rate.
- **Tests:** correct window count for a known record length; overlap arithmetic; end-of-stream handling; malformed-file rejection.

### 5.2 `DSPFilter`
- **Input:** `raw_window: float[]`; filter params (`low_cutoff=0.5`, `high_cutoff=45.0`, `filter_order`).
- **Output:** `clean_window: float[]`.
- **Responsibilities:** bandpass filtering, baseline wander removal, amplitude/noise normalization — deterministic, no learned parameters.
- **Dependencies:** DSP library (`scipy.signal`).
- *[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #2]: 4th-order Butterworth, all three params configurable via `config.yaml`.*
- **Failure modes:** filter instability on edge-of-window artifacts; NaN/Inf propagation from bad input; incorrect sample-rate assumption skewing cutoffs.
- **Tests:** known synthetic signal (sine + noise) filtered to expected SNR improvement; NaN-input rejection; output length equals input length.

### 5.3 Offline Model Pipeline (Teacher → Distillation → Student → PTQ)
Not a runtime module on the simulated MCU — a training-time pipeline producing the `.tflite`
artifact consumed by `TinyMLEngine`. Documented here because Review-1 treats it as part of
the architecture (§5.3, §7 Phase 2–3).
- **Input:** cleaned, labeled MIT-BIH windows.
- **Output:** `student_model.tflite` (INT8), plus a Teacher checkpoint (not deployed).
- **Responsibilities:** train Teacher; distill Student (soft-label + hard-label loss); run TFLite Converter Post-Training Quantization (Float32 → INT8); export `.tflite` and, if needed, a C++ header byte array.
- **Dependencies:** TensorFlow/Keras (or equivalent), TFLite Converter, TFLite Micro (for header export).
- **Failure modes:** accuracy collapse after quantization; representative-dataset omission during PTQ causing bad calibration; distillation temperature/weight misconfiguration.
- **Tests:** accuracy delta between Float32 Student and INT8 Student is measured and reported (never assumed to be zero); model file size assertion (< 1 MB); calibration-set presence check.
- *[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #3–#7]: patient/record-independent split; Teacher = moderately sized 1D-CNN (not Transformer); Student = compact 1D-CNN with GAP; KD with T=3.0, alpha=0.7 plus a required with-KD-vs-without-KD comparison; INT8 PTQ with a real representative dataset and an explicit dtype/size/functionality verification checklist.*

### 5.4 `TinyMLEngine`
- **Input:** `clean_window: float[]`; `tflite_model_array`; `confidence_threshold` (default 0.85).
- **Output:** `confidence_score: float`; anomaly label (`get_anomaly_label()`).
- **Responsibilities:** load the quantized model once, run inference per window inside the `VirtualMCU` tensor arena, expose confidence + label.
- **Dependencies:** TFLite Micro interpreter (or a Python TFLite runtime standing in for it during simulation), `VirtualMCU` tensor arena allocator.
- **Failure modes:** tensor arena overflow (`tensor_arena_size` too small); model load failure (corrupt flatbuffer); inference exceeding the 50 ms budget.
- **Tests:** inference on a known-normal window yields low confidence; on a known-anomalous window yields high confidence; arena-overflow is caught and reported, not silently truncated; latency benchmark against the 50 ms budget.
- *[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #10]: latency measured with `time.perf_counter()`, reporting mean/median/p95/min/max plus the benchmark machine's CPU/RAM/OS/Python/TensorFlow versions; explicitly labeled as host-PC simulated evidence, not physical-MCU latency.*

### 5.5 `StateScheduler`
- **Input:** `confidence_score: float` from `TinyMLEngine`.
- **Output:** `current_state: "SLEEP" | "ACTIVE"`; `is_radio_active: bool`.
- **Responsibilities:** compare score to threshold; the **sole** module allowed to authorize a call into `SecureTelemetry`; must guarantee exactly one branch fires per window (per Fig 10.3 `alt`); record every state transition (timestamp, window index, from-state, to-state, triggering confidence).
- **Dependencies:** `TinyMLEngine` output, `SecureTelemetry` (one-directional trigger only).
- **Failure modes:** threshold misconfiguration causing over/under-triggering; race condition if the pipeline is ever made concurrent; state left in ACTIVE after a failed transmission.
- **Tests:** score just below threshold → SLEEP, radio OFF, zero telemetry calls; score at/above threshold → ACTIVE, exactly one telemetry call; state resets to SLEEP after transmission per Fig 10.3.
- *[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #13]: state name "ACTIVE" adopted as the canonical name for Review-1's "WAKE" concept; threshold 0.85, configurable.*

### 5.6 `SecureTelemetry`
- **Input:** anomaly id, timestamp, confidence score (from `StateScheduler` trigger only).
- **Output:** encrypted byte payload; transmit acknowledgement (simulated).
- **Responsibilities:** `format_metadata()` restricted to {timestamp, anomaly id, confidence score}; `encrypt_payload()`; `transmit_alert()`.
- **Dependencies:** encryption primitive (lightweight symmetric cipher, chosen for MCU-class SRAM budget), simulated network sink.
- **Failure modes:** payload accidentally includes raw waveform data (must be structurally impossible, not just avoided by convention); encryption key mismanagement; transmit failure not surfaced to the scheduler.
- **Tests:** payload schema is asserted (only 3 fields, no array of raw samples); payload is not human-readable pre-encryption in transit logs; simulated transmit failure is handled without crashing the loop.

### 5.7 `VirtualMCU`
- **Input:** configured SRAM ceiling (256 KB), Flash ceiling (1 MB), a clock/latency model.
- **Output:** enforcement signals (overflow exceptions), profiling reports (peak SRAM, Flash size, per-window latency).
- **Responsibilities:** host the tensor arena and working memory for `TinyMLEngine`; track and cap memory usage; provide the timing harness for the 50 ms budget; make explicit that this is a **simulation**, not real hardware.
- **Dependencies:** memory profiler, timer.
- **Failure modes:** silently allowing SRAM overflow (must raise, not warn-and-continue); conflating PC process RAM with the simulated SRAM ceiling (see AGENTS.md — this must never be reported as equivalent).
- **Tests:** allocation beyond 256 KB raises; Flash-size check on the compiled artifact; latency harness reproducibility across repeated runs.

### 5.8 `PipelineController`
- **Input:** dataset/record selection, configuration file.
- **Output:** orchestrated end-to-end run; aggregate metrics/log.
- **Responsibilities:** wire the modules in the exact Fig 10.3 sequence; own the main loop; own configuration loading; own top-level error handling and logging.
- **Dependencies:** all other modules.
- **Failure modes:** any module failure surfacing as a silent pass-through instead of a logged, explicit failure.
- **Tests:** full end-to-end integration test on a short MIT-BIH excerpt with a known mix of normal/anomalous windows, asserting radio state transitions and telemetry call count.

## 6. State Machine

States: `SLEEP` (default), `ACTIVE` (transient, only during an active anomaly transmission —
naming per DECISIONS.md #13; Review-1's diagrams refer to this same state as "WAKE").

```
        confidence < threshold
   ┌───────────────────────────┐
   │                           ▼
[SLEEP] ──confidence ≥ threshold──▶ [ACTIVE] ──transmission_complete──▶ [SLEEP]
```

Invariant (from Fig 10.3 `alt` block): exactly one of the two branches executes per window;
no window may skip evaluation.

## 7. ML Pipeline

Teacher (heavy, feature-extraction capable) → Knowledge Distillation (soft targets +
true labels) → Student (shallow 1D-CNN) → INT8 Post-Training Quantization → `.tflite`.
The Teacher is never deployed to the edge path; it exists purely to shape the Student during
training (Review-1 §7 Phase 2).

## 8. DSP Pipeline

Deterministic, no learned parameters: bandpass filter (Butterworth or equivalent, 0.5–45 Hz)
→ baseline wander correction → amplitude/noise normalization. Runs identically at
training-data-prep time and at simulated-inference time, so train/serve skew is not
introduced by preprocessing.

## 9. Quantization Pipeline

Float32 Student → TFLite Converter with a representative dataset for calibration → INT8
weights and activations → `.tflite` flatbuffer → optional C++ header (`unsigned char[]`) for
static compilation into the MCU-style binary (Review-1 §7 Phase 3).

## 10. Virtual MCU Architecture

A software-enforced constraint layer, not a cycle-accurate CPU emulator (out of scope). It
must: (a) cap tensor-arena + working-set memory at 256 KB and raise on overflow, (b) cap the
reported Flash size of the compiled artifact at < 1 MB, (c) provide a latency harness for the
50 ms per-window budget. All numbers it reports are **PC-simulated estimates**, explicitly
labeled as such in every output (see AGENTS.md).

## 11. Security Architecture

- Raw ECG never leaves `DataIngestion`/`DSPFilter`/`TinyMLEngine` boundary.
- `StateScheduler` is the only caller of `SecureTelemetry.transmit_alert`.
- Payload schema is fixed and minimal (timestamp, anomaly id, confidence score).
- Encryption primitive chosen for a constrained-SRAM budget (Review-1 literature flags heavy
  cryptography as a cause of MCU buffer overflow — Al-Sharhan et al. 2022).
- *[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #8]:
  the simulation uses AES-GCM via the Python `cryptography` library (authenticated
  encryption: confidentiality + integrity). This proves the secure-telemetry-layer
  simulation only, not a specific physical-MCU cryptographic implementation.*

## 12. Telemetry Architecture

`format_metadata(id, time) → encrypt_payload(data) → transmit_alert(encrypted_data) → ack`.
Sink is a simulated "Hospital System / Cloud DB" endpoint (per Fig 10.1); no real hospital
integration is in scope.
*[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #9]:
the sink is an in-memory store recording timestamp, device/session ID, anomaly ID,
confidence, encrypted payload, and the transmission event; it must never receive raw ECG;
no real cloud infrastructure is introduced at this stage.*

## 13. Resource Measurement Methodology

- **SRAM:** *[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #11]*
  process RSS / Python memory usage is never used as a stand-in for MCU SRAM. Instead, SRAM is
  estimated from deployable model/tensor requirements (model memory, activation/tensor memory
  where measurable, estimated tensor arena) and reported against the 256 KB limit with
  explicit headroom. If exact tensor-arena measurement is unavailable, the report states that
  limitation rather than substituting an unrelated number. Always labeled as an MCU resource
  estimate/simulation, never equated to native PC process RAM.
- **Flash:** *[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #12]*
  measured as the actual byte size of the exported INT8 `.tflite` file. Any simulated
  runtime/application overhead (e.g., a compiled C++ harness) is reported separately and never
  conflated with "entire firmware image" size unless such an image actually exists.
- **Latency:** *[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #10]*
  measured via `time.perf_counter()` over `DSPFilter` + `TinyMLEngine.invoke_inference` per
  window, over a fixed test set, reporting mean/median/p95/min/max. The benchmark machine's
  CPU, RAM, OS, Python version, and TensorFlow version are recorded alongside every latency
  report. This is host-PC simulated benchmark evidence; it is never presented as physical-MCU
  latency unless actual MCU execution exists.
- **Bandwidth reduction:** bytes actually transmitted (anomaly windows only) compared against
  the bytes that continuous raw streaming of the same test duration would have required, at a
  stated sample rate.

## 14. Error Handling

- Every module raises explicit, typed exceptions on failure — no silent fallback that could
  mask an SRAM overflow, a corrupted model, or a scheduler bypass.
- `PipelineController` logs every exception with the window index and module name.
- A telemetry transmit failure must leave the scheduler in a well-defined state (retry or
  fail-logged), never a state that could be mistaken for "sent."

## 15. Testing Architecture

Layered:
1. **Unit tests** per module (§5, "Tests" rows above).
2. **Integration test**: full pipeline over a short, labeled MIT-BIH excerpt, asserting
   correct SLEEP/WAKE transitions and telemetry call count.
3. **Resource tests**: SRAM ceiling enforcement, Flash size assertion, latency budget
   assertion — these are pass/fail gates against PRD.md NFR-1/NFR-2/NFR-3.
4. **Security tests**: assert no raw ECG sample ever appears in any object passed to
   `SecureTelemetry` or logged to a network-facing sink.

*[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #14]:
if MIT-BIH data is unavailable locally, a clearly labeled `DEMO MODE` may run the pipeline
on synthetic data to demonstrate execution only; `DEMO MODE` output is never mixed into, or
presented as, official MIT-BIH experimental results.*

## 17. Implementation Decisions Not Explicitly Specified in Review-1

Several technical details required to make this architecture buildable are not fixed by the
Review-1 document (e.g., exact filter order, split strategy, Teacher architecture, KD
hyperparameters, the specific encryption primitive, latency/SRAM measurement methodology).
Every such choice is recorded, justified, and explicitly labeled in **`DECISIONS.md`** —
they are implementation decisions, not Review-1 requirements, and `PRD.md` is not modified to
present them as originating from Review-1. Inline markers throughout this document
(`[IMPLEMENTATION DECISION — NOT EXPLICITLY SPECIFIED IN REVIEW-1, see DECISIONS.md #N]`)
point to the relevant `DECISIONS.md` entry.
