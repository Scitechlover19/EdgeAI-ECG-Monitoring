# Plan.md — Implementation Plan

Concrete technical choices referenced below (window/overlap, filter order, split strategy,
Teacher/Student architecture family, KD hyperparameters, quantization verification,
encryption, sink, latency/SRAM/Flash measurement methodology, scheduler state names, demo
mode) are **implementation decisions**, not Review-1 requirements — see `DECISIONS.md` for
the full rationale and labeling of each.

Priorities, per the brief:
1. End-to-end pipeline → 2. DSP → 3. Student model → 4. TFLite INT8 → 5. Edge inference →
6. Scheduler → 7. Secure telemetry → 8. Resource profiling → 9. Tests → 10. Results →
11. Documentation.

Status values: `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`. All tasks start `TODO`.
**No implementation has started. This plan is for approval before large-scale build.**

---

## P0 — Must work for Review-2

| Task ID | Description | Files | Dependencies | Acceptance Criteria | Test | Status |
|---------|-------------|-------|---------------|----------------------|------|--------|
| P0-1 | Project scaffolding: repo layout per AGENTS.md §5, config loader, logging setup | `src/pipeline/config.py`, `config/config.yaml`, `pyproject.toml` | — | `config.yaml` loads window size=200, overlap=50%, filter_order=4, low_cutoff=0.5, high_cutoff=45.0, threshold=0.85, T=3.0, alpha=0.7, SRAM=256KB, Flash=1MB, latency=50ms (see DECISIONS.md #1,#2,#6,#13) | `tests/test_config.py` loads and validates all fields | DONE |
| P0-2 | `DataIngestion`: MIT-BIH reader + sliding window segmentation (200 samples, 50% overlap, per-record actual sampling frequency) | `src/ingestion/data_ingestion.py` | P0-1 | Given a MIT-BIH record, yields correct number of 200-pt windows at 50% overlap using the record's own sampling frequency (DECISIONS.md #1) | `tests/test_data_ingestion.py`: window count, overlap math, EOF handling, per-record frequency read | DONE |
| P0-3 | `DSPFilter`: 4th-order Butterworth bandpass (0.5–45 Hz) via `scipy.signal` + baseline wander correction + normalization, all params in config.yaml | `src/dsp/dsp_filter.py` | P0-1 | Synthetic sine+noise input filtered to expected SNR improvement; output same length as input; filter_order/low_cutoff/high_cutoff read from config (DECISIONS.md #2) | `tests/test_dsp_filter.py` | DONE |
| P0-4 | Data prep script: apply DSPFilter to full MIT-BIH set; patient/record-independent train/val/test split; write experiment manifest (split + seed) | `src/models/prepare_data.py`, `reports/experiment_manifest.json` | P0-2, P0-3 | No record's windows appear in more than one split; exact split and seed written to manifest (DECISIONS.md #3) | `tests/test_prepare_data.py`: split sizes, record-level leakage check | DONE |
| P0-5 | Teacher model: moderately sized 1D-CNN, define + train on labeled windows | `src/models/teacher.py`, `src/models/train_teacher.py` | P0-4 | Teacher model builds, forward pass produces correct shape, parameter count measured (DECISIONS.md #4) | `tests/test_teacher.py`: forward pass shape check, parameter count audit | DONE |
| P0-6 | Student 1D-CNN: small conv layers, low channels, global average pooling, small dense head | `src/models/student.py` | P0-4 | Student model builds, forward pass produces correct output shape, parameter count measured and logged (DECISIONS.md #5) | `tests/test_student.py` | DONE |
| P0-7 | Knowledge Distillation training loop (Teacher → Student), T=3.0, alpha=0.7 configurable | `src/models/distill.py` | P0-5, P0-6 | Student trained with soft+hard label loss; accuracy reported vs. Teacher on same held-out split | `tests/test_distill.py`: loss decreases over a few epochs on a small subset | DONE |
| P0-7b | Ablation: Student trained WITHOUT KD vs. Student WITH KD, same architecture/split, real results recorded | `src/models/train_student_no_kd.py`, `reports/kd_ablation.md` | P0-6, P0-7 | Both accuracy numbers reported side by side, no fabricated comparison (DECISIONS.md #6) | `tests/test_kd_ablation.py` | DONE |
| P0-8 | INT8 Post-Training Quantization (real representative dataset) + `.tflite` export + verification checklist | `src/models/quantize.py`, `reports/quantization_report.md` | P0-7 | `.tflite` produced, size < 1 MB; input/output/weight dtypes verified INT8; functional inference smoke test passes; Float32-vs-INT8 accuracy delta measured and logged (DECISIONS.md #7) | `tests/test_quantize.py`: file exists, size assertion, dtype checks, accuracy-delta report | DONE |
| P0-9 | `VirtualMCU`: SRAM estimate (model + tensor memory, never process RSS) + Flash size check + latency harness | `src/edge/virtual_mcu.py` | P0-1 | Reports model memory, tensor/activation memory where measurable, estimated tensor arena, estimated peak SRAM, limit, headroom; states explicitly if exact arena is unavailable (DECISIONS.md #11, #12) | `tests/test_virtual_mcu.py`: overflow raises, harness returns a duration, RSS never used as SRAM proxy | DONE |
| P0-10 | `TinyMLEngine`: load `.tflite`, run inference inside `VirtualMCU`, time with `time.perf_counter()` | `src/edge/tinyml_engine.py` | P0-8, P0-9 | Given a clean window, returns confidence score + label; known-normal vs known-anomalous windows differ correctly; per-call latency captured (DECISIONS.md #10) | `tests/test_tinyml_engine.py` | DONE |
| P0-11 | `StateScheduler`: threshold check (0.85, configurable), SLEEP/ACTIVE state machine, transition logging | `src/scheduler/state_scheduler.py` | P0-1 | Score < threshold → SLEEP, 0 telemetry calls; score ≥ threshold → ACTIVE, exactly 1 telemetry call; resets to SLEEP after; every transition logged (DECISIONS.md #13) | `tests/test_state_scheduler.py` | DONE |
| P0-12 | `SecureTelemetry`: fixed-schema metadata + AES-GCM encryption (`cryptography` lib) + in-memory sink | `src/telemetry/secure_telemetry.py`, `src/telemetry/sink.py` | P0-1 | Payload contains only {timestamp, anomaly_id, confidence}; rejects raw-array input; AES-GCM encrypts before "transmit"; sink records timestamp/session ID/anomaly ID/confidence/encrypted payload/event, never raw ECG (DECISIONS.md #8, #9) | `tests/test_secure_telemetry.py`, `tests/test_sink.py`: schema assertion, no-raw-ECG assertion, AES-GCM roundtrip | DONE |
| P0-13 | `PipelineController`: wire ingestion → DSP → inference → scheduler → telemetry | `src/pipeline/controller.py` | P0-2, P0-3, P0-10, P0-11, P0-12 | End-to-end run over a short labeled MIT-BIH excerpt produces correct SLEEP/ACTIVE sequence and telemetry call count matching ground-truth anomaly windows | `tests/test_pipeline_integration.py` | DONE |
| P0-14 | Resource profiling report: SRAM estimate, Flash size, latency stats (mean/median/p95/min/max + benchmark machine) vs. NFR-1/NFR-2/NFR-3 | `src/pipeline/profile_report.py`, `reports/resource_report.md` | P0-9, P0-10, P0-13 | Report states pass/fail against each budget explicitly, all values labeled `[SIMULATED]`, benchmark machine spec included (DECISIONS.md #10, #11, #12) | `tests/test_profile_report.py`: report generation doesn't crash and includes all metrics | DONE |
| P0-15 | Bandwidth-reduction measurement (anomaly-only vs. continuous baseline) | `src/pipeline/bandwidth_report.py` | P0-13 | Reports actual bytes transmitted vs. hypothetical continuous-streaming bytes over the same run, real numbers | `tests/test_bandwidth_report.py` | DONE |
| P0-16 | `DEMO MODE`: synthetic-data fallback path, clearly tagged, isolated from real MIT-BIH result files | `src/pipeline/demo_mode.py` | P0-13 | Runs full pipeline on synthetic data when MIT-BIH is unavailable; all output tagged `[DEMO MODE]`; never written into `reports/resource_report.md` or other official result files (DECISIONS.md #14) | `tests/test_demo_mode.py`: tagging present, output isolated from official reports | DONE |

## P1 — Important (strengthens Review-2, not blocking)

| Task ID | Description | Files | Dependencies | Acceptance Criteria | Test | Status |
|---------|-------------|-------|---------------|----------------------|------|--------|
| P1-1 | C++ header export (`.h` byte array) for the quantized model | `src/models/export_header.py` | P0-8 | Valid C-style `unsigned char[]` array file generated matching `.tflite` bytes | `tests/test_export_header.py` | TODO |
| P1-2 | CLI to replay a full MIT-BIH record end-to-end with live console output | `src/cli/run_demo.py` | P0-13 | Runs a record from the command line, prints per-window state and any triggered alerts | Manual + smoke test `tests/test_cli_smoke.py` | TODO |
| P1-3 | Metrics visualization: SRAM/Flash/latency bar charts, ROC/PR curve for classifier | `src/reports/plots.py` | P0-14 | Generates PNG/HTML plots from real report data, no fabricated data points | `tests/test_plots.py`: files generated | TODO |
| P1-4 | Encryption module hardening: symmetric cipher selection + key management via config/env | `src/telemetry/crypto.py` | P0-12 | Keys loaded from env/config, never hardcoded or logged | `tests/test_crypto.py` | TODO |
| P1-5 | Expanded test coverage: edge cases for DSPFilter (clipped signal, zero-input) and StateScheduler (threshold boundary exactly at 0.85) | `tests/test_dsp_filter.py`, `tests/test_state_scheduler.py` | P0-3, P0-11 | Boundary conditions explicitly tested and passing | Same files | TODO |
| P1-6 | Reproducibility manifest: seeds, library versions, dataset split recorded per AGENTS.md §8 | `reports/experiment_manifest.json` | P0-4–P0-8 | Manifest generated automatically by training scripts | `tests/test_manifest.py` | TODO |

## P2 — Polish

| Task ID | Description | Files | Dependencies | Acceptance Criteria | Test | Status |
|---------|-------------|-------|---------------|----------------------|------|--------|
| P2-1 | Notebook walkthrough of the full pipeline for demo/presentation | `notebooks/demo.ipynb` | P0-13 | Notebook runs top-to-bottom without error on a fresh environment | Manual run | TODO |
| P2-2 | README with setup, run, and reproduction instructions | `README.md` | P0-1 | A new contributor can set up and run the demo from README alone | Manual review | TODO |
| P2-3 | Docstring/API documentation pass across all modules | all `src/` files | P0-1–P0-13 | Every public class/function has a complete docstring | Lint check (e.g., `pydocstyle`) | TODO |
| P2-4 | Latency/SRAM sensitivity sweep (varying window size, quantization scheme) for discussion in report | `src/pipeline/sensitivity_sweep.py` | P0-14 | Real sweep data generated and plotted, not projected/estimated | `tests/test_sensitivity_sweep.py` | TODO |
| P2-5 | Final Review-2 report assembly pulling real metrics from `reports/` | `reports/review2_report.md` | P0-14, P0-15, P1-3, P1-6 | Every number in the report traces to a file in `reports/`; no hand-typed metric | Manual cross-check against `reports/` contents | TODO |

---

## Dependency Graph (high level)

```
P0-1 → P0-2, P0-3, P0-9, P0-11, P0-12
P0-2 + P0-3 → P0-4 → P0-5 → P0-7 → P0-7b
                     P0-6 ↗       ↗
P0-7 → P0-8 → P0-10 (needs P0-9 too)
P0-10 + P0-11 + P0-12 → P0-13
P0-13 → P0-14, P0-15, P0-16
P0-8 → P1-1
P0-13 → P1-2
P0-14 → P1-3, P2-4
P0-12 → P1-4
P0-3, P0-11 → P1-5
P0-4..P0-8 → P1-6
P0-13 → P2-1
P0-1 → P2-2
all → P2-3
P0-14, P0-15, P1-3, P1-6 → P2-5
```
