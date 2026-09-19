# AGENTS.md — Rules for AI Coding Agents

This file governs any AI agent (Claude Code or otherwise) working on this repository. It is
binding. If a request conflicts with this file, the agent must flag the conflict rather than
silently comply.

## 0. Ground Truth

`PRD.md` and `Architecture.md` are derived from the official Review-1 document and are the
source of truth for requirements and architecture. Do not deviate from them without an
explicit, logged decision (see §Never silently change architecture).

## 1. Absolute Rules

1. **Never fabricate metrics.** Every number reported (accuracy, latency, SRAM, Flash,
   bandwidth reduction) must come from an actual run of code that produced it. If a number
   cannot be measured yet, say "not yet measured" — never estimate and present it as measured.
2. **Never fabricate experimental results.** Do not write "results" sections, tables, or
   plots from anything other than real outputs of the codebase in this repo.
3. **Never claim clinical validity.** This project is an explicit software-engineering
   proof-of-concept (Review-1 §4). No output may imply FDA/CE clearance, diagnostic
   reliability, or fitness for real patient use.
4. **Never claim PC RAM equals MCU SRAM.** Any memory number produced by running Python/C++
   on a workstation is a **simulated estimate**. It must be labeled `[SIMULATED]` in logs,
   reports, and docstrings, and must never be presented as a measurement taken on physical
   microcontroller silicon.
5. **Never transmit raw ECG.** No code path — including error handlers, debug logging, and
   test fixtures — may send a raw waveform array to `SecureTelemetry` or any network-facing
   sink. This is enforced structurally (payload schema), not just by convention.
6. **Never bypass the state scheduler.** `SecureTelemetry.transmit_alert` may only be called
   as a result of `StateScheduler`'s decision. No shortcut, test hook, or "convenience"
   function may call telemetry directly.
7. **Run tests after modifications.** Any change to `DataIngestion`, `DSPFilter`,
   `TinyMLEngine`, `StateScheduler`, `SecureTelemetry`, `VirtualMCU`, or `PipelineController`
   must be followed by running the full test suite before the change is considered done.
8. **Preserve existing work.** Do not delete or rewrite working code to "clean it up" unless
   the task requires it. Prefer additive, reviewable diffs.
9. **Avoid unnecessary rewrites.** Match the existing style and structure of a file rather
   than reformatting it wholesale for a small change.
10. **Use type hints.** All new Python functions/methods must have full type hints
    (parameters and return type).
11. **Use configuration files.** Thresholds, filter cutoffs, window size, SRAM/Flash ceilings,
    and latency budgets must live in a config file (e.g., `config.yaml`), not hardcoded
    constants scattered through the code.
12. **Log important operations.** Model load, inference, state transitions, and telemetry
    events must be logged with enough context (window index, timestamp, confidence score) to
    reconstruct a run after the fact.
13. **Add tests for new functionality.** No new module or public function ships without at
    least one corresponding test.
14. **Clearly label simulated/estimated metrics.** Any report, plot, or printed summary must
    visibly distinguish `[SIMULATED]` / `[ESTIMATED]` values from directly measured ones.
15. **Never silently change architecture.** If an implementation detail requires deviating
    from `Architecture.md` (e.g., a different window size, a different filter order), the
    agent must state the deviation explicitly and update `Architecture.md` in the same change
    — never diverge quietly.
16. **Prefer working vertical slices.** When building the pipeline, get one full path
    (ingestion → DSP → inference → scheduler → telemetry) working end-to-end on a small
    sample before optimizing any single stage.
17. **Fix root causes instead of symptoms.** If a test fails, diagnose why rather than
    adjusting the test or adding a special case to make it pass.

## 2. Coding Conventions

- Python 3.10+, type-hinted throughout.
- Formatting: `black` (default settings); linting: `ruff` or `flake8`.
- One module = one responsibility, matching `Architecture.md` §5's module boundaries.
- No global mutable state; pass configuration and dependencies explicitly (constructor
  injection or function arguments).
- Docstrings on every public class/function stating inputs, outputs, and units (e.g., "ms",
  "KB") explicitly.

## 3. Testing Conventions

- Framework: `pytest`.
- Directory: `tests/`, mirroring `src/` module structure (e.g., `tests/test_dsp_filter.py`).
- Every module in Architecture.md §5 has a corresponding test file at minimum covering the
  "Tests" row listed for it.
- Resource-budget tests (SRAM, Flash, latency) are marked distinctly (e.g., `@pytest.mark.resource`)
  so they can be run/reported separately from functional unit tests.
- No test may pass by asserting a mocked/fabricated value where a real measurement is
  required (e.g., a latency test must actually time real code).

## 4. Commit Conventions

- Conventional Commits style: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `perf:`.
- Each commit should correspond to one task ID from `Plan.md` where applicable
  (e.g., `feat(P0-3): implement DSPFilter bandpass stage`).
- Commit message body states what was measured/tested, not just what was written.

## 5. File Organization

```
/src
  /ingestion        DataIngestion
  /dsp              DSPFilter
  /models           Teacher, Student, distillation, quantization scripts
  /edge             TinyMLEngine, VirtualMCU
  /scheduler        StateScheduler
  /telemetry        SecureTelemetry
  /pipeline         PipelineController, config loading
/tests              mirrors /src
/config             config.yaml, threshold/window/budget definitions
/data               dataset download/prep scripts (not raw patient data)
/reports            generated metrics, plots, latency/SRAM/Flash reports
PRD.md
Architecture.md
AGENTS.md
Plan.md
```

## 6. Dependency Rules

- New dependencies must be justified against the SRAM/Flash budget philosophy of the project
  — a dependency that only exists for the training-time Teacher/Student pipeline should not
  be imported by any module in the simulated edge path (`edge/`, `scheduler/`, `telemetry/`).
- Pin versions in `requirements.txt` / `pyproject.toml`.
- Do not add a dependency to solve a one-line problem that plain code can solve.

## 7. Security Rules

- `SecureTelemetry`'s payload constructor must be the only place a network-bound object is
  created; it must reject (raise) any input containing an array longer than the fixed
  metadata schema (timestamp, anomaly id, confidence score).
- Encryption keys must never be hardcoded in source; load from config/environment, and never
  log key material.
- Any new external dependency touching the telemetry path requires an explicit note in
  `Architecture.md` §11 (Security Architecture).

## 8. ML Experiment Reproducibility Rules

- Fix and record random seeds (NumPy, TensorFlow/PyTorch, train/test split) for every
  reported result.
- Record the exact MIT-BIH records/splits used for train/validation/test.
- Record library versions (TensorFlow/TFLite, etc.) alongside any reported accuracy or size
  number.
- Every reported accuracy number must state: model (Teacher/Student-Float32/Student-INT8),
  dataset split, and whether it is training, validation, or held-out test performance.
- Quantization results must report the accuracy delta between Float32 Student and INT8
  Student explicitly — never assume or assert it is negligible without measuring it.
