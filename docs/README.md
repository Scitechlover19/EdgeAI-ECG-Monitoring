# EdgeAI-ECG-Monitoring: Resource-Constrained Edge-AI Pipeline for Real-Time Privacy-Preserving Patient Monitoring

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-80%2F80%20passed-success.svg)](tests/)
[![Next.js](https://img.shields.io/badge/Next.js-16.3.5%20(Turbopack)-black.svg)](frontend/)
[![Architecture Decision](https://img.shields.io/badge/Decision%20%2315-Deployed%20No--KD%20INT8%20(%CF%84%3D0.35)-green.svg)](DECISIONS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Proof of Concept](https://img.shields.io/badge/Status-Engineering%20Proof--of--Concept-orange.svg)](#scientific-honesty--regulatory-disclaimer)

> **Academic Capstone Project**  
> Course: **SWE3004 — Software Engineering Capstone Project**  
> Institution: **Vellore Institute of Technology (VIT)**  
> Author: **Nancy Singh** (Registration No: **22MIS0027**)

---

## 1. Executive Summary & Problem Formulation

Continuous physiological cardiac monitoring is vital for detecting transient, life-threatening arrhythmias (such as Premature Ventricular Contractions, Ventricular Tachycardia, and Atrial Fibrillation). However, conventional Internet of Medical Things (IoMT) architectures rely on **continuous raw telemetry streaming**, transmitting continuous digitized voltages across wireless transceivers (Bluetooth Low Energy / Wi-Fi) to central cloud servers.

In ambulatory practice, this paradigm creates a compounding **engineering trilemma**:

```
                 Conventional IoMT Continuous Streaming
  +-------------------------------------------------------------------+
  | ECG Electrode -> ADC -> RF Radio (ALWAYS ON) -> Cloud Ingestion  |
  | 400 B/window  * 51,992 windows = 20.8 MB raw telemetry (4 hours) |
  +-------------------------------------------------------------------+
                                    |
          [CRITICAL BOTTLENECK: The IoMT Streaming Trilemma]
          1. RF Energy Drain: Radio consumes ~80% of IoT system power;
             coin-cell batteries deplete in hours rather than weeks.
          2. Privacy Exposure: Continuous raw biological waveforms
             broadcast identifiable biometric morphology over RF channels.
          3. Hardware Mismatch: Clinical CNNs require hundreds of megabytes;
             wearable MCUs are bounded by <=256 KB SRAM and <=1 MB Flash.
                                    |
                                    v
                 EdgeAI Event-Driven Anomaly Detection
  +-------------------------------------------------------------------+
  | 200-sample window -> DSP Bandpass -> INT8 1D-CNN -> State Machine |
  |     Normal Sinus Rhythm (97.09%) -> SLEEP (Radio Dormant, 0 B)    |
  |     Arrhythmia Event (tau >= 0.35) -> ACTIVE -> AES-GCM Encrypted |
  |     Transmits strictly 132-byte metadata (198.8 KB total; -99.04%)|
  +-------------------------------------------------------------------+
```

**EdgeAI-ECG-Monitoring** resolves this trilemma by shifting intelligence to the extreme edge. All signal conditioning, neural feature extraction, and state decisions execute directly within a simulated microcontroller environment (`VirtualMCU`). Normal sinus rhythms are discarded on-device, maintaining the wireless transceiver in a deep sleep state **97.09% of the time**. Only verified anomaly windows awaken the radio to transmit an **AES-256-GCM encrypted 132-byte metadata payload**. Raw cardiac waveforms are **structurally prohibited** from leaving the microcontroller.

---

## 2. System Architecture & 5-Stage Pipeline

The architecture is divided into five modular, decoupled pipeline stages adhering strictly to single-responsibility boundaries:

```
                          PHYSICAL / SENSOR DOMAIN
                                     |
                       [Raw MIT-BIH ECG Lead MLII]
                                     |
+------------------------------------+------------------------------------+
| VIRTUAL MICROCONTROLLER (VirtualMCU)                                    |
|                                                                         |
|  [Stage 01: DataIngestion]                                              |
|  * 200-sample sliding windows (360 Hz pacing, 555.5 ms window)          |
|  * Strict circular buffer management with zero dynamic heap allocation  |
|                                    |                                    |
|  [Stage 02: DSPFilter]                                                  |
|  * 2nd-order Butterworth IIR Bandpass (0.5 Hz - 45.0 Hz)                |
|  * Zero-phase forward-backward filtering (scipy/C equivalent)           |
|  * Removes respiration baseline wander & high-frequency EMG noise       |
|                                    |                                    |
|  [Stage 03: TinyMLEngine]                                               |
|  * 1D-CNN + Global Average Pooling (1,538 parameters)                   |
|  * Full-integer INT8 Post-Training Quantization (TFLite Runtime)        |
|  * Single-Softmax output with calibrated anomaly confidence c in [0, 1] |
|                                    |                                    |
|  [Stage 04: StateScheduler]                                             |
|  * Evaluates confidence against operating threshold tau = 0.35          |
|  * c < 0.35  -> SLEEP  (Radio Dormant, zero transmission)               |
|  * c >= 0.35 -> ACTIVE (Radio Awake, dispatches alert event)            |
|                                    |                                    |
|  [Stage 05: SecureTelemetry]                                            |
|  * Construct 4-field metadata payload (session_id, timestamp, id, conf) |
|  * Enforces structural schema (arrays >5 elements raise SchemaViolation)|
|  * Authenticated AES-GCM-256 cipher (12-byte nonce, 16-byte tag)        |
+------------------------------------+------------------------------------+
                                     |
                               [RF CHANNEL]
                                     |
                 Encrypted Ciphertext (122 - 133 Bytes)
                      [0 Bytes Raw ECG Transmitted]
                                     |
                                     v
                           [Secure In-Memory Sink]
```

### Stage Contracts & Invariants

| Stage | Module | Input Contract | Output Contract | Key Architectural Invariant |
| :--- | :--- | :--- | :--- | :--- |
| **01 Ingestion** | `src.ingestion` | Continuous voltage stream | Shape `(200,)` float32 window | Circular ring buffer; no heap re-allocation |
| **02 DSP Filter** | `src.dsp` | Raw `(200,)` window | Filtered `(200,)` window | Passband $0.5 - 45.0\text{ Hz}$; zero DC drift |
| **03 TinyML** | `src.edge` | Quantized `(1, 200, 1)` int8 | Anomaly confidence $c \in [0.0, 1.0]$ | Pure integer INT8; no double-softmax squashing |
| **04 Scheduler** | `src.scheduler`| Confidence score $c$ | State (`SLEEP` / `ACTIVE`) | Only entity permitted to invoke telemetry |
| **05 Telemetry**| `src.telemetry`| Metadata dictionary | Authenticated ciphertext bytes | **Structural schema: 0 raw ECG bytes allowed** |

---

## 3. Empirical Verification & Measured Benchmarks

All metrics reported below were **directly measured** on the held-out MIT-BIH Arrhythmia test dataset split across **51,992 evaluation windows (4.01 hours of continuous monitoring)**. No numbers are simulated or estimated unless explicitly marked `[SIMULATED]`.

### Non-Functional Requirements (NFR) Scorecard

| Requirement | Metric Description | Target Threshold | Measured Performance | Margin / Headroom | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NFR-1** | Peak SRAM Consumption | $\le 256.0\text{ KB}$ | **$15.16\text{ KB}$** `[SIMULATED]` | **$94.1\%\text{ headroom}$** | **PASS** |
| **NFR-2** | Flash Memory Footprint | $\le 1,024.0\text{ KB}$ | **$10.96\text{ KB}$** ($11,224\text{ B}$) | **$98.9\%\text{ headroom}$** | **PASS** |
| **NFR-3** | Edge Inference Latency | $< 50.0\text{ ms}$ | **$0.1599 \pm 0.0126\text{ ms}$** | **$312\times\text{ faster}$** | **PASS** |
| **NFR-4** | End-to-End Pipeline Latency | $< 100.0\text{ ms}$ | **$0.2505 \pm 0.0210\text{ ms}$** | **$399\times\text{ faster}$** | **PASS** |
| **NFR-5** | Biological Privacy Leakage | $0\text{ bytes raw ECG}$ | **$0\text{ bytes}$** ($1,511\text{ alerts audited}$) | **$100\%\text{ compliance}$** | **PASS** |
| **NFR-6** | Telemetry Bandwidth Reduction | $> 90.0\%$ | **$99.04\%$** ($198.8\text{ KB vs } 20.8\text{ MB}$) | **$+9.04\%\text{ margin}$** | **PASS** |
| **NFR-7** | Telemetry Cipher Security | Authenticated AES | **AES-GCM-256** ($12\text{B IV}, 16\text{B Tag}$) | Cryptographically sound | **PASS** |

---

## 4. Architectural Decision #15: The Post-Training Quantization (PTQ) Paradox

During Phase 2, a **120k-parameter Teacher Model** was compressed down to a **1,538-parameter Student Model** using Knowledge Distillation (KD with temperature $T=6.0$, $\alpha=0.5$). 

While the KD Student outperformed the un-distilled model in Float32 validation recall, **full-integer INT8 post-training quantization caused an unexpected divergence**:

```
                       The PTQ Quantization Paradox
  +---------------------------------------------------------------------+
  | Float32 Domain (Continuous Logits)                                  |
  | Teacher Soft Labels produce smooth probability distributions.       |
  | -> Validation Recall: KD Student (48.3%) > No-KD Student (38.1%)    |
  +---------------------------------------------------------------------+
                                    |
                 [Full-Integer INT8 Calibration (PTQ)]
                                    |
                                    v
  +---------------------------------------------------------------------+
  | INT8 Domain (Discrete 8-bit Tensor Grids)                           |
  | Soft labels spread probability mass across adjacent quantization    |
  | buckets. In INT8, this flattened gradient caused massive false      |
  | positive flooding upon calibration.                                 |
  | -> At tau=0.35: KD INT8 produced 1,902 FP (474 FP/hr)!              |
  | -> At tau=0.35: No-KD INT8 produced 507 FP (126.4 FP/hr) - 3.7x LESS|
  +---------------------------------------------------------------------+
```

### Full-Integer INT8 Head-to-Head Comparison ($\tau = 0.35$)

| Model Architecture | Precision | Recall | False Positives | Alert Burden (FP/hr) | Bandwidth Reduction | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KD Student INT8** | $37.52\%$ | $15.75\%$ | $1,902$ | $474.3\text{ FP/hr}$ | $96.88\%$ | **REJECTED** (Severe alert fatigue) |
| **No-KD Student INT8** | **$66.45\%$** | **$13.79\%$** | **$507$** | **$126.4\text{ FP/hr}$** | **$99.04\%$** | **DEPLOYED (Decision #15)** |

**Decision Rationale:** The No-KD model's sharp one-hot ground-truth activations survived integer affine mapping with minimal boundary drift. At $\tau = 0.35$, the No-KD INT8 model delivers **$20.5\times$ greater recall** than the legacy $0.85$ operating point ($13.79\%$ vs $0.67\%$) while reducing clinician false alert burden by **$3.75\times$** compared to KD.

---

## 5. Scientific Honesty & Regulatory Disclaimer

> [!IMPORTANT]
> **Proof-of-Concept Boundary:** This repository is an academic software-engineering proof-of-concept developed to demonstrate resource-constrained TinyML pipelines, power scheduling, and privacy-preserving payload structures. It has **not** undergone clinical trials, ISO 13485 quality management, or FDA 510(k) / CE clearance. It must **never** be used for clinical diagnosis, patient triage, or direct medical care.

> [!NOTE]
> **Minority Class Recall Transparency:** Under $14.0\%$ minority arrhythmia prevalence on the held-out test split, the 1,538-parameter INT8 model achieves an absolute recall of **$13.79\%$** ($1,004$ of $7,278$ arrhythmia windows). While this represents a $20.5\times$ sensitivity gain over the conservative $0.85$ baseline without causing alert fatigue ($126.4\text{ FP/hr}$ vs $474\text{ FP/hr}$), detecting minority arrhythmias under extreme resource constraints remains an ongoing research challenge. Patient-adaptive threshold calibration and architectural neural search are designated as future research vectors.

---

## 6. Interactive Portfolio & Showcase Website

An editorial, showcase-quality portfolio website is included in the `frontend/` directory, designed to communicate the project's technical depth, data truth, and architectural invariants without template clichés.

### Technology Stack & Design System
- **Framework:** Next.js 16 (App Router, Turbopack, React 19, TypeScript)
- **Styling:** Vanilla Tailwind CSS v4 with custom CSS variable tokens
- **Smooth Scroll:** Lenis (`lerp: 0.1`) with `prefers-reduced-motion` detection
- **Palette (Warm Putty & Ink Navy):**
  - `--bg-base` (`#E7E0D2` Warm Putty) — Warm greyed-down beige canvas
  - `--surface` (`#EFE9DC` Parchment) — Elevated paper card surface
  - `--surface-recessed` (`#D8CDB8` Sand Shadow) — Tactile code & metadata container
  - `--ink` (`#241D14` Espresso) — High-contrast warm charcoal typography
  - `--signal` (`#1F3347` Ink Navy) — Calm normal-rhythm trace and primary badges
  - `--alert` (`#7A2E22` Oxblood) — Reserved strictly for active arrhythmia transitions
  - `--accent-secondary` (`#9C7A3C` Brass) — Metadata hairlines and secondary accents

### Showcase Features
1. **Interactive Dual-State Oscilloscope:** Real-time HTML5 Canvas rendering a 200-sample sliding window at 360 Hz pacing. Includes a **"Simulate Anomaly (PVC Spike)"** trigger that demonstrates the `SLEEP` $\rightarrow$ `ACTIVE` radio wake transition in real-time.
2. **IoMT Trilemma Architectural Comparison:** Inline SVG diagram contrasting continuous raw streaming vs. event-driven edge monitoring.
3. **5-Stage Pipeline Step-Through:** Live explorer detailing I/O contracts, hardware footprints, and architectural rules for each stage.
4. **Interactive Threshold Calibrator:** Slider over $\tau \in [0.30, 0.95]$ dynamically updating recall, precision, FP/hr, and bandwidth reduction backed by `threshold-sweep.json`, featuring a No-KD vs. KD INT8 toggle to illustrate the PTQ paradox.

---

## 7. Repository Structure

```
EdgeAI-ECG-Monitoring/
├── .gitignore                      # Comprehensive exclusions (Python, Node, Data)
├── AGENTS.md                       # Binding rules for AI agents (zero fabrication)
├── Architecture.md                 # Complete architectural specification
├── DECISIONS.md                    # Architecture Decision Records (ADRs 1-15)
├── EdgeAI-ECG-Frontend-Brief.md    # Frontend design brief & palette specification
├── PRD.md                          # Product Requirements Document
├── Plan.md                         # Implementation roadmap & task tracking
├── pyproject.toml                  # Python package configuration & dependencies
├── README.md                       # Comprehensive repository documentation
│
├── config/                         # Configuration definitions
│   └── config.yaml                 # Hardware budgets, DSP cutoffs, threshold (0.35)
│
├── src/                            # Core pipeline source code
│   ├── ingestion/                  # Stage 01: Circular window buffering
│   ├── dsp/                        # Stage 02: Butterworth IIR filtering
│   ├── models/                     # Teacher, Student, and Distillation scripts
│   ├── edge/                       # Stage 03: TinyMLEngine & VirtualMCU
│   ├── scheduler/                  # Stage 04: StateScheduler (SLEEP/ACTIVE)
│   ├── telemetry/                  # Stage 05: SecureTelemetry (AES-256-GCM)
│   └── pipeline/                   # PipelineController & interactive demo runner
│
├── tests/                          # 80 unit & integration test suites
│   ├── test_bandwidth_report.py    # Bandwidth reduction arithmetic tests
│   ├── test_dsp_filter.py          # Bandpass frequency response validation
│   ├── test_pipeline_integration.py# End-to-end pipeline execution test
│   ├── test_quantize.py            # INT8 TFLite export & quantization validation
│   ├── test_secure_telemetry.py    # Ciphertext security & schema rejection
│   ├── test_state_scheduler.py     # State transition logic tests
│   └── test_tinyml_engine.py       # INT8 inference & double-softmax regression
│
├── reports/                        # Committed empirical benchmark reports
│   ├── experiment_manifest.json    # Complete reproducible experiment registry
│   ├── bandwidth_report.md         # Bandwidth reduction measurements
│   ├── resource_report.md          # Flash, SRAM, and latency benchmarks
│   ├── quantization_report.md      # Quantization error & accuracy delta
│   ├── kd_ablation.md              # Knowledge distillation ablation sweep
│   └── threshold_sweep.md          # Tables A, B, C (No-KD), and D (KD-INT8)
│
├── scripts/                        # Automation & sync utilities
│   └── sync_metrics.py             # Parses reports/*.md into frontend JSON
│
└── frontend/                       # Next.js showcase web application
    ├── src/app/                    # App router (globals.css, layout.tsx, page.tsx)
    ├── src/components/             # Hero, Problem, Pipeline, Metrics, Explorer
    ├── src/data/                   # metrics.json & threshold-sweep.json
    └── package.json                # Frontend dependencies
```

---

## 8. Quickstart & Reproduction Guide

### 8.1 Backend Setup & Test Suite Execution

```bash
# 1. Clone repository
git clone https://github.com/Scitechlover19/EdgeAI-ECG-Monitoring.git
cd EdgeAI-ECG-Monitoring

# 2. Create and activate Python virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies in editable mode
pip install -e ".[dev]"

# 4. Run the full 80-test verification suite
pytest -v
```

### 8.2 Running the Interactive CLI Pipeline Demo

```bash
# Execute the live end-to-end pipeline on sample MIT-BIH records
python -m src.pipeline.demo_mode
```

### 8.3 Running the Showcase Website

```bash
cd frontend

# Install frontend dependencies
npm install

# Run build-time metric synchronization (extracts reports/*.md to data JSONs)
python ../scripts/sync_metrics.py

# Launch development server
npm run dev

# Or build and launch production server
npm run build
npm run start
```
Open [http://localhost:3000](http://localhost:3000) (or the specified port) in your browser.

---

## 9. Academic & Ethical Declarations

- **Data Source:** MIT-BIH Arrhythmia Database, provided under the PhysioNet Open Data guidelines (Goldberger et al., 2000).
- **Security Invariant:** Evaluated and verified under AGENTS.md constraints. No raw ECG biometric traces ever reach external sinks.
- **Reproducibility:** Random seeds fixed ($42$), dependencies pinned in `pyproject.toml` and `package-lock.json`.

---

**Nancy Singh** · M.Tech Capstone Project · Department of Software Engineering, VIT
