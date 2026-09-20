# Resource Profiling & Budget Verification Report (P0-14)

**Date:** 2026-09-20 22:36:28  
**Deployable Artifact:** [`models/student_model_int8.tflite`](file:///models/student_model_int8.tflite)  
**Execution Environment:** Host PC Edge-AI Simulation Harness (`[SIMULATED]`)  
**Benchmarked Windows:** 1,000 preprocessed ECG test windows (`[MEASURED]`)

---

## 1. Latency Benchmark Statistics (`[MEASURED]`)

> [!NOTE]
> Latency was measured directly using high-precision `time.perf_counter()` over 1,000 individual 200-sample window inferences in Python 3.13 on the host development workstation. This is host-PC simulated execution timing, not physical microcontroller silicon execution.

| Latency Metric | Measured Host-PC Value `[MEASURED]` | Budget Limit (NFR-3) | Compliance Status |
|---|---|---|---|
| **Mean Latency** | **0.0600 ms** | < 50.0 ms | **PASS** |
| **Median Latency** | **0.0550 ms** | < 50.0 ms | **PASS** |
| **p95 Latency** | **0.0979 ms** | < 50.0 ms | **PASS** |
| **Minimum Latency** | **0.0506 ms** | < 50.0 ms | **PASS** |
| **Maximum Latency** | **0.1309 ms** | < 50.0 ms | **PASS** |

---

## 2. Memory & Footprint Analysis

### Model Flash Memory (`[MEASURED]`)
- **Actual Model File Size:** **11,224 bytes** (**10.96 KB**) `[MEASURED]`
- **Flash Ceiling (NFR-2):** **1,024.0 KB (1.0 MB)**
- **Flash Headroom:** **1013.04 KB** (0.989 MB)
- **Compliance Status:** **PASS**

### Simulated Peak SRAM Footprint (`[ESTIMATED]`)
> [!IMPORTANT]
> SRAM memory values are **simulated estimates** calculated from model tensor shapes and working scratchpad allocations via `VirtualMCU`. Process RSS / Python host RAM is **never** used as a stand-in for MCU SRAM.

| SRAM Memory Component | Estimated Size `[ESTIMATED]` | Bytes `[ESTIMATED]` |
|---|---|---|
| **Model Resident Memory** | **10.96 KB** | 11,224 B |
| **Input Tensor Buffer** (`1, 200, 1` int8) | **0.20 KB** | 200 B |
| **Output Tensor Buffer** (`1, 2` int8) | **0.002 KB** | 2 B |
| **Working Activation Scratchpad** | **4.00 KB** | 4,096 B |
| **Estimated Tensor Arena** | **4.20 KB** | 4,298 B |
| **Total Estimated Peak SRAM** | **15.16 KB** | **15,522 B** |

- **SRAM Ceiling (NFR-1):** **256.0 KB**
- **SRAM Headroom:** **240.84 KB** (94.1% remaining)
- **Compliance Status:** **PASS**

---

## 3. NFR Budget Compliance Matrix

| Requirement ID | Description | Measured / Estimated Value | Budget Limit | Status |
|---|---|---|---|---|
| **NFR-1** | MCU Peak SRAM Constraint | **15.16 KB** `[ESTIMATED]` | $\le$ 256.0 KB | **PASS** |
| **NFR-2** | MCU Flash Memory Constraint | **10.96 KB** `[MEASURED]` | $<$ 1.0 MB (1024 KB) | **PASS** |
| **NFR-3** | Edge Per-Window Latency Constraint | **0.0600 ms** `[MEASURED]` | $<$ 50.0 ms | **PASS** |

---

## 4. Host Benchmark Environment (`[SIMULATED]`)

- **OS Platform:** `Windows-10-10.0.26200-SP0`
- **CPU Architecture:** `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD`
- **Python Version:** `3.10.0`
- **TensorFlow Version:** `2.21.0`

---

## 5. Limitations of Host-PC Simulation

1. **Host CPU vs. MCU Core:** Latency benchmarks were performed on a x86_64 host workstation CPU. Physical microcontroller execution timing (e.g. ARM Cortex-M4 @ 64 MHz) will differ based on clock speed and TFLite Micro kernel vectorization.
2. **SRAM Estimation:** SRAM footprint is calculated from static flatbuffer structures and tensor allocations (`VirtualMCU`). Physical silicon stack/heap allocations may vary depending on the C++ toolchain used.
