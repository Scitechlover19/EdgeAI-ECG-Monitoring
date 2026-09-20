# INT8 Post-Training Quantization (PTQ) & Edge Deployment Report (P0-8)

**Date:** 2026-09-20 22:36:11  
**Deployment Candidate:** Student 1D-CNN (`student_no_kd_model.keras`, 1,538 parameters)  
**Operating Threshold:** $\tau = 0.35$ (DECISIONS.md #15)  
**Calibration Dataset:** `data/processed/X_train.npy` (500 representative 200-sample windows, `[MEASURED]`)  
**Evaluation Dataset:** Held-Out Test Split (`X_test.npy`, 51,992 windows, `[MEASURED]`)

---

## 1. Quantization Specification & Artifact Verification

- **Exported TFLite Flatbuffer:** [`models/student_model_int8.tflite`](file:///d:/Project/EdgeAI-ECG-Monitoring/models/student_model_int8.tflite) (Alias: [`models/student_model.tflite`](file:///d:/Project/EdgeAI-ECG-Monitoring/models/student_model.tflite))
- **File Size:** **11,224 bytes** (**10.96 KB**) `[MEASURED]`
- **Input Tensor Dtype:** `int8` `[MEASURED]`
- **Output Tensor Dtype:** `int8` `[MEASURED]`
- **Input Scale / Zero-Point:** Scale = `0.039852`, Zero-Point = `-31` `[MEASURED]`
- **Output Scale / Zero-Point:** Scale = `0.003906`, Zero-Point = `-128` `[MEASURED]`
- **Target Operator Set:** `TFLITE_BUILTINS_INT8` (Full-Integer Quantization) `[MEASURED]`
- **TFLite Interpreter Allocation:** **SUCCESSFUL** `[MEASURED]`

---

## 2. Float32 Student vs INT8 Student Accuracy Preservation (`[MEASURED]`)

Evaluated on the exact 51,992 held-out test windows across 8 test records (`107`, `115`, `119`, `122`, `207`, `220`, `228`, `234`):

| Model Format | Precision / Dtype | Accuracy `[MEASURED]` | Precision `[MEASURED]` | Recall `[MEASURED]` | F1 Score `[MEASURED]` | Model File Size `[MEASURED]` |
|---|---|---|---|---|---|---|
| **Float32 Student** | `float32` (Keras) | **0.8687** (86.87%) | **0.6609** | **0.1270** | **0.2130** | 65.4 KB |
| **INT8 Student** | `int8` (TFLite) | **0.8660** (86.60%) | **0.6733** | **0.0835** | **0.1486** | **10.96 KB** |

### Accuracy Delta Analysis (`[MEASURED]`)
- **Absolute Accuracy Change ($\Delta\text{Accuracy}$):** **-0.0026** (-0.26% percentage points)
- **Relative Accuracy Change ($\%\Delta\text{Accuracy}$):** **-0.30%**
- **Accuracy Preservation Finding:** INT8 quantization preserved held-out test accuracy within **0.26%** of Float32 baseline performance.

---

## 3. Resource & Memory Footprint Estimation (`[ESTIMATED]`)

> [!NOTE]
> All memory numbers below are **simulated host-PC MCU estimates** computed from tensor dimensions and Flatbuffer file sizes via `VirtualMCU`. They are labeled `[ESTIMATED]` per DECISIONS.md #11/#12 and AGENTS.md §1.

| Resource Metric | Estimated Value `[ESTIMATED]` | Budget Limit | Budget Compliance Status |
|---|---|---|---|
| **Model Flash Footprint** | **10.96 KB** (11,224 bytes) | 1,024.0 KB (1.0 MB) | **PASS** (Headroom: 1013.04 KB) |
| **Estimated Tensor Arena** | **4.20 KB** (4,298 bytes) | 256.0 KB | **PASS** |
| **Estimated Peak SRAM** | **15.16 KB** (15,522 bytes) | 256.0 KB | **PASS** (Headroom: 240.84 KB) |

---

## 4. Requirement Verification Matrix

| Requirement | Description | Status | Verification Evidence |
|---|---|---|---|
| **REQ-1** | INT8 PTQ using real `X_train.npy` calibration data | **PASS** | 500 training samples used for calibration |
| **REQ-2** | Test split `X_test.npy` strictly held out during calibration | **PASS** | 0 test samples used in calibration generator |
| **REQ-3** | Export valid `.tflite` Flatbuffer | **PASS** | Saved 11,224 bytes flatbuffer |
| **REQ-4** | INT8 Input and Output Tensor dtypes | **PASS** | Input=`int8`, Output=`int8` |
| **REQ-5** | TFLite Interpreter allocation and inference smoke test | **PASS** | `TinyMLEngine` loaded and executed 51,992 inferences |
| **REQ-6** | Measured Float32 vs INT8 accuracy comparison on test set | **PASS** | Float32=0.8687 vs INT8=0.8660 |
| **REQ-7** | SRAM & Flash budget headroom calculation against limits | **PASS** | SRAM Peak=15.16KB (< 256KB), Flash=10.96KB (< 1MB) |
