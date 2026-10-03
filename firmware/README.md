# Embedded Cortex-M4 Firmware & Renode Simulation

This folder contains the complete, self-contained embedded C firmware and Renode simulation platform for the **EdgeAI-ECG-Monitoring** capstone project.

---

## 📁 Directory Structure

```
firmware/
├── inc/
│   ├── mcu_config.h           # ARM Cortex-M4 memory bounds (256 KB SRAM, 1 MB Flash, 360 Hz)
│   ├── ecg_dsp.h              # 0.5–45 Hz Bandpass Butterworth filter & Z-score scaling
│   ├── state_scheduler.h      # Two-state power scheduler (SLEEP <-> ACTIVE)
│   ├── secure_telemetry.h     # Privacy-preserving metadata-only payload (zero raw ECG)
│   └── student_model_int8.h   # Exported INT8 Quantized Student Model C array (11,224 bytes)
├── src/
│   ├── main.c                 # Cortex-M4 pipeline loop & UART logger
│   ├── ecg_dsp.c              # Biquad filter implementation
│   ├── state_scheduler.c      # Dynamic state transition machine
│   └── secure_telemetry.c     # Metadata JSON serializer
├── renode/
│   ├── cortex_m4_ecg.repl     # Renode Platform Description (Cortex-M4, 256KB SRAM, 1MB Flash, UART)
│   └── simulate.resc          # Renode startup script
├── CMakeLists.txt             # Standard CMake configuration
├── Makefile                   # Multi-target Makefile (ARM GCC cross-compile & Host GCC)
├── platformio.ini             # 1-click PlatformIO build configuration
└── README.md
```

---

## 🎯 Review Presentation Overview

| Metric | Target Specification | Measured / Simulated In Firmware |
| :--- | :--- | :--- |
| **CPU Architecture** | ARM Cortex-M4 | ARM Cortex-M4 (`cortex_m4_ecg.repl`) |
| **SRAM Ceiling** | $\le 256\text{ KB}$ | $\approx 15.16\text{ KB}$ arena ($\approx 5.9\%$ utilization) `[SIMULATED]` |
| **Flash Ceiling** | $\le 1\text{ MB}$ (1024 KB) | $11,224\text{ bytes}$ ($10.96\text{ KB}$, $1.07\%$ utilization) `[MEASURED]` |
| **Sampling Frequency** | $360\text{ Hz}$ | $360\text{ Hz}$ window buffer (200 samples) |
| **DSP Filtering** | $0.5 - 45.0\text{ Hz}$ Bandpass | Cascaded Direct-Form II Biquad Stages |
| **State Machine** | SLEEP $\leftrightarrow$ ACTIVE | Two-state hysteresis scheduler (threshold $\tau = 0.35$) |
| **Privacy Guarantee** | Zero raw ECG transmitted | Discrete metadata payload: `{ts, anomaly_id, conf, bpm, auth}` |

---

## 🛠️ How to Build the Firmware ELF

### Option 1: Using PlatformIO (Easiest if using VS Code)
1. Install the PlatformIO IDE extension in VS Code.
2. Open the `firmware/` directory.
3. Run:
   ```bash
   pio run
   ```
   The compiled ELF binary will be generated at `.pio/build/nucleo_f401re/firmware.elf`.

### Option 2: Using ARM GCC & Make
If `arm-none-eabi-gcc` is installed:
```bash
make USE_ARM=1
```
This produces `build/firmware.elf`.

### Option 3: Using CMake
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

---

## 🚀 Running in Renode

1. Launch Renode 1.17.0.
2. In the Renode Monitor window, navigate to your project and execute:
   ```renode
   s @firmware/renode/simulate.resc
   ```
3. Type:
   ```renode
   start
   ```
4. A UART terminal window will pop up showing:
   - System startup banner with exact SRAM and Flash memory statistics.
   - Real-time window evaluations.
   - State transition from `SLEEP` to `ACTIVE` when an arrhythmia spike is detected.
   - Encrypted metadata telemetry transmissions (and suppressed transmission during normal rhythms).
   - Final run summary showing **>95% bandwidth reduction** and **zero raw ECG transmitted**.
