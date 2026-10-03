# 🌐 60-Second Wokwi Online Microcontroller Simulation

Wokwi lets you simulate an embedded microcontroller directly in Google Chrome / Edge with **zero installations**, **zero toolchains**, and **zero command-line setups**.

---

## ⚡ Quick Start Guide (3 Simple Steps)

### Step 1: Open Wokwi
Go to [wokwi.com](https://wokwi.com) and click **ESP32** (or **STM32**).
- Directly open the starter: **[wokwi.com/projects/new/esp32](https://wokwi.com/projects/new/esp32)**

### Step 2: Paste the Code
1. In the code editor on the left (`sketch.ino`), press `Ctrl + A` and delete any sample code.
2. Open [`wokwi/sketch.ino`](file:///d:/Project/EdgeAI-ECG-Monitoring/wokwi/sketch.ino) from this repository, copy everything (`Ctrl + A`, `Ctrl + C`), and paste it into Wokwi (`Ctrl + V`).

### Step 3: Click the Green Play Button ▶
1. Click the green **Play** (`▶`) button at the top.
2. The virtual board will turn on, and the **Serial Monitor** at the bottom will start running live!

---

## 🔍 What You & the Review Panel Will See Live

1. **Hardware Verification Header**:
   - MCU SRAM Ceiling: `256 KB`
   - MCU Flash Ceiling: `1024 KB (1 MB)`
   - Measured INT8 Model Size: `11,224 bytes (10.96 KB)` $\rightarrow$ **1.07% Flash utilization**
   - Simulated Tensor Arena: `15.16 KB` $\rightarrow$ **5.92% SRAM utilization**
2. **Real-Time Heartbeat LED**:
   - The onboard virtual LED flashes brief pulses on every normal sinus beat.
   - When an arrhythmia occurs (Windows 5–7), the LED turns **solid ON**!
3. **State Transitions & Suppression**:
   - **Windows 1–4**: Normal ECG $\rightarrow$ `State: SLEEP` $\rightarrow$ Telemetry suppressed (`0 bytes`).
   - **Windows 5–7**: Anomaly injected $\rightarrow$ Transition to `State: ACTIVE` $\rightarrow$ Encrypted metadata JSON dispatched over Serial.
   - **Windows 8–12**: Recovery hysteresis counts 5 consecutive normal beats $\rightarrow$ Transitions back to `SLEEP`.
4. **Summary Report**:
   - Shows **>95% bandwidth reduction** compared to streaming raw ECG.
   - Confirms strictly **ZERO raw ECG samples** were transmitted over telemetry.
