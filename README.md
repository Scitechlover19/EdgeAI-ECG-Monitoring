# Resource-Constrained Edge-AI Pipeline for Real-Time Privacy-Preserving Patient Monitoring

A software-engineering proof-of-concept for real-time ECG anomaly monitoring using a lightweight Edge-AI pipeline designed for resource-constrained devices.

The system processes ECG signals locally, performs anomaly inference using an INT8-quantized 1D-CNN, and transmits only encrypted metadata when an anomaly state is detected. Raw ECG waveforms are never transmitted through the telemetry layer.

---

## Overview

Traditional physiological monitoring can require continuous transmission of raw biosignals, creating bandwidth, privacy, and resource challenges.

This project explores an Edge-AI architecture where:

- ECG preprocessing happens locally.
- A lightweight neural network performs on-device inference.
- Normal windows remain in a `SLEEP` state with no telemetry.
- Anomaly states transition to `ACTIVE`.
- Only encrypted metadata is transmitted.
- Raw ECG data never enters the telemetry layer.
- The deployment model is quantized to INT8 for TinyML-style execution.

This is a software-engineering proof-of-concept and is **not a clinical diagnostic system**.

---

## Architecture

```text
MIT-BIH ECG
     |
     v
Data Ingestion
     |
     v
200-Sample Sliding Windows
     |
     v
Butterworth Bandpass Filter
0.5 - 45 Hz
     |
     v
Lightweight INT8 1D-CNN
     |
     v
Anomaly Confidence
     |
     v
State Scheduler
    / \
   /   \
SLEEP  ACTIVE
  |       |
  |       v
  |    AES-GCM
  |       |
  |       v
  |   Encrypted Metadata
  |       |
  |       v
  |   In-Memory Sink
  |
No telemetry
