"use client";

import { Cpu, Sliders, Shield, Database } from "lucide-react";

export default function TechStack() {
  return (
    <section id="architecture" className="py-20 border-b border-steel/15">
      <div className="mx-auto max-w-6xl px-6">
        {/* Section Header */}
        <div className="max-w-2xl mb-12">
          <span className="text-xs font-medium text-steel">Architecture & Specifications</span>
          <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink mt-2 mb-4">
            Under the hood: software stack & standards.
          </h2>
          <p className="text-base sm:text-lg text-ink/75 leading-relaxed">
            A plain-language breakdown of the actual technologies, mathematical models, and cryptographic primitives
            powering this pipeline.
          </p>
        </div>

        {/* 4-Column Technical Architecture Breakdown */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Layer 1: Edge Deep Learning */}
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs">
            <div className="flex items-center gap-2.5 mb-4 text-steel">
              <Cpu className="h-4 w-4 text-signal" />
              <h3 className="text-base font-semibold text-ink">Edge Deep Learning & Quantization</h3>
            </div>
            <ul className="space-y-3 text-xs text-ink/80 leading-relaxed font-numeric">
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Student 1D-CNN (1,538 Parameters)</span>
                <span className="text-steel">
                  2× Conv1D (kernel size 5) + BatchNorm + ReLU + Global Average Pooling + Dense. 10.96 KB flatbuffer size.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Teacher 1D-CNN (120,674 Parameters)</span>
                <span className="text-steel">
                  4-stage deep convolutional model (472 KB) trained to 90.96% test accuracy for knowledge distillation.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Full-Integer INT8 PTQ (TFLite)</span>
                <span className="text-steel">
                  Quantized with 500 representative calibration samples from X_train.npy. Preserves accuracy within 0.26%
                  of Float32 baseline.
                </span>
              </li>
            </ul>
          </div>

          {/* Layer 2: Signal Conditioning & Data */}
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs">
            <div className="flex items-center gap-2.5 mb-4 text-steel">
              <Sliders className="h-4 w-4 text-signal" />
              <h3 className="text-base font-semibold text-ink">DSP Preprocessing & Dataset Splitting</h3>
            </div>
            <ul className="space-y-3 text-xs text-ink/80 leading-relaxed font-numeric">
              <li className="flex flex-col">
                <span className="font-semibold text-ink">PhysioNet MIT-BIH Arrhythmia Database</span>
                <span className="text-steel">
                  48 full patient records (311,952 total 200-sample windows). Real clinical ECG annotated by cardiologists.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Patient-Independent Split Strategy</span>
                <span className="text-steel">
                  Strict 33 train / 7 val / 8 test patient-record partitioning. Zero patient overlap across splits.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">4th-Order Butterworth Bandpass (0.5 – 45.0 Hz)</span>
                <span className="text-steel">
                  Zero-phase filtering via SciPy signal processing + mean subtraction and Z-score standardization.
                </span>
              </li>
            </ul>
          </div>

          {/* Layer 3: Hardware Emulation & Scheduling */}
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs">
            <div className="flex items-center gap-2.5 mb-4 text-steel">
              <Database className="h-4 w-4 text-signal" />
              <h3 className="text-base font-semibold text-ink">Hardware Emulation & Power Management</h3>
            </div>
            <ul className="space-y-3 text-xs text-ink/80 leading-relaxed font-numeric">
              <li className="flex flex-col">
                <span className="font-semibold text-ink">VirtualMCU Simulation Harness</span>
                <span className="text-steel">
                  Enforces strict hardware ceilings: ≤ 256 KB SRAM, &lt; 1.0 MB Flash, &lt; 50 ms latency. RSS RAM is never
                  conflated with MCU SRAM.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">StateScheduler (SLEEP / ACTIVE)</span>
                <span className="text-steel">
                  Radio duty cycle state machine calibrated at τ = 0.35. Keeps transceiver powered down for 97.09% of test
                  duration.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">C++ Header Firmware Model Export</span>
                <span className="text-steel">
                  Exports canonical models/student_model_int8.h byte array (11,224 bytes) for embedded C/C++ compilation.
                </span>
              </li>
            </ul>
          </div>

          {/* Layer 4: Privacy & Cryptography */}
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs">
            <div className="flex items-center gap-2.5 mb-4 text-steel">
              <Shield className="h-4 w-4 text-signal" />
              <h3 className="text-base font-semibold text-ink">Privacy & Authenticated Cryptography</h3>
            </div>
            <ul className="space-y-3 text-xs text-ink/80 leading-relaxed font-numeric">
              <li className="flex flex-col">
                <span className="font-semibold text-ink">AES-GCM (256-bit Symmetric Key)</span>
                <span className="text-steel">
                  Authenticated symmetric encryption with 12-byte random nonce and 16-byte authentication tag for message
                  integrity.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Structural Schema Assertion</span>
                <span className="text-steel">
                  Code-enforced restriction: payload constructor strictly rejects any field containing arrays &gt;5 elements.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Zero Biological Waveform Leakage</span>
                <span className="text-steel">
                  All 1,511 transmitted telemetry packets audited: 0 raw waveform sample bytes transmitted across the network.
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
}
