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
                <span className="font-semibold text-ink">Dual-Branch 1D-CNN (1,698 Parameters)</span>
                <span className="text-steel">
                  Fuses beat waveform morphology (1D-CNN branch) with causal rhythm context [RR_pre, RR_ratio, RR_var] (Dense branch). 12.91 KB FlatBuffer.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Full-Integer INT8 PTQ (TFLite Micro)</span>
                <span className="text-steel">
                  Quantized with representative training set calibration. Runs in 5.4 KB SRAM tensor arena with 1.8 ms Cortex-M4 latency.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Strict Patient-Independent Distillation</span>
                <span className="text-steel">
                  Trained on subject-partitioned records (subject 201/202 grouped). Eliminates inter-patient data leakage.
                </span>
              </li>
            </ul>
          </div>

          {/* Layer 2: Signal Conditioning & Data */}
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs">
            <div className="flex items-center gap-2.5 mb-4 text-steel">
              <Sliders className="h-4 w-4 text-signal" />
              <h3 className="text-base font-semibold text-ink">DSP Preprocessing & Multi-Database Cohort</h3>
            </div>
            <ul className="space-y-3 text-xs text-ink/80 leading-relaxed font-numeric">
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Causal Pan-Tompkins QRS Peak Detector</span>
                <span className="text-steel">
                  97.75% sensitivity & 99.40% PPV on MIT-BIH; 98.77% sensitivity on external cohort with 25-sample (69.4 ms) confirmation window.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Multi-Database Validation (MIT-BIH + SVDB + INCART)</span>
                <span className="text-steel">
                  Validated across 4.01h of MIT-BIH (360 Hz) and 3.5h of external data (SVDB @ 128 Hz, INCART @ 257 Hz) via causal polyphase resampling.
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Forward-Only 4th-Order Butterworth Bandpass</span>
                <span className="text-steel">
                  0.5 Hz – 45.0 Hz passband with 0 samples lookahead. Running in-place Z-score normalization.
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
                <span className="font-semibold text-ink">Hysteresis State Scheduler (τ* = 0.52)</span>
                <span className="text-steel">
                  5-beat recovery timer lockout prevents radio thrashing, bounding false alarms to 5.24 ep/hr (MIT-BIH) and 11.71 ep/hr (External).
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Hardware Emulation (Renode + Wokwi)</span>
                <span className="text-steel">
                  Verified in Antmicro Renode 1.17.0 (STM32F4 Cortex-M4) and Wokwi (ESP32 with I2C OLED & NeoPixel).
                </span>
              </li>
              <li className="flex flex-col">
                <span className="font-semibold text-ink">Deterministic Flash & SRAM Footprint</span>
                <span className="text-steel">
                  12.91 KB Flash model (~82 KB firmware, 8.0% of 1 MB) and 5.4 KB SRAM arena (2.1% of 256 KB).
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
