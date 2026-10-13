"use client";

import { useState } from "react";
import { Activity, Sliders, Cpu, Radio, Lock, CheckCircle2 } from "lucide-react";

interface PipelineStage {
  id: string;
  number: string;
  name: string;
  module: string;
  icon: typeof Activity;
  tag: string;
  summary: string;
  input: string;
  output: string;
  hardwareSpecs: { label: string; value: string }[];
  rules: string[];
}

const STAGES: PipelineStage[] = [
  {
    id: "ingestion",
    number: "01",
    name: "Causal QRS Detection & Framing",
    module: "src.dsp.QRSDetector",
    icon: Activity,
    tag: "Real-Time Peak Detection",
    summary:
      "Performs real-time causal Pan-Tompkins QRS detection (97.75% sensitivity, 99.40% PPV) with a 25-sample (69.4 ms) confirmation delay. Segments 200-sample beat-centred windows (-72 pre-peak to +128 post-peak) and computes causal rhythm features.",
    input: "Continuous raw ECG voltage stream (16-bit ADC samples at 360 Hz)",
    output: "Beat waveform array (200 samples) + 3 causal RR features (RR_pre, RR_ratio, RR_var)",
    hardwareSpecs: [
      { label: "QRS Sensitivity", value: "97.75% (MIT) / 98.77% (Ext)" },
      { label: "QRS PPV", value: "99.40% (MIT) / 96.89% (Ext)" },
      { label: "Confirmation Delay", value: "69.4 ms (25 samples @ 360 Hz)" },
      { label: "Extraction Window", value: "200 samples [-72, +128]" },
    ],
    rules: [
      "Strict zero-lookahead causality: every feature is computed using past beats exclusively.",
      "1-to-1 bipartite Hungarian/greedy matching enforces exact integer accounting.",
    ],
  },
  {
    id: "dsp",
    number: "02",
    name: "Causal IIR Filter & Normalization",
    module: "src.dsp.DSPFilter",
    icon: Sliders,
    tag: "Signal Conditioning",
    summary:
      "Eliminates baseline wander (<0.5 Hz) and EMG noise (>45 Hz) using a 4th-order forward-only Butterworth digital bandpass filter, followed by in-place running Z-score standardization.",
    input: "200-sample raw ECG window with baseline drift & noise",
    output: "Clean, standardized 200-sample zero-mean unit-variance array",
    hardwareSpecs: [
      { label: "Filter Order", value: "4th-order Butterworth IIR" },
      { label: "Passband", value: "0.5 Hz – 45.0 Hz" },
      { label: "Lookahead", value: "0 samples (Strict Forward)" },
      { label: "Processing Time", value: "< 0.05 ms / window" },
    ],
    rules: [
      "Forward-only filtering prevents any acausal leakage into future samples.",
      "Maintains unit variance input distribution for INT8 quantized neural layers.",
    ],
  },
  {
    id: "tinyml",
    number: "03",
    name: "Dual-Branch TinyML Inference",
    module: "src.edge.TinyMLEngine",
    icon: Cpu,
    tag: "Quantized INT8 Fusion",
    summary:
      "Executes a dual-branch quantized INT8 1D-CNN fusing beat waveform morphology (1D-CNN branch) with causal rhythm context (Dense branch). Achieves 79.28% sensitivity on MIT-BIH and 86.43% on external cohorts.",
    input: "Waveform tensor (1, 200, 1) int8 + RR features tensor (1, 3) int8",
    output: "Anomaly probability p ∈ [0.0, 1.0] and binary predicted label",
    hardwareSpecs: [
      { label: "Architecture", value: "Dual-Branch Waveform + Causal RR" },
      { label: "Parameter Count", value: "1,698 weights (+160 vs baseline)" },
      { label: "Model Flash Size", value: "12.91 KB (13,216 bytes)" },
      { label: "Tensor Arena SRAM", value: "5.4 KB (Limit: 256 KB)" },
      { label: "Inference Latency", value: "1.8 ms (End-to-End: 357.4 ms)" },
    ],
    rules: [
      "Strict full-integer TFLITE_BUILTINS_INT8 post-training quantization.",
      "Activation memory double-buffered; fits in 16 KB reserved MCU arena.",
    ],
  },
  {
    id: "scheduler",
    number: "04",
    name: "Hysteresis State Scheduler",
    module: "src.scheduler.StateScheduler",
    icon: Radio,
    tag: "Power & Duty Cycle Manager",
    summary:
      "Controls SLEEP/ACTIVE radio states with a 5-beat recovery hysteresis lockout. While confidence < τ* (0.52), radio sleeps. Upon arrhythmia detection, it alerts and locks out rapid thrashing, bounding false alarms to 5.24 ep/hr.",
    input: "Per-beat anomaly confidence score p ∈ [0.0, 1.0]",
    output: "Radio state transition (SLEEP or ACTIVE) + alert dispatch trigger",
    hardwareSpecs: [
      { label: "Decision Threshold (τ*)", value: "0.52 (F1-Opt) / 0.48 (F2-Opt)" },
      { label: "Recovery Lockout", value: "5 consecutive normal beats" },
      { label: "MIT-BIH False Alarms", value: "5.24 episodes/hr (21 episodes)" },
      { label: "External False Alarms", value: "11.71 episodes/hr (41 episodes)" },
    ],
    rules: [
      "5-beat recovery timer eliminates radio thrashing and caregiver alarm fatigue.",
      "Every state change logged with microsecond timestamp and triggering confidence.",
    ],
  },
  {
    id: "telemetry",
    number: "05",
    name: "Privacy-Preserving Telemetry",
    module: "src.telemetry.SecureTelemetry",
    icon: Lock,
    tag: "AES-GCM Authenticated Cipher",
    summary:
      "Serializes structured clinical metadata exclusively (timestamp, anomaly_id, confidence, heart_rate_bpm, auth_tag). Strict schema assertions unconditionally forbid raw waveform sample arrays.",
    input: "Timestamp, anomaly classification code, confidence score, BPM",
    output: "AES-GCM authenticated JSON packet (107 to 132 bytes; 0 raw ECG)",
    hardwareSpecs: [
      { label: "Cipher", value: "AES-GCM (256-bit symmetric key)" },
      { label: "Payload Length", value: "107 – 132 bytes" },
      { label: "Raw ECG Transmitted", value: "0 bytes [STRICTLY AUDITED]" },
      { label: "Bandwidth Reduction", value: "99.04% vs continuous streaming" },
    ],
    rules: [
      "Structural schema assertion unconditionally rejects any raw sample array.",
      "Zero raw patient biosignals traverse RF channels.",
    ],
  },
];

export default function PipelineExplorer() {
  const [activeStageId, setActiveStageId] = useState<string>("tinyml");
  const currentStage = STAGES.find((s) => s.id === activeStageId) || STAGES[2];

  return (
    <section id="pipeline" className="py-20 border-b border-steel/15 bg-surface/50">
      <div className="mx-auto max-w-6xl px-6">
        {/* Section Header */}
        <div className="max-w-2xl mb-12">
          <span className="text-xs font-medium text-steel">System Architecture</span>
          <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink mt-2 mb-4">
            How the 5-stage edge pipeline executes.
          </h2>
          <p className="text-base sm:text-lg text-ink/75 leading-relaxed">
            Every 200-sample window is conditioned, classified, and scheduled directly inside the virtual
            microcontroller. Step through each architectural stage below.
          </p>
        </div>

        {/* 5-Stage Stepper Buttons */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 sm:gap-3 mb-8">
          {STAGES.map((stage) => {
            const Icon = stage.icon;
            const isActive = stage.id === activeStageId;
            return (
              <button
                key={stage.id}
                onClick={() => setActiveStageId(stage.id)}
                className={`flex flex-col items-start p-3.5 rounded-sm border text-left transition-all cursor-pointer ${
                  isActive
                    ? "bg-surface border-signal/60 shadow-xs ring-1 ring-signal/20"
                    : "bg-surface/60 border-steel/20 hover:bg-surface hover:border-steel/35"
                }`}
              >
                <div className="flex items-center justify-between w-full mb-2">
                  <span className="text-[11px] font-numeric font-semibold text-steel">{stage.number}</span>
                  <Icon
                    className={`h-4 w-4 transition-colors ${isActive ? "text-signal" : "text-steel/70"}`}
                  />
                </div>
                <span className="text-xs font-semibold text-ink line-clamp-1">{stage.name}</span>
                <span className="text-[11px] text-steel font-numeric mt-0.5">{stage.tag}</span>
              </button>
            );
          })}
        </div>

        {/* Selected Stage Detail Panel */}
        <div className="rounded-sm border border-steel/20 bg-surface p-6 sm:p-8 shadow-xs">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-steel/15 pb-4 mb-6">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-numeric font-semibold text-signal bg-signal/10 px-2 py-0.5 rounded">
                  Stage {currentStage.number}
                </span>
                <span className="text-xs font-numeric text-steel">{currentStage.module}</span>
              </div>
              <h3 className="text-xl sm:text-2xl font-semibold text-ink">{currentStage.name}</h3>
            </div>
            <div className="text-xs text-steel font-numeric bg-surface-recessed px-3 py-1.5 rounded border border-steel/15">
              {currentStage.tag}
            </div>
          </div>

          <p className="text-sm sm:text-base text-ink/80 leading-relaxed mb-6">{currentStage.summary}</p>

          {/* I/O Specifications Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="rounded-sm border border-steel/15 bg-surface-recessed/60 p-4">
              <span className="text-xs font-semibold text-steel block mb-1">Input Contract</span>
              <p className="text-xs font-numeric text-ink/90">{currentStage.input}</p>
            </div>
            <div className="rounded-sm border border-steel/15 bg-surface-recessed/60 p-4">
              <span className="text-xs font-semibold text-steel block mb-1">Output Contract</span>
              <p className="text-xs font-numeric text-ink/90">{currentStage.output}</p>
            </div>
          </div>

          {/* Hardware Specs & Invariants */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-4 border-t border-steel/15">
            <div className="lg:col-span-2">
              <span className="text-xs font-semibold text-steel block mb-3">Hardware & Algorithm Specifications</span>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {currentStage.hardwareSpecs.map((spec, i) => (
                  <div key={i} className="p-2.5 rounded bg-surface-recessed/40 border border-steel/10 font-numeric">
                    <span className="text-[11px] text-steel block">{spec.label}</span>
                    <span className="text-xs font-semibold text-ink">{spec.value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <span className="text-xs font-semibold text-steel block mb-3">Architectural Invariants</span>
              <ul className="space-y-2 text-xs text-ink/75">
                {currentStage.rules.map((rule, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-signal shrink-0 mt-0.5" />
                    <span>{rule}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
