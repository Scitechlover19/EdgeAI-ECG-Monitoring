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
    name: "Data Ingestion & Framing",
    module: "src.ingestion.DataIngestion",
    icon: Activity,
    tag: "Time-Series Windowing",
    summary:
      "Segments raw continuous analog-to-digital ECG streams into uniform 200-sample sliding windows with 50% overlap (100 sample stride) at the record's native sampling rate (360 Hz).",
    input: "Continuous raw ECG voltage stream (16-bit ADC samples)",
    output: "1D NumPy array of 200 raw float32 samples (555.5 ms window duration)",
    hardwareSpecs: [
      { label: "Window Size", value: "200 samples" },
      { label: "Overlap Ratio", value: "50% (100 samples)" },
      { label: "Sampling Rate", value: "360.0 Hz" },
      { label: "Buffer Allocation", value: "400 bytes (in-RAM ring)" },
    ],
    rules: [
      "Strict zero-loss sliding step preserves transient ectopic beat morphologies across window boundaries.",
      "Never mutates the original patient record metadata.",
    ],
  },
  {
    id: "dsp",
    number: "02",
    name: "DSP Bandpass & Conditioning",
    module: "src.dsp.DSPFilter",
    icon: Sliders,
    tag: "Signal Conditioning",
    summary:
      "Eliminates baseline wander (respiratory drift <0.5 Hz) and high-frequency EMG muscle tremor (>45 Hz) using a 4th-order zero-phase Butterworth bandpass filter, followed by mean-centering and Z-score normalization.",
    input: "200-sample raw ECG window with baseline drift & noise",
    output: "Normalized, noise-filtered 200-sample zero-mean unit-variance array",
    hardwareSpecs: [
      { label: "Filter Order", value: "4th-order Butterworth" },
      { label: "Passband", value: "0.5 Hz – 45.0 Hz" },
      { label: "Standardization", value: "Z-score (μ=0, σ=1)" },
      { label: "Processing Time", value: "< 0.05 ms / window" },
    ],
    rules: [
      "Configurable via config.yaml without code refactoring.",
      "Ensures stable numerical input distribution for INT8 quantized neural layers.",
    ],
  },
  {
    id: "tinyml",
    number: "03",
    name: "TinyML Edge Inference Engine",
    module: "src.edge.TinyMLEngine",
    icon: Cpu,
    tag: "INT8 Neural Inference",
    summary:
      "Executes full-integer INT8 post-training quantized 1D-CNN inference inside VirtualMCU. Distilled from a 120k-parameter Teacher down to 1,538 parameters (78.5x compression), returning calibrated anomaly probabilities.",
    input: "Preprocessed 1D tensor shape (1, 200, 1) int8",
    output: "Anomaly confidence score c ∈ [0.0, 1.0] and binary predicted label",
    hardwareSpecs: [
      { label: "Model Architecture", value: "1D-CNN + Global Average Pooling" },
      { label: "Parameter Count", value: "1,538 weights" },
      { label: "Model Flash Size", value: "10.96 KB (11,224 bytes)" },
      { label: "Peak SRAM Footprint", value: "15.16 KB [ESTIMATED]" },
      { label: "Inference Latency", value: "0.1599 ± 0.0126 ms [MEASURED]" },
    ],
    rules: [
      "Strict full-integer TFLITE_BUILTINS_INT8 quantization.",
      "Regression-guarded against double-softmax confidence squashing.",
    ],
  },
  {
    id: "scheduler",
    number: "04",
    name: "Anomaly-Driven State Scheduler",
    module: "src.scheduler.StateScheduler",
    icon: Radio,
    tag: "Power & Radio Duty Management",
    summary:
      "Enforces the core SLEEP/ACTIVE power-management state machine. While anomaly confidence remains below threshold (τ = 0.35), the wireless radio remains unpowered. When an anomaly breaches threshold, it wakes the radio to transmit.",
    input: "Per-window anomaly confidence score c ∈ [0.0, 1.0]",
    output: "Radio state transition (SLEEP or ACTIVE) + alert dispatch trigger",
    hardwareSpecs: [
      { label: "Decision Threshold (τ)", value: "0.35 (Decision #15)" },
      { label: "Default State", value: "SLEEP (Radio Powered Down)" },
      { label: "Active State", value: "ACTIVE (Wake, Transmit, Sleep)" },
      { label: "Radio Duty Cycle", value: "2.91% active (97.09% dormant)" },
    ],
    rules: [
      "AGENTS.md Rule 6: SecureTelemetry may ONLY be invoked by StateScheduler decision.",
      "Every state transition is logged with timestamp, window index, and triggering confidence.",
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
      "Constructs a strictly restricted 4-field metadata payload, encrypts it using AES-GCM 256-bit encryption with a 12-byte random nonce, and transmits only the encrypted bytes to the sink. Raw ECG waveforms are structurally forbidden.",
    input: "Timestamp, anomaly classification code, confidence score",
    output: "AES-GCM authenticated ciphertext (122 to 133 bytes; 0 raw ECG)",
    hardwareSpecs: [
      { label: "Cipher", value: "AES-GCM (256-bit symmetric key)" },
      { label: "Nonce Overhead", value: "12 bytes (96-bit random)" },
      { label: "Auth Tag", value: "16 bytes (128-bit integrity)" },
      { label: "Payload Length", value: "131.56 bytes avg [MEASURED]" },
      { label: "Raw ECG Transmitted", value: "0 bytes [STRICTLY AUDITED]" },
    ],
    rules: [
      "AGENTS.md Rule 5: Structural payload schema rejects any array >5 elements.",
      "Zero biological waveform data leaves the microcontroller.",
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
