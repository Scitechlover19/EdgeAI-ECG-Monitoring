"use client";

import metricsData from "@/data/metrics.json";
import { AlertCircle, CheckCircle2, Zap, Shield, Cpu, Clock, Activity, HardDrive } from "lucide-react";

export default function MetricsPanel() {
  const op = metricsData.operating_point;
  const budgets = metricsData.resource_budgets;
  const compression = metricsData.model_compression;

  return (
    <section id="metrics" className="py-20 border-b border-steel/15">
      <div className="mx-auto max-w-6xl px-6">
        {/* Header */}
        <div className="max-w-2xl mb-12">
          <span className="text-xs font-medium text-steel">Empirical Evidence</span>
          <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink mt-2 mb-4">
            Directly measured, never fabricated.
          </h2>
          <p className="text-base sm:text-lg text-ink/75 leading-relaxed">
            Every figure below traces directly to benchmark logs and held-out MIT-BIH test evaluation runs in the
            repository reports.
          </p>
        </div>

        {/* Primary Metric Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {/* Bandwidth Reduction */}
          <div className="rounded-sm border border-steel/20 bg-surface p-5 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span>Payload Reduction</span>
                <span className="font-numeric text-signal font-semibold">NFR-6 PASS</span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-1">
                {op.bandwidth_reduction_percent.toFixed(2)}%
              </div>
              <p className="text-xs text-ink/70">
                198.8 KB transmitted vs. 20.8 MB baseline across 51,992 test windows.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>Target: &gt; 90.0%</span>
              <span className="text-signal font-medium">+9.04% headroom</span>
            </div>
          </div>

          {/* Anomaly Recall */}
          <div className="rounded-sm border border-steel/20 bg-surface p-5 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span>Anomaly Sensitivity</span>
                <span className="font-numeric text-steel font-medium">τ = 0.35</span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-1">
                {op.recall_percent.toFixed(2)}%
              </div>
              <p className="text-xs text-ink/70">
                1,004 of 7,278 arrhythmia windows detected (20.5× increase over old 0.85 threshold).
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>Status Quo (0.85): 0.67%</span>
              <span className="text-signal font-medium">+13.12% delta</span>
            </div>
          </div>

          {/* Precision */}
          <div className="rounded-sm border border-steel/20 bg-surface p-5 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span>Precision (PPV)</span>
                <span className="font-numeric text-steel font-medium">Alert Quality</span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-1">
                {op.precision_percent.toFixed(2)}%
              </div>
              <p className="text-xs text-ink/70">
                2 out of every 3 dispatched telemetry alerts correspond to genuine arrhythmias.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>FDR: 33.55%</span>
              <span className="text-signal font-medium">High reliability</span>
            </div>
          </div>

          {/* Clinical Alert Fatigue */}
          <div className="rounded-sm border border-steel/20 bg-surface p-5 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span>Alert Burden</span>
                <span className="font-numeric text-steel font-medium">Clinical Tolerance</span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-1">
                {op.false_alarm_rate_per_hour.toFixed(1)} <span className="text-sm font-normal text-steel">FP/hr</span>
              </div>
              <p className="text-xs text-ink/70">
                507 false alarms over 4.01 monitored hours (~2.1 alerts/min). Avoids KD fatigue (474 FP/hr).
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>KD Alternative: 474/hr</span>
              <span className="text-signal font-medium">3.7x lower burden</span>
            </div>
          </div>
        </div>

        {/* Secondary Hardware Resource Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
          {/* Flash Size */}
          <div className="rounded-sm border border-steel/15 bg-surface-recessed/40 p-4 font-numeric">
            <div className="flex items-center gap-2 text-xs text-steel mb-1.5">
              <HardDrive className="h-3.5 w-3.5 text-signal" />
              <span>Model Flash Size</span>
            </div>
            <div className="text-xl font-semibold text-ink">{budgets.flash.model_kb.toFixed(2)} KB</div>
            <div className="text-[11px] text-steel mt-1">
              Limit: &lt; 1,024 KB · <span className="text-signal font-medium">98.9% headroom</span>
            </div>
          </div>

          {/* Peak SRAM */}
          <div className="rounded-sm border border-steel/15 bg-surface-recessed/40 p-4 font-numeric">
            <div className="flex items-center gap-2 text-xs text-steel mb-1.5">
              <Cpu className="h-3.5 w-3.5 text-signal" />
              <span>Simulated Peak SRAM</span>
            </div>
            <div className="text-xl font-semibold text-ink">{budgets.sram.peak_estimated_kb.toFixed(2)} KB</div>
            <div className="text-[11px] text-steel mt-1">
              Ceiling: ≤ 256 KB · <span className="text-signal font-medium">94.1% headroom</span>
            </div>
          </div>

          {/* Inference Latency */}
          <div className="rounded-sm border border-steel/15 bg-surface-recessed/40 p-4 font-numeric">
            <div className="flex items-center gap-2 text-xs text-steel mb-1.5">
              <Clock className="h-3.5 w-3.5 text-signal" />
              <span>Inference Latency</span>
            </div>
            <div className="text-xl font-semibold text-ink">
              {budgets.latency.mean_ms.toFixed(4)}{" "}
              <span className="text-xs text-steel font-normal">± {budgets.latency.std_ms.toFixed(4)} ms</span>
            </div>
            <div className="text-[11px] text-steel mt-1">
              Budget: &lt; 50 ms · <span className="text-signal font-medium">312× faster</span>
            </div>
          </div>

          {/* Raw Waveform Leakage */}
          <div className="rounded-sm border border-steel/15 bg-surface-recessed/40 p-4 font-numeric">
            <div className="flex items-center gap-2 text-xs text-steel mb-1.5">
              <Shield className="h-3.5 w-3.5 text-signal" />
              <span>Raw Waveform Leakage</span>
            </div>
            <div className="text-xl font-semibold text-signal">0 Bytes</div>
            <div className="text-[11px] text-steel mt-1">
              1,511 alerts audited · <span className="text-signal font-medium">100% verified</span>
            </div>
          </div>
        </div>

        {/* Honest Limitation Callout Banner */}
        <div className="rounded-sm border border-steel/25 border-l-4 border-l-signal bg-surface p-6 sm:p-7 shadow-xs">
          <div className="flex items-start gap-3.5">
            <AlertCircle className="h-5 w-5 text-signal shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm sm:text-base font-semibold text-ink mb-1.5">
                Scientific Honesty: Absolute Recall & Proof-of-Concept Boundary
              </h3>
              <p className="text-xs sm:text-sm text-ink/75 leading-relaxed">
                Under 14.0% minority arrhythmia class prevalence on the held-out MIT-BIH test split, absolute recall
                of this 1,538-parameter INT8 model is <strong>13.79%</strong>. While moving to τ = 0.35 delivers a 20.5×
                detection gain over the initial threshold and avoids the severe alert fatigue of Knowledge Distillation
                (which flooded 474 to 612 false alerts per hour), minority arrhythmia detection remains an active challenge.
              </p>
              <p className="text-xs text-steel mt-2 font-numeric">
                The bandwidth and privacy objectives are solved and verified (&gt;99% reduction, 0 raw ECG bytes leaked).
                Architecture scaling and patient-adaptive thresholding are designated as future work. This project is an
                explicit software-engineering proof-of-concept, not a clinical diagnostic device.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
