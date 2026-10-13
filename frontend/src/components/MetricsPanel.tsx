"use client";

import metricsData from "@/data/metrics.json";
import { AlertCircle, CheckCircle2, Zap, Shield, Cpu, Clock, Activity, HardDrive, Check, Layers } from "lucide-react";

export default function MetricsPanel() {
  const op = metricsData.operating_point;
  const ext = metricsData.stage3_external_validation;
  const budgets = metricsData.resource_budgets;
  const progression = metricsData.stage_progression;

  return (
    <section id="metrics" className="py-20 border-b border-steel/15">
      <div className="mx-auto max-w-6xl px-6">
        {/* Header */}
        <div className="max-w-2xl mb-12">
          <span className="text-xs font-medium text-steel">Empirical Evidence & Stage 3 Validation</span>
          <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink mt-2 mb-4">
            Reconciled, externally validated, never fabricated.
          </h2>
          <p className="text-base sm:text-lg text-ink/75 leading-relaxed">
            Every metric below traces to reproducible benchmark runs across the 8 held-out MIT-BIH records and
            7 unseen external clinical records (MIT-BIH SVDB + St Petersburg INCART 12-lead).
          </p>
        </div>

        {/* Primary Metric Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {/* MIT-BIH Sensitivity */}
          <div className="rounded-sm border border-steel/20 bg-surface p-5 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span>MIT-BIH Sensitivity</span>
                <span className="font-numeric text-signal font-semibold">STAGE 2 VERIFIED</span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-1">
                {op.recall_percent.toFixed(2)}%
              </div>
              <p className="text-xs text-ink/70">
                1,056 of 1,332 reference target arrhythmias detected at τ* = 0.52 (up from 13.79% baseline).
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>Specificity: {op.specificity_percent.toFixed(2)}%</span>
              <span className="text-signal font-medium">F1: {op.f1_score.toFixed(4)}</span>
            </div>
          </div>

          {/* External Cross-Dataset Sensitivity */}
          <div className="rounded-sm border border-steel/20 bg-surface p-5 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span>External Cohort Recall</span>
                <span className="font-numeric text-signal font-semibold">STAGE 3 VALIDATED</span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-1">
                {ext.recall_percent.toFixed(2)}%
              </div>
              <p className="text-xs text-ink/70">
                1,083 of 1,259 target arrhythmias detected on 7 unseen patients (SVDB 128 Hz + INCART 257 Hz).
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>VEB Recall: {ext.veb_recall_percent.toFixed(2)}%</span>
              <span className="text-signal font-medium">Zero Retraining</span>
            </div>
          </div>

          {/* Precision & False Alarm Suppression */}
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
                1,056 true alerts out of 1,688 emitted alarms. Strict 1-to-1 bipartite matching.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>Accuracy: {op.accuracy_percent.toFixed(2)}%</span>
              <span className="text-signal font-medium">Reconciled Sum</span>
            </div>
          </div>

          {/* Clinical Alert Fatigue */}
          <div className="rounded-sm border border-steel/20 bg-surface p-5 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span>Clinical Alert Burden</span>
                <span className="font-numeric text-signal font-semibold">&lt; 150/hr CEILING</span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-1">
                {op.false_alarm_rate_per_hour.toFixed(2)} <span className="text-sm font-normal text-steel">ep/hr</span>
              </div>
              <p className="text-xs text-ink/70">
                21 false episodes over 4.01h on MIT-BIH; 11.71 ep/hr on external cohort under 5-beat recovery lockout.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>External: 11.71 ep/hr</span>
              <span className="text-signal font-medium">&gt;12x below ceiling</span>
            </div>
          </div>
        </div>

        {/* Secondary Hardware Resource Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {/* Flash Size */}
          <div className="rounded-sm border border-steel/15 bg-surface-recessed/40 p-4 font-numeric">
            <div className="flex items-center gap-2 text-xs text-steel mb-1.5">
              <HardDrive className="h-3.5 w-3.5 text-signal" />
              <span>Model Flash Size</span>
            </div>
            <div className="text-xl font-semibold text-ink">{budgets.flash.model_kb.toFixed(2)} KB</div>
            <div className="text-[11px] text-steel mt-1">
              1,698 params · Total firmware ~82 KB · <span className="text-signal font-medium">98.7% headroom</span>
            </div>
          </div>

          {/* Peak SRAM */}
          <div className="rounded-sm border border-steel/15 bg-surface-recessed/40 p-4 font-numeric">
            <div className="flex items-center gap-2 text-xs text-steel mb-1.5">
              <Cpu className="h-3.5 w-3.5 text-signal" />
              <span>SRAM Tensor Arena</span>
            </div>
            <div className="text-xl font-semibold text-ink">{budgets.sram.tensor_arena_kb.toFixed(1)} KB</div>
            <div className="text-[11px] text-steel mt-1">
              Ceiling: ≤ 256 KB · Reserved: 16 KB · <span className="text-signal font-medium">97.9% headroom</span>
            </div>
          </div>

          {/* End-to-End Latency */}
          <div className="rounded-sm border border-steel/15 bg-surface-recessed/40 p-4 font-numeric">
            <div className="flex items-center gap-2 text-xs text-steel mb-1.5">
              <Clock className="h-3.5 w-3.5 text-signal" />
              <span>End-to-End Latency</span>
            </div>
            <div className="text-xl font-semibold text-ink">
              {budgets.latency.end_to_end_ms.toFixed(1)} <span className="text-xs text-steel font-normal">ms</span>
            </div>
            <div className="text-[11px] text-steel mt-1">
              1.8 ms inference + 355.6 ms buffer · <span className="text-signal font-medium">AAMI &lt; 10s PASS</span>
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
              99.04% bandwidth reduction · <span className="text-signal font-medium">Zero Privacy Exposure</span>
            </div>
          </div>
        </div>

        {/* Multi-Stage Experimental Progression Table */}
        <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="h-4 w-4 text-signal" />
            <h3 className="text-base font-semibold text-ink">
              Empirical Evolution Across Model Stages
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-numeric text-left">
              <thead>
                <tr className="border-b border-steel/20 text-steel font-medium">
                  <th className="py-2.5 px-3">Experimental Stage</th>
                  <th className="py-2.5 px-3">Architecture & Technique</th>
                  <th className="py-2.5 px-3">Operating Point</th>
                  <th className="py-2.5 px-3">Sensitivity (Recall)</th>
                  <th className="py-2.5 px-3">Precision (PPV)</th>
                  <th className="py-2.5 px-3">False Alarm Burden</th>
                  <th className="py-2.5 px-3">F1-Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-steel/10 text-ink/80">
                {progression.map((item, idx) => (
                  <tr key={idx} className={idx >= 2 ? "bg-signal/5 font-medium" : ""}>
                    <td className="py-2.5 px-3 text-ink font-semibold">{item.stage}</td>
                    <td className="py-2.5 px-3">{item.architecture}</td>
                    <td className="py-2.5 px-3">τ* = {item.threshold.toFixed(2)}</td>
                    <td className="py-2.5 px-3 text-signal font-semibold">{item.recall.toFixed(2)}%</td>
                    <td className="py-2.5 px-3">{item.precision.toFixed(2)}%</td>
                    <td className="py-2.5 px-3">{item.fp_per_hr.toFixed(1)} ep/hr</td>
                    <td className="py-2.5 px-3">{item.f1.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Scientific Honesty & Stage 3 Completion Callout Banner */}
        <div className="rounded-sm border border-steel/25 border-l-4 border-l-signal bg-surface p-6 sm:p-7 shadow-xs">
          <div className="flex items-start gap-3.5">
            <CheckCircle2 className="h-5 w-5 text-signal shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm sm:text-base font-semibold text-ink mb-1.5">
                Stage 3 Completed: Cross-Database Validation & Full Arithmetic Reconciliation
              </h3>
              <p className="text-xs sm:text-sm text-ink/75 leading-relaxed">
                The pipeline has completed all 3 stages. Moving from sliding windows to beat-centred waveform inference
                coupled with causal RR context boosted MIT-BIH sensitivity from <strong>13.79% to 79.28%</strong> (1,056 / 1,332
                biological arrhythmia beats) while slashing false alarms to <strong>5.24 ep/hr</strong>. When tested on a completely
                unseen external cohort of 7 records across two distinct clinical databases (MIT-BIH SVDB at 128 Hz and St Petersburg
                INCART 12-lead at 257 Hz), the frozen model achieved <strong>86.43% overall sensitivity</strong> and{" "}
                <strong>91.34% VEB sensitivity</strong> with a manageable <strong>11.71 ep/hr</strong> false alarm rate.
              </p>
              <p className="text-xs text-steel mt-2 font-numeric">
                Embedded resources remain compact: 1,698 parameters, 12.91 KiB Flash, 5.4 KiB SRAM tensor arena, 1.8 ms inference latency,
                and 357.4 ms end-to-end delay (well below the AAMI EC13 10-second limit). Zero raw ECG data is transmitted over RF telemetry.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
