"use client";

import metricsData from "@/data/metrics.json";
import { Activity, HardDrive, Cpu, Shield, Clock, CheckCircle2 } from "lucide-react";

export default function MetricsPanel() {
  const op = metricsData.operating_point;
  const budgets = metricsData.resource_budgets;

  return (
    <section id="metrics" className="py-20 border-b border-steel/15 bg-surface/30">
      <div className="mx-auto max-w-6xl px-6">
        {/* Clean Header */}
        <div className="max-w-2xl mb-12">
          <span className="text-xs font-semibold uppercase tracking-wider text-signal">
            Measured Performance
          </span>
          <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink mt-2 mb-3">
            Core Clinical & Embedded Metrics
          </h2>
          <p className="text-base text-ink/70 leading-relaxed">
            Key empirical results from the Dual-Branch INT8 TinyML pipeline running on held-out MIT-BIH
            recordings at calibrated operating threshold τ* = 0.52.
          </p>
        </div>

        {/* 4 Core Hero Metric Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
          {/* 1. Arrhythmia Recall */}
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span className="font-medium text-ink/80">Arrhythmia Sensitivity</span>
                <span className="text-[11px] font-numeric px-2 py-0.5 rounded bg-signal/15 text-signal font-semibold">
                  τ* = 0.52
                </span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-2">
                {op.recall_percent.toFixed(2)}%
              </div>
              <p className="text-xs text-ink/70 leading-relaxed">
                1,056 of 1,332 reference target arrhythmias detected on held-out test data.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>VEB Recall: 85.53%</span>
              <span className="text-signal font-medium">F1: {op.f1_score.toFixed(3)}</span>
            </div>
          </div>

          {/* 2. Bandwidth Reduction */}
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span className="font-medium text-ink/80">Bandwidth Reduction</span>
                <span className="text-[11px] font-numeric px-2 py-0.5 rounded bg-signal/15 text-signal font-semibold">
                  Data Saved
                </span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-2">
                {op.bandwidth_reduction_percent.toFixed(2)}%
              </div>
              <p className="text-xs text-ink/70 leading-relaxed">
                Transmits 198 KB in alert packets vs 20.8 MB continuous raw stream.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>Radio Dormant: &gt;90%</span>
              <span className="text-signal font-medium">0B Raw Leak</span>
            </div>
          </div>

          {/* 3. Model Flash Size */}
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span className="font-medium text-ink/80">Model Flash Footprint</span>
                <span className="text-[11px] font-numeric px-2 py-0.5 rounded bg-surface-recessed text-steel font-medium">
                  Flash ROM
                </span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-2">
                12.91 <span className="text-lg font-normal text-steel">KB</span>
              </div>
              <p className="text-xs text-ink/70 leading-relaxed">
                Dual-Branch INT8 TinyML model (1,698 params). Consumes only 1.26% of 1 MB Flash.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>Limit: 1,024 KB</span>
              <span className="text-signal font-medium">98.7% Headroom</span>
            </div>
          </div>

          {/* 4. Tensor Arena SRAM */}
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-steel mb-2">
                <span className="font-medium text-ink/80">Tensor Arena SRAM</span>
                <span className="text-[11px] font-numeric px-2 py-0.5 rounded bg-surface-recessed text-steel font-medium">
                  SRAM RAM
                </span>
              </div>
              <div className="text-3xl sm:text-4xl font-semibold text-ink font-numeric tracking-tight mb-2">
                5.40 <span className="text-lg font-normal text-steel">KB</span>
              </div>
              <p className="text-xs text-ink/70 leading-relaxed">
                Peak activation buffer on Cortex-M4. Consumes only 2.11% of 256 KB SRAM.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-steel/10 flex justify-between text-[11px] font-numeric text-steel">
              <span>Limit: 256 KB</span>
              <span className="text-signal font-medium">97.9% Headroom</span>
            </div>
          </div>
        </div>

        {/* Sleek Embedded Specs Bar (3 Columns, clean & scannable) */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="rounded-sm border border-steel/15 bg-surface p-4 flex items-center gap-3.5">
            <Clock className="h-5 w-5 text-signal shrink-0" />
            <div className="text-xs font-numeric">
              <div className="text-ink font-semibold text-sm">
                {budgets.latency.end_to_end_ms.toFixed(1)} ms Latency
              </div>
              <div className="text-steel mt-0.5">
                1.8 ms inference · AAMI EC13 limit &lt; 10s (PASS)
              </div>
            </div>
          </div>

          <div className="rounded-sm border border-steel/15 bg-surface p-4 flex items-center gap-3.5">
            <Activity className="h-5 w-5 text-signal shrink-0" />
            <div className="text-xs font-numeric">
              <div className="text-ink font-semibold text-sm">
                {op.false_alarm_rate_per_hour.toFixed(2)} ep/hr False Alarms
              </div>
              <div className="text-steel mt-0.5">
                5-beat recovery timer · Clinical limit &lt; 150/hr (PASS)
              </div>
            </div>
          </div>

          <div className="rounded-sm border border-steel/15 bg-surface p-4 flex items-center gap-3.5">
            <Shield className="h-5 w-5 text-signal shrink-0" />
            <div className="text-xs font-numeric">
              <div className="text-ink font-semibold text-sm">
                0 Bytes Raw ECG Leakage
              </div>
              <div className="text-steel mt-0.5">
                AES-GCM metadata-only payload · Strictly audited
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
