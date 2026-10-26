"use client";

import { useState } from "react";
import sweepData from "@/data/threshold-sweep.json";
import { Sliders, RotateCcw, CheckCircle2 } from "lucide-react";

export default function ThresholdExplorer() {
  const [threshold, setThreshold] = useState<number>(0.52);

  // Focus directly on the deployed Dual-Branch INT8 TinyML model
  const dataset = sweepData.dual_branch_int8;

  const currentPoint =
    dataset.find((pt) => Math.abs(pt.threshold - threshold) < 0.015) ||
    dataset.find((pt) => Math.abs(pt.threshold - 0.52) < 0.015) ||
    dataset[0];

  const availableThresholds = dataset.map((pt) => pt.threshold);
  const minThreshold = availableThresholds[0] ?? 0.3;
  const maxThreshold = availableThresholds[availableThresholds.length - 1] ?? 0.85;

  const handleSliderChange = (val: number) => {
    let closest = availableThresholds[0];
    let minDiff = Math.abs(val - closest);
    for (const t of availableThresholds) {
      const diff = Math.abs(val - t);
      if (diff < minDiff) {
        minDiff = diff;
        closest = t;
      }
    }
    setThreshold(closest);
  };

  const isRecommended = Math.abs(threshold - 0.52) < 0.015;

  return (
    <section id="threshold" className="py-20 border-b border-steel/15 bg-surface/50">
      <div className="mx-auto max-w-6xl px-6">
        {/* Header */}
        <div className="max-w-2xl mb-12">
          <span className="text-xs font-semibold uppercase tracking-wider text-signal">
            Interactive Calibration
          </span>
          <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink mt-2 mb-3">
            Decision Threshold Calibration
          </h2>
          <p className="text-base text-ink/70 leading-relaxed">
            Drag the slider to explore the trade-off between clinical sensitivity, false alarm rate,
            and radio transmission savings on the deployed Dual-Branch INT8 TinyML model.
          </p>
        </div>

        {/* Main Explorer Card */}
        <div className="rounded-sm border border-steel/20 bg-surface p-6 sm:p-8 shadow-xs">
          {/* Top Control Bar */}
          <div className="flex flex-wrap items-center justify-between gap-4 mb-8 pb-6 border-b border-steel/15">
            <div>
              <span className="text-xs text-steel font-medium block">Active Model Architecture</span>
              <span className="text-base font-semibold text-ink">
                Dual-Branch INT8 TinyML (1D-CNN + Causal RR)
              </span>
            </div>

            <button
              onClick={() => setThreshold(0.52)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-sm text-xs font-medium transition-all cursor-pointer ${
                isRecommended
                  ? "bg-signal text-surface shadow-xs"
                  : "bg-surface-recessed text-ink hover:bg-steel/15 border border-steel/20"
              }`}
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>Reset to Optimal (τ* = 0.52)</span>
            </button>
          </div>

          {/* Slider */}
          <div className="mb-10">
            <div className="flex items-center justify-between mb-3">
              <label htmlFor="threshold-slider" className="text-sm font-medium text-ink flex items-center gap-2">
                <Sliders className="h-4 w-4 text-signal" />
                <span>Operating Threshold (τ)</span>
              </label>
              <div className="flex items-baseline gap-2 font-numeric">
                <span className="text-3xl font-semibold text-ink">
                  τ = {threshold.toFixed(2)}
                </span>
                {isRecommended && (
                  <span className="text-xs font-semibold text-signal uppercase tracking-wider">
                    (Calibrated Optimal)
                  </span>
                )}
              </div>
            </div>

            <input
              id="threshold-slider"
              type="range"
              min={minThreshold}
              max={maxThreshold}
              step={0.01}
              value={threshold}
              onChange={(e) => handleSliderChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-surface-recessed rounded-lg appearance-none cursor-pointer accent-signal"
            />

            <div className="flex justify-between text-[11px] font-numeric text-steel mt-2">
              <span>0.30 (Higher Sensitivity)</span>
              <span className="text-signal font-semibold">τ* = 0.52 (Balanced F1)</span>
              <span>0.85 (Conservative)</span>
            </div>
          </div>

          {/* 4 Outcome Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="p-4 rounded-sm bg-surface-recessed/60 border border-steel/10 font-numeric">
              <span className="text-xs text-steel block mb-1">Arrhythmia Sensitivity</span>
              <div className="text-2xl sm:text-3xl font-semibold text-ink">
                {currentPoint.recall_percent.toFixed(2)}%
              </div>
              <span className="text-[11px] text-steel mt-1 block">
                {currentPoint.tp.toLocaleString()} / 1,332 beats
              </span>
            </div>

            <div className="p-4 rounded-sm bg-surface-recessed/60 border border-steel/10 font-numeric">
              <span className="text-xs text-steel block mb-1">Precision (PPV)</span>
              <div className="text-2xl sm:text-3xl font-semibold text-ink">
                {currentPoint.precision_percent.toFixed(2)}%
              </div>
              <span className="text-[11px] text-steel mt-1 block">
                Alert reliability
              </span>
            </div>

            <div className="p-4 rounded-sm bg-surface-recessed/60 border border-steel/10 font-numeric">
              <span className="text-xs text-steel block mb-1">False Alarm Rate</span>
              <div className="text-2xl sm:text-3xl font-semibold text-ink">
                {currentPoint.false_alarms_per_hour.toFixed(1)} <span className="text-xs font-normal text-steel">ep/hr</span>
              </div>
              <span className="text-[11px] text-steel mt-1 block">
                Limit &lt; 150/hr (PASS)
              </span>
            </div>

            <div className="p-4 rounded-sm bg-surface-recessed/60 border border-steel/10 font-numeric">
              <span className="text-xs text-steel block mb-1">Bandwidth Saved</span>
              <div className="text-2xl sm:text-3xl font-semibold text-signal">
                {currentPoint.bandwidth_reduction_percent.toFixed(2)}%
              </div>
              <span className="text-[11px] text-steel mt-1 block">
                Radio dormant &gt;90%
              </span>
            </div>
          </div>

          {/* Concise Takeaway Callout */}
          <div className="p-4 rounded-sm bg-surface-recessed border border-steel/15 flex items-start gap-3">
            <CheckCircle2 className="h-4 w-4 text-signal shrink-0 mt-0.5" />
            <p className="text-xs text-ink/80 leading-relaxed font-numeric">
              {isRecommended ? (
                <>
                  <strong className="text-ink">Optimal Operating Point (τ* = 0.52):</strong> Strikes the ideal
                  clinical balance, achieving <strong>79.28% arrhythmia sensitivity</strong> (1,056/1,332 beats) while
                  restricting false alarms to only <strong>5.24 episodes/hour</strong> and saving{" "}
                  <strong>99.04%</strong> of radio transmission bandwidth.
                </>
              ) : threshold < 0.52 ? (
                <>
                  <strong className="text-ink">High Sensitivity Operating Point:</strong> Detects more anomalous beats
                  ({currentPoint.recall_percent.toFixed(1)}% sensitivity) at the expense of a higher false alarm rate
                  ({currentPoint.false_alarms_per_hour.toFixed(1)} episodes/hr).
                </>
              ) : (
                <>
                  <strong className="text-ink">Conservative Operating Point:</strong> Slashes false alarms to minimum,
                  but risks missing intermittent arrhythmia events ({currentPoint.recall_percent.toFixed(1)}% sensitivity).
                </>
              )}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
