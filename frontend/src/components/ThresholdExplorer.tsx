"use client";

import { useState } from "react";
import sweepData from "@/data/threshold-sweep.json";
import { Sliders, AlertTriangle, CheckCircle2, TrendingUp, Radio } from "lucide-react";

export default function ThresholdExplorer() {
  const [modelType, setModelType] = useState<"no_kd_int8" | "kd_int8">("no_kd_int8");
  const [threshold, setThreshold] = useState<number>(0.35);

  const activeDataset = modelType === "no_kd_int8" ? sweepData.no_kd_int8 : sweepData.kd_int8;
  const currentPoint = activeDataset.find((pt) => Math.abs(pt.threshold - threshold) < 0.001) || activeDataset[1];

  const availableThresholds = activeDataset.map((pt) => pt.threshold);

  return (
    <section id="threshold" className="py-20 border-b border-steel/15 bg-surface/40">
      <div className="mx-auto max-w-6xl px-6">
        {/* Section Header */}
        <div className="max-w-2xl mb-12">
          <span className="text-xs font-medium text-steel">Interactive Trade-off Calibration</span>
          <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink mt-2 mb-4">
            Why the threshold matters.
          </h2>
          <p className="text-base sm:text-lg text-ink/75 leading-relaxed">
            In IoMT edge monitoring, the decision threshold (τ) governs the tension between clinical sensitivity,
            physician alert fatigue, and battery radio lifetime. Drag the slider to inspect the actual measured data.
          </p>
        </div>

        {/* Controls Bar: Model Toggle & Quick Jump */}
        <div className="rounded-sm border border-steel/20 bg-surface p-6 sm:p-8 shadow-xs mb-8">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
            {/* Model Selector */}
            <div className="flex items-center gap-2 text-xs font-medium">
              <span className="text-steel">Evaluated Architecture:</span>
              <div className="inline-flex rounded-sm border border-steel/20 bg-surface-recessed p-0.5">
                <button
                  onClick={() => setModelType("no_kd_int8")}
                  className={`px-3 py-1 rounded-sm text-xs transition-all cursor-pointer ${
                    modelType === "no_kd_int8"
                      ? "bg-surface text-ink font-semibold shadow-xs"
                      : "text-steel hover:text-ink"
                  }`}
                >
                  Deployed No-KD INT8
                </button>
                <button
                  onClick={() => setModelType("kd_int8")}
                  className={`px-3 py-1 rounded-sm text-xs transition-all cursor-pointer ${
                    modelType === "kd_int8"
                      ? "bg-surface text-alert font-semibold shadow-xs"
                      : "text-steel hover:text-ink"
                  }`}
                >
                  KD INT8 (Rejected Candidate)
                </button>
              </div>
            </div>

            {/* Quick Jumps */}
            <div className="flex items-center gap-2 text-xs font-numeric">
              <span className="text-steel text-[11px]">Compare:</span>
              <button
                onClick={() => setThreshold(0.35)}
                className={`px-2 py-0.5 rounded border text-[11px] transition-colors cursor-pointer ${
                  threshold === 0.35
                    ? "bg-signal text-surface border-signal font-semibold"
                    : "bg-surface text-ink border-steel/25 hover:border-steel/50"
                }`}
              >
                τ = 0.35 (Deployed)
              </button>
              <button
                onClick={() => setThreshold(0.5)}
                className={`px-2 py-0.5 rounded border text-[11px] transition-colors cursor-pointer ${
                  threshold === 0.5
                    ? "bg-signal text-surface border-signal font-semibold"
                    : "bg-surface text-ink border-steel/25 hover:border-steel/50"
                }`}
              >
                τ = 0.50 (Standard)
              </button>
              <button
                onClick={() => setThreshold(0.85)}
                className={`px-2 py-0.5 rounded border text-[11px] transition-colors cursor-pointer ${
                  threshold === 0.85
                    ? "bg-alert text-surface border-alert font-semibold"
                    : "bg-surface text-ink border-steel/25 hover:border-steel/50"
                }`}
              >
                τ = 0.85 (Old Baseline)
              </button>
            </div>
          </div>

          {/* Interactive Threshold Slider */}
          <div className="mb-10">
            <div className="flex items-center justify-between mb-3">
              <label htmlFor="threshold-slider" className="text-sm font-medium text-ink flex items-center gap-2">
                <Sliders className="h-4 w-4 text-signal" />
                <span>Decision Threshold (τ)</span>
              </label>
              <span className="text-2xl font-semibold font-numeric text-ink">
                τ = {threshold.toFixed(2)}
              </span>
            </div>

            <input
              id="threshold-slider"
              type="range"
              min={0.3}
              max={0.95}
              step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-surface-recessed rounded-lg appearance-none cursor-pointer accent-signal"
            />

            <div className="flex justify-between text-[11px] font-numeric text-steel mt-2">
              <span>0.30 (Sensitive)</span>
              <span className={threshold === 0.35 ? "text-signal font-bold" : ""}>0.35 (Deployed)</span>
              <span>0.50</span>
              <span className={threshold === 0.85 ? "text-alert font-bold" : ""}>0.85 (Old Baseline)</span>
              <span>0.95 (Conservative)</span>
            </div>
          </div>

          {/* Dynamic Live Metric Readouts Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="p-4 rounded-sm border border-steel/15 bg-surface-recessed/50 font-numeric">
              <span className="text-xs text-steel block mb-1">Anomaly Recall</span>
              <div className="text-2xl font-semibold text-ink">{currentPoint.recall_percent.toFixed(2)}%</div>
              <span className="text-[11px] text-steel mt-1 block">
                {currentPoint.tp.toLocaleString()} of 7,278 anomalies
              </span>
            </div>

            <div className="p-4 rounded-sm border border-steel/15 bg-surface-recessed/50 font-numeric">
              <span className="text-xs text-steel block mb-1">Precision (PPV)</span>
              <div className="text-2xl font-semibold text-ink">{currentPoint.precision_percent.toFixed(2)}%</div>
              <span className="text-[11px] text-steel mt-1 block">
                {currentPoint.fp.toLocaleString()} false alarms
              </span>
            </div>

            <div className="p-4 rounded-sm border border-steel/15 bg-surface-recessed/50 font-numeric">
              <span className="text-xs text-steel block mb-1">False Alarm Rate</span>
              <div
                className={`text-2xl font-semibold ${
                  currentPoint.false_alarms_per_hour > 300 ? "text-alert" : "text-ink"
                }`}
              >
                {currentPoint.false_alarms_per_hour.toFixed(1)}{" "}
                <span className="text-xs font-normal text-steel">FP/hr</span>
              </div>
              <span className="text-[11px] text-steel mt-1 block">
                ~{(currentPoint.false_alarms_per_hour / 60).toFixed(1)} false alerts / min
              </span>
            </div>

            <div className="p-4 rounded-sm border border-steel/15 bg-surface-recessed/50 font-numeric">
              <span className="text-xs text-steel block mb-1">Bandwidth Reduction</span>
              <div className="text-2xl font-semibold text-signal">
                {currentPoint.bandwidth_reduction_percent.toFixed(2)}%
              </div>
              <span className="text-[11px] text-steel mt-1 block">
                {currentPoint.bytes_transmitted.toLocaleString()} bytes transmitted
              </span>
            </div>
          </div>

          {/* Contextual Insight Callout */}
          <div className="p-4 rounded-sm bg-surface-recessed border border-steel/15 text-xs text-ink/80 leading-relaxed">
            {threshold === 0.35 && modelType === "no_kd_int8" && (
              <div className="flex items-start gap-2.5">
                <CheckCircle2 className="h-4 w-4 text-signal shrink-0 mt-0.5" />
                <div>
                  <strong className="text-ink">Selected Operating Point (Decision #15):</strong> At τ = 0.35, the No-KD
                  INT8 model achieves <strong>13.79% recall</strong> (a 20.5× gain over 0.85) while keeping precision at{" "}
                  <strong>66.45%</strong> and false alarms to <strong>126.4 FP/hr</strong>, saving{" "}
                  <strong>99.04%</strong> of network bandwidth.
                </div>
              </div>
            )}

            {threshold === 0.85 && (
              <div className="flex items-start gap-2.5">
                <AlertTriangle className="h-4 w-4 text-alert shrink-0 mt-0.5" />
                <div>
                  <strong className="text-alert">Sensitivity Starvation Failure:</strong> At τ = 0.85, the model
                  triggered only 137 times, detecting just 49 anomalies (<strong>0.67% recall</strong>). While bandwidth
                  reduction was 99.91%, 99.33% of arrhythmias were missed entirely.
                </div>
              </div>
            )}

            {modelType === "kd_int8" && (
              <div className="flex items-start gap-2.5">
                <AlertTriangle className="h-4 w-4 text-alert shrink-0 mt-0.5" />
                <div>
                  <strong className="text-alert">The Knowledge Distillation PTQ Paradox:</strong> While KD showed higher
                  Float32 recall, full-integer INT8 quantization produced a severe false alarm explosion (+61.9% FPs,{" "}
                  <strong>{currentPoint.false_alarms_per_hour.toFixed(1)} FP/hr</strong>). At τ = 0.50, KD emits one false
                  alarm every 7.6 seconds, making it clinically unusable.
                </div>
              </div>
            )}

            {threshold !== 0.35 && threshold !== 0.85 && modelType === "no_kd_int8" && (
              <div className="flex items-start gap-2.5">
                <Radio className="h-4 w-4 text-steel shrink-0 mt-0.5" />
                <div>
                  At threshold τ = {threshold.toFixed(2)}, the pipeline triggers {currentPoint.active_alerts} times (
                  {currentPoint.active_alerts_percent.toFixed(2)}% duty cycle) across the 51,992 test windows. Network
                  payload reduction remains {currentPoint.bandwidth_reduction_percent.toFixed(2)}%, well above the 90.0%
                  NFR-6 target.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
