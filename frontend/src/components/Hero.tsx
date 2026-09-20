"use client";

import { useEffect, useRef, useState } from "react";
import { Radio, ShieldAlert, Cpu, Lock, ArrowDown } from "lucide-react";

export default function Hero() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [systemState, setSystemState] = useState<"SLEEP" | "ACTIVE">("SLEEP");
  const [anomalyConfidence, setAnomalyConfidence] = useState<number>(0.08);
  const [transmissionsCount, setTransmissionsCount] = useState<number>(1);
  const [lastAlertTime, setLastAlertTime] = useState<string>("00:04:12.4");

  // State machine loop for the oscilloscope waveform
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.offsetWidth * window.devicePixelRatio);
    let height = (canvas.height = canvas.offsetHeight * window.devicePixelRatio);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      height = canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    };
    window.addEventListener("resize", handleResize);

    // Waveform simulation buffers
    const pointsCount = 360;
    const buffer = new Float32Array(pointsCount).fill(0);
    let step = 0;
    let isAnomalous = false;
    let anomalyCooldown = 0;

    // Synthetic PQRST normal template vs Ventricular Ectopic spike
    const getSample = (t: number, anomaly: boolean): number => {
      const phase = (t % 100) / 100;
      if (!anomaly) {
        // Normal P-Q-R-S-T rhythm
        if (phase > 0.15 && phase < 0.25) return 0.15 * Math.sin(((phase - 0.15) / 0.1) * Math.PI); // P
        if (phase > 0.32 && phase < 0.35) return -0.15; // Q
        if (phase >= 0.35 && phase < 0.40) return 0.95; // R peak
        if (phase >= 0.40 && phase < 0.43) return -0.35; // S
        if (phase > 0.55 && phase < 0.70) return 0.25 * Math.sin(((phase - 0.55) / 0.15) * Math.PI); // T
        return 0.02 * (Math.random() - 0.5); // baseline noise
      } else {
        // Broad, inverted polymorphic ectopic arrhythmia complex
        if (phase > 0.25 && phase < 0.42) return 1.4 * Math.sin(((phase - 0.25) / 0.17) * Math.PI);
        if (phase >= 0.42 && phase < 0.58) return -0.85 * Math.sin(((phase - 0.42) / 0.16) * Math.PI);
        return 0.04 * (Math.random() - 0.5);
      }
    };

    // Cycle timer: 10s period (8.5s sleep, 1.5s active)
    const stateInterval = setInterval(() => {
      isAnomalous = true;
      anomalyCooldown = 75; // frames of anomaly spike
      setSystemState("ACTIVE");
      setAnomalyConfidence(0.89);
      setTransmissionsCount((prev) => prev + 1);
      const now = new Date();
      setLastAlertTime(
        `${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}.${String(Math.floor(now.getMilliseconds() / 100))}`
      );

      // Reset back to sleep after brief transmission window
      setTimeout(() => {
        isAnomalous = false;
        setSystemState("SLEEP");
        setAnomalyConfidence(0.08);
      }, 1800);
    }, 8500);

    const render = () => {
      step++;
      const isCurrentlySpiking = isAnomalous && anomalyCooldown > 0;
      if (anomalyCooldown > 0) anomalyCooldown--;

      // Shift buffer
      for (let i = 0; i < pointsCount - 1; i++) {
        buffer[i] = buffer[i + 1];
      }
      buffer[pointsCount - 1] = getSample(step, isCurrentlySpiking);

      // Clear
      ctx.clearRect(0, 0, width, height);

      // Background subtle grid lines
      ctx.strokeStyle = "rgba(156, 122, 60, 0.12)";
      ctx.lineWidth = 1;
      const gridSize = 24 * window.devicePixelRatio;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Draw ECG Trace
      ctx.beginPath();
      const centerY = height * 0.5;
      const scaleY = height * 0.38;

      for (let i = 0; i < pointsCount; i++) {
        const x = (i / (pointsCount - 1)) * width;
        const y = centerY - buffer[i] * scaleY;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      // Dynamic color based on SLEEP vs ACTIVE
      if (systemState === "ACTIVE") {
        ctx.strokeStyle = "#7A2E22"; // Oxblood
        ctx.lineWidth = 2.8 * window.devicePixelRatio;
        ctx.shadowColor = "rgba(122, 46, 34, 0.4)";
        ctx.shadowBlur = 8 * window.devicePixelRatio;
      } else {
        ctx.strokeStyle = "#1F3347"; // Ink Navy
        ctx.lineWidth = 2.0 * window.devicePixelRatio;
        ctx.shadowColor = "rgba(31, 51, 71, 0.25)";
        ctx.shadowBlur = 4 * window.devicePixelRatio;
      }

      ctx.stroke();
      ctx.shadowBlur = 0;

      // Leading scanhead indicator
      const headX = width;
      const headY = centerY - buffer[pointsCount - 1] * scaleY;
      ctx.fillStyle = systemState === "ACTIVE" ? "#7A2E22" : "#1F3347";
      ctx.beginPath();
      ctx.arc(headX - 3, headY, 3.5 * window.devicePixelRatio, 0, Math.PI * 2);
      ctx.fill();

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
      clearInterval(stateInterval);
    };
  }, [systemState]);

  return (
    <section className="relative overflow-hidden pt-12 pb-20 md:pt-20 md:pb-28 border-b border-steel/15">
      <div className="mx-auto max-w-6xl px-6">
        {/* Top Eyebrow / Hardware Constraint Pills */}
        <div className="flex flex-wrap items-center gap-2.5 text-xs text-steel mb-6">
          <span className="inline-flex items-center gap-1.5 rounded bg-surface px-2.5 py-1 border border-steel/20 font-numeric">
            <Cpu className="h-3.5 w-3.5 text-steel" />
            256 KB SRAM Limit
          </span>
          <span className="inline-flex items-center gap-1.5 rounded bg-surface px-2.5 py-1 border border-steel/20 font-numeric">
            <Lock className="h-3.5 w-3.5 text-signal" />
            Zero Raw Waveform Transmission
          </span>
          <span className="inline-flex items-center gap-1.5 rounded bg-surface px-2.5 py-1 border border-steel/20 font-numeric">
            MIT-BIH Arrhythmia Dataset
          </span>
        </div>

        {/* Main Headline & Context */}
        <div className="max-w-3xl mb-10">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-semibold tracking-tight text-ink leading-[1.08] mb-6">
            Most of the time, it says nothing.
          </h1>
          <p className="text-lg md:text-xl text-ink/75 leading-relaxed">
            An edge-AI cardiac monitoring pipeline that filters 99% of normal rhythms directly on-device,
            waking its radio transceiver only when verified arrhythmias occur — solving continuous telemetry power
            drain and eliminating biological waveform privacy exposure.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-4">
            <a
              href="#pipeline"
              className="inline-flex items-center justify-center rounded-sm bg-signal px-5 py-2.5 text-sm font-medium text-[#EDEFEA] shadow-xs transition-all hover:bg-signal/90"
            >
              See how it works
            </a>
            <a
              href="#metrics"
              className="inline-flex items-center justify-center rounded-sm border border-steel/30 bg-surface px-5 py-2.5 text-sm font-medium text-ink transition-all hover:border-steel/50 hover:bg-surface-recessed"
            >
              View measured results
            </a>
          </div>
        </div>

        {/* Oscilloscope State Machine Container */}
        <div className="relative rounded-sm border border-steel/20 bg-surface p-4 sm:p-6 shadow-xs">
          {/* Header Bar of Oscilloscope */}
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-steel/15 pb-4 mb-4">
            <div className="flex items-center gap-3">
              <span className="text-xs uppercase tracking-wider text-steel font-medium">Edge State Engine</span>
              <div
                className={`inline-flex items-center gap-2 rounded-full px-3 py-0.5 text-xs font-numeric font-medium transition-colors ${
                  systemState === "ACTIVE"
                    ? "bg-alert/15 text-alert border border-alert/30"
                    : "bg-signal/15 text-signal border border-signal/25"
                }`}
              >
                <span
                  className={`h-2 w-2 rounded-full transition-all ${
                    systemState === "ACTIVE" ? "bg-alert animate-ping" : "bg-signal"
                  }`}
                />
                {systemState === "ACTIVE" ? "STATE: ACTIVE (Wake & Transmit)" : "STATE: SLEEP (Radio Dormant)"}
              </div>
            </div>

            <div className="flex items-center gap-4 text-xs font-numeric text-steel">
              <div>
                Threshold <span className="text-ink font-medium">τ = 0.35</span>
              </div>
              <div>
                Score{" "}
                <span className={`font-semibold ${systemState === "ACTIVE" ? "text-alert" : "text-signal"}`}>
                  {anomalyConfidence.toFixed(2)}
                </span>
              </div>
              <div className="hidden sm:block">
                Radio{" "}
                <span className={`font-medium ${systemState === "ACTIVE" ? "text-alert" : "text-steel"}`}>
                  {systemState === "ACTIVE" ? "ON (132B Packet)" : "OFF (0B/s)"}
                </span>
              </div>
            </div>
          </div>

          {/* Canvas Waveform Viewport */}
          <div className="relative h-44 sm:h-56 w-full rounded-sm bg-surface-recessed/60 overflow-hidden border border-steel/10">
            <canvas ref={canvasRef} className="h-full w-full block" />

            {/* In-Canvas State Overlay */}
            <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded bg-surface/85 px-2.5 py-1 text-xs border border-steel/15 backdrop-blur-xs font-numeric text-ink/80">
              <Radio
                className={`h-3.5 w-3.5 transition-colors ${
                  systemState === "ACTIVE" ? "text-alert animate-pulse" : "text-steel/60"
                }`}
              />
              <span>
                {systemState === "ACTIVE"
                  ? "Emitting AES-GCM encrypted alert metadata..."
                  : "Normal sinus rhythm — RF radio powered down"}
              </span>
            </div>

            <div className="absolute top-3 right-3 text-[11px] font-numeric text-steel/75 bg-surface/75 px-2 py-0.5 rounded border border-steel/10">
              200-sample sliding window · 360 Hz
            </div>
          </div>

          {/* Bottom Live Schema Banner */}
          <div className="mt-4 pt-3 border-t border-steel/10 flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2 text-steel">
              <ShieldAlert className="h-3.5 w-3.5 text-signal" />
              <span>Transmitted Schema:</span>
              <code className="font-numeric text-ink/90 bg-surface-recessed px-1.5 py-0.5 rounded text-[11px]">
                {`{"session_id", "timestamp_sec", "anomaly_id", "confidence_score"}`}
              </code>
            </div>
            <div className="text-steel font-numeric text-[11px]">
              Active transmissions: <span className="text-ink font-semibold">{transmissionsCount}</span> · Last event:{" "}
              <span className="text-ink">{lastAlertTime}</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
