"use client";

import { Shield, Zap, Database } from "lucide-react";

export default function ProblemStatement() {
  return (
    <section id="problem" className="py-20 border-b border-steel/15">
      <div className="mx-auto max-w-6xl px-6">
        {/* Section Header */}
        <div className="max-w-2xl mb-14">
          <span className="text-xs font-medium text-steel">The Remote Monitoring Trilemma</span>
          <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink mt-2 mb-4">
            Continuous raw streaming is an architectural dead end.
          </h2>
          <p className="text-base sm:text-lg text-ink/75 leading-relaxed">
            Conventional wearable cardiac monitors continuously stream raw biometric voltages to remote cloud
            servers. In practice, this creates three compounding engineering bottlenecks for ambulatory care.
          </p>
        </div>

        {/* Three Core Challenges Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs">
            <div className="flex items-center gap-2.5 mb-3 text-steel">
              <Zap className="h-4 w-4 text-alert" />
              <h3 className="text-base font-semibold text-ink">RF Power Depletion</h3>
            </div>
            <p className="text-sm text-ink/70 leading-relaxed">
              Wireless transceivers (BLE / Wi-Fi) consume up to 80% of total system energy during active transmission.
              Continuous unbuffered transmission drains wearable coin-cell batteries in hours rather than weeks.
            </p>
          </div>

          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs">
            <div className="flex items-center gap-2.5 mb-3 text-steel">
              <Shield className="h-4 w-4 text-alert" />
              <h3 className="text-base font-semibold text-ink">Biometric Privacy Exposure</h3>
            </div>
            <p className="text-sm text-ink/70 leading-relaxed">
              Continuous biological ECG waveforms broadcast identifiable patient physiological morphology over public RF
              channels, exposing personal health records to interception and violating data sovereignty standards.
            </p>
          </div>

          <div className="rounded-sm border border-steel/20 bg-surface p-6 shadow-xs">
            <div className="flex items-center gap-2.5 mb-3 text-steel">
              <Database className="h-4 w-4 text-alert" />
              <h3 className="text-base font-semibold text-ink">Hardware Constraint Mismatch</h3>
            </div>
            <p className="text-sm text-ink/70 leading-relaxed">
              Clinical deep learning models exceed hundreds of megabytes. Edge microcontrollers operate with strict limits
              of &le; 256 KB SRAM and &le; 1 MB Flash, requiring aggressive, disciplined compression.
            </p>
          </div>
        </div>

        {/* Architectural Comparison Visual */}
        <div className="rounded-sm border border-steel/20 bg-surface p-6 sm:p-8">
          <div className="max-w-xl mb-6">
            <h3 className="text-lg font-semibold text-ink">System Comparison on Held-Out MIT-BIH Test Data</h3>
            <p className="text-xs text-steel mt-1 font-numeric">
              Measured over 51,992 test windows across 4.01 hours of continuous monitoring
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">
            {/* Approach A: Conventional Raw Streaming */}
            <div className="rounded-sm border border-steel/25 bg-surface-recessed/50 p-5 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-semibold text-ink/80">Conventional Continuous Streaming</span>
                  <span className="text-[11px] font-numeric px-2 py-0.5 rounded bg-surface border border-steel/20 text-steel">
                    Radio Always ON
                  </span>
                </div>
                <p className="text-xs text-ink/70 mb-4">
                  Streams every 200-sample window continuously. Transmits all 51,992 windows regardless of cardiac status.
                </p>

                {/* Inline SVG Diagram */}
                <div className="bg-surface rounded-sm p-4 border border-steel/15 my-4">
                  <div className="flex items-center justify-between text-xs text-steel font-numeric">
                    <span className="flex items-center gap-1.5 font-medium text-ink">
                      <span className="h-2 w-2 rounded-full bg-steel"></span>
                      ECG Electrode
                    </span>
                    <span className="text-[11px] text-alert">Continuous 400B/window</span>
                    <span className="flex items-center gap-1.5 font-medium text-ink">
                      <span className="h-2 w-2 rounded-full bg-alert"></span>
                      Cloud Sink
                    </span>
                  </div>
                  <div className="w-full bg-surface-recessed h-1.5 rounded-full mt-2 overflow-hidden">
                    <div className="bg-alert/70 h-full w-full animate-pulse"></div>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-steel/15 space-y-1.5 font-numeric text-xs">
                <div className="flex justify-between text-ink/80">
                  <span>Network Payload:</span>
                  <span className="font-semibold text-ink">20,796,800 bytes (19.83 MB)</span>
                </div>
                <div className="flex justify-between text-ink/80">
                  <span>Radio Duty Cycle:</span>
                  <span className="font-semibold text-alert">100.0% Continuous</span>
                </div>
                <div className="flex justify-between text-ink/80">
                  <span>Waveform Privacy:</span>
                  <span className="text-alert font-medium">Unencrypted / Interceptable</span>
                </div>
              </div>
            </div>

            {/* Approach B: EdgeAI-ECG Anomaly-Driven Mode */}
            <div className="rounded-sm border border-signal/30 bg-surface-recessed/80 p-5 flex flex-col justify-between ring-1 ring-signal/15">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-semibold text-signal flex items-center gap-1.5">
                    <span className="h-2 w-2 rounded-full bg-signal"></span>
                    EdgeAI Anomaly-Driven Mode
                  </span>
                  <span className="text-[11px] font-numeric px-2 py-0.5 rounded bg-signal/15 border border-signal/30 text-signal font-semibold">
                    99.04% Reduction
                  </span>
                </div>
                <p className="text-xs text-ink/70 mb-4">
                  Normal sinus rhythm (50,481 windows) is filtered locally. The radio powers awake only for 1,511 anomaly
                  events, sending encrypted metadata.
                </p>

                {/* Inline SVG Diagram */}
                <div className="bg-surface rounded-sm p-4 border border-steel/15 my-4">
                  <div className="flex items-center justify-between text-xs text-steel font-numeric">
                    <span className="flex items-center gap-1.5 font-medium text-ink">
                      <span className="h-2 w-2 rounded-full bg-signal"></span>
                      On-Chip Inference
                    </span>
                    <span className="text-[11px] text-signal font-semibold">Radio Dormant 97.1%</span>
                    <span className="flex items-center gap-1.5 font-medium text-ink">
                      <span className="h-2 w-2 rounded-full bg-signal"></span>
                      Secure Sink
                    </span>
                  </div>
                  <div className="w-full bg-surface-recessed h-1.5 rounded-full mt-2 overflow-hidden">
                    <div className="bg-signal h-full w-[2.91%]"></div>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-steel/15 space-y-1.5 font-numeric text-xs">
                <div className="flex justify-between text-ink/80">
                  <span>Network Payload:</span>
                  <span className="font-semibold text-signal">198,791 bytes (0.19 MB)</span>
                </div>
                <div className="flex justify-between text-ink/80">
                  <span>Payload Saved:</span>
                  <span className="font-semibold text-signal">20,598,009 bytes (99.04%)</span>
                </div>
                <div className="flex justify-between text-ink/80">
                  <span>Raw Waveform Transmitted:</span>
                  <span className="font-semibold text-signal">0 bytes (Strictly Verified)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
