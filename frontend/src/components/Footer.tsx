"use client";

import { Activity, FileText, Code2, AlertCircle } from "lucide-react";
import { GithubIcon } from "./icons/GithubIcon";

export default function Footer() {
  return (
    <footer className="border-t border-steel/15 bg-surface-recessed/60 py-16 text-xs text-ink/75">
      <div className="mx-auto max-w-6xl px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* Brand & Capstone Summary */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 mb-3">
              <Activity className="h-4 w-4 text-signal" />
              <span className="font-semibold text-ink text-sm">EdgeAI-ECG-Monitoring</span>
            </div>
            <p className="text-xs text-ink/70 max-w-md leading-relaxed mb-4">
              Resource-Constrained Edge-AI Pipeline for Real-Time Privacy-Preserving Patient Monitoring. Developed as an
              academic capstone project for SWE3004 at Vellore Institute of Technology (VIT).
            </p>
            <div className="text-[11px] font-numeric text-steel">
              Author: <span className="text-ink font-medium">Nancy Singh</span> (Reg: 22MIS0027)
            </div>
          </div>

          {/* Repository Links */}
          <div>
            <span className="font-semibold text-ink text-xs uppercase tracking-wider block mb-3">Project Assets</span>
            <ul className="space-y-2 font-numeric text-xs">
              <li>
                <a
                  href="https://github.com/Scitechlover19/EdgeAI-ECG-Monitoring"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-ink transition-colors inline-flex items-center gap-1.5"
                >
                  <GithubIcon className="h-3.5 w-3.5 text-steel" />
                  <span>GitHub Repository</span>
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/Scitechlover19/EdgeAI-ECG-Monitoring/blob/main/REVIEW2_DEMO.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-ink transition-colors inline-flex items-center gap-1.5"
                >
                  <FileText className="h-3.5 w-3.5 text-steel" />
                  <span>Review-2 Demo Script</span>
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/Scitechlover19/EdgeAI-ECG-Monitoring/blob/main/DECISIONS.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-ink transition-colors inline-flex items-center gap-1.5"
                >
                  <Code2 className="h-3.5 w-3.5 text-steel" />
                  <span>Decision #15 Rationale</span>
                </a>
              </li>
            </ul>
          </div>

          {/* Reports Index */}
          <div>
            <span className="font-semibold text-ink text-xs uppercase tracking-wider block mb-3">Committed Reports</span>
            <ul className="space-y-2 font-numeric text-xs text-ink/70">
              <li>bandwidth_report.md (99.04%)</li>
              <li>resource_report.md (0.16 ms, 10.96 KB)</li>
              <li>quantization_report.md (INT8 PTQ)</li>
              <li>threshold_sweep.md (Table C &amp; D)</li>
              <li>kd_ablation.md (Hyperparameter sweep)</li>
            </ul>
          </div>
        </div>

        {/* Regulatory & Proof-of-Concept Disclaimer */}
        <div className="pt-8 border-t border-steel/15 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-[11px] text-steel">
          <div className="flex items-start gap-2 max-w-2xl leading-relaxed">
            <AlertCircle className="h-3.5 w-3.5 text-steel shrink-0 mt-0.5" />
            <span>
              <strong>Proof-of-Concept Disclaimer:</strong> This project is an explicit software-engineering and
              embedded TinyML proof-of-concept. It does not possess FDA 510(k), CE mark, or regulatory clinical clearance,
              and is not intended for real-time patient diagnosis or clinical medical use.
            </span>
          </div>
          <div className="font-numeric text-steel whitespace-nowrap">
            MIT License · Python 3.10+ &amp; TensorFlow Lite
          </div>
        </div>
      </div>
    </footer>
  );
}
