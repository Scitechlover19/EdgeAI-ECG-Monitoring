"use client";

import Link from "next/link";
import { Activity, ShieldCheck } from "lucide-react";
import { GithubIcon } from "./icons/GithubIcon";

export default function Navigation() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-steel/15 bg-canvas/90 backdrop-blur-md transition-colors">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        {/* Brand Wordmark */}
        <Link href="/" className="group flex items-center gap-2.5">
          <div className="flex h-7 w-7 items-center justify-center rounded-sm bg-surface border border-steel/20 shadow-xs transition-colors group-hover:border-signal/40">
            <Activity className="h-4 w-4 text-signal" />
          </div>
          <span className="text-base font-semibold tracking-tight text-ink">
            EdgeAI<span className="text-signal font-normal">.ecg</span>
          </span>
          <span className="hidden sm:inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs text-steel bg-surface border border-steel/15 font-numeric">
            <ShieldCheck className="h-3 w-3 text-signal" />
            Decision #15 Deployed
          </span>
        </Link>

        {/* Section Navigation */}
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-ink/75">
          <a href="#problem" className="transition-colors hover:text-ink">
            The Problem
          </a>
          <a href="#pipeline" className="transition-colors hover:text-ink">
            How It Works
          </a>
          <a href="#metrics" className="transition-colors hover:text-ink">
            Verified Metrics
          </a>
          <a href="#threshold" className="transition-colors hover:text-ink">
            Threshold Sweep
          </a>
          <a href="#architecture" className="transition-colors hover:text-ink">
            Architecture
          </a>
        </nav>

        {/* Right CTA / GitHub link */}
        <div className="flex items-center gap-3">
          <a
            href="https://github.com/Scitechlover19/EdgeAI-ECG-Monitoring"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-sm border border-steel/25 bg-surface px-3 py-1.5 text-xs font-medium text-ink transition-all hover:border-steel/40 hover:bg-surface-recessed"
          >
            <GithubIcon className="h-3.5 w-3.5 text-ink/70" />
            <span>Repository</span>
          </a>
        </div>
      </div>
    </header>
  );
}
