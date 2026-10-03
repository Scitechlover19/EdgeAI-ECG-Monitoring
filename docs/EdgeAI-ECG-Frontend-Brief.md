# Frontend / Website Design Brief — EdgeAI-ECG-Monitoring

**For:** Antigravity (or any agent building this)
**Project:** Resource-Constrained Edge-AI Pipeline for Real-Time Privacy-Preserving Patient Monitoring
**Repo:** https://github.com/Scitechlover19/EdgeAI-ECG-Monitoring
**Goal of this document:** build a real, professional, portfolio/showcase-quality website for this capstone
project — not a dashboard mockup, not a hackathon README-as-a-page. This needs to look like it was
designed by someone who does this for a living, using this project's own real subject matter and real
measured numbers as the content. No placeholder numbers, no invented metrics — everything numeric on
this site must trace back to a file in `reports/` in the repo.

---

## 0. Read this first — what "not vibe-coded" actually means

"Vibe-coded" has a specific, recognizable look. Before building anything, actively avoid these tells —
they're the fastest way a reviewer (or anyone who's seen a lot of AI-generated sites) clocks a site as
generated rather than designed:

1. **Cream background + terracotta/clay accent** (roughly `#F4F1EA` + `#D97757`-ish). This is the single
   most common AI-generated-site signature right now. Do not use it.
2. **Near-black background + one neon accent** (acid green or vermillion). The other most common one.
3. **The SaaS-card kit** — everything chopped into identical rounded cards, one border-radius applied
   uniformly regardless of hierarchy, the same soft grey box-shadow (`rgba(0,0,0,.1)`) under every card,
   gradient washes used as pure decoration.
4. **Template chrome that shows up regardless of subject matter:** tracked-out ALL-CAPS eyebrow labels
   above every heading; meta text joined with middle-dots ("Edge AI · TinyML · Privacy"); labels like
   "PIPELINE — Stage 01"; a monospace font slapped on small labels just for a "technical" feel (monospace
   is fine, but only when it's doing a real job — see §3); an arrow "→" appended to every button/link;
   numbered markers (01/02/03) on content that isn't actually a sequence.
5. **Fade-and-slide-up on every section, hover-lift on every card.** Scattered, uniform micro-animation
   everywhere is the generic default. One deliberate, orchestrated motion moment (see §5) beats twelve
   small ones.
6. **Accenting a single word in a headline** with italics/bold/color for emphasis. Don't.

If in doubt at any point: ask "would this exact choice show up if I generated ten different SaaS landing
pages with the same prompt?" If yes, make a different choice specific to *this* project's actual subject
matter (ECG waveforms, sleep/wake state transitions, edge hardware constraints, privacy).

---

## 1. Design plan (color, type, layout, principles)

### 1.1 Color — 6 named tokens, grounded in the product's actual SLEEP/ACTIVE dichotomy

This is not an arbitrary palette — it's built directly from the one thing that makes this project's
subject matter distinctive: a device that spends 97%+ of its life dormant (`SLEEP`) and briefly, rarely
wakes to transmit (`ACTIVE`). The palette should visually embody that rhythm rather than being generic
"tech blue" or "medical white."

| Token | Hex | Role |
|---|---|---|
| `--bg-base` (Mist Sage) | `#E3E7E0` | Page background. Light, desaturated green-grey — calm, dormant, NOT white, NOT cream. This is the "sleeping" color. |
| `--surface` (Paper Sage) | `#EDEFEA` | Card/panel surfaces, slightly lighter than base, for subtle depth without borders or shadows everywhere. |
| `--surface-recessed` (Moss Whisper) | `#D7DDD3` | Footer, code blocks, recessed/secondary panels — slightly darker than base. |
| `--ink` (Deep Pine Ink) | `#16241D` | Primary text. Near-black but green-tinted, not pure `#000` or `#111` — softer, warmer under the sage backdrop. |
| `--signal` (Signal Jade) | `#2F6B52` | Primary brand/action color. A desaturated, sophisticated take on classic oscilloscope-green — used for primary CTAs, links, and the visual "normal rhythm / SLEEP state" indicator. This is the calm color. |
| `--alert` (Ember Clay) | `#BF6141` | Reserved *exclusively* for anomaly/`ACTIVE`-state moments — the wake-transmit event, alert badges, the one moment of visual tension on the page. Use it sparingly and only where the product itself would "wake up." Never use it as a generic accent for unrelated UI. |
| `--accent-secondary` (Slate Steel) | `#4C6570` | Cool secondary accent for technical/data elements: chart secondary series, borders, metadata text, hairline dividers. |

Rules:
- No pure white (`#FFFFFF`) and no pure black anywhere on the page, including form inputs and code blocks.
- `--alert` (Ember Clay) is the one color allowed to feel like tension. If it starts appearing in more
  than ~2 places outside an actual anomaly/alert context, that's a sign it's being used decoratively —
  pull it back.
- Dark mode (if built) should not simply invert this — it should use a genuinely dark, desaturated
  pine/forest base (not navy, not pure black) with the same Jade/Clay accent logic preserved.

### 1.2 Typography

One sans family for UI/body + headlines, one monospace used *only* for live numeric readouts (not as a
decorative label font). Do not default to Inter — pick something with more presence:

- **Primary (headlines + body):** `General Sans` or `Neue Montreal` (both are real, licensable/free-tier
  variable fonts with genuine character — geometric but humanist, not a generic system-font look). Use
  two weights max: a heavier weight for headlines, regular for body.
- **Numeric/technical readout only:** `IBM Plex Mono` or `JetBrains Mono` — reserved strictly for actual
  live numbers: confidence scores, byte counts, latency figures, the threshold value. If a monospace
  span isn't displaying a real measured number, it shouldn't be monospace.
- Line length under ~80 characters for body copy. No forced letter-spacing/tracking on body text.
- No ALL-CAPS section eyebrows. If a section needs a label, use sentence case, regular weight, and let
  spacing/hierarchy do the work instead of caps.

### 1.3 Layout concept

The hero is not a generic headline-plus-gradient. This project's single most characteristic visual moment
is the SLEEP → ACTIVE transition itself — that IS the product's thesis. Build the hero around it:

```
┌──────────────────────────────────────────────────────────┐
│  [small wordmark, top-left]                [nav, top-right]│
│                                                              │
│         A flat, calm ECG line drifts quietly across          │
│         the hero (Rive state-machine, see §5.2) — mostly     │
│         idle in Signal Jade, occasionally spiking and        │
│         the whole hero briefly shifts toward Ember Clay      │
│         with a small radio icon pulsing awake, then           │
│         settles back to sleep. This loop IS the headline.    │
│                                                              │
│         Headline (short, plain, no jargon):                  │
│         "Most of the time, it says nothing."                 │
│         Subhead: one real sentence about what the system      │
│         actually does (from the real abstract, reworded       │
│         to plain language, not marketing voice)               │
│                                                              │
│         [View the system →... no — just: "See how it works"] │
└──────────────────────────────────────────────────────────┘
```

Below the hero, left-aligned single-column reading rhythm for narrative sections, breaking to a wider
layout only for the pipeline diagram and the live metrics panel. Suggested section order:

1. **Hero** — the SLEEP/ACTIVE Rive moment (above).
2. **The problem** — short, plain-language version of the real problem statement (cloud streaming =
   privacy exposure + battery drain). No stock icons; if you need a visual here, a simple inline SVG
   diagram of "sensor → constant stream → cloud" vs "sensor → mostly silent → occasional alert" works
   better than icon soup.
3. **How it works** — the real 5-stage pipeline (Ingestion → DSP → TinyML → State Scheduler → Secure
   Telemetry), shown as an actual interactive diagram people can step through, not a static image. This
   is a good spot for a restrained Spline 3D object (see §5.3) — e.g. an abstract representation of the
   MCU/chip — but only if it earns its place; a flat, well-designed SVG diagram is genuinely fine here too
   if 3D doesn't add clarity.
4. **The numbers** — a real metrics panel pulling directly from `reports/*.md` (see §4). This is where
   the monospace numeric type gets used. Include the honest trade-off, not just the flattering number —
   show bandwidth reduction (99.04%) *next to* recall (13.79%), not one without the other.
5. **Why the threshold matters** — a small interactive element letting a visitor drag/select a threshold
   value and see recall/precision/bandwidth-reduction update live, using the real `threshold_sweep.md`
   data baked in as a static dataset (not live inference in the browser). This is a genuinely good use of
   real project data as an interactive teaching moment, not decoration.
6. **Architecture / tech stack** — plain list of what was actually used (1D-CNN, knowledge distillation,
   INT8 PTQ, AES-GCM, MIT-BIH), no logo-soup unless the logos are real and relevant.
7. **Footer** — links to the repo, the write-up/report, contact.

---

## 2. Content rules — non-negotiable

- Every number on the site must come from a file in `reports/` in the repo at the time of writing the
  page (`bandwidth_report.md`, `resource_report.md`, `quantization_report.md`, `kd_ablation.md`,
  `threshold_sweep.md`). Do not let the page designer round up, cherry-pick without context, or state a
  number without its honest counterpart (e.g. never show "99% bandwidth reduction" without recall nearby).
- Current canonical operating point to reflect: **No-KD INT8 model, threshold 0.35, 13.79% recall, 66.45%
  precision, 99.04% bandwidth reduction, 10.96 KB flash, ~15 KB SRAM, 126.4 false alerts/hour.** If these
  numbers change in the repo before the site ships, update the site to match — don't let stale figures
  sit on a live page.
- State the recall limitation plainly somewhere visible, not buried in a footnote. This is a strength of
  the story (honesty), not a weakness to hide.
- This is a **software-engineering proof-of-concept**, not a diagnostic medical device — say so
  explicitly somewhere on the page (the repo's own README already states this; carry it forward).

---

## 3. Reference points (since I can't browse Pinterest directly)

Rather than a Pinterest mood-board, use these as concrete, real examples of restrained, light-palette,
professional sites with considered motion — study their *restraint*, not their literal colors:

- **Linear** (linear.app) — light mode: extremely disciplined type hierarchy, motion used only where it
  clarifies state change.
- **Stripe** (stripe.com) — how real product data/diagrams are presented without icon-soup.
- **Resend** (resend.com) — small, technical product, light UI, monospace used correctly (for actual code/
  data, not decoration).
- **Cuberto-designed sites** (cuberto.com portfolio) — this studio popularized a lot of the
  Lenis-smooth-scroll + orchestrated-hero-motion language; look at their portfolio for restraint-with-one-
  bold-moment pacing, not to copy any single site.
- **Arc browser** (arc.net) — an example of a calm, light, non-white palette done well.

Do not copy layouts from these directly — use them to calibrate "how much restraint is enough."

---

## 4. Data wiring

Pull real numbers from the repo at build time rather than hand-typing them into components:

- Parse `reports/threshold_sweep.md`'s Table C (No-KD INT8) into a small static JSON at build time (a
  short script reading the markdown table into `data/threshold-sweep.json`) so the interactive threshold
  section in §1.3 point 5 is backed by real data, not a hardcoded array a developer typed by hand and
  will forget to update.
- Same for the headline metrics panel — read from `reports/bandwidth_report.md`,
  `reports/resource_report.md`, `reports/quantization_report.md` rather than hardcoding.
- Add a small `scripts/sync-metrics.ts` (or `.py`) that regenerates this JSON whenever the reports change,
  and document running it before each deploy.

---

## 5. Motion & interaction — Lenis, Rive, Spline

Use these three deliberately, each for what it's actually good at. Do not stack all three into the same
section "because we have them."

### 5.1 Lenis — smooth scroll (whole site, subtle)

Current package is simply `lenis` (the old `@studio-freight/lenis` name was retired/renamed under
`darkroom.engineering`; use the current one).

```bash
npm i lenis
```

```ts
import Lenis from 'lenis'

const lenis = new Lenis({
  lerp: 0.1,
  smoothWheel: true,
})

function raf(time: number) {
  lenis.raf(time)
  requestAnimationFrame(raf)
}
requestAnimationFrame(raf)
```

Required CSS:
```css
html.lenis, html.lenis body { height: auto; }
.lenis.lenis-smooth { scroll-behavior: auto !important; }
.lenis.lenis-smooth [data-lenis-prevent] { overscroll-behavior: contain; }
.lenis.lenis-stopped { overflow: hidden; }
.lenis.lenis-smooth iframe { pointer-events: none; }
```

If using React, `lenis/react` (or `@studio-freight/react-lenis` on older setups) provides a `<ReactLenis>`
wrapper — check current docs at install time since this changes.

Use it for: general page-scroll feel only. Keep `lerp` modest (~0.08–0.12) — Lenis should feel like a
slightly weighted scroll, not a slow-motion scroll-jacked experience. Respect `prefers-reduced-motion`:
disable smoothing entirely (`lenis.destroy()` or don't instantiate) when that media query is set.

### 5.2 Rive — the hero SLEEP/ACTIVE state machine (the one signature moment)

```bash
npm i @rive-app/react-canvas
```
(or the newer unified `rive-react` package — check current npm at install time, both exist.)

```tsx
import { useRive, useStateMachineInput } from '@rive-app/react-canvas'

function EcgHero() {
  const { RiveComponent, rive } = useRive({
    src: '/animations/ecg-state.riv',
    stateMachines: 'SleepWake',
    autoplay: true,
  })
  return <RiveComponent style={{ width: '100%', height: '420px' }} />
}
```

Build the actual `.riv` file in the Rive editor as a state machine with two states — `sleep` (calm,
looping flat-ish trace in Signal Jade, very slow) and `active` (a sharper spike, brief shift toward Ember
Clay, small radio-wave pulse icon) — with a timed or triggered transition so it demonstrates the real
product behavior: mostly sleeping, rarely and briefly waking. This is the one place on the page allowed to
be the "bold, memorable" element per the restraint principle — everything else on the page should be
comparatively quiet so this lands.

Don't use Rive for generic decorative icon animations elsewhere on the page (hover states, small nav
icons, etc.) — one deliberate use, not scattered.

### 5.3 Spline — optional, one restrained 3D moment (architecture section only)

```bash
npm i @splinetool/react-spline @splinetool/runtime
```

```tsx
import Spline from '@splinetool/react-spline'

function ChipVisual() {
  return (
    <div style={{ width: '100%', height: '500px' }}>
      <Spline scene="https://prod.spline.design/YOUR-SCENE-ID/scene.splinecode" />
    </div>
  )
}
```

Use only in the "How it works" architecture section (§1.3 point 3), as an abstract, low-poly
representation of the MCU/edge device — not a literal ECG monitor render, not a generic "AI brain" cliché.
If it doesn't clearly add clarity over a well-designed static/SVG diagram, skip it — a flat diagram
executed well beats a 3D scene that doesn't earn its weight in load time or clarity. If used, self-host
the exported `.splinecode` file (download from Spline's export panel) to avoid CORS/latency from their
CDN, and lazy-load it below the fold so it never blocks the hero's first paint.

### 5.4 Motion budget, overall

One orchestrated hero moment (Rive) + smooth scroll feel (Lenis) + at most one restrained 3D moment
(Spline, optional) = the entire motion budget for this page. Everything else (card hovers, nav
transitions) should be simple, fast (150–200ms), and purely functional — confirming what changed, not
performing.

---

## 6. Recommended stack

- **Next.js** (React) — best support surface for all three libraries above, easy static export or
  Vercel deploy, good for the data-wiring step in §4 (can run the metrics-sync script at build time).
- **Tailwind CSS**, but with the 6 color tokens above defined as custom theme colors (`bg-base`,
  `surface`, `ink`, `signal`, `alert`, `accent-secondary`) rather than reaching for Tailwind's default
  palette — the default slate/zinc/blue scale is itself a common "looks templated" tell if used unedited.
- Deploy target: Vercel (simplest for Next.js) or any static host if using static export.

---

## 7. Build checklist for Antigravity

1. Scaffold Next.js + Tailwind, wire the 6 custom color tokens and the two type families into the theme
   config before writing any component.
2. Write `scripts/sync-metrics` to pull real numbers from `reports/*.md` in the repo into a `data/`
   JSON the site reads from — do this before building any UI that displays a number.
3. Build the Rive `.riv` SLEEP/ACTIVE state machine asset (this can be done in parallel/by a human in the
   Rive editor if Antigravity can't author `.riv` files directly — flag this as a dependency rather than
   faking it with a CSS animation that doesn't actually look like Rive's state-machine quality).
4. Build sections in the order in §1.3, plain HTML/CSS structure first, states/data wired second, motion
   (Lenis/Rive/Spline) added last — motion should be the final layer on top of a page that already reads
   well completely static.
5. Self-critique pass before calling it done: take screenshots at desktop and mobile widths, check against
   §0's anti-pattern list line by line, confirm `prefers-reduced-motion` is respected, confirm every
   number on the page traces to an actual file in `reports/`, confirm no pure white/black anywhere.
6. Confirm mobile responsiveness and keyboard focus visibility before considering this done.
