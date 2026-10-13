"""Causal QRS detector and cardiac rhythm feature extractor for streaming Edge-AI.

Implements an embedded-friendly, strictly causal Pan-Tompkins QRS peak detector
and extracts causal pre-beat RR timing context:
1. pre_rr_norm: Preceding RR interval in seconds (normalized).
2. rr_ratio: Ratio of current RR to local rolling median of previous 8 intervals.
3. rr_variability: Local coefficient of variation (std / mean) across previous 8 intervals.

Strictly causal: zero lookahead to future beats or cardiac cycles.
Governed by PRD.md (FR-3, FR-4), Architecture.md §5.2, and embedded Cortex-M4 constraints.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class DetectedBeat:
    """Represents a causally detected cardiac beat and its rhythm context."""
    sample_idx: int
    timestamp_s: float
    rr_pre_s: float
    rr_ratio: float
    rr_var: float


class CausalQRSDetector:
    """Causal QRS peak detector and RR interval rhythm feature extractor."""

    def __init__(
        self,
        sample_rate: float = 360.0,
        mwi_window_ms: float = 100.0,
        refractory_ms: float = 200.0,
        history_len: int = 8,
    ) -> None:
        """Initialize CausalQRSDetector.

        Args:
            sample_rate: Signal sampling rate in Hz (default 360.0 Hz).
            mwi_window_ms: Moving-window integration window in ms (default 100 ms).
            refractory_ms: Minimum refractory period between consecutive beats in ms (default 200 ms).
            history_len: Number of previous RR intervals tracked for running median/variability (default 8).
        """
        self.fs = float(sample_rate)
        self.mwi_len = max(3, int((mwi_window_ms / 1000.0) * self.fs))
        self.refractory_samples = max(10, int((refractory_ms / 1000.0) * self.fs))
        self.history_len = int(history_len)

        # 5-point derivative filter coefficients: [-2, -1, 0, 1, 2] / 8.0 (causal FIR)
        self.diff_kernel = np.array([2.0, 1.0, 0.0, -1.0, -2.0], dtype=np.float32) / 8.0
        self.mwi_kernel = np.ones(self.mwi_len, dtype=np.float32) / float(self.mwi_len)

    def detect_peaks(self, filtered_signal: np.ndarray) -> List[DetectedBeat]:
        """Detect R-peaks and compute causal RR features across a filtered continuous signal.

        Operates with strictly zero lookahead:
        - Derivative and MWI are causal FIR filters (lfilter).
        - Local peak is confirmed at step i by checking i-1 against i-2 and i (1-sample delay = 2.78 ms).
        - R-peak refinement searches strictly backward in the past buffer [cand_idx - mwi_len, cand_idx].
        - Adaptive dual-threshold searchback recovers ectopic/small beats if elapsed time > 1.66 * RR_avg.

        Args:
            filtered_signal: Bandpass-filtered 1D float32 numpy array.

        Returns:
            List of DetectedBeat instances.
        """
        if len(filtered_signal) < self.refractory_samples:
            return []

        from scipy.signal import lfilter

        # Step 1: Strictly causal 5-point derivative FIR filter
        diff = lfilter(self.diff_kernel, [1.0], filtered_signal)

        # Step 2: Non-linear squaring operator
        sq = diff ** 2

        # Step 3: Strictly causal moving-window integration (MWI) FIR filter
        mwa = lfilter(self.mwi_kernel, [1.0], sq)

        # Step 4: Adaptive dual-threshold state machine with searchback
        init_samples = min(len(mwa), int(2.0 * self.fs))
        peaki = float(np.mean(mwa[:init_samples]) * 2.0) if init_samples > 0 else 1.0
        noisei = float(np.mean(mwa[:init_samples]) * 0.5) if init_samples > 0 else 0.1

        detected_beats: List[DetectedBeat] = []
        recent_rr_samples: List[float] = [300.0]
        rr_avg_samples: float = 300.0
        sb_limit_samples: int = int(1.66 * 300.0)
        last_r_idx: Optional[int] = None
        last_search_idx = -self.refractory_samples

        best_cand_idx = -1
        best_cand_val = -1.0

        for i in range(2, len(mwa)):
            cand_idx = i - 1
            # Check local peak in MWI at cand_idx (1-sample causal delay)
            if mwa[cand_idx] > mwa[i - 2] and mwa[cand_idx] >= mwa[i]:
                cand_val = float(mwa[cand_idx])
                th1 = noisei + 0.25 * (peaki - noisei)
                th2 = 0.5 * th1

                if cand_val > th1 and (cand_idx - last_search_idx) > self.refractory_samples:
                    # Primary detection: search backward in clean signal past buffer
                    search_start = max(0, cand_idx - self.mwi_len)
                    r_idx = search_start + int(np.argmax(np.abs(filtered_signal[search_start:cand_idx + 1])))

                    rr_samples = float(r_idx - last_r_idx) if last_r_idx is not None else 300.0
                    rr_s = rr_samples / self.fs

                    if len(recent_rr_samples) >= 3:
                        med_rr = float(np.median(recent_rr_samples[-self.history_len:]))
                        rr_ratio = float(np.clip(rr_samples / med_rr, 0.2, 2.5))
                        mean_rr = float(np.mean(recent_rr_samples[-self.history_len:]))
                        std_rr = float(np.std(recent_rr_samples[-self.history_len:]))
                        rr_var = float(np.clip(std_rr / (mean_rr + 1e-6), 0.0, 1.0))
                    else:
                        rr_ratio = 1.0
                        rr_var = 0.0

                    recent_rr_samples.append(rr_samples)
                    if len(recent_rr_samples) > 32:
                        recent_rr_samples.pop(0)

                    rr_avg_samples = float(np.mean(recent_rr_samples[-8:]))
                    sb_limit_samples = int(1.66 * rr_avg_samples)

                    detected_beats.append(
                        DetectedBeat(
                            sample_idx=r_idx,
                            timestamp_s=float(r_idx / self.fs),
                            rr_pre_s=float(np.clip(rr_s, 0.2, 2.5)),
                            rr_ratio=rr_ratio,
                            rr_var=rr_var,
                        )
                    )

                    last_r_idx = r_idx
                    last_search_idx = cand_idx
                    peaki = 0.125 * cand_val + 0.875 * peaki
                    best_cand_idx = -1
                    best_cand_val = -1.0
                elif cand_val > th2 and (cand_idx - last_search_idx) > self.refractory_samples:
                    if cand_val > best_cand_val:
                        best_cand_idx = cand_idx
                        best_cand_val = cand_val
                else:
                    noisei = 0.125 * cand_val + 0.875 * noisei

            # Searchback check: if elapsed time > 1.66 * rr_avg and candidate available
            if (i - last_search_idx) > sb_limit_samples and best_cand_idx != -1:
                if (best_cand_idx - last_search_idx) > self.refractory_samples:
                    search_start = max(0, best_cand_idx - self.mwi_len)
                    r_idx = search_start + int(np.argmax(np.abs(filtered_signal[search_start:best_cand_idx + 1])))

                    rr_samples = float(r_idx - last_r_idx) if last_r_idx is not None else 300.0
                    rr_s = rr_samples / self.fs

                    if len(recent_rr_samples) >= 3:
                        med_rr = float(np.median(recent_rr_samples[-self.history_len:]))
                        rr_ratio = float(np.clip(rr_samples / med_rr, 0.2, 2.5))
                        mean_rr = float(np.mean(recent_rr_samples[-self.history_len:]))
                        std_rr = float(np.std(recent_rr_samples[-self.history_len:]))
                        rr_var = float(np.clip(std_rr / (mean_rr + 1e-6), 0.0, 1.0))
                    else:
                        rr_ratio = 1.0
                        rr_var = 0.0

                    recent_rr_samples.append(rr_samples)
                    if len(recent_rr_samples) > 32:
                        recent_rr_samples.pop(0)

                    rr_avg_samples = float(np.mean(recent_rr_samples[-8:]))
                    sb_limit_samples = int(1.66 * rr_avg_samples)

                    detected_beats.append(
                        DetectedBeat(
                            sample_idx=r_idx,
                            timestamp_s=float(r_idx / self.fs),
                            rr_pre_s=float(np.clip(rr_s, 0.2, 2.5)),
                            rr_ratio=rr_ratio,
                            rr_var=rr_var,
                        )
                    )

                    last_r_idx = r_idx
                    last_search_idx = best_cand_idx
                    peaki = 0.25 * best_cand_val + 0.75 * peaki

                best_cand_idx = -1
                best_cand_val = -1.0

        # Sort chronologically and filter any refractory duplicates
        detected_beats.sort(key=lambda b: b.sample_idx)
        cleaned_beats: List[DetectedBeat] = []
        for b in detected_beats:
            if not cleaned_beats or (b.sample_idx - cleaned_beats[-1].sample_idx) >= self.refractory_samples:
                cleaned_beats.append(b)

        return cleaned_beats


    def extract_beat_segments(
        self,
        clean_signal: np.ndarray,
        pre_samples: int = 72,
        post_samples: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, List[DetectedBeat]]:
        """Extract beat-centred 200-sample waveform windows and causal RR vectors.

        Timing Note: A beat segment [R - pre_samples, R + post_samples] is physically
        available only once R + post_samples arrives at the receiver.
        For pre=72, post=128 at 360 Hz:
        Algorithmic window delay = 128 / 360 = 355.6 ms.
        Inference compute time on Cortex-M4 = ~1.8 ms.
        Total response latency from R-peak = 357.4 ms.
        """
        beats = self.detect_peaks(clean_signal)
        window_size = pre_samples + post_samples

        raw_windows = []
        rr_feats = []
        valid_beats = []

        for b in beats:
            start = b.sample_idx - pre_samples
            end = b.sample_idx + post_samples
            if start >= 0 and end <= len(clean_signal):
                raw_windows.append(clean_signal[start:end])
                rr_feats.append([b.rr_pre_s, b.rr_ratio, b.rr_var])
                valid_beats.append(b)

        if not raw_windows:
            return (
                np.zeros((0, window_size), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                [],
            )

        win_arr = np.array(raw_windows, dtype=np.float32)
        means = np.mean(win_arr, axis=1, keepdims=True)
        stds = np.std(win_arr, axis=1, keepdims=True)
        stds = np.where(stds < 1e-6, 1.0, stds)
        norm_windows = ((win_arr - means) / stds).astype(np.float32)

        return norm_windows, np.array(rr_feats, dtype=np.float32), valid_beats
