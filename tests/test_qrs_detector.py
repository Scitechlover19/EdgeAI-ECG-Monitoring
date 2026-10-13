"""Unit tests for CausalQRSDetector and causal rhythm features."""

import numpy as np
import pytest
from src.dsp.qrs_detector import CausalQRSDetector, DetectedBeat


def test_causal_qrs_detector_synthetic_regular():
    """Verify QRS detector on synthetic regular heartbeats (72 bpm, 300 sample spacing)."""
    fs = 360.0
    duration_s = 10.0
    t = np.arange(int(duration_s * fs)) / fs
    signal = np.zeros_like(t, dtype=np.float32)

    # Place sharp QRS pulses every 300 samples (~72 bpm)
    beat_samples = np.arange(300, len(signal) - 300, 300)
    for b in beat_samples:
        # Simple QRS spike: negative Q, sharp positive R, negative S
        signal[b - 4 : b] = -0.3
        signal[b] = 2.0
        signal[b + 1 : b + 5] = -0.4

    detector = CausalQRSDetector(sample_rate=fs)
    detected_beats = detector.detect_peaks(signal)

    assert len(detected_beats) >= len(beat_samples) - 1
    # Check that RR ratio settles to ~1.0 for steady rhythm
    steady_ratios = [b.rr_ratio for b in detected_beats[5:]]
    for r in steady_ratios:
        assert 0.95 <= r <= 1.05


def test_causal_qrs_detector_premature_beat():
    """Verify that a premature beat produces an RR ratio < 0.85 causally."""
    fs = 360.0
    duration_s = 10.0
    t = np.arange(int(duration_s * fs)) / fs
    signal = np.zeros_like(t, dtype=np.float32)

    # Regular beats at: 300, 600, 900, 1200, 1500, 1800, 2100
    # Premature beat injected at: 2320 (gap of 220 samples instead of 300, ratio ~ 0.73)
    beat_samples = [300, 600, 900, 1200, 1500, 1800, 2100, 2320, 2700]
    for b in beat_samples:
        signal[b - 4 : b] = -0.3
        signal[b] = 2.0
        signal[b + 1 : b + 5] = -0.4

    detector = CausalQRSDetector(sample_rate=fs)
    detected_beats = detector.detect_peaks(signal)

    # Find detected beat near sample 2320
    pac_beat = None
    for b in detected_beats:
        if abs(b.sample_idx - 2320) < 10:
            pac_beat = b
            break

    assert pac_beat is not None, "Premature beat was not detected!"
    assert pac_beat.rr_ratio < 0.85, f"Expected premature RR ratio < 0.85, got {pac_beat.rr_ratio:.3f}"


def test_extract_beat_segments_shapes():
    """Verify extract_beat_segments returns correct tensor shapes."""
    fs = 360.0
    signal = np.zeros(2000, dtype=np.float32)
    for b in [300, 600, 900, 1200, 1500, 1800]:
        signal[b] = 2.0

    detector = CausalQRSDetector(sample_rate=fs)
    waveforms, rr_feats, beats = detector.extract_beat_segments(signal, pre_samples=72, post_samples=128)

    assert waveforms.ndim == 2
    assert waveforms.shape[1] == 200
    assert rr_feats.ndim == 2
    assert rr_feats.shape[1] == 3
    assert len(waveforms) == len(rr_feats) == len(beats)
