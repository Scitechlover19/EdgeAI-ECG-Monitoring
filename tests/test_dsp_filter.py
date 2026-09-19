"""Unit tests for DSPFilter module (P0-3)."""

import numpy as np
import pytest

from src.dsp.dsp_filter import DSPFilter, DSPFilterError
from src.pipeline.config import load_config


def test_dsp_filter_initialization() -> None:
    """Test DSPFilter parameter initialization and validation."""
    dsp = DSPFilter(low_cutoff=0.5, high_cutoff=45.0, filter_order=4, sample_rate=360.0)
    assert dsp.low_cutoff == 0.5
    assert dsp.high_cutoff == 45.0
    assert dsp.filter_order == 4
    assert dsp.sample_rate == 360.0


def test_dsp_filter_from_config() -> None:
    """Test DSPFilter loading parameters directly from PipelineConfig."""
    config = load_config()
    dsp = DSPFilter(dsp_config=config.dsp, sample_rate=config.ingestion.default_sample_rate)

    assert dsp.low_cutoff == config.dsp.low_cutoff
    assert dsp.high_cutoff == config.dsp.high_cutoff
    assert dsp.filter_order == config.dsp.filter_order


def test_dsp_filter_invalid_params() -> None:
    """Test DSPFilter raises DSPFilterError on invalid cutoff values."""
    with pytest.raises(DSPFilterError, match="low_cutoff .* must be positive"):
        DSPFilter(low_cutoff=0.0)

    with pytest.raises(DSPFilterError, match="high_cutoff .* must be less than Nyquist"):
        DSPFilter(high_cutoff=200.0, sample_rate=360.0)

    with pytest.raises(DSPFilterError, match="low_cutoff .* must be strictly less than high_cutoff"):
        DSPFilter(low_cutoff=50.0, high_cutoff=45.0)


def test_process_window_shape_and_dtype() -> None:
    """Test that output shape and dtype match raw input window."""
    dsp = DSPFilter()
    raw_window = np.random.randn(200).astype(np.float32)
    clean_window = dsp.process_window(raw_window)

    assert clean_window.shape == (200,)
    assert clean_window.dtype == np.float32
    assert not np.isnan(clean_window).any()
    assert not np.isinf(clean_window).any()


def test_snr_improvement_on_synthetic_signal() -> None:
    """Test filtering on a synthetic signal (10 Hz cardiac wave + 0.1 Hz baseline + 60 Hz powerline noise)."""
    fs = 360.0
    n_samples = 360
    t = np.arange(n_samples) / fs

    # Signals
    pure_cardiac = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz (in passband)
    baseline_wander = 2.0 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz (out of passband, low)
    powerline_noise = 0.5 * np.sin(2 * np.pi * 60.0 * t)  # 60 Hz (out of passband, high)

    noisy_signal = pure_cardiac + baseline_wander + powerline_noise

    dsp = DSPFilter(low_cutoff=0.5, high_cutoff=45.0, filter_order=4, sample_rate=fs)
    cleaned = dsp.process_window(noisy_signal)

    # FFT spectral verification
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    fft_noisy = np.abs(np.fft.rfft(noisy_signal))
    fft_cleaned = np.abs(np.fft.rfft(cleaned))

    # Baseline energy (0.1 Hz) and Noise energy (60 Hz) should drop significantly relative to 10 Hz peak
    idx_10hz = np.argmin(np.abs(freqs - 10.0))
    idx_01hz = np.argmin(np.abs(freqs - 0.1))
    idx_60hz = np.argmin(np.abs(freqs - 60.0))

    # In noisy signal, 0.1 Hz component is dominant (amplitude 2.0 vs 1.0)
    assert fft_noisy[idx_01hz] > fft_noisy[idx_10hz]

    # In cleaned signal, 10 Hz component should dominate, 0.1 Hz and 60 Hz suppressed
    assert fft_cleaned[idx_10hz] > fft_cleaned[idx_01hz] * 5.0
    assert fft_cleaned[idx_10hz] > fft_cleaned[idx_60hz] * 5.0


def test_nan_and_inf_handling() -> None:
    """Test that NaN or Inf inputs raise DSPFilterError."""
    dsp = DSPFilter()
    bad_nan = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    bad_inf = np.array([1.0, np.inf, 3.0, 4.0], dtype=np.float32)

    with pytest.raises(DSPFilterError, match="contains NaN or Inf"):
        dsp.apply_bandpass(bad_nan)

    with pytest.raises(DSPFilterError, match="contains NaN or Inf"):
        dsp.apply_bandpass(bad_inf)


def test_zero_variance_flat_signal() -> None:
    """Test that flat signal (zero variance) is handled without divide-by-zero."""
    dsp = DSPFilter()
    flat_signal = np.ones(200, dtype=np.float32) * 5.0
    cleaned = dsp.process_window(flat_signal)

    assert cleaned.shape == (200,)
    assert not np.isnan(cleaned).any()
    assert not np.isinf(cleaned).any()
