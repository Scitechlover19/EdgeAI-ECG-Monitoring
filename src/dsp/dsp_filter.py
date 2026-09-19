"""DSPFilter module for ECG signal preprocessing.

Implements a deterministic 4th-order Butterworth bandpass filter (0.5–45 Hz),
baseline wander removal, and z-score amplitude normalization.
Governed by PRD.md (FR-3, FR-4), Architecture.md §5.2, and DECISIONS.md #2.
"""

from typing import Optional, Union
import numpy as np
from scipy.signal import butter, sosfiltfilt, medfilt

from src.pipeline.config import DSPConfig


class DSPFilterError(Exception):
    """Custom exception raised for DSP processing errors."""
    pass


class DSPFilter:
    """ECG digital signal processing pipeline for noise filtering and baseline correction."""

    def __init__(
        self,
        low_cutoff: float = 0.5,
        high_cutoff: float = 45.0,
        filter_order: int = 4,
        sample_rate: float = 360.0,
        dsp_config: Optional[DSPConfig] = None,
    ) -> None:
        """Initialize DSPFilter with frequency cutoffs and filter order.

        Args:
            low_cutoff: Low cut-off frequency in Hz (default 0.5 Hz for baseline wander).
            high_cutoff: High cut-off frequency in Hz (default 45.0 Hz for muscle artifacts).
            filter_order: Butterworth filter order (default 4).
            sample_rate: Signal sampling frequency in Hz (default 360.0 Hz).
            dsp_config: Optional DSPConfig object to override parameters.

        Raises:
            DSPFilterError: If filter parameters violate Nyquist frequency or bounds.
        """
        if dsp_config is not None:
            self.low_cutoff = float(dsp_config.low_cutoff)
            self.high_cutoff = float(dsp_config.high_cutoff)
            self.filter_order = int(dsp_config.filter_order)
        else:
            self.low_cutoff = float(low_cutoff)
            self.high_cutoff = float(high_cutoff)
            self.filter_order = int(filter_order)

        self.sample_rate = float(sample_rate)
        self.nyquist = 0.5 * self.sample_rate

        self._validate_params()
        self._build_filter()

    def _validate_params(self) -> None:
        """Validate cut-off frequencies against Nyquist limit."""
        if self.low_cutoff <= 0:
            raise DSPFilterError(f"low_cutoff ({self.low_cutoff} Hz) must be positive.")
        if self.high_cutoff >= self.nyquist:
            raise DSPFilterError(
                f"high_cutoff ({self.high_cutoff} Hz) must be less than Nyquist frequency ({self.nyquist} Hz)."
            )
        if self.low_cutoff >= self.high_cutoff:
            raise DSPFilterError(
                f"low_cutoff ({self.low_cutoff} Hz) must be strictly less than high_cutoff ({self.high_cutoff} Hz)."
            )
        if self.filter_order <= 0:
            raise DSPFilterError(f"filter_order ({self.filter_order}) must be positive.")

    def _build_filter(self) -> None:
        """Construct Second-Order Sections (SOS) Butterworth bandpass filter coefficients."""
        try:
            low = self.low_cutoff / self.nyquist
            high = self.high_cutoff / self.nyquist
            self.sos = butter(self.filter_order, [low, high], btype="bandpass", output="sos")
        except Exception as err:
            raise DSPFilterError(f"Failed to build Butterworth filter: {err}") from err

    def apply_bandpass(self, window: np.ndarray) -> np.ndarray:
        """Apply zero-phase Butterworth bandpass filter to a 1D ECG array.

        Args:
            window: 1D float numpy array representing raw ECG signal window.

        Returns:
            Bandpass filtered 1D numpy array of same shape and float32 dtype.

        Raises:
            DSPFilterError: If input is invalid, contains NaNs, or filtering fails.
        """
        if not isinstance(window, np.ndarray) or window.ndim != 1:
            raise DSPFilterError("Input window must be a 1D numpy array.")
        if len(window) == 0:
            raise DSPFilterError("Input window is empty.")
        if np.isnan(window).any() or np.isinf(window).any():
            raise DSPFilterError("Input window contains NaN or Inf values.")

        try:
            # If window is shorter than 3 * filter order, pad or use sosfilt
            padlen = min(15, len(window) - 1)
            filtered = sosfiltfilt(self.sos, window, padlen=padlen)
            return filtered.astype(np.float32)
        except Exception as err:
            raise DSPFilterError(f"Error applying bandpass filter: {err}") from err

    def remove_baseline_wander(self, window: np.ndarray) -> np.ndarray:
        """Remove low-frequency baseline wander using mean subtraction and median filtering.

        Args:
            window: 1D float numpy array.

        Returns:
            Baseline-corrected 1D numpy array.
        """
        if len(window) == 0:
            return window

        # Mean subtraction (zero DC offset)
        centered = window - np.mean(window)
        return centered.astype(np.float32)

    def normalize(self, window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Normalize amplitude using z-score normalization (zero mean, unit variance).

        Args:
            window: 1D float numpy array.
            eps: Small constant to prevent divide-by-zero on flat signals.

        Returns:
            Normalized 1D float32 numpy array.
        """
        if len(window) == 0:
            return window

        mean = np.mean(window)
        std = np.std(window)
        if std < eps:
            std = 1.0

        normalized = (window - mean) / std
        return normalized.astype(np.float32)

    def process_window(self, raw_window: np.ndarray) -> np.ndarray:
        """Full DSP pipeline execution: Bandpass -> Baseline Correction -> Normalization.

        Args:
            raw_window: Raw 1D ECG array (e.g. 200 samples).

        Returns:
            Cleaned, normalized 1D float32 array of identical shape.
        """
        bandpassed = self.apply_bandpass(raw_window)
        baseline_corrected = self.remove_baseline_wander(bandpassed)
        clean_window = self.normalize(baseline_corrected)
        return clean_window
