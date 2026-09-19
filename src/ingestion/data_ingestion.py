"""DataIngestion module for loading MIT-BIH Arrhythmia Database ECG signals.

Handles signal loading via `wfdb`, sliding window segmentation (200 samples, 50% overlap),
sampling rate extraction, and streaming iteration for the edge AI monitoring pipeline.
Governed by PRD.md (FR-1, FR-2), Architecture.md §5.1, and DECISIONS.md #1.
"""

from typing import Generator, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import wfdb


class DataIngestionError(Exception):
    """Custom exception raised for DataIngestion failures."""
    pass


class DataIngestion:
    """Sliding-window ECG signal generator and reader."""

    def __init__(
        self,
        source: Union[str, Path, np.ndarray],
        window_size: int = 200,
        overlap_ratio: float = 0.5,
        channel: int = 0,
        sample_rate: Optional[float] = None,
    ) -> None:
        """Initialize DataIngestion instance.

        Args:
            source: Record name/path (for wfdb load) or direct 1D numpy array of ECG signal.
            window_size: Number of samples per window (default 200).
            overlap_ratio: Fraction of window overlap (default 0.5 = 50%).
            channel: Lead channel index to ingest (default 0, usually MLII).
            sample_rate: Sampling frequency in Hz (if None, extracted from wfdb header or 360.0 default).

        Raises:
            DataIngestionError: If input is invalid or file cannot be read.
        """
        if window_size <= 0:
            raise DataIngestionError(f"Invalid window_size {window_size}. Must be positive.")
        if not (0.0 <= overlap_ratio < 1.0):
            raise DataIngestionError(f"Invalid overlap_ratio {overlap_ratio}. Must be in [0.0, 1.0).")

        self.window_size: int = window_size
        self.overlap_ratio: float = overlap_ratio
        self.step_size: int = int(window_size * (1.0 - overlap_ratio))
        if self.step_size < 1:
            self.step_size = 1

        self.channel: int = channel
        self.record_name: Optional[str] = None
        self.sample_rate: float = 360.0

        if isinstance(source, (str, Path)):
            self.record_name = str(source)
            self.signal, extracted_fs = self._load_wfdb_record(str(source))
            self.sample_rate = sample_rate if sample_rate is not None else extracted_fs
        elif isinstance(source, np.ndarray):
            if source.ndim != 1:
                raise DataIngestionError(f"Input numpy array must be 1D, got shape {source.shape}")
            if len(source) == 0:
                raise DataIngestionError("Input numpy array is empty.")
            self.signal = source.astype(np.float32)
            if sample_rate is not None:
                self.sample_rate = float(sample_rate)
        else:
            raise DataIngestionError(f"Unsupported source type: {type(source)}")

    def _load_wfdb_record(self, record_path: str) -> Tuple[np.ndarray, float]:
        """Load a MIT-BIH record using wfdb.

        Args:
            record_path: Path or record name string.

        Returns:
            Tuple of (1D float32 numpy array signal, sampling frequency in Hz).

        Raises:
            DataIngestionError: If wfdb fails to parse the file.
        """
        try:
            record = wfdb.rdrecord(record_path)
            if record.p_signal is None or record.p_signal.size == 0:
                raise DataIngestionError(f"Record {record_path} has empty signal data.")

            n_channels = record.p_signal.shape[1]
            if self.channel >= n_channels:
                raise DataIngestionError(
                    f"Requested channel {self.channel} out of bounds (record has {n_channels} channels)."
                )

            signal_1d = record.p_signal[:, self.channel].astype(np.float32)
            # Remove any trailing NaNs if present
            signal_1d = np.nan_to_num(signal_1d, nan=0.0)
            fs = float(record.fs) if record.fs else 360.0
            return signal_1d, fs
        except Exception as err:
            raise DataIngestionError(f"Failed to read wfdb record '{record_path}': {err}") from err

    @property
    def total_samples(self) -> int:
        """Total sample count of the loaded signal."""
        return len(self.signal)

    @property
    def total_windows(self) -> int:
        """Calculate total number of full sliding windows available."""
        if len(self.signal) < self.window_size:
            return 0
        return (len(self.signal) - self.window_size) // self.step_size + 1

    def stream_windows(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Generator yielding sliding windows one by one.

        Yields:
            Tuple containing:
            - window_index (int): 0-based index of the window.
            - start_timestamp_sec (float): Start time of the window in seconds.
            - raw_window (np.ndarray): 1D float32 array of shape (window_size,).

        Raises:
            DataIngestionError: If signal length is shorter than window_size.
        """
        if self.total_windows == 0:
            raise DataIngestionError(
                f"Signal length ({len(self.signal)}) is shorter than window_size ({self.window_size})."
            )

        for i in range(self.total_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + self.window_size
            raw_window = self.signal[start_idx:end_idx].copy()
            timestamp_sec = start_idx / self.sample_rate
            yield i, timestamp_sec, raw_window

    def get_window(self, index: int) -> Tuple[float, np.ndarray]:
        """Retrieve a specific window by index.

        Args:
            index: 0-based window index.

        Returns:
            Tuple of (start_timestamp_sec, raw_window).

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= self.total_windows:
            raise IndexError(f"Window index {index} out of bounds (0..{self.total_windows - 1}).")

        start_idx = index * self.step_size
        end_idx = start_idx + self.window_size
        raw_window = self.signal[start_idx:end_idx].copy()
        timestamp_sec = start_idx / self.sample_rate
        return timestamp_sec, raw_window
