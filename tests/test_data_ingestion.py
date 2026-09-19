"""Unit tests for DataIngestion module (P0-2)."""

import numpy as np
import pytest
from pathlib import Path
import wfdb

from src.ingestion.data_ingestion import DataIngestion, DataIngestionError


def test_numpy_array_ingestion() -> None:
    """Test DataIngestion with a direct 1D NumPy array."""
    # 1000 samples, window_size=200, overlap=0.5 -> step=100
    # Expected total windows: (1000 - 200) // 100 + 1 = 9 windows
    signal = np.arange(1000, dtype=np.float32)
    ingestion = DataIngestion(signal, window_size=200, overlap_ratio=0.5, sample_rate=360.0)

    assert ingestion.total_samples == 1000
    assert ingestion.total_windows == 9
    assert ingestion.sample_rate == 360.0

    windows = list(ingestion.stream_windows())
    assert len(windows) == 9

    # Verify first window
    w_idx, t_sec, raw_w = windows[0]
    assert w_idx == 0
    assert t_sec == 0.0
    assert len(raw_w) == 200
    assert np.array_equal(raw_w, signal[:200])

    # Verify second window (50% overlap -> starts at 100)
    w_idx2, t_sec2, raw_w2 = windows[1]
    assert w_idx2 == 1
    assert t_sec2 == 100 / 360.0
    assert np.array_equal(raw_w2, signal[100:300])

    # Verify last window (starts at 800)
    w_idx_last, t_sec_last, raw_w_last = windows[-1]
    assert w_idx_last == 8
    assert t_sec_last == 800 / 360.0
    assert np.array_equal(raw_w_last, signal[800:1000])


def test_get_window_direct() -> None:
    """Test get_window method for specific window indexing."""
    signal = np.linspace(0.0, 10.0, 500, dtype=np.float32)
    ingestion = DataIngestion(signal, window_size=200, overlap_ratio=0.5)

    # total_windows = (500 - 200) // 100 + 1 = 4
    assert ingestion.total_windows == 4

    t_sec, w = ingestion.get_window(2)
    assert t_sec == 200 / 360.0
    assert np.array_equal(w, signal[200:400])

    with pytest.raises(IndexError):
        ingestion.get_window(5)


def test_invalid_parameters() -> None:
    """Test error handling for bad initialization parameters."""
    signal = np.ones(500, dtype=np.float32)

    with pytest.raises(DataIngestionError, match="Invalid window_size"):
        DataIngestion(signal, window_size=0)

    with pytest.raises(DataIngestionError, match="Invalid overlap_ratio"):
        DataIngestion(signal, overlap_ratio=1.5)

    with pytest.raises(DataIngestionError, match="Input numpy array must be 1D"):
        DataIngestion(np.ones((10, 10)), window_size=200)

    with pytest.raises(DataIngestionError, match="Input numpy array is empty"):
        DataIngestion(np.array([]), window_size=200)


def test_short_signal_error() -> None:
    """Test streaming when signal length is less than window_size."""
    signal = np.ones(150, dtype=np.float32)
    ingestion = DataIngestion(signal, window_size=200)

    assert ingestion.total_windows == 0
    with pytest.raises(DataIngestionError, match="Signal length .* is shorter than window_size"):
        list(ingestion.stream_windows())


def test_wfdb_record_loading(tmp_path: Path) -> None:
    """Test DataIngestion with a synthetic wfdb record written to disk."""
    record_name = "test_rec"
    record_path = str(tmp_path / record_name)

    # Generate synthetic 2-channel ECG signal
    fs = 250.0
    n_samples = 1000
    t = np.arange(n_samples) / fs
    p_signal = np.column_stack([np.sin(2 * np.pi * 1.0 * t), np.cos(2 * np.pi * 1.0 * t)])

    # Write wfdb record
    wfdb.wrsamp(
        record_name=record_name,
        fs=int(fs),
        units=["mV", "mV"],
        sig_name=["MLII", "V1"],
        p_signal=p_signal,
        write_dir=str(tmp_path),
    )

    ingestion = DataIngestion(record_path, window_size=200, overlap_ratio=0.5, channel=0)
    assert ingestion.sample_rate == fs
    assert ingestion.total_samples == n_samples
    assert ingestion.total_windows == (1000 - 200) // 100 + 1

    windows = list(ingestion.stream_windows())
    assert len(windows) == 9
    assert windows[0][1] == 0.0
    assert pytest.approx(windows[1][1]) == 100 / fs
