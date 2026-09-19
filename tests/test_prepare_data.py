"""Unit tests for prepare_data module and record-independent dataset split (P0-4)."""

import numpy as np
import pytest
from pathlib import Path
import wfdb

from src.dsp.dsp_filter import DSPFilter
from src.models.prepare_data import (
    MITDB_RECORDS,
    extract_record_windows,
    perform_record_independent_split,
)


def test_patient_record_independent_split() -> None:
    """Test perform_record_independent_split enforces zero record-level leakage across splits (DECISIONS.md #3)."""
    train_recs, val_recs, test_recs = perform_record_independent_split(
        MITDB_RECORDS, train_ratio=0.70, val_ratio=0.15, seed=42
    )

    # Convert to sets
    train_set = set(train_recs)
    val_set = set(val_recs)
    test_set = set(test_recs)

    # Assert strict disjointness (zero leakage)
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

    # Assert total record count
    assert len(train_recs) + len(val_recs) + len(test_recs) == len(MITDB_RECORDS)
    assert len(train_recs) == 33
    assert len(val_recs) == 7
    assert len(test_recs) == 8


def test_split_reproducibility() -> None:
    """Test that perform_record_independent_split is deterministic given the same seed."""
    train1, val1, test1 = perform_record_independent_split(MITDB_RECORDS, seed=42)
    train2, val2, test2 = perform_record_independent_split(MITDB_RECORDS, seed=42)

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_extract_record_windows_synthetic(tmp_path: Path) -> None:
    """Test extract_record_windows on a synthetic WFDB record with annotations."""
    rec_name = "test_extract"
    rec_path = str(tmp_path / rec_name)
    fs = 360.0
    n_samples = 1000

    # 1000 samples signal
    t = np.arange(n_samples) / fs
    p_signal = np.column_stack([np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 1.0 * t)])

    # Write sample record
    wfdb.wrsamp(
        record_name=rec_name,
        fs=int(fs),
        units=["mV", "mV"],
        sig_name=["MLII", "V1"],
        p_signal=p_signal,
        write_dir=str(tmp_path),
    )

    # Write sample annotation with normal 'N' and anomaly 'V' beats
    ann_samples = np.array([100, 450, 800], dtype=np.int32)
    ann_symbols = ["N", "V", "N"]
    wfdb.wrann(
        record_name=rec_name,
        extension="atr",
        sample=ann_samples,
        symbol=ann_symbols,
        write_dir=str(tmp_path),
    )

    dsp_filter = DSPFilter(sample_rate=fs)
    windows, labels = extract_record_windows(rec_path, dsp_filter, window_size=200, overlap_ratio=0.5)

    # (1000 - 200) // 100 + 1 = 9 windows
    assert windows.shape == (9, 200)
    assert labels.shape == (9,)
    assert not np.isnan(windows).any()
    assert not np.isinf(windows).any()

    # Window index 4 (samples 400..600) contains the 'V' beat at sample 450 -> label 1
    assert labels[4] == 1
    # Window index 0 (samples 0..200) contains 'N' beat at sample 100 -> label 0
    assert labels[0] == 0
