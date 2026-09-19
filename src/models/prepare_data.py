"""Data preparation script for MIT-BIH Arrhythmia Database.

Downloads MIT-BIH records via `wfdb` if not present locally, applies DSPFilter preprocessing,
segments signal into 200-sample sliding windows (50% overlap), extracts AAMI-compliant binary labels,
performs strict patient/record-independent train/validation/test splitting, and exports
`reports/experiment_manifest.json` and preprocessed `.npy` datasets.

Governed by PRD.md (FR-1 to FR-4), Architecture.md §5.3, AGENTS.md §8, and DECISIONS.md #1, #2, #3.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import wfdb

from src.dsp.dsp_filter import DSPFilter
from src.pipeline.config import load_config, setup_logging

logger = setup_logging()

# Standard MIT-BIH record IDs (48 total half-hour 2-channel ECG records)
MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

# Normal/Baseline beat symbol annotations
NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j", "."}

# Anomaly/Arrhythmia beat symbol annotations (Ventricular, Supraventricular, Ectopic, Paced)
ANOMALY_BEAT_SYMBOLS = {"V", "A", "F", "E", "a", "J", "S", "Q", "/", "f", "x", "!"}


def ensure_mitdb_downloaded(raw_dir: Path) -> List[str]:
    """Download MIT-BIH Arrhythmia Database records via wfdb if not already available locally.

    Args:
        raw_dir: Directory path to store/check raw MIT-BIH files.

    Returns:
        List of record path strings.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    available_records: List[str] = []

    missing_records = []
    for rec_id in MITDB_RECORDS:
        header_file = raw_dir / f"{rec_id}.hea"
        dat_file = raw_dir / f"{rec_id}.dat"
        atr_file = raw_dir / f"{rec_id}.atr"
        if header_file.is_file() and dat_file.is_file() and atr_file.is_file():
            available_records.append(str(raw_dir / rec_id))
        else:
            missing_records.append(rec_id)

    if missing_records:
        logger.info(f"Downloading {len(missing_records)} missing MIT-BIH records to {raw_dir}...")
        try:
            wfdb.dl_database("mitdb", dl_dir=str(raw_dir), records=missing_records)
            logger.info("MIT-BIH records downloaded successfully!")
        except Exception as err:
            logger.error(f"Failed to download MIT-BIH records: {err}")
            raise RuntimeError(f"WFDB download failed: {err}") from err

    # Re-verify all records
    final_records = [str(raw_dir / rec_id) for rec_id in MITDB_RECORDS]
    return final_records


def extract_record_windows(
    record_path: str,
    dsp_filter: DSPFilter,
    window_size: int = 200,
    overlap_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a MIT-BIH record, segment into 200-sample windows, apply DSP filter, and extract labels.

    Args:
        record_path: File path string to the WFDB record.
        dsp_filter: DSPFilter instance.
        window_size: Samples per window (default 200).
        overlap_ratio: Overlap fraction (default 0.5 = 50%).

    Returns:
        Tuple of (X_windows: np.ndarray shape (N, 200), y_labels: np.ndarray shape (N,)).
    """
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    signal = record.p_signal[:, 0].astype(np.float32)  # MLII lead channel 0
    signal = np.nan_to_num(signal, nan=0.0)

    # Annotation sample indices and symbol types
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    step_size = int(window_size * (1.0 - overlap_ratio))
    total_windows = (len(signal) - window_size) // step_size + 1

    windows = []
    labels = []

    for i in range(total_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        raw_win = signal[start_idx:end_idx]

        # Determine window label: check annotations falling inside [start_idx, end_idx)
        mask = (ann_samples >= start_idx) & (ann_samples < end_idx)
        win_symbols = set(np.array(ann_symbols)[mask])

        # Label 1 (Anomaly) if any anomaly beat symbol is present in window
        if win_symbols.intersection(ANOMALY_BEAT_SYMBOLS):
            label = 1
        else:
            label = 0

        clean_win = dsp_filter.process_window(raw_win)
        windows.append(clean_win)
        labels.append(label)

    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


def perform_record_independent_split(
    record_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Perform patient/record-independent dataset split per DECISIONS.md #3.

    Ensures no record's windows appear in more than one split!

    Args:
        record_ids: List of record identifier strings.
        train_ratio: Ratio of records for training (default 0.70).
        val_ratio: Ratio of records for validation (default 0.15).
        seed: Random seed for reproducible record assignment.

    Returns:
        Tuple of (train_records, val_records, test_records).
    """
    rng = np.random.RandomState(seed)
    shuffled = list(record_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_recs = sorted(shuffled[:n_train])
    val_recs = sorted(shuffled[n_train : n_train + n_val])
    test_recs = sorted(shuffled[n_train + n_val :])

    # Assert strict disjoint sets (zero leakage)
    assert len(set(train_recs).intersection(set(val_recs))) == 0
    assert len(set(train_recs).intersection(set(test_recs))) == 0
    assert len(set(val_recs).intersection(set(test_recs))) == 0
    assert len(train_recs) + len(val_recs) + len(test_recs) == n_total

    return train_recs, val_recs, test_recs


def prepare_dataset() -> Dict[str, Any]:
    """Execute complete data prep pipeline: download, filter, window, split, and manifest.

    Returns:
        Dict summarizing dataset statistics and manifest path.
    """
    config = load_config()
    data_dir = config.paths.data_dir
    raw_dir = data_dir / "raw" / "mitdb"
    processed_dir = data_dir / "processed"
    reports_dir = config.paths.reports_dir

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ensure raw MIT-BIH files are present
    ensure_mitdb_downloaded(raw_dir)

    # Step 2: Record-independent split
    train_recs, val_recs, test_recs = perform_record_independent_split(
        MITDB_RECORDS, train_ratio=0.70, val_ratio=0.15, seed=config.model.random_seed
    )

    dsp_filter = DSPFilter(dsp_config=config.dsp, sample_rate=config.ingestion.default_sample_rate)

    def _load_split_data(split_recs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        all_x = []
        all_y = []
        for rec_id in split_recs:
            rec_path = str(raw_dir / rec_id)
            x_win, y_lbl = extract_record_windows(
                rec_path,
                dsp_filter,
                window_size=config.ingestion.window_size,
                overlap_ratio=config.ingestion.overlap_ratio,
            )
            all_x.append(x_win)
            all_y.append(y_lbl)
        return np.vstack(all_x), np.concatenate(all_y)

    logger.info("Processing training records...")
    X_train, y_train = _load_split_data(train_recs)

    logger.info("Processing validation records...")
    X_val, y_val = _load_split_data(val_recs)

    logger.info("Processing test records...")
    X_test, y_test = _load_split_data(test_recs)

    # Save processed numpy arrays
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "X_val.npy", X_val)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "X_test.npy", X_test)
    np.save(processed_dir / "y_test.npy", y_test)

    # Generate experiment manifest JSON per AGENTS.md §8 and DECISIONS.md #3
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": "MIT-BIH Arrhythmia Database (PhysioNet)",
        "split_method": "Patient/Record-Independent Split (DECISIONS.md #3)",
        "random_seed": config.model.random_seed,
        "records": {
            "total_records": len(MITDB_RECORDS),
            "train_records": train_recs,
            "val_records": val_recs,
            "test_records": test_recs,
        },
        "windows": {
            "window_size": config.ingestion.window_size,
            "overlap_ratio": config.ingestion.overlap_ratio,
            "train_windows": len(X_train),
            "val_windows": len(X_val),
            "test_windows": len(X_test),
            "total_windows": len(X_train) + len(X_val) + len(X_test),
        },
        "class_distribution": {
            "train": {
                "normal_0": int(np.sum(y_train == 0)),
                "anomaly_1": int(np.sum(y_train == 1)),
            },
            "val": {
                "normal_0": int(np.sum(y_val == 0)),
                "anomaly_1": int(np.sum(y_val == 1)),
            },
            "test": {
                "normal_0": int(np.sum(y_test == 0)),
                "anomaly_1": int(np.sum(y_test == 1)),
            },
        },
        "dsp_parameters": {
            "low_cutoff_hz": config.dsp.low_cutoff,
            "high_cutoff_hz": config.dsp.high_cutoff,
            "filter_order": config.dsp.filter_order,
            "sample_rate_hz": config.ingestion.default_sample_rate,
        },
    }

    manifest_path = reports_dir / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Data preparation complete! Total windows: {manifest['windows']['total_windows']}. Manifest saved to {manifest_path}"
    )
    return manifest


if __name__ == "__main__":
    prepare_dataset()
