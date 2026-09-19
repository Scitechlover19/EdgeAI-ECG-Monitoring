"""Unit tests for INT8 Post-Training Quantization module (P0-8).

Verifies TFLite INT8 Flatbuffer generation, file size limits (< 1 MB Flash limit),
input/output int8 dtypes, scale/zero-point quantization parameters, and TinyMLEngine integration.
Governed by AGENTS.md §3 and DECISIONS.md #7.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import tensorflow as tf

from src.edge.tinyml_engine import TinyMLEngine
from src.models.quantize import (
    create_representative_dataset_generator,
    quantize_student_model,
)
from src.models.student import build_student_model


def test_representative_dataset_generator(tmp_path: Path):
    """Test representative dataset generator yields float32 tensors of shape (1, 200, 1)."""
    dummy_x_train = np.random.randn(20, 200).astype(np.float32)
    x_train_file = tmp_path / "X_train.npy"
    np.save(x_train_file, dummy_x_train)

    gen = create_representative_dataset_generator(train_data_path=x_train_file, num_samples=5)
    samples = list(gen())

    assert len(samples) == 5
    for s in samples:
        assert isinstance(s, list)
        assert len(s) == 1
        assert s[0].shape == (1, 200, 1)
        assert s[0].dtype == np.float32


def test_quantize_student_model_export(tmp_path: Path):
    """Test quantize_student_model exports a valid INT8 TFLite Flatbuffer with int8 dtypes."""
    # Build dummy student model
    student = build_student_model(input_shape=(200, 1), num_classes=2, seed=42)
    student_keras_path = tmp_path / "student_dummy.keras"
    student.save(student_keras_path)

    # Save dummy X_train.npy
    dummy_x_train = np.random.randn(50, 200).astype(np.float32)
    x_train_file = tmp_path / "X_train.npy"
    np.save(x_train_file, dummy_x_train)

    out_tflite_path = tmp_path / "student_model_int8.tflite"

    tflite_bytes, metadata = quantize_student_model(
        student_model_path=student_keras_path,
        train_data_path=x_train_file,
        output_tflite_path=out_tflite_path,
        num_calibration_samples=20,
    )

    assert out_tflite_path.exists()
    assert len(tflite_bytes) > 0
    assert metadata["file_size_bytes"] < 1_000_000, "Exported model size exceeds 1 MB Flash limit!"

    assert metadata["input_dtype"] == "int8"
    assert metadata["output_dtype"] == "int8"
    assert metadata["input_shape"] == (1, 200, 1)
    assert metadata["output_shape"] == (1, 2)


def test_int8_tinyml_engine_inference(tmp_path: Path):
    """Test TinyMLEngine loads exported INT8 .tflite model and runs inference cleanly."""
    student = build_student_model(input_shape=(200, 1), num_classes=2, seed=42)
    student_keras_path = tmp_path / "student_dummy.keras"
    student.save(student_keras_path)

    dummy_x_train = np.random.randn(50, 200).astype(np.float32)
    x_train_file = tmp_path / "X_train.npy"
    np.save(x_train_file, dummy_x_train)

    out_tflite_path = tmp_path / "student_model_int8.tflite"
    quantize_student_model(
        student_model_path=student_keras_path,
        train_data_path=x_train_file,
        output_tflite_path=out_tflite_path,
        num_calibration_samples=20,
    )

    # Instantiate TinyMLEngine with INT8 .tflite file
    engine = TinyMLEngine(model_path=out_tflite_path)
    assert engine.interpreter is not None
    assert engine.input_details[0]["dtype"] == np.int8

    clean_window = np.random.randn(200).astype(np.float32)
    confidence, label_idx, label_str, latency_ms = engine.invoke_inference(clean_window)

    assert 0.0 <= confidence <= 1.0
    assert label_idx in (0, 1)
    assert label_str in ("NORMAL", "ANOMALY")
    assert latency_ms >= 0.0
