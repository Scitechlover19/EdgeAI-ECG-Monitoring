"""Tests for Student 1D-CNN model architecture.

Verifies construction, compact parameter bounds, capacity reduction ratio vs Teacher,
forward pass finiteness, TFLite conversion compatibility, and serialization.
Governed by AGENTS.md §3 and DECISIONS.md #5.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import tensorflow as tf

from src.models.student import build_student_model
from src.models.teacher import build_teacher_model


def test_student_model_construction():
    """Test Student model instantiation and output layer compilation."""
    model = build_student_model(input_shape=(200, 1), num_classes=2)
    assert model is not None
    assert model.name == "Student_1D_CNN"
    assert len(model.layers) > 4
    assert model.output_shape == (None, 2)


def test_student_input_output_shapes():
    """Test explicit input and output shape contracts."""
    model = build_student_model(input_shape=(200, 1), num_classes=2)
    assert model.input_shape == (None, 200, 1)
    assert model.output_shape == (None, 2)


def test_student_parameter_count_and_teacher_ratio():
    """Test Student parameter count is compact and substantially smaller than Teacher."""
    student_model = build_student_model(input_shape=(200, 1), num_classes=2)
    teacher_model = build_teacher_model(input_shape=(200, 1), num_classes=2)

    student_params = student_model.count_params()
    teacher_params = teacher_model.count_params()

    assert student_params > 0
    assert student_params < 10000, f"Student model parameters too large ({student_params})"
    
    # Assert numerical parameter compression ratio >= 10x
    param_ratio = teacher_params / student_params
    assert param_ratio >= 10.0, f"Teacher/Student parameter ratio ({param_ratio:.1f}x) is below 10x threshold"


def test_student_forward_pass_finite_outputs():
    """Test forward pass on a batch of dummy preprocessed ECG windows."""
    model = build_student_model(input_shape=(200, 1), num_classes=2)
    batch_size = 8
    dummy_input = np.random.randn(batch_size, 200, 1).astype(np.float32)

    outputs = model.predict(dummy_input, verbose=0)

    assert outputs.shape == (batch_size, 2)
    assert np.all(np.isfinite(outputs)), "Forward pass output contains NaN or Inf values"
    row_sums = np.sum(outputs, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(batch_size), atol=1e-5)


def test_student_tflite_conversion_compatibility():
    """Test that Student model converts to a valid TFLite Flatbuffer with zero op failures."""
    model = build_student_model(input_shape=(200, 1), num_classes=2)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_bytes = converter.convert()

    assert tflite_model_bytes is not None
    assert len(tflite_model_bytes) > 0

    # Smoke test flatbuffer tensor allocation using TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    assert tuple(input_details[0]["shape"]) == (1, 200, 1)
    assert tuple(output_details[0]["shape"]) == (1, 2)


def test_student_model_serialization():
    """Test saving and loading Student model checkpoint."""
    model = build_student_model(input_shape=(200, 1), num_classes=2)
    dummy_input = np.random.randn(2, 200, 1).astype(np.float32)
    original_pred = model.predict(dummy_input, verbose=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "student_test.keras"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = tf.keras.models.load_model(save_path, compile=False)
        loaded_pred = loaded_model.predict(dummy_input, verbose=0)
        np.testing.assert_allclose(original_pred, loaded_pred, atol=1e-5)
