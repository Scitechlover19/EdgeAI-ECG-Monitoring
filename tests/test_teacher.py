"""Tests for Teacher 1D-CNN model architecture.

Verifies construction, parameter counts, forward pass shapes, output finiteness,
reproducibility, and serialization.
Governed by AGENTS.md §3 and DECISIONS.md #4.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import tensorflow as tf

from src.models.teacher import build_teacher_model


def test_teacher_model_construction():
    """Test Teacher model instantiation and output layer compilation."""
    model = build_teacher_model(input_shape=(200, 1), num_classes=2)
    assert model is not None
    assert model.name == "Teacher_1D_CNN"
    assert len(model.layers) > 5
    assert model.output_shape == (None, 2)


def test_teacher_input_output_shapes():
    """Test exact input and output shape contracts."""
    model = build_teacher_model(input_shape=(200, 1), num_classes=2)
    assert model.input_shape == (None, 200, 1)
    assert model.output_shape == (None, 2)


def test_teacher_parameter_count():
    """Test Teacher model parameter count is non-zero and within expected capacity bounds."""
    model = build_teacher_model(input_shape=(200, 1), num_classes=2)
    total_params = model.count_params()
    assert total_params > 50000, f"Teacher model params too small ({total_params})"
    assert total_params < 300000, f"Teacher model params too large ({total_params})"


def test_teacher_forward_pass_finite_outputs():
    """Test forward pass on a batch of dummy preprocessed ECG windows."""
    model = build_teacher_model(input_shape=(200, 1), num_classes=2)
    batch_size = 8
    dummy_input = np.random.randn(batch_size, 200, 1).astype(np.float32)

    outputs = model.predict(dummy_input, verbose=0)

    assert outputs.shape == (batch_size, 2)
    assert np.all(np.isfinite(outputs)), "Forward pass output contains NaN or Inf values"
    # Verify probabilities sum to 1 per sample
    row_sums = np.sum(outputs, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(batch_size), atol=1e-5)


def test_teacher_deterministic_construction():
    """Test that model building with a fixed seed produces identical initial weights."""
    model1 = build_teacher_model(input_shape=(200, 1), num_classes=2, seed=42)
    model2 = build_teacher_model(input_shape=(200, 1), num_classes=2, seed=42)

    for w1, w2 in zip(model1.get_weights(), model2.get_weights()):
        np.testing.assert_array_equal(w1, w2)


def test_teacher_model_serialization():
    """Test saving and loading Teacher model checkpoint."""
    model = build_teacher_model(input_shape=(200, 1), num_classes=2)
    dummy_input = np.random.randn(2, 200, 1).astype(np.float32)
    original_pred = model.predict(dummy_input, verbose=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "teacher_test.keras"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = tf.keras.models.load_model(save_path, compile=False)
        loaded_pred = loaded_model.predict(dummy_input, verbose=0)
        np.testing.assert_allclose(original_pred, loaded_pred, atol=1e-5)
