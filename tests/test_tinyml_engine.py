"""Unit tests for TinyMLEngine module (P0-10)."""

import numpy as np
import pytest
import tensorflow as tf
from pathlib import Path

from src.edge.tinyml_engine import TinyMLEngine, TinyMLEngineError
from src.edge.virtual_mcu import VirtualMCU


def test_heuristic_fallback_inference() -> None:
    """Test TinyMLEngine without a model file using the fallback heuristic model."""
    mcu = VirtualMCU()
    engine = TinyMLEngine(virtual_mcu=mcu)

    # Normal baseline window (low amplitude, standard normalized)
    normal_window = np.sin(np.linspace(0, 2 * np.pi, 200), dtype=np.float32)
    conf_norm, label_norm, label_str_norm, lat_norm = engine.invoke_inference(normal_window)

    assert 0.0 <= conf_norm <= 1.0
    assert label_norm in (0, 1)
    assert label_str_norm in ("NORMAL", "ANOMALY")
    assert lat_norm >= 0.0

    # Anomalous high-amplitude window
    anom_window = np.sin(np.linspace(0, 2 * np.pi, 200), dtype=np.float32) * 4.0
    conf_anom, label_anom, label_str_anom, lat_anom = engine.invoke_inference(anom_window)

    assert conf_anom > conf_norm
    assert label_anom == 1
    assert label_str_anom == "ANOMALY"


def test_custom_fallback_fn() -> None:
    """Test TinyMLEngine with a custom python model callback."""

    def custom_model(window: np.ndarray):
        mean_val = float(np.mean(window))
        return (0.95, 1) if mean_val > 0.5 else (0.10, 0)

    engine = TinyMLEngine(fallback_model_fn=custom_model)

    win_high = np.ones(200, dtype=np.float32)
    conf, label, label_str, lat = engine.invoke_inference(win_high)
    assert conf == 0.95
    assert label == 1
    assert label_str == "ANOMALY"


def test_invalid_input_window() -> None:
    """Test error handling for bad window inputs."""
    engine = TinyMLEngine()

    # Wrong shape
    with pytest.raises(TinyMLEngineError, match="must be a 1D numpy array"):
        engine.invoke_inference(np.ones((200, 1), dtype=np.float32))

    # Wrong length
    with pytest.raises(TinyMLEngineError, match="length must be 200"):
        engine.invoke_inference(np.ones(150, dtype=np.float32))


def test_tflite_model_loading_and_inference(tmp_path: Path) -> None:
    """Test creating, exporting, loading, and inferring a simple TFLite model."""
    # Build a simple Keras 1D-CNN model matching input shape (1, 200, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(200, 1)),
        tf.keras.layers.Conv1D(filters=4, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    # Save to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = tmp_path / "simple_test.tflite"
    tflite_path.write_bytes(tflite_model)

    engine = TinyMLEngine(model_path=tflite_path)
    clean_window = np.random.randn(200).astype(np.float32)

    conf, label, label_str, lat_ms = engine.invoke_inference(clean_window)
    assert 0.0 <= conf <= 1.0
    assert label in (0, 1)
    assert label_str in ("NORMAL", "ANOMALY")
    assert lat_ms > 0.0


def test_regression_confidence_not_squashed_by_double_softmax(tmp_path: Path) -> None:
    """Regression test ensuring confidence is not artificially squashed into [0.269, 0.731] by double-softmax.

    If model output is already softmax probabilities [0.018, 0.982], TinyMLEngine must return ~0.982,
    NOT re-apply softmax which would squash 0.982 down to <= 0.731.
    """
    # 1. Structural check with controlled Keras model producing high-confidence softmax
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(200, 1)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(2, activation="softmax", bias_initializer=tf.keras.initializers.Constant([-2.0, 2.0]))
    ])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    tflite_path = tmp_path / "softmax_test.tflite"
    tflite_path.write_bytes(tflite_bytes)

    engine = TinyMLEngine(model_path=tflite_path)
    clean_window = np.zeros(200, dtype=np.float32)
    conf, label, label_str, _ = engine.invoke_inference(clean_window)

    # Under double-softmax bug, max possible confidence was squashed to ~0.7217 (<= 0.731)
    assert conf > 0.85, f"Confidence {conf} was squashed below 0.85 (double-softmax bug regressed!)"
    assert label == 1
    assert label_str == "ANOMALY"

    # 2. Check against real deployed model and real anomalous window if available
    real_model_path = Path("models/student_model_int8.tflite")
    test_data_path = Path("data/processed/X_test.npy")
    if real_model_path.exists() and test_data_path.exists():
        real_engine = TinyMLEngine(model_path=real_model_path)
        X_te = np.load(test_data_path, mmap_mode="r")
        real_conf, real_label, _, _ = real_engine.invoke_inference(np.array(X_te[26723], dtype=np.float32))
        assert real_conf > 0.85, f"Real window confidence {real_conf} failed to exceed 0.85 threshold"
        assert real_label == 1

