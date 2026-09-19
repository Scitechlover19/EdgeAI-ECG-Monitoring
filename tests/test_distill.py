"""Unit tests for Knowledge Distillation module (P0-7).

Verifies Distiller class instantiation, loss formulation, Teacher weight freezing,
Student weight updates, hyperparameter assignment, and mini-batch train step behavior.
Governed by AGENTS.md §3 and DECISIONS.md #6.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import tensorflow as tf

from src.models.distill import Distiller, create_distiller
from src.models.student import build_student_model
from src.models.teacher import build_teacher_model


def test_distiller_initialization():
    """Test Distiller initialization, parameter assignment, and Teacher freezing."""
    teacher = build_teacher_model(input_shape=(200, 1), num_classes=2, seed=42)
    student = build_student_model(input_shape=(200, 1), num_classes=2, seed=42)

    distiller = create_distiller(
        teacher_model=teacher,
        student_model=student,
        temperature=3.0,
        alpha=0.7,
        learning_rate=1e-3,
    )

    assert distiller.temperature == 3.0
    assert distiller.alpha == 0.7
    assert distiller.teacher.trainable is False
    assert len(distiller.metrics) == 4


def test_teacher_frozen_student_updates():
    """Test that Teacher weights remain strictly unchanged while Student weights update during train_step."""
    teacher = build_teacher_model(input_shape=(200, 1), num_classes=2, seed=42)
    student = build_student_model(input_shape=(200, 1), num_classes=2, seed=42)

    distiller = create_distiller(teacher, student, temperature=3.0, alpha=0.7)

    teacher_weights_before = [w.numpy().copy() for w in teacher.weights]
    student_weights_before = [w.numpy().copy() for w in student.trainable_weights]

    x_batch = np.random.randn(16, 200, 1).astype(np.float32)
    y_batch = np.random.choice([0, 1], size=(16,)).astype(np.int64)

    metrics_res = distiller.train_step((x_batch, y_batch))

    assert "loss" in metrics_res
    assert "student_loss" in metrics_res
    assert "distill_loss" in metrics_res
    assert "accuracy" in metrics_res

    # 1. Assert Teacher weights are 100% identical (zero change)
    for w_before, w_after in zip(teacher_weights_before, teacher.weights):
        np.testing.assert_array_equal(w_before, w_after)

    # 2. Assert Student trainable weights HAVE changed (received non-zero gradient update)
    weight_changed = False
    for w_before, w_after in zip(student_weights_before, student.trainable_weights):
        if not np.array_equal(w_before, w_after.numpy()):
            weight_changed = True
            break

    assert weight_changed, "Student trainable weights did not change after train_step!"


def test_distillation_loss_math():
    """Test soft-target loss formula: (1 - alpha) * L_ce + alpha * T^2 * L_kl."""
    teacher = build_teacher_model(input_shape=(200, 1), num_classes=2, seed=42)
    student = build_student_model(input_shape=(200, 1), num_classes=2, seed=42)

    temperature = 2.0
    alpha = 0.6
    distiller = Distiller(student, teacher, temperature=temperature, alpha=alpha)

    x_dummy = np.random.randn(4, 200, 1).astype(np.float32)
    y_dummy = np.array([0, 1, 0, 1], dtype=np.int64)

    # Compute manual reference loss
    t_pred = teacher(x_dummy, training=False)
    s_pred = student(x_dummy, training=False)

    ce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    kl_loss_fn = tf.keras.losses.KLDivergence()

    manual_ce = ce_loss_fn(y_dummy, s_pred).numpy()

    t_soft = distiller._compute_soft_targets(t_pred)
    s_soft = distiller._compute_soft_targets(s_pred)
    manual_kl = kl_loss_fn(t_soft, s_soft).numpy() * (temperature ** 2)

    expected_total_loss = (1.0 - alpha) * manual_ce + alpha * manual_kl

    # Compile and execute 1 step
    distiller.compile(optimizer=tf.keras.optimizers.Adam())
    metrics_out = distiller.train_step((x_dummy, y_dummy))

    np.testing.assert_allclose(metrics_out["loss"].numpy(), expected_total_loss, rtol=1e-4)


def test_distiller_fit_smoke_run():
    """Test fit execution on a tiny synthetic dataset."""
    teacher = build_teacher_model(input_shape=(200, 1), num_classes=2, seed=42)
    student = build_student_model(input_shape=(200, 1), num_classes=2, seed=42)
    distiller = create_distiller(teacher, student, temperature=3.0, alpha=0.7)

    x_dummy = np.random.randn(32, 200, 1).astype(np.float32)
    y_dummy = np.random.choice([0, 1], size=(32,)).astype(np.int64)

    history = distiller.fit(x_dummy, y_dummy, epochs=2, batch_size=16, verbose=0)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 2
    assert np.all(np.isfinite(history.history["loss"]))
