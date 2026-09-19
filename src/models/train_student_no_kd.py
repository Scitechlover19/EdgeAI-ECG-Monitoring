"""Baseline training script for Student 1D-CNN WITHOUT Knowledge Distillation.

Trains Student model using standard hard labels for ablation comparison against KD Student.
Governed by PRD.md (FR-7, FR-8), Architecture.md §5.3, AGENTS.md §8, and DECISIONS.md #6.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import tensorflow as tf

from src.models.student import build_student_model
from src.models.train_teacher import compute_binary_metrics, load_processed_datasets
from src.pipeline.config import load_config, setup_logging

logger = setup_logging()


def train_student_no_kd_pipeline(
    epochs: int = 15,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
    smoke_test: bool = False,
    output_dir: Optional[Path] = None,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Execute baseline Student training pipeline (Without KD).

    Args:
        epochs: Number of training epochs (default 15).
        batch_size: Batch size (default 256).
        learning_rate: Adam optimizer learning rate.
        seed: Random seed for reproducibility.
        smoke_test: If True, run on small subset for 1 epoch.
        output_dir: Directory to save model checkpoint.

    Returns:
        Tuple of (trained_student_model, metrics_dict).
    """
    config = load_config()
    data_dir = config.paths.data_dir
    models_dir = output_dir if output_dir is not None else Path(config.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(seed)
    logger.info("Loading preprocessed dataset splits...")
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_processed_datasets(data_dir)

    if smoke_test:
        logger.info("Running SMOKE TEST training on small subset (500 samples)...")
        X_tr, y_tr = X_tr[:500], y_tr[:500]
        X_va, y_va = X_va[:200], y_va[:200]
        X_te, y_te = X_te[:200], y_te[:200]
        epochs = 1
        batch_size = 64

    X_tr_in = np.expand_dims(X_tr, axis=-1)
    X_va_in = np.expand_dims(X_va, axis=-1)
    X_te_in = np.expand_dims(X_te, axis=-1)

    logger.info(f"Building Student model (Without KD, Seed: {seed})...")
    student = build_student_model(
        input_shape=(200, 1),
        num_classes=2,
        learning_rate=learning_rate,
        seed=seed,
    )

    logger.info(f"Training Student (No KD) for {epochs} epochs...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]

    history = student.fit(
        X_tr_in,
        y_tr,
        validation_data=(X_va_in, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks if not smoke_test else [],
        verbose=1 if not smoke_test else 0,
    )

    logger.info("Evaluating Student (No KD) on held-out test split...")
    test_probs = student.predict(X_te_in, batch_size=batch_size, verbose=0)
    test_metrics = compute_binary_metrics(y_te, test_probs)

    checkpoint_path = models_dir / "student_no_kd_model.keras"
    student.save(checkpoint_path)
    logger.info(
        f"Student (No KD) checkpoint saved to {checkpoint_path}. "
        f"Measured Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}"
    )

    result_summary = {
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
    }
    return student, result_summary


if __name__ == "__main__":
    train_student_no_kd_pipeline(smoke_test=True)
