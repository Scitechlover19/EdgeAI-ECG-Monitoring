"""Training script for Teacher 1D-CNN model on preprocessed MIT-BIH dataset.

Trains the reference Teacher model, evaluates performance on held-out test split,
and saves checkpoint to `models/teacher_model.keras`.
Governed by PRD.md (FR-5, FR-6), Architecture.md §5.3, AGENTS.md §8, and DECISIONS.md #4.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import tensorflow as tf

from src.models.teacher import build_teacher_model
from src.pipeline.config import load_config, setup_logging

logger = setup_logging()


def load_processed_datasets(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed MIT-BIH dataset splits.

    Args:
        data_dir: Base data directory containing `processed/` subfolder.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    proc_dir = data_dir / "processed"
    if not proc_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found at {proc_dir}")

    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    X_val = np.load(proc_dir / "X_val.npy")
    y_val = np.load(proc_dir / "y_val.npy")
    X_test = np.load(proc_dir / "X_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_binary_metrics(
    y_true: np.ndarray, y_pred_probs: np.ndarray
) -> Dict[str, float]:
    """Compute measured classification metrics on binary labels.

    Args:
        y_true: Ground truth binary labels array (0 or 1).
        y_pred_probs: Predicted probability array of shape (N, 2) or (N,).

    Returns:
        Dictionary of measured metrics: accuracy, precision, recall, f1_score.
    """
    if y_pred_probs.ndim == 2:
        y_pred = np.argmax(y_pred_probs, axis=1)
    else:
        y_pred = (y_pred_probs >= 0.5).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    acc = float(np.mean(y_true == y_pred))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def train_teacher_pipeline(
    epochs: int = 15,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
    smoke_test: bool = False,
    output_dir: Optional[Path] = None,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Execute complete Teacher model training and evaluation pipeline.

    Args:
        epochs: Number of training epochs (default 15).
        batch_size: Batch size (default 256).
        learning_rate: Adam optimizer learning rate.
        seed: Random seed for reproducibility.
        smoke_test: If True, run on a small subset for 1 epoch to verify pipeline.
        output_dir: Directory to save model checkpoint.

    Returns:
        Tuple of (trained_teacher_model, metrics_dict).
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

    # Expand channel dimension (N, 200) -> (N, 200, 1)
    X_tr_in = np.expand_dims(X_tr, axis=-1)
    X_va_in = np.expand_dims(X_va, axis=-1)
    X_te_in = np.expand_dims(X_te, axis=-1)

    logger.info(
        f"Building Teacher model (Input: {X_tr_in.shape}, Classes: 2, Seed: {seed})..."
    )
    teacher = build_teacher_model(
        input_shape=(200, 1),
        num_classes=2,
        learning_rate=learning_rate,
        seed=seed,
    )

    logger.info(
        f"Training Teacher model for {epochs} epochs (Batch size: {batch_size})..."
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]

    history = teacher.fit(
        X_tr_in,
        y_tr,
        validation_data=(X_va_in, y_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks if not smoke_test else [],
        verbose=1 if not smoke_test else 0,
    )

    logger.info("Evaluating Teacher model on held-out test split...")
    test_probs = teacher.predict(X_te_in, batch_size=batch_size, verbose=0)
    test_metrics = compute_binary_metrics(y_te, test_probs)

    checkpoint_path = models_dir / "teacher_model.keras"
    teacher.save(checkpoint_path)
    logger.info(
        f"Teacher model checkpoint saved to {checkpoint_path}. "
        f"Measured Test Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}"
    )

    result_summary = {
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
    }
    return teacher, result_summary


if __name__ == "__main__":
    train_teacher_pipeline(smoke_test=True)
