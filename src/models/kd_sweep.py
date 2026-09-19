"""Knowledge Distillation Hyperparameter Sweep Script.

Sweeps (T, alpha) combinations to evaluate whether Knowledge Distillation
can improve performance over the no-KD baseline on patient-independent MIT-BIH test split.
Governed by AGENTS.md §8 and DECISIONS.md #6.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import tensorflow as tf

from src.models.distill import create_distiller
from src.models.student import build_student_model
from src.models.train_student_kd import load_teacher_checkpoint
from src.models.train_teacher import compute_binary_metrics, load_processed_datasets
from src.pipeline.config import load_config, setup_logging

logger = setup_logging()


def run_kd_sweep(
    combinations: List[Tuple[float, float]],
    epochs: int = 8,
    batch_size: int = 256,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Execute KD hyperparameter sweep across specified (T, alpha) pairs.

    Args:
        combinations: List of (T, alpha) tuples.
        epochs: Maximum epochs per configuration (default 8).
        batch_size: Batch size (default 256).
        seed: Random seed for reproducibility.

    Returns:
        List of result dictionaries.
    """
    config = load_config()
    data_dir = config.paths.data_dir
    models_dir = Path(config.paths.models_dir)

    logger.info("Loading preprocessed dataset splits for KD sweep...")
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_processed_datasets(data_dir)

    X_tr_in = np.expand_dims(X_tr, axis=-1)
    X_va_in = np.expand_dims(X_va, axis=-1)
    X_te_in = np.expand_dims(X_te, axis=-1)

    # Load frozen Teacher checkpoint
    t_path = models_dir / "teacher_model.keras"
    logger.info(f"Loading Teacher checkpoint from {t_path}...")
    teacher = load_teacher_checkpoint(t_path, seed=seed)
    teacher.trainable = False

    results: List[Dict[str, Any]] = []

    for t_val, a_val in combinations:
        logger.info(f"\n{'='*60}\nStarting KD Run: T={t_val}, alpha={a_val} (Epochs={epochs}, Seed={seed})\n{'='*60}")
        tf.keras.utils.set_random_seed(seed)

        student = build_student_model(input_shape=(200, 1), num_classes=2, seed=seed)
        distiller = create_distiller(
            teacher_model=teacher,
            student_model=student,
            temperature=t_val,
            alpha=a_val,
            learning_rate=1e-3,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ]

        t_start = time.time()
        history = distiller.fit(
            X_tr_in,
            y_tr,
            validation_data=(X_va_in, y_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        elapsed_sec = time.time() - t_start

        # Evaluate on held-out test split
        test_probs = student.predict(X_te_in, batch_size=batch_size, verbose=0)
        metrics = compute_binary_metrics(y_te, test_probs)

        run_result = {
            "temperature": t_val,
            "alpha": a_val,
            "epochs_run": len(history.history["loss"]),
            "training_time_sec": elapsed_sec,
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1_score"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
        }
        results.append(run_result)
        logger.info(
            f"Result for (T={t_val}, alpha={a_val}): Acc={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}"
        )

    return results


if __name__ == "__main__":
    test_combos = [
        (2.0, 0.3),
        (2.0, 0.5),
        (4.0, 0.3),
        (4.0, 0.5),
        (6.0, 0.3),
        (6.0, 0.5),
    ]
    sweep_results = run_kd_sweep(test_combos, epochs=6)
    print("\n--- KD SWEEP SUMMARY ---")
    for r in sweep_results:
        print(f"T={r['temperature']}, alpha={r['alpha']} | Acc={r['test_accuracy']:.4f} | F1={r['test_f1']:.4f} | Recall={r['test_recall']:.4f}")
