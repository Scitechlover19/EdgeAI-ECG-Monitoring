"""Comprehensive Validation Selection and Threshold Sweep Execution Script.

Implements:
1. Validation-set hyperparameter selection for Knowledge Distillation (leakage-free).
2. Evaluation at actual scheduler threshold (0.85) for No-KD and KD models.
3. Full threshold sweep from 0.30 to 0.95 in 0.05 steps recording recall, precision, F1,
   active triggers, bytes transmitted, and bandwidth reduction percentage.
4. Generates ROC/PR visualization plots and documentation reports.

Governed by PRD.md, Architecture.md §5, AGENTS.md §8, and DECISIONS.md #6, #13, #15.
"""

from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Tuple
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import tensorflow as tf

from src.models.distill import create_distiller
from src.models.student import build_student_model
from src.models.train_student_kd import load_teacher_checkpoint
from src.models.train_teacher import compute_binary_metrics, load_processed_datasets
from src.pipeline.config import load_config, setup_logging

logger = setup_logging()


def load_student_checkpoint(path: Path, seed: int = 42) -> tf.keras.Model:
    """Load Student model checkpoint with Keras 3 deserialization fallback."""
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        student = build_student_model(input_shape=(200, 1), num_classes=2, seed=seed)
        with zipfile.ZipFile(path, "r") as z:
            with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
                tmp.write(z.read("model.weights.h5"))
                tmp_path = tmp.name
        student.load_weights(tmp_path)
        return student


def evaluate_at_threshold(y_true: np.ndarray, anomaly_probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute classification and transmission metrics for a specific decision threshold."""
    preds = (anomaly_probs >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    total = len(y_true)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    active_transmissions = tp + fp
    sleep_windows = total - active_transmissions
    active_rate_pct = (active_transmissions / total * 100.0) if total > 0 else 0.0

    # Measured payload parameters from reports/bandwidth_report.md
    bytes_per_alert = 132.23
    bytes_per_raw_window = 400.0

    telemetry_bytes = active_transmissions * bytes_per_alert
    baseline_bytes = total * bytes_per_raw_window
    reduction_pct = ((baseline_bytes - telemetry_bytes) / baseline_bytes * 100.0) if baseline_bytes > 0 else 0.0

    return {
        "threshold": round(threshold, 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "active_transmissions": active_transmissions,
        "sleep_windows": sleep_windows,
        "active_rate_pct": round(active_rate_pct, 2),
        "telemetry_bytes": round(telemetry_bytes, 1),
        "baseline_bytes": int(baseline_bytes),
        "reduction_pct": round(reduction_pct, 4),
        "meets_nfr6": reduction_pct > 90.0,
    }


def run_full_experiment_pipeline() -> Dict[str, Any]:
    """Run validation selection, threshold 0.85 evaluation, and full threshold sweep."""
    config = load_config()
    data_dir = config.paths.data_dir
    models_dir = Path(config.paths.models_dir)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading preprocessed dataset splits...")
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_processed_datasets(data_dir)

    X_tr_in = np.expand_dims(X_tr, axis=-1)
    X_va_in = np.expand_dims(X_va, axis=-1)
    X_te_in = np.expand_dims(X_te, axis=-1)

    # -------------------------------------------------------------------------
    # 1. Evaluate Baselines (Teacher and Student No KD) on Val and Test
    # -------------------------------------------------------------------------
    logger.info("Evaluating Teacher model...")
    teacher = load_teacher_checkpoint(models_dir / "teacher_model.keras", seed=42)
    teacher.trainable = False

    t_val_probs = teacher.predict(X_va_in, batch_size=512, verbose=0)
    t_test_probs = teacher.predict(X_te_in, batch_size=512, verbose=0)
    teacher_val_metrics = compute_binary_metrics(y_va, t_val_probs)
    teacher_test_metrics = compute_binary_metrics(y_te, t_test_probs)

    logger.info("Evaluating Student (No KD) baseline...")
    student_no_kd = load_student_checkpoint(models_dir / "student_no_kd_model.keras", seed=42)
    nokd_val_probs = student_no_kd.predict(X_va_in, batch_size=512, verbose=0)
    nokd_test_probs = student_no_kd.predict(X_te_in, batch_size=512, verbose=0)
    nokd_val_metrics = compute_binary_metrics(y_va, nokd_val_probs)
    nokd_test_metrics = compute_binary_metrics(y_te, nokd_test_probs)

    # -------------------------------------------------------------------------
    # 2. Knowledge Distillation Hyperparameter Sweep on VALIDATION SPLIT
    # -------------------------------------------------------------------------
    combos = [
        (2.0, 0.3),
        (2.0, 0.5),
        (4.0, 0.3),
        (4.0, 0.5),
        (6.0, 0.3),
        (6.0, 0.5),
    ]

    sweep_results: List[Dict[str, Any]] = []
    trained_kd_models: Dict[Tuple[float, float], tf.keras.Model] = {}
    kd_val_probs_dict: Dict[Tuple[float, float], np.ndarray] = {}
    kd_test_probs_dict: Dict[Tuple[float, float], np.ndarray] = {}

    for t_val, a_val in combos:
        logger.info(f"Training KD candidate: T={t_val}, alpha={a_val}...")
        tf.keras.utils.set_random_seed(42)

        student_kd = build_student_model(input_shape=(200, 1), num_classes=2, seed=42)
        distiller = create_distiller(
            teacher_model=teacher,
            student_model=student_kd,
            temperature=t_val,
            alpha=a_val,
            learning_rate=1e-3,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ]

        t0 = time.time()
        distiller.fit(
            X_tr_in,
            y_tr,
            validation_data=(X_va_in, y_va),
            epochs=6,
            batch_size=256,
            callbacks=callbacks,
            verbose=0,
        )
        elapsed = time.time() - t0

        # Predict on validation split (for selection)
        val_probs = student_kd.predict(X_va_in, batch_size=512, verbose=0)
        val_m = compute_binary_metrics(y_va, val_probs)

        # Predict on held-out test split (for reporting)
        test_probs = student_kd.predict(X_te_in, batch_size=512, verbose=0)
        test_m = compute_binary_metrics(y_te, test_probs)

        entry = {
            "temperature": t_val,
            "alpha": a_val,
            "training_time_sec": round(elapsed, 2),
            "val_accuracy": round(val_m["accuracy"], 4),
            "val_precision": round(val_m["precision"], 4),
            "val_recall": round(val_m["recall"], 4),
            "val_f1": round(val_m["f1_score"], 4),
            "val_tp": val_m["tp"],
            "val_fp": val_m["fp"],
            "val_fn": val_m["fn"],
            "val_tn": val_m["tn"],
            "test_accuracy": round(test_m["accuracy"], 4),
            "test_precision": round(test_m["precision"], 4),
            "test_recall": round(test_m["recall"], 4),
            "test_f1": round(test_m["f1_score"], 4),
            "test_tp": test_m["tp"],
            "test_fp": test_m["fp"],
            "test_fn": test_m["fn"],
            "test_tn": test_m["tn"],
        }
        sweep_results.append(entry)
        trained_kd_models[(t_val, a_val)] = student_kd
        kd_val_probs_dict[(t_val, a_val)] = val_probs
        kd_test_probs_dict[(t_val, a_val)] = test_probs

        logger.info(
            f"KD (T={t_val}, alpha={a_val}) -> Val F1: {val_m['f1_score']:.4f}, Val Recall: {val_m['recall']:.4f} | "
            f"Test F1: {test_m['f1_score']:.4f}, Test Recall: {test_m['recall']:.4f}"
        )

    # -------------------------------------------------------------------------
    # 3. Model Selection based strictly on Validation Split
    # -------------------------------------------------------------------------
    # Best KD configuration on validation F1
    best_kd_entry = max(sweep_results, key=lambda x: x["val_f1"])
    best_t = best_kd_entry["temperature"]
    best_a = best_kd_entry["alpha"]
    logger.info(f"\nWINNER selected on Validation Split: T={best_t}, alpha={best_a} (Val F1={best_kd_entry['val_f1']})")

    best_kd_model = trained_kd_models[(best_t, best_a)]
    best_kd_model.save(models_dir / "student_kd_model.keras")

    best_kd_test_probs = kd_test_probs_dict[(best_t, best_a)]

    # -------------------------------------------------------------------------
    # 4. Metrics @ Actual Scheduler Threshold (0.85) on Held-Out Test Set
    # -------------------------------------------------------------------------
    nokd_test_anomaly_probs = nokd_test_probs[:, 1]
    best_kd_test_anomaly_probs = best_kd_test_probs[:, 1]
    teacher_test_anomaly_probs = t_test_probs[:, 1]

    t85_teacher = evaluate_at_threshold(y_te, teacher_test_anomaly_probs, 0.85)
    t85_nokd = evaluate_at_threshold(y_te, nokd_test_anomaly_probs, 0.85)
    t85_kd = evaluate_at_threshold(y_te, best_kd_test_anomaly_probs, 0.85)

    # -------------------------------------------------------------------------
    # 5. Full Threshold Sweep on Test Split (0.30 to 0.95 in 0.05 steps)
    # -------------------------------------------------------------------------
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]
    nokd_sweep: List[Dict[str, Any]] = []
    kd_sweep: List[Dict[str, Any]] = []

    for th in thresholds:
        nokd_sweep.append(evaluate_at_threshold(y_te, nokd_test_anomaly_probs, th))
        kd_sweep.append(evaluate_at_threshold(y_te, best_kd_test_anomaly_probs, th))

    # -------------------------------------------------------------------------
    # 6. Plot ROC and PR Curves
    # -------------------------------------------------------------------------
    fpr_nokd, tpr_nokd, _ = roc_curve(y_te, nokd_test_anomaly_probs)
    roc_auc_nokd = auc(fpr_nokd, tpr_nokd)

    fpr_kd, tpr_kd, _ = roc_curve(y_te, best_kd_test_anomaly_probs)
    roc_auc_kd = auc(fpr_kd, tpr_kd)

    prec_nokd, rec_nokd, _ = precision_recall_curve(y_te, nokd_test_anomaly_probs)
    pr_auc_nokd = auc(rec_nokd, prec_nokd)

    prec_kd, rec_kd, _ = precision_recall_curve(y_te, best_kd_test_anomaly_probs)
    pr_auc_kd = auc(rec_kd, prec_kd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    ax1.plot(fpr_nokd, tpr_nokd, label=f"Student (No KD) AUC = {roc_auc_nokd:.4f}", color="#1f77b4", lw=2)
    ax1.plot(fpr_kd, tpr_kd, label=f"Student KD (T={best_t}, α={best_a}) AUC = {roc_auc_kd:.4f}", color="#ff7f0e", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("ROC Curve on Held-Out Test Set")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Precision-Recall Curve
    ax2.plot(rec_nokd, prec_nokd, label=f"Student (No KD) AUC-PR = {pr_auc_nokd:.4f}", color="#1f77b4", lw=2)
    ax2.plot(rec_kd, prec_kd, label=f"Student KD (T={best_t}, α={best_a}) AUC-PR = {pr_auc_kd:.4f}", color="#ff7f0e", lw=2)
    ax2.axhline(y=7278 / 51992, color="gray", linestyle=":", label="Prevalence (14.0%)")
    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision (PPV)")
    ax2.set_title("Precision-Recall Curve on Held-Out Test Set")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = reports_dir / "threshold_roc_pr.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC/PR plot to {plot_path}")

    return {
        "teacher_val": teacher_val_metrics,
        "teacher_test": teacher_test_metrics,
        "nokd_val": nokd_val_metrics,
        "nokd_test": nokd_test_metrics,
        "kd_sweep": sweep_results,
        "best_kd_config": (best_t, best_a),
        "best_kd_val_f1": best_kd_entry["val_f1"],
        "t85_teacher": t85_teacher,
        "t85_nokd": t85_nokd,
        "t85_kd": t85_kd,
        "nokd_sweep": nokd_sweep,
        "kd_sweep_threshold": kd_sweep,
        "roc_auc_nokd": roc_auc_nokd,
        "roc_auc_kd": roc_auc_kd,
        "pr_auc_nokd": pr_auc_nokd,
        "pr_auc_kd": pr_auc_kd,
    }


if __name__ == "__main__":
    results = run_full_experiment_pipeline()
    print("\n=== EXPERIMENT PIPELINE COMPLETE ===")
