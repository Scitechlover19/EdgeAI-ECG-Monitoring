"""
scripts/train_imbalance_experiments.py
Experiment harness for Stage 1: Imbalance-Aware Training & Event-Level Evaluation.

Evaluates:
- Baseline (Unweighted w=1.0)
- Weighted Cross-Entropy (w=2.0, 3.0, 5.0, 8.0)
- Sparse Focal Loss (gamma=2.0, alpha_pos=0.75)

Computes both:
- Window-Level Metrics (Accuracy, Precision, Recall, F1)
- Event-Level Clinical Metrics (Event Recall, False Alarm Episodes/hr)
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.student import build_student_model
from src.models.train_teacher import load_processed_datasets


def extract_episodes(binary_array, max_gap=1):
    """Group contiguous positive markers into clinical episodes.
    
    Args:
        binary_array: 1D array of 0s and 1s.
        max_gap: Max number of intervening zeros allowed within the same episode.
                 (default 1 for ground truth; 5 for alarm hysteresis ~ 1.4s).
    """
    episodes = []
    in_episode = False
    start_idx = 0
    last_pos_idx = -1
    for i, val in enumerate(binary_array):
        if val == 1:
            if not in_episode:
                in_episode = True
                start_idx = i
            last_pos_idx = i
        else:
            if in_episode and (i - last_pos_idx > max_gap):
                episodes.append((start_idx, last_pos_idx + 1))
                in_episode = False
    if in_episode:
        episodes.append((start_idx, last_pos_idx + 1))
    return episodes


def evaluate_window_and_event_metrics(y_true, probs, threshold=0.50, hours=4.01):
    """Compute comprehensive window-level and event-level metrics."""
    preds = (probs >= threshold).astype(int)
    
    # 1. Window-Level
    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    
    acc = float(np.mean(y_true == preds))
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fp_per_hr = float(fp / hours)
    
    # 2. Event-Level
    gt_episodes = extract_episodes(y_true, max_gap=1)
    detected_episodes = 0
    for s, e in gt_episodes:
        if np.any(preds[s:e] == 1):
            detected_episodes += 1
            
    total_gt_episodes = len(gt_episodes)
    event_recall = float(detected_episodes / total_gt_episodes) if total_gt_episodes > 0 else 0.0
    
    # Alert episodes with 5-window (~1.4s) hysteresis merge
    alert_episodes = extract_episodes(preds, max_gap=5)
    false_alert_episodes = 0
    for s, e in alert_episodes:
        if not np.any(y_true[s:e] == 1):
            false_alert_episodes += 1
            
    false_ep_per_hr = float(false_alert_episodes / hours)
    
    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_per_hr": fp_per_hr,
        "gt_episodes": total_gt_episodes,
        "detected_episodes": detected_episodes,
        "event_recall": event_recall,
        "alert_episodes": len(alert_episodes),
        "false_alert_episodes": false_alert_episodes,
        "false_ep_per_hr": false_ep_per_hr,
    }


def make_sparse_focal_loss(alpha_pos=0.75, gamma=2.0):
    """Sparse categorical focal loss for 2-class softmax output."""
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_true = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
        alpha = tf.where(tf.equal(y_true, 1), alpha_pos, 1.0 - alpha_pos)
        loss = -alpha * tf.pow(1.0 - p_true, gamma) * tf.math.log(p_true)
        return tf.reduce_mean(loss)
    return focal_loss


def run_experiments():
    data_dir = Path("data")
    print("Loading datasets...")
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_processed_datasets(data_dir)
    
    X_tr_in = np.expand_dims(X_tr, -1)
    X_va_in = np.expand_dims(X_va, -1)
    X_te_in = np.expand_dims(X_te, -1)
    
    # Calculate duration
    val_hours = len(y_va) * 100 / (360 * 3600)  # ~3.51 hours
    test_hours = len(y_te) * 100 / (360 * 3600) # ~4.01 hours
    
    experiments = [
        {"name": "Baseline (Unweighted)", "type": "ce", "weight": 1.0},
        {"name": "Weighted CE (w=2.0)",   "type": "ce", "weight": 2.0},
        {"name": "Weighted CE (w=3.0)",   "type": "ce", "weight": 3.0},
        {"name": "Weighted CE (w=5.0)",   "type": "ce", "weight": 5.0},
        {"name": "Weighted CE (w=8.0)",   "type": "ce", "weight": 8.0},
        {"name": "Focal Loss (gamma=2)",  "type": "focal", "alpha": 0.75, "gamma": 2.0},
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n=======================================================")
        print(f" Running Experiment: {exp['name']}")
        print(f"=======================================================")
        
        tf.keras.utils.set_random_seed(42)
        student = build_student_model(input_shape=(200, 1), num_classes=2, learning_rate=1e-3, seed=42)
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ]
        
        if exp["type"] == "ce":
            class_weight = {0: 1.0, 1: exp["weight"]}
            student.fit(
                X_tr_in, y_tr,
                validation_data=(X_va_in, y_va),
                epochs=15,
                batch_size=256,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=1
            )
        else: # focal loss
            loss_fn = make_sparse_focal_loss(alpha_pos=exp["alpha"], gamma=exp["gamma"])
            student.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=loss_fn,
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
            )
            student.fit(
                X_tr_in, y_tr,
                validation_data=(X_va_in, y_va),
                epochs=15,
                batch_size=256,
                callbacks=callbacks,
                verbose=1
            )
            
        # Predict Validation & Test
        va_probs = student.predict(X_va_in, batch_size=256, verbose=0)[:, 1]
        te_probs = student.predict(X_te_in, batch_size=256, verbose=0)[:, 1]
        
        # Evaluate at standard tau=0.50 and calibrated tau=0.35
        for tau in [0.50, 0.35]:
            va_m = evaluate_window_and_event_metrics(y_va, va_probs, threshold=tau, hours=val_hours)
            te_m = evaluate_window_and_event_metrics(y_te, te_probs, threshold=tau, hours=test_hours)
            
            entry = {
                "name": exp["name"],
                "tau": tau,
                "val_win_rec": va_m["rec"],
                "val_win_prec": va_m["prec"],
                "val_ev_rec": va_m["event_recall"],
                "val_ev_fp_hr": va_m["false_ep_per_hr"],
                "test_win_rec": te_m["rec"],
                "test_win_prec": te_m["prec"],
                "test_ev_rec": te_m["event_recall"],
                "test_ev_fp_hr": te_m["false_ep_per_hr"],
                "test_ev_detected": te_m["detected_episodes"],
                "test_ev_total": te_m["gt_episodes"],
            }
            results.append(entry)
            print(f"[@ tau={tau:.2f}] VAL Event Recall: {va_m['event_recall']:.2%}, False Ep/hr: {va_m['false_ep_per_hr']:.1f} | TEST Event Recall: {te_m['event_recall']:.2%}, False Ep/hr: {te_m['false_ep_per_hr']:.1f}")

    # Print Full Comparative Table
    print("\n\n" + "="*85)
    print("FINAL COMPARISON TABLE: WINDOW-LEVEL VS EVENT-LEVEL PERFORMANCE")
    print("="*85)
    header = f"{'Experiment':<24} | {'tau':<4} | {'Win Rec':<7} | {'Win Prec':<8} | {'EVENT RECALL':<12} | {'False Ep/hr':<11} | {'Constraint <=150'}"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "PASS" if r['test_ev_fp_hr'] <= 150 else "EXCEEDED"
        print(f"{r['name']:<24} | {r['tau']:<4.2f} | {r['test_win_rec']:<7.2%} | {r['test_win_prec']:<8.2%} | {r['test_ev_rec']:<12.2%} | {r['test_ev_fp_hr']:<11.1f} | {status}")


if __name__ == "__main__":
    run_experiments()
