"""scripts/audit_alerts_and_qrs.py
Deep audit of:
1. Exact QRS detection metrics across all 8 test records (1-to-1 bipartite matching).
2. Exact composition of every emitted alert beat (TP vs FP vs non-target annotations).
3. Complete standard confusion matrix (TP, FP, TN, FN, Sensitivity, Specificity, Precision, F1).
4. False alarm episodes under the 5-beat hysteresis merging rule.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import wfdb
import tensorflow as tf

from src.dsp.dsp_filter import DSPFilter
from src.dsp.qrs_detector import CausalQRSDetector
from src.models.prepare_data import (
    select_lead_channel,
    BIOLOGICAL_ARRHYTHMIA_SYMBOLS,
    AAMI_VEB_SYMBOLS,
    AAMI_SVEB_SYMBOLS,
    AAMI_FUSION_SYMBOLS,
    AAMI_PACED_UNKNOWN_SYMBOLS,
)
from scripts.train_imbalance_experiments import extract_episodes

def audit_qrs_detector(test_recs, raw_mitdb_dir=Path("data/raw/mitdb"), tolerance_samples=54):
    print("="*80)
    print("AUDIT 1: EXACT CAUSAL QRS DETECTOR VALIDATION (1-to-1 MATCHING)")
    print("="*80)
    dsp_filter = DSPFilter(sample_rate=360.0)
    qrs_detector = CausalQRSDetector(sample_rate=360.0)

    qrs_results = {}
    tot_ref = 0
    tot_det = 0
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    tot_duration_min = 0.0

    for rec_id in test_recs:
        rec_path = str(raw_mitdb_dir / rec_id)
        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")

        ch_idx, _ = select_lead_channel(record)
        raw_signal = record.p_signal[:, ch_idx].astype(np.float32)
        duration_min = len(raw_signal) / (360.0 * 60.0)
        tot_duration_min += duration_min

        clean_signal, _ = dsp_filter.apply_bandpass_causal(raw_signal)
        detected_peaks = qrs_detector.detect_peaks(clean_signal)

        ref_samples = np.array(ann.sample)
        det_samples = np.array([p.sample_idx for p in detected_peaks])

        # 1-to-1 bipartite matching
        matched_ref = set()
        matched_det = set()

        for d_i, d_pos in enumerate(det_samples):
            dists = np.abs(ref_samples - d_pos)
            cand = np.where(dists <= tolerance_samples)[0]
            cand = cand[np.argsort(dists[cand])]
            for c in cand:
                if c not in matched_ref:
                    matched_ref.add(c)
                    matched_det.add(d_i)
                    break

        n_ref = len(ref_samples)
        n_det = len(det_samples)
        tp = len(matched_ref)
        fp = n_det - tp
        fn = n_ref - tp

        se = (tp / n_ref * 100.0) if n_ref > 0 else 0.0
        ppv = (tp / n_det * 100.0) if n_det > 0 else 0.0
        fp_rate = fp / duration_min

        qrs_results[rec_id] = {
            "ref_beats": n_ref,
            "detected_peaks": n_det,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "sensitivity": se,
            "ppv": ppv,
            "fp_per_min": fp_rate,
            "duration_min": duration_min,
        }

        tot_ref += n_ref
        tot_det += n_det
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

        print(f"Record {rec_id:<4} | Ref: {n_ref:<5} | Det: {n_det:<5} | TP: {tp:<5} | FP: {fp:<4} | FN: {fn:<4} | Se: {se:6.2f}% | PPV: {ppv:6.2f}% | FP/min: {fp_rate:5.2f}")

    tot_se = (tot_tp / tot_ref * 100.0) if tot_ref > 0 else 0.0
    tot_ppv = (tot_tp / tot_det * 100.0) if tot_det > 0 else 0.0
    tot_fp_rate = tot_fp / tot_duration_min

    qrs_results["TOTAL"] = {
        "ref_beats": tot_ref,
        "detected_peaks": tot_det,
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "sensitivity": tot_se,
        "ppv": tot_ppv,
        "fp_per_min": tot_fp_rate,
        "duration_min": tot_duration_min,
    }
    print("-" * 80)
    print(f"TOTAL       | Ref: {tot_ref:<5} | Det: {tot_det:<5} | TP: {tot_tp:<5} | FP: {tot_fp:<4} | FN: {tot_fn:<4} | Se: {tot_se:6.2f}% | PPV: {tot_ppv:6.2f}% | FP/min: {tot_fp_rate:5.2f}")
    return qrs_results


def audit_alert_composition(test_recs, tflite_path="models/student_dual_branch_int8.tflite"):
    print("\n" + "="*80)
    print("AUDIT 2: EXACT COMPOSITION OF DETECTED BEATS & PREDICTED ALERTS")
    print("="*80)

    # Load test data
    data_dir = Path("data/processed_beats_causal")
    X_w = np.load(data_dir / "X_wave_test.npy")
    X_rr = np.load(data_dir / "X_rr_test.npy")
    y_te = np.load(data_dir / "y_test.npy")
    meta = np.load(data_dir / "meta_test.npz")
    boundaries = json.load(open(data_dir / "boundaries_test.json"))

    beat_samples = meta["beat_samples"]
    symbols = meta["symbols"]
    record_ids = meta["record_ids"]

    # Run INT8 inference
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    wave_idx = 0 if len(input_details[0]["shape"]) == 3 and input_details[0]["shape"][1] == 200 else 1
    rr_idx = 1 - wave_idx

    w_scale, w_zero = input_details[wave_idx]["quantization"]
    r_scale, r_zero = input_details[rr_idx]["quantization"]
    out_scale, out_zero = output_details[0]["quantization"]

    probs = []
    for i in range(len(X_w)):
        w_in = np.clip(np.round(X_w[i:i+1] / w_scale) + w_zero, -128, 127).astype(np.int8)
        r_in = np.clip(np.round(X_rr[i:i+1] / r_scale) + r_zero, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[wave_idx]["index"], w_in)
        interpreter.set_tensor(input_details[rr_idx]["index"], r_in)
        interpreter.invoke()
        q_out = interpreter.get_tensor(output_details[0]["index"])
        p = (q_out.astype(np.float32) - out_zero) * out_scale
        probs.append(p[0, 1])

    probs = np.array(probs, dtype=np.float32)

    # Evaluate at F1-optimal (0.52) and F2-optimal (0.48)
    for tau in [0.52, 0.48]:
        preds = (probs >= tau).astype(int)
        print(f"\n--- EVALUATION AT THRESHOLD tau = {tau:.2f} ---")

        total_beats = len(preds)
        alert_mask = (preds == 1)
        non_alert_mask = (preds == 0)

        # What are the symbols of alert beats?
        alert_symbols = symbols[alert_mask]
        alert_recs = record_ids[alert_mask]

        from collections import Counter
        sym_counts = Counter(alert_symbols)
        print(f"Total Beats Evaluated by TinyML: {total_beats}")
        print(f"Total Emitted Alerts (Pred=1): {np.sum(alert_mask)}")
        print(f"Alert Symbol Breakdown: {dict(sym_counts)}")

        # Breakdown into:
        # 1. Target Arrhythmias (VEB + SVEB + Fusion)
        target_alerts = [s for s in alert_symbols if s in BIOLOGICAL_ARRHYTHMIA_SYMBOLS]
        # 2. Normal Sinus Beats (N, L, R)
        normal_alerts = [s for s in alert_symbols if s in {'N', 'L', 'R', 'B'}]
        # 3. Paced / Artifact / Flutter / Unclassified (/, f, F, Q, ?, !, x)
        other_alerts = [s for s in alert_symbols if s not in BIOLOGICAL_ARRHYTHMIA_SYMBOLS and s not in {'N', 'L', 'R', 'B'}]

        print(f"  - Target Arrhythmia Detections (True Positives, TP): {len(target_alerts)}")
        print(f"  - Normal Sinus Misclassifications (Normal FP): {len(normal_alerts)} {dict(Counter(normal_alerts))}")
        print(f"  - Paced / Flutter / Artifact Triggers (Non-target FP): {len(other_alerts)} {dict(Counter(other_alerts))}")

        # Check total target beats in the test set
        target_mask = np.isin(symbols, list(BIOLOGICAL_ARRHYTHMIA_SYMBOLS))
        non_target_mask = ~target_mask

        tp = np.sum(alert_mask & target_mask)
        fn = np.sum(non_alert_mask & target_mask)
        fp_total = np.sum(alert_mask & non_target_mask)
        tn = np.sum(non_alert_mask & non_target_mask)

        se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp_total) if (tn + fp_total) > 0 else 0.0
        ppv_strict = tp / (tp + fp_total) if (tp + fp_total) > 0 else 0.0
        f1_strict = (2 * ppv_strict * se) / (ppv_strict + se) if (ppv_strict + se) > 0 else 0.0

        print(f"\nStrict Binary Confusion Matrix (Arrhythmia vs. Non-Arrhythmia):")
        print(f"  TP: {tp:<6} | FN: {fn:<6} (Sensitivity / Recall: {se:.2%})")
        print(f"  FP: {fp_total:<6} | TN: {tn:<6} (Specificity: {sp:.2%})")
        print(f"  Strict Precision (PPV = TP / (TP + FP_total)): {ppv_strict:.2%}")
        print(f"  Strict F1-Score: {f1_strict:.4f}")

        # Per record breakdown of FP
        print(f"\nPer-Record Distribution of the {fp_total} False Positives:")
        for r in test_recs:
            r_mask = (record_ids == r)
            r_alert = alert_mask & r_mask
            r_target = target_mask & r_mask
            r_tp = np.sum(r_alert & r_target)
            r_fn = np.sum((~alert_mask) & r_target)
            r_fp = np.sum(r_alert & (~r_target))
            r_fp_syms = dict(Counter(symbols[r_alert & (~r_target)]))
            print(f"  Record {r:<4} | TP: {r_tp:<4} | FN: {r_fn:<4} | FP: {r_fp:<4} | Non-target symbols: {r_fp_syms}")

        # False alarm episodes with 5-beat hysteresis merging
        print(f"\nFalse Alarm Episodes (5-beat hysteresis merging rule):")
        tot_episodes = 0
        tot_false_episodes = 0
        tot_false_ep_beats = 0
        for r in test_recs:
            s_i, e_i = boundaries[r]
            r_preds = preds[s_i:e_i]
            r_targets = target_mask[s_i:e_i]
            eps = extract_episodes(r_preds, max_gap=5)
            r_false_eps = 0
            r_false_beats = 0
            for ep_s, ep_e in eps:
                tot_episodes += 1
                if not np.any(r_targets[ep_s:ep_e]):
                    r_false_eps += 1
                    r_false_beats += (ep_e - ep_s)
                    tot_false_episodes += 1
                    tot_false_ep_beats += (ep_e - ep_s)
            print(f"  Record {r:<4} | Total Alarm Episodes: {len(eps):<4} | False Alarm Episodes: {r_false_eps:<4} | False Episode Beats: {r_false_beats}")
        
        fp_ep_per_hr = tot_false_episodes / 4.01
        print(f"  TOTAL False Alarm Episodes: {tot_false_episodes} ({fp_ep_per_hr:.2f} ep/hr)")
        print(f"  TOTAL False Episode Beats: {tot_false_ep_beats} beats")

if __name__ == "__main__":
    recs = ["107", "115", "119", "122", "207", "220", "228", "234"]
    audit_qrs_detector(recs)
    audit_alert_composition(recs)
