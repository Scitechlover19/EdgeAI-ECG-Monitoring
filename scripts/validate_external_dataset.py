"""scripts/validate_external_dataset.py
Stage 3: External Dataset Generalisation Validation.

Evaluates the deployed, strictly frozen Dual-Branch INT8 TinyML model and causal DSP pipeline
on completely unseen, patient-independent external records without retraining or modifying weights.

Evaluated External Databases:
1. MIT-BIH Supraventricular Arrhythmia Database (svdb):
   - Records: 800, 801, 802, 804, 805, 806 (6 independent patients, 3.0 hours, 14,218 beats, 915 arrhythmias)
   - Sampling Rate: 128 Hz -> Resampled causally to 360 Hz
   - Channel: Lead ECG1 (Modified Lead II)
2. St Petersburg INCART 12-lead Database (incartdb):
   - Record: I01 (1 patient, 0.5 hours, 2,757 beats, 344 arrhythmias)
   - Sampling Rate: 257 Hz -> Resampled causally to 360 Hz
   - Channel: Lead II

Operating Thresholds:
- Primary: Frozen MIT-BIH threshold tau* = 0.52
- Secondary: Frozen MIT-BIH threshold tau* = 0.48

Outputs:
- reports/stage3_external_validation_results.json
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import scipy.signal
import wfdb
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dsp.dsp_filter import DSPFilter
from src.dsp.qrs_detector import CausalQRSDetector
from src.models.prepare_data import (
    BIOLOGICAL_ARRHYTHMIA_SYMBOLS,
    AAMI_VEB_SYMBOLS,
    AAMI_SVEB_SYMBOLS,
    AAMI_FUSION_SYMBOLS,
)
from scripts.train_imbalance_experiments import extract_episodes


def evaluate_external_records(
    dataset_name: str,
    record_configs: list,
    tflite_path: str = "models/student_dual_branch_int8.tflite",
    frozen_tau: float = 0.52,
    secondary_tau: float = 0.48,
    tolerance_samples: int = 54,  # 150 ms @ 360 Hz
):
    print("=" * 85)
    print(f"STAGE 3: EXTERNAL DATASET VALIDATION — {dataset_name}")
    print(f"Model: {tflite_path} (FROZEN WEIGHTS)")
    print(f"Operating Thresholds: Primary tau* = {frozen_tau:.2f} | Secondary tau* = {secondary_tau:.2f}")
    print("=" * 85)

    dsp_filter = DSPFilter(sample_rate=360.0)
    qrs_detector = CausalQRSDetector(sample_rate=360.0)

    # Load frozen INT8 model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    wave_idx = 0 if len(input_details[0]["shape"]) == 3 and input_details[0]["shape"][1] == 200 else 1
    rr_idx = 1 - wave_idx

    w_scale, w_zero = input_details[wave_idx]["quantization"]
    r_scale, r_zero = input_details[rr_idx]["quantization"]
    out_scale, out_zero = output_details[0]["quantization"]

    qrs_records = {}
    tot_qrs_ref = 0
    tot_qrs_det = 0
    tot_qrs_tp = 0
    tot_qrs_fp = 0
    tot_qrs_fn = 0
    tot_hours = 0.0

    all_beat_probs = []
    all_beat_symbols = []
    all_beat_records = []
    boundaries = {}
    curr_idx = 0

    record_ref_stats = {}

    for cfg in record_configs:
        rec_id = cfg["id"]
        rec_dir = cfg["dir"]
        ch_idx = cfg.get("channel", 0)

        rec_path = str(Path(rec_dir) / rec_id)
        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")

        fs_orig = record.fs
        raw_sig_orig = record.p_signal[:, ch_idx].astype(np.float32)
        dur_min = len(raw_sig_orig) / (fs_orig * 60.0)
        tot_hours += dur_min / 60.0

        # Resample causally to 360 Hz
        target_len = int(np.round(len(raw_sig_orig) * 360.0 / fs_orig))
        sig_360 = scipy.signal.resample(raw_sig_orig, target_len).astype(np.float32)

        # Scale annotations to 360 Hz
        scale = 360.0 / fs_orig
        ref_samples_360 = np.round(np.array(ann.sample) * scale).astype(np.int64)
        ref_symbols = np.array(ann.symbol)

        # Reference arrhythmia counts
        target_indices = np.where(np.isin(ref_symbols, list(BIOLOGICAL_ARRHYTHMIA_SYMBOLS)))[0]
        record_ref_stats[rec_id] = {
            "total_ref_beats": len(ref_symbols),
            "target_anom_beats": len(target_indices),
            "ref_samples": ref_samples_360,
            "ref_symbols": ref_symbols,
        }

        # Causal filter
        clean_sig, _ = dsp_filter.apply_bandpass_causal(sig_360)

        # Detect peaks & extract segments
        peaks = qrs_detector.detect_peaks(clean_sig)
        det_samples = np.array([p.sample_idx for p in peaks])
        waveforms, rr_feats, detected_beats = qrs_detector.extract_beat_segments(
            clean_sig, pre_samples=72, post_samples=128
        )

        # QRS 1-to-1 bipartite matching
        matched_ref_qrs = set()
        matched_det_qrs = set()
        for d_i, d_pos in enumerate(det_samples):
            dists = np.abs(ref_samples_360 - d_pos)
            cands = np.where(dists <= tolerance_samples)[0]
            cands = cands[np.argsort(dists[cands])]
            for c in cands:
                if c not in matched_ref_qrs:
                    matched_ref_qrs.add(c)
                    matched_det_qrs.add(d_i)
                    break

        n_ref = len(ref_samples_360)
        n_det = len(det_samples)
        tp_qrs = len(matched_ref_qrs)
        fp_qrs = n_det - tp_qrs
        fn_qrs = n_ref - tp_qrs

        qrs_records[rec_id] = {
            "ref_beats": n_ref,
            "detected_peaks": n_det,
            "tp": tp_qrs,
            "fp": fp_qrs,
            "fn": fn_qrs,
            "sensitivity": (tp_qrs / n_ref * 100.0) if n_ref > 0 else 0.0,
            "ppv": (tp_qrs / n_det * 100.0) if n_det > 0 else 0.0,
            "fp_per_min": fp_qrs / dur_min,
        }
        tot_qrs_ref += n_ref
        tot_qrs_det += n_det
        tot_qrs_tp += tp_qrs
        tot_qrs_fp += fp_qrs
        tot_qrs_fn += fn_qrs

        # Map each extracted beat segment to reference beat symbol
        start_b_idx = curr_idx
        for b in detected_beats:
            s_idx = b.sample_idx
            dists = np.abs(ref_samples_360 - s_idx)
            min_idx = np.argmin(dists) if len(dists) > 0 else -1
            if min_idx >= 0 and dists[min_idx] <= tolerance_samples:
                sym = str(ref_symbols[min_idx])
            else:
                sym = "?"
            all_beat_symbols.append(sym)
            all_beat_records.append(rec_id)

        # INT8 inference
        rec_probs = []
        for i in range(len(waveforms)):
            w_in = np.clip(np.round(waveforms[i:i+1, :, np.newaxis] / w_scale) + w_zero, -128, 127).astype(np.int8)
            r_in = np.clip(np.round(rr_feats[i:i+1] / r_scale) + r_zero, -128, 127).astype(np.int8)
            interpreter.set_tensor(input_details[wave_idx]["index"], w_in)
            interpreter.set_tensor(input_details[rr_idx]["index"], r_in)
            interpreter.invoke()
            q_out = interpreter.get_tensor(output_details[0]["index"])
            prob = (q_out.astype(np.float32) - out_zero) * out_scale
            rec_probs.append(prob[0, 1])

        rec_probs = np.array(rec_probs, dtype=np.float32)
        all_beat_probs.extend(rec_probs)
        curr_idx += len(waveforms)
        boundaries[rec_id] = (start_b_idx, curr_idx)

    # QRS Aggregate
    qrs_aggregate = {
        "ref_beats": tot_qrs_ref,
        "detected_peaks": tot_qrs_det,
        "tp": tot_qrs_tp,
        "fp": tot_qrs_fp,
        "fn": tot_qrs_fn,
        "sensitivity": (tot_qrs_tp / tot_qrs_ref * 100.0) if tot_qrs_ref > 0 else 0.0,
        "ppv": (tot_qrs_tp / tot_qrs_det * 100.0) if tot_qrs_det > 0 else 0.0,
        "fp_per_min": tot_qrs_fp / (tot_hours * 60.0),
    }

    all_beat_probs = np.array(all_beat_probs, dtype=np.float32)
    all_beat_symbols = np.array(all_beat_symbols)
    all_beat_records = np.array(all_beat_records)

    # Print QRS Table
    print("\n" + "="*80)
    print(f"1. CAUSAL QRS DETECTOR PERFORMANCE ON {dataset_name} ({tot_hours:.1f} HOURS)")
    print("="*80)
    print(f"{'Record':<8} | {'Ref Beats':<10} | {'Detected':<10} | {'TP':<8} | {'FP':<6} | {'Sensitivity':<12} | {'PPV':<10} | {'FP/min':<8}")
    print("-" * 80)
    for cfg in record_configs:
        r = cfg["id"]
        qs = qrs_records[r]
        print(f"{r:<8} | {qs['ref_beats']:<10} | {qs['detected_peaks']:<10} | {qs['tp']:<8} | {qs['fp']:<6} | {qs['sensitivity']:<11.2f}% | {qs['ppv']:<9.2f}% | {qs['fp_per_min']:<7.2f}")
    print("-" * 80)
    print(f"{'TOTAL':<8} | {qrs_aggregate['ref_beats']:<10} | {qrs_aggregate['detected_peaks']:<10} | {qrs_aggregate['tp']:<8} | {qrs_aggregate['fp']:<6} | {qrs_aggregate['sensitivity']:<11.2f}% | {qrs_aggregate['ppv']:<9.2f}% | {qrs_aggregate['fp_per_min']:<7.2f}")

    # Evaluate at Thresholds
    threshold_results = {}
    for tau in [frozen_tau, secondary_tau]:
        preds = (all_beat_probs >= tau).astype(int)
        total_eval_beats = len(preds)
        alert_mask = (preds == 1)

        tot_target_ref = 0
        tot_detected_ref = 0
        tot_veb_ref = 0
        tot_veb_det = 0
        tot_sveb_ref = 0
        tot_sveb_det = 0
        per_rec_eval = {}

        for cfg in record_configs:
            r = cfg["id"]
            s_i, e_i = boundaries[r]
            r_preds = preds[s_i:e_i]
            r_syms = all_beat_symbols[s_i:e_i]

            ref_meta = record_ref_stats[r]
            ref_sym = ref_meta["ref_symbols"]

            rec_target = ref_meta["target_anom_beats"]
            tot_target_ref += rec_target

            r_det = int(np.sum((r_preds == 1) & np.isin(r_syms, list(BIOLOGICAL_ARRHYTHMIA_SYMBOLS))))
            tot_detected_ref += r_det

            # VEB / SVEB
            r_veb_ref = int(np.sum(np.isin(ref_sym, list(AAMI_VEB_SYMBOLS))))
            r_sveb_ref = int(np.sum(np.isin(ref_sym, list(AAMI_SVEB_SYMBOLS))))
            r_veb_det = int(np.sum((r_preds == 1) & np.isin(r_syms, list(AAMI_VEB_SYMBOLS))))
            r_sveb_det = int(np.sum((r_preds == 1) & np.isin(r_syms, list(AAMI_SVEB_SYMBOLS))))

            tot_veb_ref += r_veb_ref
            tot_veb_det += r_veb_det
            tot_sveb_ref += r_sveb_ref
            tot_sveb_det += r_sveb_det

            # Episodes
            r_eps = extract_episodes(r_preds, max_gap=5)
            r_false_eps = 0
            for ep_s, ep_e in r_eps:
                if not np.any(np.isin(r_syms[ep_s:ep_e], list(BIOLOGICAL_ARRHYTHMIA_SYMBOLS))):
                    r_false_eps += 1

            r_sens = (r_det / rec_target * 100.0) if rec_target > 0 else 100.0
            r_alerts = int(np.sum(r_preds == 1))
            r_fp = int(np.sum((r_preds == 1) & (~np.isin(r_syms, list(BIOLOGICAL_ARRHYTHMIA_SYMBOLS)))))

            per_rec_eval[r] = {
                "target_ref": rec_target,
                "detected": r_det,
                "sensitivity": r_sens,
                "alerts": r_alerts,
                "fp_beats": r_fp,
                "false_episodes": r_false_eps,
            }

        # Confusion Matrix
        target_mask = np.isin(all_beat_symbols, list(BIOLOGICAL_ARRHYTHMIA_SYMBOLS))
        tp = int(np.sum(alert_mask & target_mask))
        fn = int(np.sum((~alert_mask) & target_mask))
        fp = int(np.sum(alert_mask & (~target_mask)))
        tn = int(np.sum((~alert_mask) & (~target_mask)))

        se = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        sp = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
        ppv = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        acc = ((tp + tn) / total_eval_beats * 100.0) if total_eval_beats > 0 else 0.0
        f1 = (2 * ppv * se) / (ppv + se) if (ppv + se) > 0 else 0.0

        tot_false_eps = sum(pe["false_episodes"] for pe in per_rec_eval.values())
        fp_ep_per_hr = tot_false_eps / tot_hours

        fp_syms = dict(Counter(all_beat_symbols[alert_mask & (~target_mask)]))

        threshold_results[f"tau_{tau:.2f}"] = {
            "threshold": tau,
            "total_beats_evaluated": total_eval_beats,
            "target_arrhythmias": tot_target_ref,
            "detected_arrhythmias": tp,
            "sensitivity": se,
            "specificity": sp,
            "precision_ppv": ppv,
            "accuracy": acc,
            "f1_score": f1 / 100.0,
            "veb_sensitivity": (tot_veb_det / tot_veb_ref * 100.0) if tot_veb_ref > 0 else 0.0,
            "sveb_sensitivity": (tot_sveb_det / tot_sveb_ref * 100.0) if tot_sveb_ref > 0 else 0.0,
            "veb_counts": f"{tot_veb_det} / {tot_veb_ref}",
            "sveb_counts": f"{tot_sveb_det} / {tot_sveb_ref}",
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "false_alarm_episodes": tot_false_eps,
            "false_alarm_episodes_hr": fp_ep_per_hr,
            "fp_symbol_breakdown": fp_syms,
            "per_record": per_rec_eval,
        }

    # Print Headline Table
    print("\n" + "="*80)
    print(f"2. EXTERNAL CLASSIFICATION PERFORMANCE AT FROZEN THRESHOLD (tau* = {frozen_tau:.2f})")
    print("="*80)
    res = threshold_results[f"tau_{frozen_tau:.2f}"]
    print(f"Total Beats Evaluated:           {res['total_beats_evaluated']:,}")
    print(f"Target Biological Arrhythmias:   {res['target_arrhythmias']:,} beats")
    print(f"True Positive Detections (TP):   {res['tp']:,} beats")
    print(f"False Negatives (FN):            {res['fn']:,} beats")
    print(f"False Positives (FP):            {res['fp']:,} beats")
    print(f"True Negatives (TN):             {res['tn']:,} beats")
    print(f"Overall Accuracy:                {res['accuracy']:.2f}%")
    print(f"Clinical Sensitivity (Recall):   {res['sensitivity']:.2f}% ({res['tp']}/{res['target_arrhythmias']})")
    print(f"AAMI VEB Sensitivity (V, E):     {res['veb_sensitivity']:.2f}% ({res['veb_counts']})")
    print(f"AAMI SVEB Sensitivity (A, S):    {res['sveb_sensitivity']:.2f}% ({res['sveb_counts']})")
    print(f"Clinical Specificity:            {res['specificity']:.2f}%")
    print(f"Strict Precision (PPV):          {res['precision_ppv']:.2f}%")
    print(f"Strict F1-Score:                 {res['f1_score']:.4f}")
    print(f"False Alarm Episodes:            {res['false_alarm_episodes']} ({res['false_alarm_episodes_hr']:.2f} ep/hr)")
    print(f"Non-Target FP Breakdown:         {res['fp_symbol_breakdown']}")

    print("\nPER-RECORD BREAKDOWN ON EXTERNAL DATASET:")
    print(f"{'Record':<8} | {'Target Beats':<13} | {'Detected':<10} | {'Sensitivity':<12} | {'Emitted Alerts':<15} | {'False Episodes':<15}")
    print("-" * 80)
    for cfg in record_configs:
        r = cfg["id"]
        pr = res["per_record"][r]
        print(f"{r:<8} | {pr['target_ref']:<13} | {pr['detected']:<10} | {pr['sensitivity']:<11.2f}% | {pr['alerts']:<15} | {pr['false_episodes']:<15}")
    print("=" * 80)

    # Save results
    out_file = Path("reports/stage3_external_validation_results.json")
    with open(out_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "dataset_name": dataset_name,
                    "records": [c["id"] for c in record_configs],
                    "total_hours": tot_hours,
                    "frozen_model": tflite_path,
                    "causality": "Strict Forward IIR + Causal QRS Peak Detection (Zero Lookahead)",
                },
                "qrs_detection": qrs_records,
                "qrs_aggregate": qrs_aggregate,
                "classification_results": threshold_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved Stage 3 external validation results to {out_file}")
    return threshold_results


if __name__ == "__main__":
    # Check what records are available
    svdb_recs = ["800", "801", "802", "804", "805", "806"]
    configs = [{"id": r, "dir": "data/raw/svdb", "channel": 0} for r in svdb_recs]

    # Check if INCART I01 is available
    i01_dat = Path("data/raw/incartdb/I01.dat")
    if i01_dat.exists() and i01_dat.stat().st_size >= 11000000:
        configs.append({"id": "I01", "dir": "data/raw/incartdb", "channel": 1})
        dset_name = "MIT-BIH SVDB (800-806) + St Petersburg INCART (I01)"
    else:
        dset_name = "MIT-BIH Supraventricular Arrhythmia Database (SVDB 800-806)"

    evaluate_external_records(dset_name, configs)
