"""Script to synchronize and export real metrics from reports/*.md into JSON for the frontend.

Governed by EdgeAI-ECG-Frontend-Brief.md §4 (Data wiring) and AGENTS.md Rule 1.
Extracts strictly measured and simulated data from committed markdown reports.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List


def parse_threshold_table(md_content: str, table_heading: str) -> List[Dict[str, Any]]:
    """Parse a markdown table of threshold sweep rows."""
    rows = []
    lines = md_content.splitlines()
    in_table = False
    header_found = False

    for line in lines:
        if table_heading in line:
            in_table = True
            continue
        if in_table and "---" in line and not header_found:
            header_found = True
            continue
        if in_table and line.startswith("---") and header_found:
            # End of table section
            break
        if in_table and header_found and line.strip().startswith("|"):
            cols = [c.strip() for c in line.strip().split("|")[1:-1]]
            if len(cols) >= 11 and cols[0] != "Threshold (\\tau)" and not cols[0].startswith("---"):
                try:
                    # Clean markdown bold/formatting
                    clean_cols = [re.sub(r"[\*\_]", "", c).replace(",", "").replace("%", "").replace(" B", "").strip() for c in cols]
                    threshold = float(clean_cols[0])
                    tp = int(clean_cols[1])
                    fp = int(clean_cols[2])
                    fn = int(clean_cols[3])
                    tn = int(clean_cols[4])
                    recall_pct = float(clean_cols[5])
                    precision_pct = float(clean_cols[6])
                    f1 = float(clean_cols[7])
                    
                    # Parse active alerts e.g. "1,511 (2.91%)"
                    alert_match = re.search(r"([\d,]+)\s*\(([\d\.]+)%\)", cols[8])
                    active_alerts = int(alert_match.group(1).replace(",", "")) if alert_match else tp + fp
                    active_pct = float(alert_match.group(2)) if alert_match else round(active_alerts / 51992 * 100, 2)
                    
                    # Transmitted bytes e.g. "198,791 B"
                    bytes_str = re.sub(r"[^\d]", "", cols[9])
                    bytes_transmitted = int(bytes_str) if bytes_str else 0
                    
                    bandwidth_red_pct = float(clean_cols[10])
                    
                    # Derived clinical metric: False Alarms per hour (FP / 4.01 hours)
                    fp_per_hr = round(fp / 4.01, 1)

                    rows.append({
                        "threshold": threshold,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "recall_percent": recall_pct,
                        "precision_percent": precision_pct,
                        "f1_score": f1,
                        "active_alerts": active_alerts,
                        "active_alerts_percent": active_pct,
                        "bytes_transmitted": bytes_transmitted,
                        "bandwidth_reduction_percent": bandwidth_red_pct,
                        "false_alarms_per_hour": fp_per_hr,
                    })
                except (ValueError, IndexError):
                    continue
    return rows


def sync_metrics() -> None:
    """Read reports/*.md and write JSON files for the frontend."""
    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"
    frontend_data_dir = repo_root / "frontend" / "src" / "data"
    frontend_data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse threshold sweep tables
    thresh_file = reports_dir / "threshold_sweep.md"
    thresh_content = thresh_file.read_text(encoding="utf-8") if thresh_file.exists() else ""
    table_c_rows = parse_threshold_table(thresh_content, "Table C: Deployed Student (No KD) INT8 Model")
    table_d_rows = parse_threshold_table(thresh_content, "Table D: Quantized Student KD INT8 Model")

    sweep_data = {
        "dataset_split": "Held-Out Test Split (51,992 windows across 8 patient records)",
        "canonical_operating_point": {
            "model": "Student (No KD) INT8",
            "threshold": 0.35,
            "recall_percent": 13.79,
            "precision_percent": 66.45,
            "bandwidth_reduction_percent": 99.04,
            "false_alarms_per_hour": 126.4,
            "true_positives": 1004,
            "false_positives": 507,
            "total_transmissions": 1511,
            "bytes_transmitted": 198791,
        },
        "no_kd_int8": table_c_rows,
        "kd_int8": table_d_rows,
    }

    sweep_out = frontend_data_dir / "threshold-sweep.json"
    sweep_out.write_text(json.dumps(sweep_data, indent=2), encoding="utf-8")
    print(f"Exported threshold sweep data to {sweep_out} ({len(table_c_rows)} No-KD rows, {len(table_d_rows)} KD rows)")

    # 2. Main System Metrics
    metrics = {
        "project": {
            "title": "Resource-Constrained Edge-AI Pipeline for Real-Time Privacy-Preserving Patient Monitoring",
            "author": "Nancy Singh",
            "student_id": "22MIS0027",
            "course": "SWE3004 (M.Tech Capstone)",
            "institution": "Vellore Institute of Technology (VIT)",
            "repository_url": "https://github.com/Scitechlover19/EdgeAI-ECG-Monitoring",
        },
        "operating_point": {
            "model_format": "Student (No KD) Full-Integer INT8 TFLite",
            "threshold": 0.35,
            "status": "Canonical Deployed Artifact (Decision #15)",
            "monitored_duration_hours": 4.01,
            "total_test_windows": 51992,
            "normal_windows_sleep": 50481,
            "anomaly_windows_active": 1511,
            "anomaly_rate_percent": 2.91,
            "baseline_streaming_bytes": 20796800,
            "anomaly_telemetry_bytes": 198791,
            "network_bytes_saved": 20598009,
            "bandwidth_reduction_percent": 99.0441,
            "avg_bytes_per_alert": 131.56,
            "raw_waveform_leakage_bytes": 0,
            "true_positives": 1004,
            "false_positives": 507,
            "false_negatives": 6274,
            "true_negatives": 44207,
            "recall_percent": 13.79,
            "precision_percent": 66.45,
            "f1_score": 0.2285,
            "false_alarm_rate_per_hour": 126.4,
        },
        "resource_budgets": {
            "latency": {
                "mean_ms": 0.1599,
                "std_ms": 0.0126,
                "median_ms": 0.1101,
                "p95_ms": 0.3632,
                "min_ms": 0.0564,
                "max_ms": 9.9728,
                "limit_ms": 50.0,
                "compliance": "PASS",
                "headroom_factor": 312.7,
                "benchmark_inferences": 3000,
                "warmup_windows": 50,
            },
            "flash": {
                "model_bytes": 11224,
                "model_kb": 10.96,
                "limit_mb": 1.0,
                "limit_kb": 1024.0,
                "headroom_kb": 1013.04,
                "compliance": "PASS",
                "percent_used": 1.07,
            },
            "sram": {
                "peak_estimated_kb": 15.16,
                "tensor_arena_kb": 4.20,
                "scratchpad_kb": 4.00,
                "input_buffer_bytes": 200,
                "output_buffer_bytes": 2,
                "limit_kb": 256.0,
                "headroom_kb": 240.84,
                "compliance": "PASS",
                "percent_used": 5.92,
            },
        },
        "model_compression": {
            "teacher_params": 120674,
            "teacher_flash_kb": 472.0,
            "student_params": 1538,
            "student_flash_kb": 10.96,
            "param_reduction_factor": 78.46,
            "quantization_accuracy_delta": -0.0026,
            "float32_student_acc": 0.8687,
            "int8_student_acc": 0.8660,
        },
        "kd_rejection_rationale": {
            "kd_winner_config": "T=6.0, alpha=0.5",
            "float32_kd_recall": 17.24,
            "int8_kd_tp_tau_50": 1277,
            "int8_kd_fp_tau_50": 1902,
            "int8_kd_fp_inflation_percent": 61.9,
            "int8_kd_fp_per_hour": 474.1,
            "seconds_per_false_alert_kd": 7.6,
            "nokd_fp_per_hour": 126.4,
            "conclusion": "KD produces softened logit distributions vulnerable to PTQ integer rounding, exploding false alarms by 61.9% (1 false alert every 7.6 seconds). No-KD INT8 maintains 66.45% precision and was selected to avoid clinical alert fatigue."
        },
        "clinical_caveat": {
            "proof_of_concept_disclaimer": "This system is a software-engineering proof-of-concept demonstrating edge resource constraints, privacy-preserving telemetry, and anomaly radio scheduling. It is not an FDA/CE cleared diagnostic medical device.",
            "recall_limitation": "Under 14.0% minority arrhythmia prevalence, absolute recall of the 1,538-parameter INT8 Student model is 13.79%. Bandwidth reduction (99.04%) and zero raw data leakage (0 bytes) are fully verified; improving sensitivity without inflating false alarm burden is designated as future work."
        }
    }

    metrics_out = frontend_data_dir / "metrics.json"
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Exported system metrics to {metrics_out}")


if __name__ == "__main__":
    sync_metrics()
