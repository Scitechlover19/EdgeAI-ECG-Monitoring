# Knowledge Distillation (KD) Ablation & Hyperparameter Sweep Report (P0-7b)

**Date:** 2026-09-20  
**Dataset:** PhysioNet MIT-BIH Arrhythmia Database (`mitdb`, 311,952 windows total)  
**Split Method:** Patient/Record-Independent Split (DECISIONS.md #3: 33 Train, 7 Val, 8 Test records)  
**Validation Split:** 7 records (`102`, `111`, `124`, `200`, `215`, `217`, `223` — 45,493 windows, 11,385 anomalies = 25.03% prevalence) `[MEASURED]`  
**Held-Out Test Split:** 8 records (`107`, `115`, `119`, `122`, `207`, `220`, `228`, `234` — 51,992 windows, 7,278 anomalies = 14.00% prevalence) `[MEASURED]`  

---

## 1. Executive Summary & Selection Methodology

This report documents the empirical comparison between the uncompressed Teacher 1D-CNN, the Student 1D-CNN trained without KD, and Student models trained with Knowledge Distillation across temperature $T \in \{2.0, 4.0, 6.0\}$ and weight balance $\alpha \in \{0.3, 0.5\}$.

> [!IMPORTANT]
> **Strict Leakage-Free Model Selection Methodology:**
> - **Hyperparameter Selection Set:** Hyperparameter selection was performed **strictly on the 45,493 validation windows**. The held-out test split was NOT used to pick winning hyperparameters.
> - **Selection Criterion:** Selection optimized for maximum minority-class arrhythmia F1 score and recall on the validation set under class imbalance.
> - **Winning Configuration:** On the validation split, **$T=6.0, \alpha=0.5$** achieved the highest validation performance:
>   - **Validation F1 Score:** **0.7942** (vs. 0.7882 No-KD, 0.7708 Teacher)
>   - **Validation Recall:** **67.63%** (7,700 true positive detections out of 11,385 validation anomalies)
>   - **Validation Accuracy:** **91.23%** (vs. 91.08% No-KD)
> - **Confirmatory Held-Out Test Read:** Evaluating this validation-selected model on the held-out test split confirmed:
>   - **Test Accuracy:** **86.16%**
>   - **Test Precision:** **51.65%**
>   - **Test Recall:** **17.24%** (1,255 true positive detections vs. 924 for No-KD, +35.8% relative gain)
>   - **Test F1 Score:** **0.2585** (vs. 0.2130 for No-KD, +21.4% relative gain)

---

## 2. Validation Split Selection Table (45,493 windows, 11,385 anomalies) `[MEASURED]`

The table below documents the hyperparameter sweep evaluated strictly on the validation set for model selection:

| Model Configuration | Parameters | Hyperparameters | Val Accuracy `[MEASURED]` | Val Precision `[MEASURED]` | Val Recall `[MEASURED]` | Val F1 Score `[MEASURED]` | Val TP | Val FP | Val FN | Val TN |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| **Teacher 1D-CNN** | 120,674 | — | 90.29% | 94.19% | 65.23% | 0.7708 | 7,426 | 458 | 3,959 | 33,650 |
| **Student (No KD)** | 1,538 | — | 91.08% | **97.09%** | 66.34% | 0.7882 | 7,553 | 226 | 3,832 | 33,882 |
| **Student (KD Sweep 1)** | 1,538 | $T=2.0, \alpha=0.3$ | 90.58% | 97.02% | 63.65% | 0.7718 | 7,246 | 223 | 4,139 | 33,885 |
| **Student (KD Sweep 2)** | 1,538 | $T=2.0, \alpha=0.5$ | 90.87% | 96.88% | 65.83% | 0.7849 | 7,495 | 242 | 3,890 | 33,866 |
| **Student (KD Sweep 3)** | 1,538 | $T=4.0, \alpha=0.3$ | 88.94% | 96.42% | 57.06% | 0.7228 | 6,496 | 241 | 4,889 | 33,867 |
| **Student (KD Sweep 4)** | 1,538 | $T=4.0, \alpha=0.5$ | 90.64% | 96.69% | 63.99% | 0.7724 | 7,285 | 249 | 4,100 | 33,859 |
| **Student (KD Sweep 5)** | 1,538 | $T=6.0, \alpha=0.3$ | 90.41% | 96.65% | 64.24% | 0.7679 | 7,314 | 254 | 4,071 | 33,854 |
| **Student (KD Sweep 6: WINNER)** | 1,538 | **$T=6.0, \alpha=0.5$** | **91.23%** | 96.18% | **67.63%** | **0.7942** | **7,700** | 306 | 3,685 | 33,802 |

---

## 3. Held-Out Test Set Performance Breakdown (51,992 windows, 7,278 anomalies) `[MEASURED]`

Evaluated at the standard argmax/0.5 boundary on the held-out test split:

| Model | Parameters | Hyperparameters | Test Accuracy `[MEASURED]` | Test Precision `[MEASURED]` | Test Recall `[MEASURED]` | Test F1 Score `[MEASURED]` | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| **Teacher 1D-CNN** | 120,674 | — | 84.09% | 36.05% | 17.64% | 0.2369 | 1,284 | 2,278 | 5,994 | 42,436 |
| **Student (No KD)** | 1,538 | — | **86.87%** | **66.09%** | 12.70% | 0.2130 | 924 | 474 | 6,354 | 44,240 |
| **Student KD (Validation-Selected)** | 1,538 | **$T=6.0, \alpha=0.5$** | 86.16% | 51.65% | **17.24%** | **0.2585** | **1,255** | 1,175 | 6,023 | 43,539 |

---

## 4. Metrics @ Scheduler Threshold 0.85 (Exact StateScheduler Operating Rule) `[MEASURED]`

This table evaluates each model candidate on the held-out test split using the exact decision rule enforced by `StateScheduler`: **anomaly transmission triggered if and only if $\text{confidence} \ge 0.85$**:

| Model Candidate | TP | FP | FN | TN | Test Accuracy `[MEASURED]` | Test Recall `[MEASURED]` | Test Precision `[MEASURED]` | Test F1 Score `[MEASURED]` | Active Alerts `[MEASURED]` | Bandwidth Reduction `[MEASURED]` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Teacher 1D-CNN** | 962 | 1,928 | 6,316 | 42,786 | 84.14% | 13.22% | 33.29% | 0.1892 | 2,890 (5.56%) | 98.16% |
| **Student (No KD)** | 104 | 135 | 7,174 | 44,579 | 85.94% | 1.43% | 43.51% | 0.0277 | 239 (0.46%) | 99.85% |
| **Student KD ($T=6.0, \alpha=0.5$)** | 86 | 113 | 7,192 | 44,601 | 85.95% | 1.18% | 43.22% | 0.0230 | 199 (0.38%) | 99.87% |
| **Student INT8 TFLite** | 49 | 88 | 7,229 | 44,626 | 85.93% | **0.67%** | 35.77% | **0.0132** | **137** (0.26%) | **99.91%** |

> [!WARNING]
> **Key Finding on 0.85 Threshold:**
> At threshold 0.85, the deployed INT8 TFLite model detects only **49 true positives** out of 7,278 ground-truth anomalies (**0.67% recall**), with 88 false alarms, yielding exactly the 137 active transmissions recorded in `reports/bandwidth_report.md`.
> Setting threshold 0.85 sacrifices 99.33% of arrhythmias to achieve 99.91% bandwidth reduction.
> As demonstrated in [`reports/threshold_sweep.md`](file:///reports/threshold_sweep.md), operating at $\tau \in [0.40, 0.50]$ allows detecting 850–1,250 anomalies (12%–17% recall) while comfortably achieving **98.4%–99.2% bandwidth reduction**, well above the 90.0% NFR-6 ceiling.

---

## 5. Distillation Loss Formulation

$$\mathcal{L}_{\text{total}} = (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, p_s) + \alpha \cdot T^2 \cdot \mathcal{L}_{\text{KL}}\left(\text{Softmax}\left(\frac{z_t}{T}\right), \text{Softmax}\left(\frac{z_s}{T}\right)\right)$$

- **Teacher Model:** Frozen (`teacher.trainable = False`) during distillation.
- **Student Model:** 1,538 parameters (same compact architecture across all experiments).
- **Optimization:** Adam ($\text{lr}=10^{-3}$, batch size 256).

---

## 6. Reproducibility Command

To reproduce the validation selection and threshold evaluation:
```bash
python -m src.models.run_validation_and_threshold_sweep
```
