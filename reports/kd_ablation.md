# Knowledge Distillation (KD) Ablation Report (P0-7b)

**Date:** 2026-09-19  
**Dataset:** PhysioNet MIT-BIH Arrhythmia Database (`mitdb`, 311,952 windows total)  
**Split Method:** Patient/Record-Independent Split (DECISIONS.md #3: 33 Train, 7 Val, 8 Test records)  
**Test Split:** 8 held-out records (`107`, `115`, `119`, `122`, `207`, `220`, `228`, `234` — 51,992 windows)

---

## 1. Executive Summary

This report documents the factual, empirical comparison between:
1. **Teacher 1D-CNN** (120,674 parameters)
2. **Student 1D-CNN WITHOUT KD** (1,538 parameters, standard hard cross-entropy loss)
3. **Student 1D-CNN WITH KD** (1,538 parameters, soft-target distillation loss with $T=3.0$, $\alpha=0.7$)

All three models were trained on the exact same 214,467 training windows and evaluated on the exact same 51,992 held-out test windows.

> [!IMPORTANT]
> **Factual Observation:** In this 15-epoch experiment ($T=3.0, \alpha=0.7$), Knowledge Distillation did **not** improve held-out test accuracy or F1 score over the baseline Student model trained without KD. All reported numbers below are `[MEASURED]` directly from the evaluation pipeline run.

---

## 2. Quantitative Performance Comparison

| Model | Architecture Family | Total Parameters `[MEASURED]` | Parameter Reduction `[MEASURED]` | Test Accuracy `[MEASURED]` | Test Precision `[MEASURED]` | Test Recall `[MEASURED]` | Test F1 Score `[MEASURED]` |
|---|---|---|---|---|---|---|---|
| **Teacher 1D-CNN** | 1D-CNN (3 Conv Blocks) | **120,674** | Baseline (1.0x) | **0.8409** (84.09%) | **0.3605** (36.05%) | **0.1764** (17.64%) | **0.2369** |
| **Student (No KD)** | Compact 1D-CNN | **1,538** | **78.46x smaller** | **0.8687** (86.87%) | **0.6609** (66.09%) | **0.1270** (12.70%) | **0.2130** |
| **Student (With KD)** | Compact 1D-CNN | **1,538** | **78.46x smaller** | **0.8376** (83.76%) | **0.2785** (27.85%) | **0.1006** (10.06%) | **0.1478** |

---

## 3. Confusion Matrix Breakdown (`[MEASURED]`)

| Model | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Test Windows Total |
|---|---|---|---|---|---|
| **Teacher 1D-CNN** | 1,284 | 2,278 | 5,994 | 42,436 | 51,992 |
| **Student (No KD)** | 924 | 474 | 6,354 | 44,240 | 51,992 |
| **Student (With KD)** | 732 | 1,896 | 6,546 | 42,818 | 51,992 |

---

## 4. Distillation Hyperparameters & Loss Formulation

- **Temperature ($T$):** `3.0` (DECISIONS.md #6)
- **Alpha ($\alpha$):** `0.7` (DECISIONS.md #6)
- **Loss Equation:**
  $$\mathcal{L}_{\text{total}} = (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, p_s) + \alpha \cdot T^2 \cdot \mathcal{L}_{\text{KL}}\left(\text{Softmax}\left(\frac{z_t}{T}\right), \text{Softmax}\left(\frac{z_s}{T}\right)\right)$$
- **Teacher Status:** Frozen (`teacher.trainable = False`) during Student distillation.

---

## 5. Key Findings & Observations

1. **Parameter Compression:** The Student 1D-CNN achieves a **78.46x parameter reduction** (1,538 params vs 120,674 params) while maintaining baseline test accuracy.
2. **Ablation Performance:** Student trained WITHOUT KD achieved higher test accuracy (0.8687 vs 0.8376) and F1 score (0.2130 vs 0.1478) than Student trained WITH KD at $T=3.0, \alpha=0.7$.
3. **Class Imbalance Sensitivity:** ECG anomaly detection on patient-independent held-out records exhibits high class imbalance (7,278 anomalies out of 51,992 test windows = 14.0% anomalies).
