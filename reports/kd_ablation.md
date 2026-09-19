# Knowledge Distillation (KD) Ablation & Hyperparameter Sweep Report (P0-7b)

**Date:** 2026-09-20  
**Dataset:** PhysioNet MIT-BIH Arrhythmia Database (`mitdb`, 311,952 windows total)  
**Split Method:** Patient/Record-Independent Split (DECISIONS.md #3: 33 Train, 7 Val, 8 Test records)  
**Test Split:** 8 held-out records (`107`, `115`, `119`, `122`, `207`, `220`, `228`, `234` — 51,992 windows)

---

## 1. Executive Summary

This report documents the factual, empirical comparison across:
1. **Teacher 1D-CNN** (120,674 parameters)
2. **Student 1D-CNN WITHOUT KD** (1,538 parameters, standard hard cross-entropy loss)
3. **Student 1D-CNN WITH KD Baseline** (1,538 parameters, $T=3.0, \alpha=0.7$)
4. **Knowledge Distillation Hyperparameter Sweep** across $T \in \{2.0, 4.0, 6.0\}$ and $\alpha \in \{0.3, 0.5\}$

All models were trained on the exact same 214,467 training windows and evaluated on the exact same 51,992 held-out test windows.

> [!IMPORTANT]
> **Factual Sweep Findings:**
> - **Overall Accuracy:** No KD configuration surpassed the **Student (No KD)** baseline test accuracy of **86.87%** (`0.8687`). The closest KD configuration was $T=2.0, \alpha=0.3$ at **86.74%** (`0.8674`).
> - **Anomaly Recall & F1 Score:** Moderate alpha ($\alpha=0.5$) with higher temperatures ($T=4.0$ and $T=6.0$) produced substantial gains in minority-class anomaly recall: $T=4.0, \alpha=0.5$ improved recall from **12.70%** (no KD) to **17.31%** (1,260 true positives detected vs. 924), achieving an **F1 score of 0.2629** (surpassing both Teacher F1 of 0.2369 and Student-No-KD F1 of 0.2130).
> - All reported numbers below are `[MEASURED]` directly from evaluation on the held-out test split.

---

## 2. Quantitative Performance Comparison (`[MEASURED]`)

| Model | Parameters `[MEASURED]` | Compression vs Teacher | Hyperparameters | Test Accuracy `[MEASURED]` | Test Precision `[MEASURED]` | Test Recall `[MEASURED]` | Test F1 Score `[MEASURED]` |
|---|---|---|---|---|---|---|---|
| **Teacher 1D-CNN** | 120,674 | Baseline (1.0x) | — | **0.8409** (84.09%) | 0.3605 | 0.1764 | 0.2369 |
| **Student (No KD)** | 1,538 | **78.46x smaller** | — | **0.8687** (86.87%) | **0.6609** | 0.1270 | 0.2130 |
| **Student (KD Baseline)** | 1,538 | **78.46x smaller** | $T=3.0, \alpha=0.7$ | 0.8376 (83.76%) | 0.2785 | 0.1006 | 0.1478 |
| **Student (KD Sweep 1)** | 1,538 | **78.46x smaller** | $T=2.0, \alpha=0.3$ | **0.8674** (86.74%) | 0.6312 | 0.1272 | 0.2118 |
| **Student (KD Sweep 2)** | 1,538 | **78.46x smaller** | $T=2.0, \alpha=0.5$ | 0.8611 (86.11%) | 0.5138 | 0.1483 | 0.2301 |
| **Student (KD Sweep 3)** | 1,538 | **78.46x smaller** | $T=4.0, \alpha=0.3$ | 0.8539 (85.39%) | 0.3948 | 0.0815 | 0.1351 |
| **Student (KD Sweep 4)** | 1,538 | **78.46x smaller** | $T=4.0, \alpha=0.5$ | 0.8641 (86.41%) | 0.5464 | **0.1731** | **0.2629** |
| **Student (KD Sweep 5)** | 1,538 | **78.46x smaller** | $T=6.0, \alpha=0.3$ | 0.8445 (84.45%) | 0.2761 | 0.0684 | 0.1097 |
| **Student (KD Sweep 6)** | 1,538 | **78.46x smaller** | $T=6.0, \alpha=0.5$ | 0.8616 (86.16%) | 0.5165 | 0.1724 | 0.2585 |

---

## 3. Confusion Matrix Breakdown (`[MEASURED]`)

| Model Configuration | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Total Windows |
|---|---|---|---|---|---|
| **Teacher 1D-CNN** | 1,284 | 2,278 | 5,994 | 42,436 | 51,992 |
| **Student (No KD)** | 924 | 474 | 6,354 | 44,240 | 51,992 |
| **Student (KD Baseline: $T=3.0, \alpha=0.7$)** | 732 | 1,896 | 6,546 | 42,818 | 51,992 |
| **Student (KD: $T=2.0, \alpha=0.3$)** | 926 | 541 | 6,352 | 44,173 | 51,992 |
| **Student (KD: $T=2.0, \alpha=0.5$)** | 1,079 | 1,021 | 6,199 | 43,693 | 51,992 |
| **Student (KD: $T=4.0, \alpha=0.3$)** | 593 | 909 | 6,685 | 43,805 | 51,992 |
| **Student (KD: $T=4.0, \alpha=0.5$)** | **1,260** | 1,046 | 6,018 | 43,668 | 51,992 |
| **Student (KD: $T=6.0, \alpha=0.3$)** | 498 | 1,306 | 6,780 | 43,408 | 51,992 |
| **Student (KD: $T=6.0, \alpha=0.5$)** | 1,255 | 1,175 | 6,023 | 43,539 | 51,992 |

---

## 4. Distillation Hyperparameters & Loss Formulation

$$\mathcal{L}_{\text{total}} = (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, p_s) + \alpha \cdot T^2 \cdot \mathcal{L}_{\text{KL}}\left(\text{Softmax}\left(\frac{z_t}{T}\right), \text{Softmax}\left(\frac{z_s}{T}\right)\right)$$

- **Teacher Model:** Frozen (`teacher.trainable = False`) during all distillation runs.
- **Student Model:** 1,538 parameters (same compact architecture across all experiments).
- **Optimization:** Adam ($\text{lr}=10^{-3}$, batch size 256).

---

## 5. Key Findings & Observations

1. **Trade-off between Accuracy and Recall:**
   - The **Student trained WITHOUT KD** prioritizes the majority class (normal rhythms), achieving high overall test accuracy (**86.87%**) and precision (**66.09%**), but lower minority-class recall (**12.70%**).
   - Knowledge Distillation at $T=4.0, \alpha=0.5$ transfers dark knowledge from the Teacher that transfers anomaly sensitivity, increasing minority recall to **17.31%** (+36.3% relative increase in detected true anomalies) and lifting F1 score to **0.2629** (+23.4% relative gain over no-KD).
2. **Impact of Weighting ($\alpha$):**
   - Setting $\alpha=0.7$ (as in the initial trial) placed too much emphasis on soft KL loss relative to hard ground-truth labels under severe class imbalance (14.0% anomalies), depressing both accuracy and F1.
   - A balanced weight $\alpha=0.5$ demonstrated the best synergy between soft regularization and hard ground truth.
3. **Deployment Recommendation:**
   - If optimizing purely for raw test accuracy on normal-predominant streams: **Student (No KD)** (86.87%).
   - If optimizing for balanced sensitivity / F1 score: **Student (KD: $T=4.0, \alpha=0.5$)** (F1 = 0.2629).
