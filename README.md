# Credit Card Fraud Detection

End-to-end imbalanced classification study covering the production ML concepts that a basic classification project skips: temporal splits, preprocessing leakage, four imbalance strategies, TimeSeriesSplit cross-validation, probability calibration, threshold economics, SHAP explainability, and business value translation.

**284,807 transactions · 0.17% fraud rate · 9 models · 14 charts · 18 key concepts**

---

## What This Project Covers

| Concept | What you learn |
|---|---|
| **Temporal train/test split** | Why random splits create look-ahead bias on time-ordered fraud data |
| **Data leakage in scaling** | Why `RobustScaler` must be fit on training rows only |
| **RobustScaler vs StandardScaler** | How outliers in `Amount` break mean/std scaling |
| **4 imbalance strategies** | No handling · class_weight · SMOTE (tuned) · RandomUnderSampler |
| **SMOTE sampling_strategy** | Why the default 1.0 creates a 543:1 synthetic ratio; tuning to 0.1 |
| **Isolation Forest** | Unsupervised anomaly detection without labels; contamination tuning |
| **TimeSeriesSplit CV** | Why `StratifiedKFold(shuffle=True)` leaks future data into validation folds |
| **imblearn Pipeline** | Running SMOTE and undersampling inside CV folds without contamination |
| **Bootstrap confidence intervals** | Quantifying uncertainty with only 75 fraud test cases |
| **PR-AUC vs ROC-AUC** | Why ROC-AUC is inflated by 56,887 true negatives at 0.17% fraud rate |
| **Threshold tuning** | Sweeping 300 thresholds to find the F1-optimal decision boundary for free |
| **Recall-constrained threshold** | Setting thresholds from a minimum recall SLA — the production pattern |
| **Probability calibration** | Why XGBoost underestimates fraud probability; fixing with isotonic regression |
| **Business value analysis** | Translating TP/FP/FN counts into net dollars saved across all thresholds |
| **Cost sensitivity** | How the optimal threshold shifts when investigation cost changes |
| **SHAP explainability** | Per-transaction feature contributions with direction (not just global rank) |
| **CV growing-data effect** | Why TimeSeriesSplit fold 1 underperforms fold 5 (45K vs 182K training rows) |
| **Temporal drift** | Train fraud rate (0.183%) vs test fraud rate (0.132%) and its implications |

---

## Dataset

Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Transactions | 284,807 |
| Fraud | 492 (0.17%) |
| Legitimate | 284,315 (99.83%) |
| Time window | 48 hours (sequential) |
| Features | `Time`, `V1`–`V28` (PCA-anonymised), `Amount`, `Class` |

> `V1`–`V28` were PCA-transformed by the dataset authors, making them unusually discriminative. On raw real-world features, the performance gap between imbalance strategies is typically much more pronounced.

---

## Results

Evaluated on the **temporal test set** — last 20% of transactions by time (56,962 rows, **75 fraud cases**).

| Model | Recall | F1 | PR-AUC |
|---|---|---|---|
| LR — No Handling | 0.560 | 0.456 | 0.606 |
| LR — Balanced | 0.893 | 0.073 | 0.590 |
| LR — SMOTE | 0.893 | 0.072 | 0.583 |
| LR — Undersampled | 0.907 | 0.079 | 0.672 |
| XGB — No Handling | 0.693 | **0.806** | 0.787 |
| XGB — Balanced | 0.787 | 0.652 | **0.796** |
| XGB — SMOTE | 0.760 | 0.481 | 0.768 |
| XGB — Undersampled | 0.893 | 0.104 | 0.786 |
| Isolation Forest | 0.013 | 0.014 | 0.045 |

**Threshold tuning (XGB — Balanced):** default threshold 0.50 → F1 = 0.652 · optimal threshold 0.95 → F1 = **0.821** — no retraining required.

**Post-hoc calibration (isotonic regression):** calibrated XGB at the default 0.50 threshold → F1 = **0.778** · PR-AUC unchanged at 0.797. Calibration removes the need to sweep thresholds manually.

**Business value optimum:** at avg fraud = $122 and investigation cost = $5/alert, the business-optimal threshold is 0.44 — lower than the F1-optimal 0.95, because catching one extra fraud ($122) outweighs the cost of 24 additional false alarms ($120).

**Top SHAP drivers:** V14 (mean |SHAP| = 1.15), V4 (1.14), V12 (0.71), V10 (0.69), V1 (0.61).

---

## Charts

| File | Content |
|---|---|
| `fraud_0_eda.png` | 2×2 EDA grid: amount distribution, box plot, hourly transaction volume, fraud rate by hour |
| `fraud_1_class_distribution.png` | Class imbalance bar chart with exact counts and percentages |
| `fraud_2_confusion_matrices.png` | 3×3 grid — confusion matrix for each of the 9 models |
| `fraud_3_roc_curves.png` | All 9 ROC curves with AUC values and random-classifier baseline |
| `fraud_4_pr_curves.png` | All 9 PR curves with PR-AUC values and random baseline (0.0017) |
| `fraud_5_metrics_comparison.png` | Recall / F1 / PR-AUC grouped bar chart across all 9 models |
| `fraud_6_cv_results.png` | 5-fold TimeSeriesSplit CV mean ± std for all 8 CV models |
| `fraud_7_threshold_tuning.png` | Precision / Recall / F1 vs threshold; marks default (0.50) and optimal |
| `fraud_8_feature_importance.png` | Top 15 XGBoost gain-based feature importances |
| `fraud_9_calibration.png` | Reliability diagrams: uncalibrated XGB vs calibrated (isotonic) vs perfect |
| `fraud_10_business_value.png` | Net $ value vs threshold; marks default, F1-optimal, and business-optimal |
| `fraud_11_shap_beeswarm.png` | SHAP beeswarm: per-transaction impact of top 15 features; colour = feature value |
| `fraud_12_shap_bar.png` | SHAP mean \|SHAP\| global importance with direction preserved |
| `fraud_13_cv_per_fold.png` | Per-fold CV scores showing the TimeSeriesSplit growing-data effect |

---

## Script Structure

`fraud_detection.py` runs end-to-end in approximately 8–10 minutes. Every step prints a `CONCEPT` explanation.

| Step | What it does |
|---|---|
| 1 | Load data · 2×2 EDA · demonstrate the useless 99.83% accuracy baseline |
| 2 | Extract `Hour` from `Time` — no scaling yet (leakage prevention) |
| 3 | Temporal split: first 80% by `Time` → train, last 20% → test; detect drift |
| 4 | Fit `RobustScaler` on training data only, then transform both splits |
| 5 | Train 9 models: 4 imbalance strategies × 2 classifiers (LR, XGB) + Isolation Forest |
| 6 | 5-fold `TimeSeriesSplit` CV on 8 models — `ImbPipeline` for all resampling variants |
| 7 | Evaluate all 9 models on the held-out temporal test set |
| 8 | Results table + CV vs test comparison (flag temporal drift where gap > 0.05) |
| 9 | Bootstrap 95% CIs — 1,000 stratified resamples for the top 5 models |
| 10 | Threshold tuning: F1-optimal sweep (300 thresholds) + recall-constrained SLA table |
| 11 | Cost-sensitive business value: net $ across all thresholds + cost sensitivity table |
| 11.5 | Post-hoc calibration: `CalibratedClassifierCV` (isotonic, 3-fold) vs uncalibrated |
| 12 | Save all 14 charts |
| 13 | Print 18 key takeaways |

---

## How to Run

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost shap
python3 fraud_detection.py
```

Place `creditcard.csv` (download from Kaggle — link above) in the same directory as the script. All 14 charts are saved to that directory automatically.

---

## Key Concepts

**Why PR-AUC, not accuracy or ROC-AUC?**
Predicting "all legitimate" gives 99.83% accuracy and catches zero fraud. ROC-AUC is inflated by 56,887 true negatives — a model that does nothing scores 0.50 on ROC but is useless in practice. PR-AUC measures only positive-class performance; a random classifier scores PR-AUC ≈ 0.0017 (the fraud rate), so any real improvement is clearly visible.

**Why temporal split instead of random split?**
The dataset covers 48 hours of sequential transactions. A random split lets the model see future transactions during training — look-ahead bias that never exists in production. The first 80% of transactions by `Time` form the training set; the last 20% form the test set. The same principle applies inside cross-validation: `TimeSeriesSplit` keeps each fold's validation window chronologically after its training window.

**Why imblearn Pipeline inside CV?**
Fitting SMOTE or `RandomUnderSampler` before calling `cross_val_score` leaks resampled data across folds, inflating CV scores. Wrapping the resampler and model in `ImbPipeline([resampler, model])` ensures resampling happens only on each fold's training portion — the validation fold sees only real, untouched data.

**Why calibration matters?**
XGBoost on imbalanced data compresses probability outputs toward zero — the model is underconfident, so the optimal decision threshold ends up at 0.95 rather than 0.50. `CalibratedClassifierCV` (isotonic regression) remaps raw scores to reliable probabilities. After calibration, the default 0.50 threshold achieves F1 = 0.778, very close to the uncalibrated tuned result of 0.821, without any threshold sweeping.

**Threshold economics**
The F1-optimal threshold (0.95) and the business-optimal threshold (0.44) are different because F1 weights precision and recall equally. In practice, missing one fraud costs ~$122 (the average fraud amount) while one false alarm costs ~$5 (investigation time). That 24:1 asymmetry pushes the business optimum toward a lower threshold — catch more fraud, accept more false alarms. When investigation cost rises to $50/alert, the optimal threshold rises toward 0.92. There is no universal threshold: it is an economic decision.
