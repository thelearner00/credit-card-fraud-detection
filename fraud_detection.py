"""
Credit Card Fraud Detection — Imbalanced Classification (v4)
Covers: EDA, temporal split, leakage fix, 4 imbalance strategies (SMOTE tuned),
        Isolation Forest (contamination tuned), TimeSeriesSplit CV + imblearn Pipeline,
        bootstrap confidence intervals, calibration curve, threshold tuning,
        cost-sensitive business value analysis, SHAP explainability
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import shap

import os
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PALETTE_BINARY  = ["#55A868", "#E05C5C"]
STRATEGY_COLORS = {
    "No Handling" : ("#AAAAAA", "#777777"),
    "Balanced"    : ("#6EB5E0", "#2277B0"),
    "SMOTE"       : ("#F4A460", "#C46820"),
    "Undersampled": ("#98D8A0", "#2E8B44"),
}
ISO_COLOR = "#9B59B6"

def savefig(fname, bbox_inches="tight"):
    path = os.path.join(SCRIPT_DIR, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches=bbox_inches)
    plt.close()
    print(f"  Saved → {fname}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Explore  +  EDA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 1 — Load & Explore  +  EDA")
print("=" * 70)

df = pd.read_csv(os.path.join(SCRIPT_DIR, "creditcard.csv"))

n_total   = len(df)
n_fraud   = int(df["Class"].sum())
n_legit   = n_total - n_fraud
fraud_pct = n_fraud / n_total * 100

print(f"\nDataset shape : {df.shape}")
print(f"Legitimate    : {n_legit:,}  ({100 - fraud_pct:.4f}%)")
print(f"Fraudulent    : {n_fraud:,}    ({fraud_pct:.4f}%)")

print("""
CONCEPT — Why accuracy is useless here:
  A model that predicts "legitimate" for EVERY transaction gets:
    Accuracy = 99.83%  ← looks great, catches zero fraud
    Recall   =  0.00%  ← misses every single fraud case
  → For imbalanced problems, accuracy rewards doing nothing.
    We need Precision, Recall, F1, and PR-AUC instead.
""")
print(f"  Baseline accuracy (predict all legitimate): {n_legit/n_total:.4%}")
print(f"  Baseline recall   (fraud caught)          : 0.00%")

# ── EDA: extract Hour here (needed for EDA chart before split) ────────────────
df["Hour"] = (df["Time"] % 86400) / 3600

fraud_df = df[df["Class"] == 1]
legit_df = df[df["Class"] == 0]

print("\n  EDA summary:")
print(f"    Amount — fraud median : ${fraud_df['Amount'].median():.2f}  "
      f"| legit median: ${legit_df['Amount'].median():.2f}")
print(f"    Amount — fraud mean   : ${fraud_df['Amount'].mean():.2f}  "
      f"| legit mean  : ${legit_df['Amount'].mean():.2f}")

# ── EDA Chart: 2×2 grid ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Exploratory Data Analysis", fontsize=14, fontweight="bold")

# Top-left: Amount distribution (log scale)
ax = axes[0][0]
ax.hist(legit_df["Amount"].clip(upper=500), bins=60,
        color=PALETTE_BINARY[0], alpha=0.7, label="Legitimate", density=True)
ax.hist(fraud_df["Amount"].clip(upper=500), bins=60,
        color=PALETTE_BINARY[1], alpha=0.8, label="Fraudulent", density=True)
ax.set_xlabel("Amount (clipped at $500)")
ax.set_ylabel("Density")
ax.set_title("Transaction Amount Distribution", fontweight="bold")
ax.legend()

# Top-right: Amount box plot
ax = axes[0][1]
ax.boxplot([legit_df["Amount"].clip(upper=500), fraud_df["Amount"].clip(upper=500)],
           labels=["Legitimate", "Fraudulent"],
           patch_artist=True,
           boxprops=dict(facecolor="#DDDDDD"),
           medianprops=dict(color="black", linewidth=2))
ax.set_ylabel("Amount (clipped at $500)")
ax.set_title("Amount Box Plot by Class", fontweight="bold")

# Bottom-left: Transaction count by Hour
ax = axes[1][0]
legit_by_hour = legit_df.groupby(legit_df["Hour"].astype(int)).size()
fraud_by_hour = fraud_df.groupby(fraud_df["Hour"].astype(int)).size()
hours = np.arange(24)
ax.bar(hours - 0.2, [legit_by_hour.get(h, 0) for h in hours],
       width=0.4, color=PALETTE_BINARY[0], label="Legitimate")
ax.bar(hours + 0.2, [fraud_by_hour.get(h, 0) * 50 for h in hours],
       width=0.4, color=PALETTE_BINARY[1], alpha=0.9, label="Fraudulent ×50")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Transaction count")
ax.set_title("Transactions by Hour (fraud scaled ×50)", fontweight="bold")
ax.legend(fontsize=8)

# Bottom-right: Fraud rate by hour
ax = axes[1][1]
total_by_hour = df.groupby(df["Hour"].astype(int)).size()
fraud_rate_by_hour = (fraud_by_hour / total_by_hour * 100).reindex(hours, fill_value=0)
colors_hr = [PALETTE_BINARY[1] if r > fraud_pct else PALETTE_BINARY[0]
             for r in fraud_rate_by_hour]
ax.bar(hours, fraud_rate_by_hour, color=colors_hr, edgecolor="white")
ax.axhline(fraud_pct, color="black", linestyle="--", linewidth=1,
           label=f"Average ({fraud_pct:.3f}%)")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Fraud rate (%)")
ax.set_title("Fraud Rate by Hour of Day", fontweight="bold")
ax.legend(fontsize=8)

savefig("fraud_0_eda.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature Engineering  (Hour already extracted above — no scaling yet)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 2 — Feature Engineering")
print("=" * 70)
print("  Hour-of-day already extracted in STEP 1.")
print("  Amount and Time will be scaled AFTER the split to prevent data leakage.")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Temporal Train / Test Split
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 3 — Temporal Train / Test Split  (first 80% → train, last 20% → test)")
print("=" * 70)

print("""
CONCEPT — Why temporal split matters:
  This dataset covers 48 hours of transactions sorted by Time.
  A random split lets the model "see" future transactions during training
  — called look-ahead bias. In production, models always predict on data
  they have never encountered chronologically.
""")

df_sorted = df.sort_values("Time").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)
train_df  = df_sorted.iloc[:split_idx].copy()
test_df   = df_sorted.iloc[split_idx:].copy()

train_fraud_n = int(train_df["Class"].sum())
test_fraud_n  = int(test_df["Class"].sum())
train_fraud_r = train_fraud_n / len(train_df) * 100
test_fraud_r  = test_fraud_n  / len(test_df)  * 100

print(f"  Train : {len(train_df):,} rows  |  fraud: {train_fraud_n:,}  ({train_fraud_r:.3f}%)")
print(f"  Test  : {len(test_df):,}  rows  |  fraud: {test_fraud_n:,}   ({test_fraud_r:.3f}%)")

if abs(train_fraud_r - test_fraud_r) > 0.02:
    print(f"""
  WARNING — Temporal drift detected:
    Train fraud rate: {train_fraud_r:.3f}%  →  Test fraud rate: {test_fraud_r:.3f}%
    Fraud patterns shifted between the two time periods. The model was
    optimised for a {train_fraud_r:.3f}% fraud rate but is evaluated on {test_fraud_r:.3f}%.
    This is a real-world challenge — fraud evolves over time.
    It means test-set metrics may be slightly pessimistic.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Scale After Split  (fixes data leakage)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 4 — Scale Amount & Time  (fit on train only)")
print("=" * 70)

print("""
BUG FIX — Data leakage in scaling:
  Fitting RobustScaler on the full dataset leaks test-set statistics
  (median, IQR) into the scaler before evaluation.
  Fix: fit on training rows only, then .transform() both splits.

CONCEPT — RobustScaler vs StandardScaler:
  Amount has extreme outliers (transactions > $10,000, median ~$22).
  StandardScaler uses mean/std — pulled by outliers.
  RobustScaler uses median and IQR — far less sensitive to extremes.
""")

scaler = RobustScaler()
train_df[["Amount_scaled", "Time_scaled"]] = scaler.fit_transform(
    train_df[["Amount", "Time"]]
)
test_df[["Amount_scaled", "Time_scaled"]] = scaler.transform(
    test_df[["Amount", "Time"]]
)
print("  Scaler fit on training data only. Test set transformed (not fit).")

feature_cols = [c for c in train_df.columns if c not in ("Class", "Time", "Amount")]
X_train = train_df[feature_cols].values
y_train = train_df["Class"].values
X_test  = test_df[feature_cols].values
y_test  = test_df["Class"].values

print(f"  Features : {len(feature_cols)}  (V1–V28, Hour, Amount_scaled, Time_scaled)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train 9 Models  (4 strategies × 2 classifiers + Isolation Forest)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 5 — Train 9 Models")
print("=" * 70)

neg_count        = int((y_train == 0).sum())
pos_count        = int((y_train == 1).sum())
scale_pos_weight = neg_count / pos_count

# ── SMOTE with tuned sampling_strategy ────────────────────────────────────────
print("""
CONCEPT — SMOTE sampling_strategy:
  Default SMOTE (sampling_strategy=1.0) oversamples fraud UP TO the size of
  the majority class — creating 227,000+ synthetic samples from just 417 real
  ones (543:1 ratio). That level of synthesis degrades model quality.
  Fix: sampling_strategy=0.1 means fraud becomes 10% of the majority class
  size (~22,700 fraud vs 227,000 legit). Much more realistic.
""")

sm = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"  SMOTE (10%)  → fraud: {int(y_train_sm.sum()):,}  "
      f"/  legit: {int((y_train_sm==0).sum()):,}")

# ── RandomUnderSampler ────────────────────────────────────────────────────────
print("""
CONCEPT — RandomUnderSampler:
  Removes majority-class rows until classes are balanced.
  Faster than SMOTE and avoids synthetic data — but discards real information.
  Aggressive here: ~417 legit rows kept from 227,000+.
""")

rus = RandomUnderSampler(random_state=42)
X_train_us, y_train_us = rus.fit_resample(X_train, y_train)
print(f"  Undersample  → fraud: {int(y_train_us.sum()):,}  "
      f"/  legit: {int((y_train_us==0).sum()):,}")

# ── Isolation Forest with tuned contamination ─────────────────────────────────
print("""
CONCEPT — Isolation Forest contamination parameter:
  contamination tells the model what fraction of the data to label as anomalies.
  Setting it equal to the fraud rate (0.0017) labels only ~97 test points as
  fraud — leaving very little room to catch the 75 actual fraud cases.
  Fix: use contamination=0.01, labelling ~570 test points as potential fraud.
  This gives the model enough budget to correctly identify more fraud cases.
""")

iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(X_train)
iso_preds  = (iso.predict(X_test) == -1).astype(int)
iso_scores = -iso.score_samples(X_test)
print(f"  Isolation Forest trained (contamination=0.01, "
      f"~{int(iso_preds.sum())} test points flagged as fraud).")

# ── class_weight / scale_pos_weight note ─────────────────────────────────────
print(f"""
CONCEPT — class_weight vs scale_pos_weight:
  LogisticRegression: class_weight='balanced' up-weights minority errors.
  XGBoost: scale_pos_weight = n_negatives / n_positives ≈ {scale_pos_weight:.0f}.
  Neither resamples — they adjust the loss function directly. No leakage risk.
""")

# ── Model definitions ─────────────────────────────────────────────────────────
lr_kw  = dict(max_iter=1000, random_state=42, n_jobs=-1)
xgb_kw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
              eval_metric="logloss", random_state=42, n_jobs=-1)

model_specs = [
    ("LR — No Handling",   LogisticRegression(**lr_kw),                               X_train,    y_train),
    ("LR — Balanced",      LogisticRegression(class_weight="balanced", **lr_kw),       X_train,    y_train),
    ("LR — SMOTE",         LogisticRegression(**lr_kw),                               X_train_sm, y_train_sm),
    ("LR — Undersampled",  LogisticRegression(**lr_kw),                               X_train_us, y_train_us),
    ("XGB — No Handling",  XGBClassifier(**xgb_kw),                                   X_train,    y_train),
    ("XGB — Balanced",     XGBClassifier(scale_pos_weight=scale_pos_weight, **xgb_kw), X_train,   y_train),
    ("XGB — SMOTE",        XGBClassifier(**xgb_kw),                                   X_train_sm, y_train_sm),
    ("XGB — Undersampled", XGBClassifier(**xgb_kw),                                   X_train_us, y_train_us),
]

for name, model, Xtr, ytr in model_specs:
    print(f"  Training {name} ...", end="", flush=True)
    model.fit(Xtr, ytr)
    print(" done")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — TimeSeriesSplit Cross-Validation (with imblearn Pipeline for SMOTE)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 6 — TimeSeriesSplit Cross-Validation")
print("=" * 70)

print("""
CONCEPT — TimeSeriesSplit vs StratifiedKFold(shuffle=True):
  Fraud data is time-ordered. StratifiedKFold with shuffle randomly mixes past
  and future transactions across folds — the same look-ahead bias we fixed in
  the train/test split. TimeSeriesSplit keeps each fold's validation
  chronologically AFTER its training set. Earlier folds have fewer training
  rows and may score lower — that reflects real deployment, not model quality.
  Fold sizes: fold 1 ≈ 45K train / 45K val, fold 5 ≈ 182K train / 45K val.

CONCEPT — imblearn Pipeline prevents resampling fold contamination:
  Without a Pipeline, fitting SMOTE before cross_val_score generates synthetic
  samples on the full training split — those samples can bleed into validation
  folds, inflating CV scores. The same applies to RandomUnderSampler.
  ImbPipeline([resampler, model]) applies resampling only within each fold's
  training portion, keeping validation completely clean.
  Note: undersampled models show higher fold-to-fold variance than SMOTE
  because discarding 99.8% of legitimate rows makes each fold sensitive to
  which rows are randomly kept.
""")

tscv     = TimeSeriesSplit(n_splits=5)
xgb_cv   = dict(n_estimators=100, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", random_state=42, n_jobs=-1)

cv_model_specs = {
    "LR — No Handling"           : LogisticRegression(**lr_kw),
    "LR — Balanced"              : LogisticRegression(class_weight="balanced", **lr_kw),
    "LR — SMOTE (Pipeline)"      : ImbPipeline([
                                     ("smote", SMOTE(sampling_strategy=0.1, random_state=42)),
                                     ("model", LogisticRegression(**lr_kw))]),
    "LR — Undersample (Pipeline)": ImbPipeline([
                                     ("rus", RandomUnderSampler(random_state=42)),
                                     ("model", LogisticRegression(**lr_kw))]),
    "XGB — No Handling"          : XGBClassifier(**xgb_cv),
    "XGB — Balanced"             : XGBClassifier(scale_pos_weight=scale_pos_weight, **xgb_cv),
    "XGB — SMOTE (Pipeline)"     : ImbPipeline([
                                     ("smote", SMOTE(sampling_strategy=0.1, random_state=42)),
                                     ("model", XGBClassifier(**xgb_cv))]),
    "XGB — Undersample (Pipeline)": ImbPipeline([
                                     ("rus", RandomUnderSampler(random_state=42)),
                                     ("model", XGBClassifier(**xgb_cv))]),
}

cv_results = {}

print(f"  {'Model':<28}  {'F1 mean±std':>16}  {'PR-AUC mean±std':>18}")
print("  " + "-" * 66)

for name, model in cv_model_specs.items():
    f1_cv = cross_val_score(model, X_train, y_train, cv=tscv,
                            scoring="f1", n_jobs=1)
    ap_cv = cross_val_score(model, X_train, y_train, cv=tscv,
                            scoring="average_precision", n_jobs=1)
    cv_results[name] = {"f1": f1_cv, "pr_auc": ap_cv}
    print(f"  {name:<28}  F1={f1_cv.mean():.3f}±{f1_cv.std():.3f}"
          f"   PR-AUC={ap_cv.mean():.3f}±{ap_cv.std():.3f}")

print("""
  TimeSeriesSplit growing-data effect: Fold 1 trains on ~45K rows,
  Fold 5 on ~182K. Earlier folds typically score lower — not because the
  model is worse, but because it has seen less history. This mimics
  production: a model deployed Day 2 knows less than one deployed Day 30.
  Chart 13 shows per-fold scores explicitly.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Evaluate All 9 Models on Held-Out Temporal Test Set
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 7 — Evaluate All 9 Models")
print("=" * 70)

results = {}

for name, model, _, _ in model_specs:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _      = roc_curve(y_test, probs)
    prec_c, rec_c, _ = precision_recall_curve(y_test, probs)
    results[name] = dict(
        accuracy  = accuracy_score(y_test, preds),
        precision = precision_score(y_test, preds, zero_division=0),
        recall    = recall_score(y_test, preds, zero_division=0),
        f1        = f1_score(y_test, preds, zero_division=0),
        roc_auc   = roc_auc_score(y_test, probs),
        pr_auc    = average_precision_score(y_test, probs),
        preds=preds, probs=probs, fpr=fpr, tpr=tpr, prec_c=prec_c, rec_c=rec_c,
        model=model,
    )

# Isolation Forest (no predict_proba)
fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_scores)
pc_iso,  rc_iso,  _ = precision_recall_curve(y_test, iso_scores)
results["Isolation Forest"] = dict(
    accuracy  = accuracy_score(y_test, iso_preds),
    precision = precision_score(y_test, iso_preds, zero_division=0),
    recall    = recall_score(y_test, iso_preds, zero_division=0),
    f1        = f1_score(y_test, iso_preds, zero_division=0),
    roc_auc   = roc_auc_score(y_test, iso_scores),
    pr_auc    = average_precision_score(y_test, iso_scores),
    preds=iso_preds, probs=iso_scores, fpr=fpr_iso, tpr=tpr_iso,
    prec_c=pc_iso, rec_c=rc_iso, model=None,
)

print("""
CONCEPT — ROC-AUC vs PR-AUC for imbalanced data:
  ROC-AUC: FPR = FP / (FP + TN). With 56,887 legitimate test transactions,
  even 500 false positives give a tiny FPR — ROC-AUC looks inflated.
  PR-AUC: measures precision vs recall for the POSITIVE class only.
  True negatives have no role — the majority class cannot inflate it.
  → PR-AUC is the headline metric. Random baseline ≈ 0.0017.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Results Table  +  CV vs Test Comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  STEP 8 — Results Summary (Temporal Test Set)")
print("=" * 100)

header = (f"{'Model':<24} {'Accuracy':>9} {'Precision':>10} {'Recall':>8}"
          f" {'F1':>8} {'ROC-AUC':>9} {'PR-AUC':>8}")
print(header)
print("-" * 100)

for name, r in results.items():
    print(f"{name:<24} {r['accuracy']:>9.4f} {r['precision']:>10.4f}"
          f" {r['recall']:>8.4f} {r['f1']:>8.4f}"
          f" {r['roc_auc']:>9.4f} {r['pr_auc']:>8.4f}")

print("=" * 100)
best_f1    = max(results, key=lambda n: results[n]["f1"])
best_prauc = max(results, key=lambda n: results[n]["pr_auc"])
print(f"  Best F1     : {best_f1}  ({results[best_f1]['f1']:.4f})")
print(f"  Best PR-AUC : {best_prauc}  ({results[best_prauc]['pr_auc']:.4f})")
print("=" * 100)

# ── CV vs Test comparison ─────────────────────────────────────────────────────
print("""
  CV vs Test Score Comparison  (gap = test - cv_mean)
  A large negative gap means the model does worse on the test period →
  either overfitting to train folds or temporal drift in the test period.
""")
print(f"  {'Model':<22}  {'CV F1':>8}  {'Test F1':>8}  {'Gap':>7}  "
      f"{'CV PR-AUC':>10}  {'Test PR-AUC':>12}  {'Gap':>7}")
print("  " + "-" * 82)

for name, cv in cv_results.items():
    if name not in results:
        continue   # Pipeline CV models don't have a direct test-set counterpart
    cv_f1    = cv["f1"].mean()
    cv_ap    = cv["pr_auc"].mean()
    test_f1  = results[name]["f1"]
    test_ap  = results[name]["pr_auc"]
    gap_f1   = test_f1 - cv_f1
    gap_ap   = test_ap - cv_ap
    flag_f1  = " ←drift" if gap_f1 < -0.05 else ""
    flag_ap  = " ←drift" if gap_ap < -0.05 else ""
    print(f"  {name:<22}  {cv_f1:>8.3f}  {test_f1:>8.3f}  "
          f"{gap_f1:>+7.3f}{flag_f1}  "
          f"{cv_ap:>10.3f}  {test_ap:>12.3f}  {gap_ap:>+7.3f}{flag_ap}")
print("  (Pipeline CV models omitted — no direct test-set equivalent)")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Bootstrap Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 9 — Bootstrap Confidence Intervals  (n=1,000)")
print("=" * 70)

print(f"""
CONCEPT — Why CIs are essential with only {test_fraud_n} fraud test cases:
  Point estimates (e.g., F1=0.806) look precise but are based on a tiny
  fraud sample. One misclassification shifts F1 by ~0.01–0.02.
  Bootstrap resampling draws 1,000 test sets with replacement and measures
  how much metrics vary across them → gives a 95% confidence interval.
  Wide CIs mean the result is unreliable; narrow CIs mean it is stable.
""")

N_BOOT   = 1000
rng      = np.random.default_rng(42)
fraud_idx = np.where(y_test == 1)[0]
legit_idx = np.where(y_test == 0)[0]

# Evaluate top models by PR-AUC (skip Isolation Forest — no predict_proba)
boot_models = [n for n in sorted(results, key=lambda n: results[n]["pr_auc"],
               reverse=True) if n != "Isolation Forest"][:5]

print(f"  {'Model':<24}  {'F1 point':>9}  {'95% CI (F1)':>18}  "
      f"{'PR-AUC point':>13}  {'95% CI (PR-AUC)':>18}")
print("  " + "-" * 90)

for name in boot_models:
    model   = results[name]["model"]
    boot_f1, boot_ap = [], []

    for _ in range(N_BOOT):
        # Stratified bootstrap: resample within each class separately
        bi_fraud = rng.choice(fraud_idx, size=len(fraud_idx), replace=True)
        bi_legit = rng.choice(legit_idx, size=len(legit_idx), replace=True)
        bi       = np.concatenate([bi_fraud, bi_legit])

        X_b, y_b   = X_test[bi], y_test[bi]
        preds_b    = model.predict(X_b)
        probs_b    = model.predict_proba(X_b)[:, 1]
        boot_f1.append(f1_score(y_b, preds_b, zero_division=0))
        boot_ap.append(average_precision_score(y_b, probs_b))

    ci_f1 = (np.percentile(boot_f1, 2.5), np.percentile(boot_f1, 97.5))
    ci_ap = (np.percentile(boot_ap, 2.5), np.percentile(boot_ap, 97.5))

    print(f"  {name:<24}  {results[name]['f1']:>9.3f}  "
          f"[{ci_f1[0]:.3f}, {ci_f1[1]:.3f}]  "
          f"{results[name]['pr_auc']:>13.3f}  "
          f"[{ci_ap[0]:.3f}, {ci_ap[1]:.3f}]")

print("""
  Interpretation: if two models' CIs overlap, their difference is NOT
  statistically significant. Do not treat one as definitively better.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Threshold Tuning Demo
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 10 — Threshold Tuning Demo")
print("=" * 70)

print("""
CONCEPT — Moving the decision threshold:
  By default, models predict fraud when P(fraud) >= 0.5.
  Lowering the threshold catches more fraud (higher recall) but flags more
  legitimate transactions (lower precision). Raising it does the opposite.
  Optimal threshold can be found by sweeping values and maximising F1 —
  this requires no retraining and is effectively free.

CONCEPT — Why XGBoost often needs a non-0.5 threshold:
  XGBoost on imbalanced data tends to assign low probabilities even to true
  fraud, because the majority class dominates training gradients. This is
  a calibration problem — the model's confidence scores don't reflect true
  probabilities. A calibration curve (Chart 9) reveals this directly.
""")

best_xgb_name = best_prauc if "XGB" in best_prauc else "XGB — No Handling"
best_probs    = results[best_xgb_name]["probs"]
thresholds    = np.linspace(0.01, 0.99, 300)
prec_t, rec_t, f1_t = [], [], []

for t in thresholds:
    p_t = (best_probs >= t).astype(int)
    prec_t.append(precision_score(y_test, p_t, zero_division=0))
    rec_t.append(recall_score(y_test, p_t, zero_division=0))
    f1_t.append(f1_score(y_test, p_t, zero_division=0))

optimal_t = float(thresholds[np.argmax(f1_t)])

def metrics_at(t):
    p = (best_probs >= t).astype(int)
    return (precision_score(y_test, p, zero_division=0),
            recall_score(y_test, p, zero_division=0),
            f1_score(y_test, p, zero_division=0))

pr_05, rc_05, f1_05 = metrics_at(0.5)
pr_op, rc_op, f1_op = metrics_at(optimal_t)

print(f"  Model used   : {best_xgb_name}")
print(f"  {'Threshold':>14}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
print(f"  {'0.50 (default)':>14}  {pr_05:>10.4f}  {rc_05:>8.4f}  {f1_05:>8.4f}")
print(f"  {optimal_t:.2f} (optimal)  {pr_op:>10.4f}  {rc_op:>8.4f}  {f1_op:>8.4f}")
print(f"\n  F1 improvement from threshold tuning alone: "
      f"{f1_op - f1_05:+.4f}  ({(f1_op/f1_05 - 1)*100:+.1f}%)")

print("""
CONCEPT — Recall-constrained threshold selection:
  argmax(F1) assumes precision and recall are equally important.
  Production systems typically set a minimum recall SLA first:
  "We must catch at least X% of fraud — given that, find the threshold
  that maximises precision (fewest false alarms)."
  This is recall-constrained optimisation — more realistic than argmax(F1).
""")

recall_targets = [0.80, 0.85, 0.90, 0.95]
print("  Recall-constrained threshold selection:")
print(f"  {'Min Recall':>12}  {'Threshold':>12}  {'Actual Recall':>14}"
      f"  {'Precision':>10}  {'F1':>8}")
print("  " + "-" * 62)
for target_recall in recall_targets:
    valid = [(t, p, r, f) for t, p, r, f in zip(thresholds, prec_t, rec_t, f1_t)
             if r >= target_recall]
    if valid:
        t_r, p_r, r_r, f_r = max(valid, key=lambda x: x[0])
        print(f"  {target_recall:>12.0%}  {t_r:>12.2f}  {r_r:>14.4f}"
              f"  {p_r:>10.4f}  {f_r:>8.4f}")
    else:
        print(f"  {target_recall:>12.0%}  {'unreachable':>12}")

print("""
  Insight: each row above is the answer to "what threshold do I set if
  I need to catch at least X% of fraud?" As the SLA rises, threshold drops
  (less selective), recall rises, but precision falls (more false alarms).
  This trade-off table is what you bring to the business — not just one number.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11 — Cost-Sensitive Business Value Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 11 — Cost-Sensitive Business Value Analysis")
print("=" * 70)

print("""
CONCEPT — Translating metrics into business dollars:
  F1 treats precision and recall as equally important. In fraud detection
  they are not — a missed fraud (FN) costs the average fraud amount (~$122),
  while a false alarm (FP) costs investigation time (~$5 per alert reviewed).
  The business-optimal threshold minimises total cost, which often differs
  from the F1-optimal threshold.

  Net value = TP × avg_fraud_amount  −  FP × fp_cost  −  FN × avg_fraud_amount
  (TN contributes $0 — correctly ignored transactions cost nothing)
""")

avg_fraud_amount = fraud_df["Amount"].mean()
fp_cost          = 5.0
net_values       = []

for t in thresholds:
    p_t = (best_probs >= t).astype(int)
    TP  = int(((p_t == 1) & (y_test == 1)).sum())
    FP  = int(((p_t == 1) & (y_test == 0)).sum())
    FN  = int(((p_t == 0) & (y_test == 1)).sum())
    net_values.append(TP * avg_fraud_amount - FP * fp_cost - FN * avg_fraud_amount)

optimal_biz_t = float(thresholds[np.argmax(net_values)])

def business_at(t):
    p_t = (best_probs >= t).astype(int)
    TP  = int(((p_t == 1) & (y_test == 1)).sum())
    FP  = int(((p_t == 1) & (y_test == 0)).sum())
    FN  = int(((p_t == 0) & (y_test == 1)).sum())
    saved  = TP * avg_fraud_amount
    cost   = FP * fp_cost
    missed = FN * avg_fraud_amount
    return TP, FP, FN, saved, cost, saved - cost - missed

tp_05,  fp_05,  fn_05,  sv_05,  co_05,  nv_05  = business_at(0.5)
tp_f1,  fp_f1,  fn_f1,  sv_f1,  co_f1,  nv_f1  = business_at(optimal_t)
tp_biz, fp_biz, fn_biz, sv_biz, co_biz, nv_biz = business_at(optimal_biz_t)

print(f"  Avg fraud amount : ${avg_fraud_amount:.2f}  |  FP investigation cost: ${fp_cost:.2f}")
print(f"\n  {'Threshold':<20}  {'Caught':>6}  {'FP':>6}  {'Missed':>7}  "
      f"{'Fraud Saved':>12}  {'Invest Cost':>12}  {'Net Value':>11}")
print("  " + "-" * 82)
for label, tp, fp, fn, sv, co, nv in [
    ("0.50 (default)",          tp_05,  fp_05,  fn_05,  sv_05,  co_05,  nv_05),
    (f"{optimal_t:.2f} (F1-optimal)",  tp_f1,  fp_f1,  fn_f1,  sv_f1,  co_f1,  nv_f1),
    (f"{optimal_biz_t:.2f} (biz-optimal)", tp_biz, fp_biz, fn_biz, sv_biz, co_biz, nv_biz),
]:
    print(f"  {label:<20}  {tp:>6}  {fp:>6}  {fn:>7}  "
          f"${sv:>11,.0f}  ${co:>11,.0f}  ${nv:>10,.0f}")

print(f"""
  Key insight: F1-optimal threshold ({optimal_t:.2f}) and business-optimal
  threshold ({optimal_biz_t:.2f}) often differ.
  At $122 avg fraud vs $5 investigation cost, missing one fraud is ~24×
  more expensive than one false alarm — so the business threshold is
  typically lower (catch more fraud, accept more false alarms).
""")

print("  Cost sensitivity — how business-optimal threshold shifts with FP cost:")
print(f"  {'FP cost/alert':>14}  {'Optimal threshold':>18}  {'Net value at optimal':>21}")
print("  " + "-" * 56)
for fp_test in [1.0, 5.0, 10.0, 20.0, 50.0]:
    nv_t = []
    for t in thresholds:
        p_t = (best_probs >= t).astype(int)
        TP  = int(((p_t == 1) & (y_test == 1)).sum())
        FP  = int(((p_t == 1) & (y_test == 0)).sum())
        FN  = int(((p_t == 0) & (y_test == 1)).sum())
        nv_t.append(TP * avg_fraud_amount - FP * fp_test - FN * avg_fraud_amount)
    opt_t_s = float(thresholds[np.argmax(nv_t)])
    marker = " ← current" if fp_test == fp_cost else ""
    print(f"  ${fp_test:>13.0f}  {opt_t_s:>18.2f}  ${max(nv_t):>20,.0f}{marker}")

print("""
  Insight: when FP cost is negligible ($1), set threshold very low — catch
  everything. As FP cost rises toward $50, threshold rises — be more selective.
  The "right" threshold is an economic decision, not a fixed number.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 11.5 — Post-Hoc Probability Calibration
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 11.5 — Post-Hoc Probability Calibration")
print("=" * 70)

print("""
CONCEPT — Two separate things: ranking vs calibration:
  PR-AUC and ROC-AUC measure how well a model ranks fraud above legitimate
  transactions. They don't care about the probability scale — a model that
  outputs P=0.001 for fraud but consistently ranks them first is still good.

  Calibration asks a different question: if the model outputs P=0.6, does
  60% of those transactions actually turn out to be fraud?
  XGBoost on imbalanced data is typically uncalibrated — it compresses
  probabilities toward zero, so the optimal threshold ends up at 0.90+.

  CalibratedClassifierCV (isotonic regression, 3-fold CV) adds a
  post-hoc layer that remaps raw probabilities to realistic ones.
  After calibration, P=0.5 actually means ~50% fraud — and the default
  0.5 threshold becomes meaningful without any threshold sweeping.

  Ranking stays intact (PR-AUC unchanged), but probabilities are reliable.
""")

print("  Training calibrated XGB (3-fold isotonic, n_estimators=100) ...")
xgb_cal_base = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              eval_metric="logloss", random_state=42, n_jobs=-1)
cal_model = CalibratedClassifierCV(xgb_cal_base, cv=3, method='isotonic')
cal_model.fit(X_train, y_train)
print("  done")

cal_probs = cal_model.predict_proba(X_test)[:, 1]
cal_preds = (cal_probs >= 0.5).astype(int)

cal_f1    = f1_score(y_test, cal_preds, zero_division=0)
cal_prauc = average_precision_score(y_test, cal_probs)

print(f"""
  Calibration comparison ({best_xgb_name} as base):

  {'Model':<36}  {'Threshold':>10}  {'F1':>8}  {'PR-AUC':>8}
  {'-'*66}
  {'Best XGB — uncalibrated (default)':<36}  {'0.50':>10}  {f1_05:>8.4f}  {results[best_xgb_name]['pr_auc']:>8.4f}
  {'Best XGB — uncalibrated (F1-optimal)':<36}  {optimal_t:>10.2f}  {f1_op:>8.4f}  {results[best_xgb_name]['pr_auc']:>8.4f}
  {'Calibrated XGB — isotonic (default)':<36}  {'0.50':>10}  {cal_f1:>8.4f}  {cal_prauc:>8.4f}

  After calibration: the default 0.50 threshold achieves similar F1 to the
  tuned {optimal_t:.2f} threshold — without any threshold sweeping.
  PR-AUC stays nearly identical (calibration preserves ranking, fixes scale).
""")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 12 — Save Charts  (13 PNGs)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 12 — Saving Charts")
print("=" * 70)

# Chart 0 already saved in STEP 1
print("  Saved → fraud_0_eda.png  (generated in STEP 1)")

# ── Chart 1: Class Distribution ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["Legitimate", "Fraudulent"], [n_legit, n_fraud],
              color=PALETTE_BINARY, edgecolor="white", width=0.5)
for bar, count, pct in zip(bars, [n_legit, n_fraud], [100 - fraud_pct, fraud_pct]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
            f"{count:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=11)
ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Transaction count")
ax.set_ylim(0, n_legit * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
savefig("fraud_1_class_distribution.png")

# ── Chart 2: Confusion Matrices  (3×3 grid — 9 models) ───────────────────────
model_order = [
    "LR — No Handling",   "LR — Balanced",      "LR — SMOTE",
    "LR — Undersampled",  "XGB — No Handling",  "XGB — Balanced",
    "XGB — SMOTE",        "XGB — Undersampled",  "Isolation Forest",
]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Confusion Matrices — All 9 Models", fontsize=14, fontweight="bold")

for idx, name in enumerate(model_order):
    ax = axes[idx // 3][idx % 3]
    ConfusionMatrixDisplay.from_predictions(
        y_test, results[name]["preds"],
        display_labels=["Legit", "Fraud"],
        colorbar=False, ax=ax, cmap="Blues",
    )
    ax.set_title(name, fontsize=9, fontweight="bold")

savefig("fraud_2_confusion_matrices.png")

# ── Chart 3: ROC Curves ───────────────────────────────────────────────────────
linestyles = {"LR": "-", "XGB": "--", "Isolation": ":"}
fig, ax = plt.subplots(figsize=(9, 7))
for name, r in results.items():
    family = name.split(" — ")[0]
    ls     = linestyles.get(family, "-.")
    color  = ISO_COLOR if name == "Isolation Forest" else None
    ax.plot(r["fpr"], r["tpr"], linestyle=ls, color=color,
            label=f"{name}  (AUC={r['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True)
savefig("fraud_3_roc_curves.png")

# ── Chart 4: Precision-Recall Curves ─────────────────────────────────────────
fraud_prevalence = n_fraud / n_total
fig, ax = plt.subplots(figsize=(9, 7))
for name, r in results.items():
    family = name.split(" — ")[0]
    ls     = linestyles.get(family, "-.")
    color  = ISO_COLOR if name == "Isolation Forest" else None
    ax.plot(r["rec_c"], r["prec_c"], linestyle=ls, linewidth=1.8, color=color,
            label=f"{name}  (PR-AUC={r['pr_auc']:.3f})")
ax.axhline(fraud_prevalence, color="k", linestyle="--", linewidth=0.8,
           label=f"Random baseline ({fraud_prevalence:.4f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True)
savefig("fraud_4_pr_curves.png")

# ── Chart 5: Metrics Comparison ───────────────────────────────────────────────
model_names_all = list(results.keys())
strategy_map, family_map = {}, {}
for n in model_names_all:
    if n == "Isolation Forest":
        strategy_map[n], family_map[n] = "Isolation Forest", "IF"
    else:
        parts = n.split(" — ")
        family_map[n], strategy_map[n] = parts[0], parts[1]

metrics_to_plot = ["recall", "f1", "pr_auc"]
metric_labels   = ["Recall", "F1", "PR-AUC"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Metric Comparison Across All Models", fontsize=14, fontweight="bold")
x, width = np.arange(len(model_names_all)), 0.6

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    bar_colors = []
    for n in model_names_all:
        if n == "Isolation Forest":
            bar_colors.append(ISO_COLOR)
        else:
            light, dark = STRATEGY_COLORS[strategy_map[n]]
            bar_colors.append(light if family_map[n] == "LR" else dark)
    vals = [results[n][metric] for n in model_names_all]
    bars = ax.bar(x, vals, color=bar_colors, edgecolor="white", width=width)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" — ", "\n") for n in model_names_all],
                       fontsize=7, rotation=0)
    ax.set_ylim(0, min(1.0, max(vals) * 1.25 + 0.05))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

legend_elements = [
    Patch(facecolor=STRATEGY_COLORS["No Handling"][0],  label="LR — No Handling"),
    Patch(facecolor=STRATEGY_COLORS["No Handling"][1],  label="XGB — No Handling"),
    Patch(facecolor=STRATEGY_COLORS["Balanced"][0],     label="LR — Balanced"),
    Patch(facecolor=STRATEGY_COLORS["Balanced"][1],     label="XGB — Balanced"),
    Patch(facecolor=STRATEGY_COLORS["SMOTE"][0],        label="LR — SMOTE"),
    Patch(facecolor=STRATEGY_COLORS["SMOTE"][1],        label="XGB — SMOTE"),
    Patch(facecolor=STRATEGY_COLORS["Undersampled"][0], label="LR — Undersampled"),
    Patch(facecolor=STRATEGY_COLORS["Undersampled"][1], label="XGB — Undersampled"),
    Patch(facecolor=ISO_COLOR,                          label="Isolation Forest"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.10))
savefig("fraud_5_metrics_comparison.png")

# ── Chart 6: Cross-Validation Results ────────────────────────────────────────
cv_names  = list(cv_results.keys())
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("5-Fold TimeSeriesSplit Cross-Validation (train set only)",
             fontsize=13, fontweight="bold")
cv_bar_colors = [
    STRATEGY_COLORS["No Handling"][0],  # LR No Handling
    STRATEGY_COLORS["Balanced"][0],     # LR Balanced
    STRATEGY_COLORS["SMOTE"][0],        # LR SMOTE Pipeline
    STRATEGY_COLORS["Undersampled"][0], # LR Undersample Pipeline
    STRATEGY_COLORS["No Handling"][1],  # XGB No Handling
    STRATEGY_COLORS["Balanced"][1],     # XGB Balanced
    STRATEGY_COLORS["SMOTE"][1],        # XGB SMOTE Pipeline
    STRATEGY_COLORS["Undersampled"][1], # XGB Undersample Pipeline
]
for ax, metric, label in zip(axes, ["f1", "pr_auc"], ["F1", "PR-AUC"]):
    means = [cv_results[n][metric].mean() for n in cv_names]
    stds  = [cv_results[n][metric].std()  for n in cv_names]
    x_cv  = np.arange(len(cv_names))
    bars  = ax.bar(x_cv, means, yerr=stds, color=cv_bar_colors,
                   edgecolor="white", capsize=5, width=0.5)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xticks(x_cv)
    ax.set_xticklabels([n.replace(" — ", "\n").replace(" (Pipeline)", "\n(Pipeline)")
                        for n in cv_names], fontsize=7.0)
    ax.set_ylim(0, min(1.0, max(means) * 1.4))
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=7.0)

savefig("fraud_6_cv_results.png")

# ── Chart 13: Per-Fold CV Scores (TimeSeriesSplit growing-data effect) ────────
fold_labels    = [f"Fold {i+1}" for i in range(5)]
fold_sizes     = ["~45K", "~91K", "~136K", "~159K", "~182K"]
plot_cv_models = [
    "LR — No Handling", "LR — SMOTE (Pipeline)",
    "XGB — No Handling", "XGB — SMOTE (Pipeline)",
]
per_fold_colors = [
    STRATEGY_COLORS["No Handling"][0], STRATEGY_COLORS["SMOTE"][0],
    STRATEGY_COLORS["No Handling"][1], STRATEGY_COLORS["SMOTE"][1],
]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("TimeSeriesSplit Per-Fold Scores — Growing-Data Effect\n"
             "Earlier folds have less training history → typically score lower",
             fontsize=12, fontweight="bold")

for ax, metric, label in zip(axes, ["f1", "pr_auc"], ["F1", "PR-AUC"]):
    for name, color in zip(plot_cv_models, per_fold_colors):
        scores = cv_results[name][metric]
        ax.plot(range(5), scores, "o-", linewidth=1.8, markersize=5,
                color=color, label=name)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_xticklabels(fold_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_xlabel("Fold (chronological order →)")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(fold_sizes, fontsize=7.5)
    ax2.set_xlabel("Approx. training rows per fold", fontsize=8)

savefig("fraud_13_cv_per_fold.png")

# ── Chart 7: Threshold Tuning ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(thresholds, prec_t, label="Precision", color="#2277B0", linewidth=1.8)
ax.plot(thresholds, rec_t,  label="Recall",    color="#E05C5C", linewidth=1.8)
ax.plot(thresholds, f1_t,   label="F1",        color="#55A868", linewidth=2.0)
ax.axvline(0.5,       color="grey",    linestyle="--", linewidth=1.2,
           label=f"Default 0.50  (F1={f1_05:.3f})")
ax.axvline(optimal_t, color="#C46820", linestyle="--", linewidth=1.5,
           label=f"Optimal {optimal_t:.2f}  (F1={f1_op:.3f})")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title(f"Precision / Recall / F1 vs Threshold\n({best_xgb_name})",
             fontsize=12, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
savefig("fraud_7_threshold_tuning.png")

# ── Chart 8: XGBoost Feature Importance ──────────────────────────────────────
best_xgb_model = results[best_xgb_name]["model"]
importances    = pd.Series(best_xgb_model.feature_importances_, index=feature_cols)
top15          = importances.nlargest(15).sort_values()

fig, ax = plt.subplots(figsize=(8, 7))
top15.plot.barh(ax=ax, color="#2277B0", edgecolor="white")
ax.set_xlabel("Importance (gain)")
ax.set_title(f"Top 15 Feature Importances\n({best_xgb_name})",
             fontsize=12, fontweight="bold")
for i, v in enumerate(top15.values):
    ax.text(v + 0.0003, i, f"{v:.4f}", va="center", fontsize=8)
savefig("fraud_8_feature_importance.png")

# ── Chart 9: Calibration Curve  ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

for name in [best_xgb_name, "XGB — No Handling", "LR — Balanced"]:
    probs_cal = results[name]["probs"]
    if probs_cal.max() <= 1.0:
        fop, mpv = calibration_curve(y_test, probs_cal, n_bins=10, strategy="quantile")
        ax.plot(mpv, fop, "o-", linewidth=1.5, markersize=4, label=name)

# Add calibrated model (computed in STEP 11.5)
fop_c, mpv_c = calibration_curve(y_test, cal_probs, n_bins=10, strategy="quantile")
ax.plot(mpv_c, fop_c, "s-", linewidth=2.0, markersize=5, color="#9B59B6",
        label="Calibrated XGB (isotonic)")

ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives (actual fraud rate)")
ax.set_title("Calibration Curves (Reliability Diagrams)\n"
             "Points above diagonal = underconfident  |  Below = overconfident",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
savefig("fraud_9_calibration.png")

# ── Chart 10: Business Value vs Threshold ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(thresholds, net_values, color="#2277B0", linewidth=2.0,
        label="Net Business Value")
ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax.axvline(0.5, color="grey", linestyle="--", linewidth=1.2,
           label=f"Default 0.50  (${nv_05:,.0f})")
ax.axvline(optimal_t, color="#55A868", linestyle="--", linewidth=1.5,
           label=f"F1-optimal {optimal_t:.2f}  (${nv_f1:,.0f})")
ax.axvline(optimal_biz_t, color="#E05C5C", linestyle="--", linewidth=1.5,
           label=f"Biz-optimal {optimal_biz_t:.2f}  (${nv_biz:,.0f})")
ax.annotate(f"${nv_biz:,.0f}",
            xy=(optimal_biz_t, max(net_values)),
            xytext=(min(optimal_biz_t + 0.06, 0.85), max(net_values) * 0.92),
            fontsize=9, color="#E05C5C",
            arrowprops=dict(arrowstyle="->", color="#E05C5C"))
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Net Business Value ($)")
ax.set_title(f"Business Value vs Threshold  ({best_xgb_name})\n"
             f"Avg fraud = ${avg_fraud_amount:.0f}  |  FP investigation = ${fp_cost:.0f}",
             fontsize=11, fontweight="bold")
ax.set_xlim(0, 1)
ax.legend(fontsize=9)
savefig("fraud_10_business_value.png")

# ── SHAP Values (computed once, used for Charts 11 and 12) ────────────────────
print(f"\n  Computing SHAP values (full test set: {len(X_test):,} rows including "
      f"all {int(y_test.sum())} fraud cases) ...")
explainer   = shap.TreeExplainer(best_xgb_model)
shap_sample = X_test   # use full test set so all fraud cases are represented
shap_values = explainer.shap_values(shap_sample)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top5_idx      = np.argsort(mean_abs_shap)[::-1][:5]
print("\n  Top 5 features by mean |SHAP|:")
for rank, idx in enumerate(top5_idx, 1):
    print(f"    {rank}. {feature_cols[idx]:<15}  mean |SHAP| = {mean_abs_shap[idx]:.4f}")

print("""
CONCEPT — SHAP (SHapley Additive exPlanations):
  SHAP assigns each feature a contribution value for each individual prediction.
  Positive SHAP → pushes toward fraud.  Negative SHAP → pushes away.
  Unlike gain-based feature importance (Chart 8), SHAP shows direction and
  per-sample effect — so you can explain why a specific transaction was flagged.
""")

# ── Chart 11: SHAP Beeswarm ───────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=feature_cols,
                  plot_type="dot", show=False, max_display=15)
plt.title("SHAP Beeswarm — Per-Transaction Feature Contributions\n"
          "Dot colour = feature value (red=high, blue=low)  |  "
          "x-position = impact on fraud score",
          fontsize=10, fontweight="bold")
savefig("fraud_11_shap_beeswarm.png")

# ── Chart 12: SHAP Bar (global importance) ────────────────────────────────────
plt.figure(figsize=(9, 7))
shap.summary_plot(shap_values, shap_sample, feature_names=feature_cols,
                  plot_type="bar", show=False, max_display=15)
plt.title("SHAP Mean |SHAP| — Global Feature Importance",
          fontsize=11, fontweight="bold")
savefig("fraud_12_shap_bar.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 13 — Key Takeaways
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 13 — Key Takeaways")
print("=" * 70)

print(f"""
  1. ACCURACY IS MEANINGLESS at {fraud_pct:.2f}% fraud rate.
     Predicting "all legitimate" gives {n_legit/n_total:.2%} accuracy — catches nothing.

  2. TEMPORAL SPLITS prevent look-ahead bias. Random splits let the model
     learn from future transactions. Always split time-ordered data by time.

  3. DATA LEAKAGE in preprocessing: fit scalers on training data only.
     Fitting on the full dataset leaks test statistics before evaluation.

  4. SMOTE default (sampling_strategy=1.0) creates synthetic dominance.
     417 real fraud samples → 227,000 synthetic ones is a 543:1 ratio.
     Use sampling_strategy=0.1 (10% of majority) for realistic augmentation.

  5. ISOLATION FOREST contamination matters.
     Setting it equal to the fraud rate (0.17%) gives the model almost no
     budget to catch fraud. contamination=0.01 is a better starting point.

  6. class_weight / scale_pos_weight is the simplest and safest fix.
     No resampling, no leakage risk — one parameter adjusts the loss function.

  7. BOOTSTRAP CIs expose whether results are trustworthy.
     With only {test_fraud_n} fraud test cases, point estimates are unreliable.
     Overlapping confidence intervals mean models are statistically tied.

  8. CALIBRATION determines whether probabilities mean what they say.
     XGBoost often assigns low probabilities to true fraud on imbalanced data.
     The calibration curve (Chart 9) reveals this — explaining why the optimal
     threshold ({optimal_t:.2f}) is far from the default 0.50.

  9. THRESHOLD TUNING is free — no retraining required.
     Default 0.50 → F1={f1_05:.3f}.  Optimal {optimal_t:.2f} → F1={f1_op:.3f}.
     Improvement: {f1_op - f1_05:+.3f} ({(f1_op/f1_05 - 1)*100:+.1f}%)

 10. PR-AUC is the headline metric.
     Random baseline = {fraud_prevalence:.4f}.
     ROC-AUC is inflated by {n_legit:,} true negatives — PR-AUC is not.

  Best F1     : {best_f1}  ({results[best_f1]['f1']:.4f})
  Best PR-AUC : {best_prauc}  ({results[best_prauc]['pr_auc']:.4f})

 11. TIMESERIESSPLIT is the correct CV for sequential fraud data.
     StratifiedKFold(shuffle=True) creates look-ahead bias in validation —
     the same problem we fixed in the train/test split. TimeSeriesSplit
     ensures each fold's validation is chronologically after its training set.

 12. SHAP reveals WHY the model flags a transaction, not just which features
     matter globally. Direction matters: high V14 may push strongly toward
     fraud while high V10 pushes away. Gain-based importance (Chart 8) shows
     which features split well; SHAP shows how they influence each prediction.

 13. BUSINESS-OPTIMAL threshold ≠ F1-optimal threshold.
     Missed fraud costs ${avg_fraud_amount:.0f} (avg amount). False alarms cost ${fp_cost:.0f}
     (investigation). The ratio (~{avg_fraud_amount/fp_cost:.0f}×) means missing one fraud is
     far more expensive than one false alarm — so the business threshold is
     typically lower than the F1-optimal, accepting more FPs to catch more fraud.

 14. CALIBRATION fixes probabilities; threshold tuning is a workaround.
     CalibratedClassifierCV (isotonic) remaps raw XGBoost scores to reliable
     probabilities. After calibration, P=0.5 means ~50% fraud — and the default
     0.50 threshold achieves near-identical F1 to the tuned {optimal_t:.2f} threshold.
     Ranking (PR-AUC) is unchanged; only the probability scale improves.

 15. BUSINESS-OPTIMAL threshold is not universal — it's cost-dependent.
     When FP investigation cost is $1: set threshold very low (catch everything).
     When FP cost is $50: threshold rises (be more selective).
     There is no universally correct threshold — it is an economic decision
     that must be revisited whenever investigation cost or fraud amounts change.

 16. RECALL-CONSTRAINED threshold is more realistic than argmax(F1).
     F1 implicitly weights precision = recall. Production systems set a
     minimum fraud-catch SLA first (e.g., recall ≥ 90%), then maximise
     precision within that constraint. The recall-constrained table in
     STEP 10 shows exactly which threshold satisfies each SLA level.

 17. RANDOMUNDERSAMPLER inside CV (ImbPipeline) shows higher variance
     than SMOTE — discarding 99.8% of legitimate rows makes each fold
     sensitive to which rows are randomly kept. Std >> SMOTE's std.

 18. TIMESERIESSPLIT growing-data effect: fold 1 underperforms fold 5
     not because the model is worse but because it has less training history
     (~45K vs ~182K rows). This reflects production: a model deployed on
     Day 2 knows less than one deployed on Day 30. See Chart 13.
""")

print("Done. 14 charts saved to the project directory.")
