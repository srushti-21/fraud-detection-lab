# 🔍 Fraud Detection Learning Lab

> A structured, hands-on ML project built on the IEEE-CIS Fraud Detection dataset. Every experiment is designed to fail first — then fix it. Four dangerous ML failure modes, systematically exposed and resolved.

---

## Why This Project

Fraud detection is the canonical data science problem. A single dataset teaches you:

- **Metric deception** — how accuracy lies on imbalanced data
- **Data leakage** — why your best model collapses in production
- **Feature engineering** — extracting signal from 400+ noisy features
- **Imbalance handling** — why SMOTE fails and threshold tuning wins

Every mistake you can make in production ML shows up here. That's exactly why we start here.

---

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle)

| Property | Value |
|---|---|
| Total transactions | 590,540 |
| Features | 432 (after merge) |
| Fraud rate | 3.50% |
| Source | Vesta Corporation real-world data |

- `train_transaction.csv` + `train_identity.csv` → merged on `TransactionID`
- Mix of Vesta-engineered V-features (V1–V340), card features, device/identity features
- Extreme class imbalance — makes every naive decision wrong

---

## Project Structure

```
fraud-detection-lab/
├── notebooks/
│   ├── 00_failure_experiments.ipynb    ← Phase 1: intentional failures
│   ├── 01_eda_feature_audit.ipynb      ← Phase 2: feature audit
│   ├── 02_feature_engineering.ipynb    ← Phase 3: feature engineering
│   └── 03_imbalance_experiments.ipynb  ← Phase 4: imbalance strategies
├── src/
│   ├── features.py       ← shared feature computation
│   ├── metrics.py        ← custom metric functions
│   └── validation.py     ← leakage detection utilities
├── data/                 ← raw CSVs (not committed)
├── experiments/
│   └── results.csv       ← tracked experiment outcomes
└── reports/
    ├── feature_audit.csv
    ├── feature_audit_plots.png
    ├── temporal_fraud_patterns.png
    └── imbalance_comparison.png
```

---

## Phase 1 — Intentional Failures

> Build broken models deliberately. Experience the failure before the fix.

### Experiment 0A — The Naive Model

Train XGBoost on raw features with zero imbalance handling.

| Metric | Value |
|---|---|
| Accuracy | 97.98% |
| Fraud Recall | 0.49 |
| Missed Frauds | 2,176 |
| **Business Cost** | **₹10.89 crore** |

**The lie:** 97.98% accuracy sounds great. The model misses 51% of actual fraud.

**The math:** Do-nothing baseline (predict nothing is fraud) already gets **96.41% accuracy**. The gap between doing nothing and building a model is only **1.57% accuracy** — but **₹10.32 crore in fraud losses**.

```
Accuracy is a vanity metric on imbalanced data.
The right question is never "how often are we right?"
It's "how much does being wrong cost?"
```

---

### Experiment 0B — The Leaky Model

Artificially create a feature using full-dataset aggregation before splitting (`card_fraud_rate` computed across all rows). Watch AUC spike. Then simulate production.

| Metric | Offline (validation) | Production (30% new cards) |
|---|---|---|
| ROC-AUC | 0.9556 | 0.8030 |
| Recall | 0.54 | 0.38 |

**The danger:** Leakage doesn't look wrong — it looks like improvement. AUC jumped from 0.94 → 0.96, model looked like your best performer, then collapsed silently in production when new cards appeared with no historical fraud rate.

**Rule:** Any feature computed using the target variable across the full dataset before splitting = leakage. Always compute aggregates after splitting, using only training data.

---

### Experiment 0C — The Wrong Metric

| Model | Accuracy | Fraud Recall | Business Cost |
|---|---|---|---|
| Do-Nothing (predict all legit) | 96.41% | 0.00 | ₹21.21 crore |
| Naive XGBoost | 97.98% | 0.49 | ₹10.89 crore |
| Leaky XGBoost | 98.12% | 0.54 | ₹9.79 crore* |

*Leaky model's offline cost looks best — but production recall drops to 0.38, making it the worst.

**Always report:** Recall, Precision, PR-AUC, and Expected Business Cost. Never accuracy alone.

---

## Phase 2 — Feature Audit

Profile all 432 features before touching a model.

### Key Findings

| Finding | Detail |
|---|---|
| High-signal features (AUC ≥ 0.62) | 156 |
| Medium-signal features (AUC 0.55–0.62) | 95 |
| Low-signal features (AUC < 0.55) | 150 |
| Features with >50% nulls | 214 |
| Drop candidates (>90% null, AUC ~0.50) | 12 |
| Leakage detected in raw features | None (max AUC = 0.69) |

### Missingness Is Structured, Not Random

Three distinct null clusters exist: 0%, ~47%, ~80%. This isn't data quality — it's different data collection systems.

**Critical finding:** High-null V-series features (76% null) have fraud rate **7.91% when present** vs **2.13% when null** (baseline: 3.50%).

```
Being present is the fraud signal, not being null.
Missingness pattern tells you which system processed the transaction.
That system correlates with fraud.
```

This means: don't just fill nulls with -999 and forget. The null pattern is meaningful.

---

## Phase 3 — Feature Engineering

Built 3 feature families from scratch.

### Family 1: Missingness Indicators
Binary flags: `V167_present`, `V168_present`, etc. for 15 high-null, high-signal V-features.

- All 15 indicators: AUC 0.65+ consistently
- Fraud when present: 7.91% | Fraud when null: 2.13%

### Family 2: Transaction Velocity
Leakage-safe cumulative features per card:

```python
card1_time_diff         # seconds since last transaction on this card
card1_txn_count         # cumulative transaction count per card
card1_amt_cummean       # card's rolling average transaction amount
card1_amt_deviation     # how far this transaction deviates from card history
```

| Feature | Single-Feature AUC |
|---|---|
| card1_amt_deviation | 0.5784 |
| card1_time_diff | 0.5628 |
| card1_amt_cummean | 0.5567 |
| card1_txn_count | 0.5286 |

### Family 3: Temporal Features
Hour of day, day of week, `is_night`, `is_weekend`.

**Finding:** All AUCs 0.50–0.51. Temporal features alone don't predict fraud well. Fraud spikes at hours 5–9 but this reflects low transaction volume inflating the rate, not a clean signal.

### Key Learning
Engineered features got near-zero importance when competing with 400 raw V-features. Feature engineering matters most on curated feature sets, not alongside 400 raw alternatives. In production: replace raw features with engineered ones rather than adding on top.

---

## Phase 4 — Imbalance Experiments

6 strategies, same base model, same time-based split.

**Cost matrix:**
- False Positive (blocked legit transaction): ₹500
- False Negative (missed fraud): ₹50,000

| Strategy | AUC | PR-AUC | Recall | Missed Frauds | Business Cost |
|---|---|---|---|---|---|
| No Handling | 0.8873 | 0.4941 | 0.37 | 2,546 | ₹12.76 crore |
| scale_pos_weight | 0.8867 | 0.4990 | 0.68 | 1,314 | ₹7.05 crore |
| SMOTE (20%) | 0.8851 | 0.5052 | 0.36 | 2,617 | ₹13.11 crore |
| ADASYN (20%) | 0.8877 | 0.5084 | 0.35 | 2,641 | ₹13.23 crore |
| BalancedRandomForest | 0.8910 | 0.4689 | 0.65 | 1,433 | ₹7.59 crore |
| **Cost-Optimized Threshold (0.13)** | **0.8867** | **0.4990** | **0.92** | **329** | **₹4.16 crore** |

### Key Insights

**1. Threshold optimization beats every resampling strategy.**
Same model as `scale_pos_weight`. Moved threshold from 0.5 → 0.13. Recall jumped 0.68 → 0.92. Business cost dropped from ₹7.05 crore → ₹4.16 crore. **One line of code saved ₹8.6 crore.**

**2. SMOTE and ADASYN failed.**
Both performed worse than no handling. Reason: on 400-dimensional data, synthetic minority samples created by interpolation don't represent real fraud patterns. SMOTE is a low-dimensional technique used on a high-dimensional problem.

**3. There is no universally best strategy.**
Winner depends entirely on the cost matrix. With equal FP/FN costs, SMOTE might win. With FN cost 100x FP cost (our setup), threshold tuning dominates.

**4. ROC-AUC flatters imbalanced models.**
All strategies score 0.88–0.89 AUC but have wildly different business outcomes. PR-AUC (0.47–0.51) is more discriminating and more honest on imbalanced data.

**5. Never set threshold at 0.5.**
Default threshold assumes FP and FN are equally costly. They never are in fraud. Always derive threshold from the cost matrix.

---

## Setup & Reproduction

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap mlflow matplotlib seaborn jupyter kaggle
```

### Data Download

```bash
kaggle competitions download -c ieee-fraud-detection -p data/
cd data && tar -xf ieee-fraud-detection.zip
```

Requires accepting competition rules at: https://www.kaggle.com/competitions/ieee-fraud-detection/rules

### Run Notebooks in Order

```
00_failure_experiments.ipynb  → Phase 1
01_eda_feature_audit.ipynb    → Phase 2
02_feature_engineering.ipynb  → Phase 3
03_imbalance_experiments.ipynb → Phase 4
```

**Important:** Use time-based splits, not random splits. Random splits inflate AUC by ~0.06 on this dataset due to structural card-level leakage.

---

## Skills Demonstrated

- Systematic ML experiment design with hypothesis → result → insight tracking
- Leakage detection and elimination (both explicit and structural)
- Feature auditing at scale (432 features profiled, signal ranked, buckets defined)
- Translating model metrics into business cost outcomes
- Imbalance handling: resampling, class weighting, threshold optimization
- Temporal feature engineering (leakage-safe cumulative aggregates)
- Identifying when engineered features are drowned by raw feature volume

---

## Real-World Relevance

Every fintech company — Razorpay, PhonePe, CRED, Groww, Paytm — has a fraud team solving this exact problem: imbalanced real-time scoring with regulatory constraints and hard cost asymmetry between false positives and false negatives.

The ability to articulate metric choice, threshold optimization, leakage diagnosis, and business cost translation directly differentiates candidates in data science interviews for these roles.

---

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `lightgbm` · `imbalanced-learn` · `matplotlib` · `seaborn` · `jupyter`

---

*Built as part of a structured real-world DS problem-solving series.*
