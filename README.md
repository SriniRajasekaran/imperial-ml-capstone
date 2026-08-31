# Imperial College London — Professional Certificate in ML & AI
## Capstone Repository

**Owner:** Srini Rajasekaran
**Programme:** Imperial College London Professional Certificate in Machine Learning & AI

---

## Overview

This repository covers both stages of the capstone project. Stage 1 is a free-exploration credit model risk audit toolkit. Stage 2 is a structured black-box optimisation challenge run over 13 weekly query cycles.

---

## Stage 1 — Credit Model Risk Audit Toolkit

### Problem Statement

Corporate bankruptcy prediction using financial ratios is a well-established credit risk problem. The harder challenge is understanding and auditing the model's behaviour — ensuring its explanations are stable, consistent, and trustworthy enough for use in a governance context.

This stage builds an Automated Model Risk Audit Toolkit with quantitative explainability diagnostics, directly motivated by SR 26-2, PRA SS1/23, EU AI Act Article 13, and NIST AI RMF governance frameworks.

### Data

Polish Companies Bankruptcy dataset (UCI Machine Learning Repository, ID 365). Approximately 10,500 company observations, 64 financial ratios, binary bankruptcy label. Roughly 5% of companies are bankrupt — a realistic wholesale credit portfolio default rate.

### Models

Logistic Regression, KNN, Decision Trees, Random Forest, Gradient Boosting, Naive Bayes, SVM, and XGBoost. XGBoost (n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=19.3) is the primary model under audit. Logistic Regression is maintained as a SHAP calibration anchor — its coefficients are a closed-form analogue of SHAP values under linearity, so divergence between LR and XGBoost SHAP signals non-linear behaviour requiring investigation.

### Hyperparameter Optimisation

Grid search over max_depth (3, 5, 7), n_estimators (100, 200, 300), and learning_rate (0.01, 0.05, 0.1) using 5-fold stratified cross-validation. SMOTE (k=5) applied to training folds only. RobustScaler applied to all features; medians imputed on training data only.

### Audit Diagnostics (Novel Contribution)

Four quantitative diagnostics with calibrated RAG thresholds:

**A — Explanation Stability**
Bootstrap variance of SHAP attributions across 100 resamples. Amber: variance > 0.15. Red: variance > 0.30.

**B — Explanation Drift**
Wasserstein, KL, and KS distances between SHAP distributions across time-partitioned subsets. Amber: distance > 0.10. Red: distance > 0.25.

**C — Explanation Sensitivity Ratio (ESR) — Novel Contribution**
ESR_i = |dSHAP_i(epsilon)| / (|S_i| + delta). Measures how sensitive a feature's explanation is to a small perturbation relative to the magnitude of its explanation. Amber: ESR > 1.5. Red: ESR > 3.0. A RED finding was produced on the net financial debt ratio feature in the primary audit run.

**D — Local Consistency**
k-means clustering of local SHAP vectors to identify behaviourally inconsistent subpopulations. More than 2 clusters triggers a governance review flag.

### Regulatory Alignment

- PRA SS1/23 Section 3.4 — explanation drift diagnostic
- SR 26-2 Section IV — independent challenge and ESR
- EU AI Act Article 13 — local SHAP and transparency requirements
- NIST AI RMF GOVERN 1.2 — stability and stress-shift monitoring

Note: SR 11-7 was superseded by SR 26-2 in April 2026.

### Repository Structure (Stage 1)

```
credit_model_risk_audit_toolkit/
    notebooks/
        01_eda.ipynb
        02_preprocessing.ipynb
        03_knn_trees_ensembles.ipynb
        04_nb_lr_svm.ipynb
        05_xgboost_shap_audit.ipynb
    data/
        polish_bankruptcy_data.arff
        feature_map.csv
    data_sheet.md
    model_card.md
```

---

## Stage 2 — Black-Box Optimisation Challenge

### Overview

Eight unknown functions of increasing dimensionality (F1: 2D to F8: 8D). One portal query per function per week across 13 weekly cycles (Modules 12-24). All functions are maximisation tasks, including those with negative outputs — the portal transformed minimisation objectives into maximisation by negation.

### Surrogate Methods

**Gaussian Process with UCB Acquisition**
Primary surrogate. UCB(x) = GP_mean(x) + beta x GP_std(x). Beta starts at 2.0 (exploration) and decreases to 0.5 (exploitation) across the campaign. The C2 Kernel Challenger selects the best kernel from RBF, Matern-3/2, and Matern-5/2 using leave-one-out cross-validation Q2 on the full combined dataset each week — run as the mandatory first step of every session.

**Probability of Improvement (PI) Acquisition**
Permanently applied to F8 after a 7/7 historical backtest confirmed PI beats UCB across all prior rounds.

**OLS Regression (C1 Challenger)**
Tested weekly against the GP. Four eligibility gates: R2 > 0.30, Shapiro-Wilk p > 0.05, Durbin-Watson in [1.5, 2.5], predicted y exceeding current best. Used in early rounds; progressively superseded by the GP as surface structure became clearer.

### Candidate Generation

Sobol low-discrepancy sequences — primary generator throughout the campaign, chosen for their space-filling properties in high-dimensional unit hypercubes, consistent with their use in quasi-Monte Carlo integration in derivatives model risk. Default pool: 2^14 candidates per session.

### Visual Inspection Layer

Weekly HTML dashboards tracked output trajectories and input coordinate movements across successive queries. Plotting X-coordinate trajectories per function made drift patterns visible before statistical tests flagged them, and directly influenced several query decisions. Dashboards were extended to include SHAP-based function performance assessment in the final sessions.

### Key Methodological Findings

**Combined dataset is mandatory.** Initial data plus all weekly observations must always be used together. The weekly-only dataset produces signal reversals: F6 x4 inverts (r = +0.617 combined vs r = -0.30 weekly), F7 x1 inverts (r = -0.634 combined vs r = +0.33 weekly), F8 x2 inverts — confirmed by bootstrap analysis.

**The most important diagnostic finding** was the F5 x1 signal reversal. The initial 20-point dataset gave r = -0.28 for x1. The combined dataset gives r = +0.75. Root cause: the initial design covered x1 only up to 0.84, leaving the high-yield corner entirely unsampled. Six consecutive new campaign bests followed once the correct direction was confirmed.

**Professional practice analogies applied directly:** four-corners testing from derivatives stress testing; trust regions from domain-of-model-validity bounds; scenario matrix analysis from volatility surface construction; Sobol sequences from quasi-Monte Carlo in structured products risk.

### Campaign Results

| Fn | Description | Dims | Initial Best | Campaign Best | Best Week | Gain |
|----|------------|------|-------------|--------------|-----------|------|
| F1 | Radiation field detection | 2 | 0.000000 | 1.846721 | Week 13 | Located and exceeded spike |
| F2 | Noisy ML log-likelihood | 2 | 0.611205 | 0.745754 | Week 6 | +22% |
| F3 | Drug discovery (side-effect min) | 3 | -0.034835 | -0.007182 | Week 8 | +79% |
| F4 | Warehouse allocation (cost min) | 4 | -4.025542 | +0.491925 | Week 9 | +112% |
| F5 | Chemical yield optimisation | 4 | 1,088.860 | 8,662.405 | Week 13 | +696% |
| F6 | Recipe scoring | 5 | -0.714265 | -0.058449 | Week 13 | +92% |
| F7 | ML hyperparameter tuning | 6 | 1.364968 | 3.148939 | Week 12 | +130% |
| F8 | Neural network tuning | 8 | 9.598482 | 9.984774 | Week 13 | +4% |

Week 13 (Module 24) results confirmed. Five new campaign bests: F1, F4, F5, F6, F8.

### Repository Structure (Stage 2)

```
stage2_bbo/
    src/                         GP surrogate, diagnostics, data loading, session runner
    notebooks/                   Full diagnostic battery notebook with all outputs rendered
    reflections/                 Weekly discussion board posts, Modules 12-24
    evidence/
        week01/ ... week13/      Evidence workbooks, dashboards, SHAP assessments
    datasheet_bbo_dataset.md
    model_card_bbo_optimiser.md
    README.md
```

---

## AI Use Disclosure

AI tools were used to support code development, analysis, and drafting in this capstone, in accordance with Imperial College London's confirmed guidance (ticket 2821471). See AI_USE_DISCLOSURE.md for the complete working framework.

---

## Contact

Srini Rajasekaran
Imperial College London Professional Certificate in ML & AI, 2025-2026
