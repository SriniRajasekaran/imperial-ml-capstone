# Imperial College London — Professional Certificate in ML & AI
## Capstone Repository

**Owner:** Srini Rajasekaran  
**Programme:** Imperial College London Professional Certificate in Machine Learning & AI  
**Repository:** [github.com/SriniRajasekaran/imperial-ml-capstone](https://github.com/SriniRajasekaran/imperial-ml-capstone)

---

## Overview

This repository covers both stages of the capstone project for the Imperial College London Professional Certificate in ML & AI. Stage 1 is a free-exploration audit toolkit applied to a real credit dataset. Stage 2 is a structured black-box optimisation challenge run over 13 weekly query cycles.

---

## Stage 1 — Credit Model Risk Audit Toolkit

### Problem Statement

Corporate bankruptcy prediction using financial ratios is a well-established credit risk problem. The harder challenge is understanding and auditing the model's behaviour — ensuring its explanations are stable, consistent, and trustworthy enough for use in a governance context.

This stage builds an Automated Model Risk Audit Toolkit with quantitative explainability diagnostics, directly motivated by the SR 26-2, PRA SS1/23, EU AI Act Article 13, and NIST AI RMF governance frameworks.

### Data

Polish Companies Bankruptcy dataset (UCI Machine Learning Repository, ID 365). Approximately 10,500 company observations, 64 financial ratios, binary bankruptcy label. Roughly 5% of companies are bankrupt — a realistic wholesale credit portfolio default rate.

### Models

Logistic Regression, KNN, Decision Trees, Random Forest, Gradient Boosting, Naive Bayes, SVM, and XGBoost. XGBoost (n\_estimators=300, max\_depth=5, learning\_rate=0.05, subsample=0.8, colsample\_bytree=0.8, scale\_pos\_weight=19.3) is the primary model under audit. Logistic Regression is maintained as a SHAP calibration anchor — its coefficients are a closed-form analogue of SHAP values under linearity, so divergence between LR and XGBoost SHAP signals non-linear behaviour requiring investigation.

### Hyperparameter Optimisation

Grid search over max\_depth (3, 5, 7), n\_estimators (100, 200, 300), and learning\_rate (0.01, 0.05, 0.1) using 5-fold stratified cross-validation. SMOTE (k=5) applied to training folds only to address class imbalance. RobustScaler applied to all features; medians imputed on training data only.

### Audit Diagnostics (Novel Contribution)

Four quantitative diagnostics with calibrated RAG thresholds:

**A — Explanation Stability**  
Bootstrap variance of SHAP attributions across 100 resamples. Amber threshold: variance > 0.15. Red threshold: variance > 0.30. Flags features whose SHAP values are sensitive to the training sample.

**B — Explanation Drift**  
Wasserstein, KL, and KS distances between SHAP distributions across time-partitioned or randomly-split data subsets. Amber threshold: distance > 0.10. Red threshold: distance > 0.25. Flags temporal instability in the model's explanatory behaviour.

**C — Explanation Sensitivity Ratio (ESR) — Novel Contribution**  
ESR\_i = |dSHAP\_i(epsilon)| / (|S\_i| + delta). Measures how sensitive a feature's explanation is to a small perturbation of that feature's value, relative to the magnitude of its explanation. Amber threshold: ESR > 1.5. Red threshold: ESR > 3.0. A RED finding on net financial debt ratio was produced in the primary audit run — the feature's explanation was more volatile than its prediction, a direct violation of the stability principle in SR 26-2 model risk guidance.

**D — Local Consistency**  
k-means clustering of local SHAP vectors to identify behaviourally inconsistent subpopulations. More than 2 clusters triggers a governance review flag.

### Regulatory Alignment

- PRA SS1/23 Section 3.4 — explanation drift diagnostic  
- SR 26-2 Section IV — independent challenge and ESR  
- EU AI Act Article 13 — local SHAP and transparency requirements  
- NIST AI RMF GOVERN 1.2 — stability and stress-shift monitoring  

Note: SR 11-7 was superseded by SR 26-2 in April 2026. All references in this repository use SR 26-2.

### Results

All five technique notebooks executed successfully and produce correct outputs. The ESR diagnostic produced a RED audit finding on the net financial debt ratio feature. Full results documented in notebook outputs.

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
    reports/
        (PNG charts from audit runs)
    data_sheet.md
    model_card.md
```

---

## Stage 2 — Black-Box Optimisation Challenge

### Overview

Eight unknown functions of increasing dimensionality (F1: 2D to F8: 8D). One portal query per function per week across 13 weekly cycles (Modules 12–24). All functions are maximisation tasks, including those with negative outputs — the portal transformed minimisation objectives into maximisation by negation.

### Surrogate Methods

**Gaussian Process (GP) with UCB Acquisition**  
Primary surrogate. UCB(x) = GP\_mean(x) + beta × GP\_std(x). Beta starts at 2.0 (exploration) and decreases to 0.5 (exploitation) across the campaign. The C2 Kernel Challenger test selects the best kernel from RBF, Matern-3/2, and Matern-5/2 using leave-one-out cross-validation Q² on the full combined dataset each week.

**Probability of Improvement (PI) Acquisition**  
Permanently applied to F8 after historical backtesting confirmed PI beats UCB in 7 of 7 rounds.

**OLS Regression (C1 Challenger)**  
Tested weekly as a challenger to the GP. Eligibility requires all four gates to pass: R² > 0.30, Shapiro-Wilk residual normality p > 0.05, Durbin-Watson statistic in [1.5, 2.5], and predicted y at the significance-gated query exceeding the current best. Regression was used in early rounds and then consistently superseded by the GP as data grew and landscape structure became clearer.

### Candidate Generation

Sobol low-discrepancy sequences (primary, from Week 2 onward). Default pool: 2^14 = 16,384 candidates per session. Convergence verified at 2^15 for each function. Latin Hypercube Sampling used as comparison only.

### Key Methodological Findings

**Combined dataset is mandatory.** Initial data plus all weekly observations must always be used together. The weekly-only dataset produces critical signal reversals: F6 x4 direction inverts (r = +0.617 combined vs r = -0.30 weekly), F7 x1 inverts (r = -0.634 combined vs r = +0.33 weekly), F8 x2 inverts. These reversals were confirmed by bootstrap analysis and directly motivated the combined-dataset rule.

**The most important diagnostic finding of the campaign** was the F5 x1 signal reversal. The initial 20-point dataset gave r = -0.28 for x1 (negative association). The combined dataset gives r = +0.75 (strongly positive). Root cause: the initial design covered x1 only up to 0.84; the high-yield corner (x1 > 0.84) was entirely unsampled. Strategy based on initial data alone would have pushed x1 in the wrong direction. The combined dataset confirmed all four dimensions as positive monotone drivers, producing six consecutive new campaign bests in Weeks 7–12.

**Four-corners testing maps to derivatives stress testing.** The methodology used four extreme scenario probes (corners of the input space) to identify boundary structure, directly analogous to extreme scenario testing in derivatives risk.

**Trust regions map to domain-of-model-validity bounds.** Exploitation phases used bounded trust regions around the best observed point, analogous to the domain-of-model-validity constraint in model risk frameworks.

**Adaptive mesh refinement** (from PDE-based pricing models) was identified as the natural conceptual extension — the stratified topographic assessment used in the final session is a first step toward true AMR, which would allocate more samples to high-uncertainty, high-value regions adaptively.

### Campaign Results Summary

| Fn | Description | Dims | Initial Best | Campaign Best | Best Week | Improvement |
|----|------------|------|-------------|--------------|-----------|------------|
| F1 | Radiation field detection | 2 | 0.000000 | 1.452581 | Week 7 | Located spike |
| F2 | Noisy ML log-likelihood | 2 | 0.611205 | 0.745754 | Week 6 | +22% |
| F3 | Drug discovery (side-effect min) | 3 | -0.034835 | -0.007182 | Week 8 | +79% |
| F4 | Warehouse allocation (cost min) | 4 | -4.025542 | +0.491925 | Week 9 | +112% |
| F5 | Chemical yield optimisation | 4 | 1,088.860 | 8,636.766 | Week 12 | +694% |
| F6 | Recipe scoring | 5 | -0.714265 | -0.094440 | Week 12 | +87% |
| F7 | ML hyperparameter tuning | 6 | 1.364968 | 3.148939 | Week 12 | +130% |
| F8 | Neural network tuning | 8 | 9.598482 | 9.983487 | Week 12 | +4% |

Week 13 (Module 24) results pending. Table will be updated on receipt.

### Repository Structure (Stage 2)

```
stage2_bbo/
    src/
        data_loader.py       Data loading and combined dataset construction
        gp_core.py           GP surrogate, UCB and PI acquisition, kernel fitting
        diagnostics.py       21-test diagnostic battery (IV, C, A series)
        week_runner.py       Weekly session orchestration
    reflections/
        module_12_week01_reflection.md
        module_13_week02_reflection.md
        ...
        module_23_week12_reflection.md
        module_24_week13_reflection.md   (pending Week 13 results)
    evidence/
        week01/ ... week13/  Evidence workbooks and dashboards
    context/
        PROJECT_CONTEXT_Wk13_Final.md   Complete campaign state document
    datasheet_bbo_dataset.md
    model_card_bbo_optimiser.md
    README.md
```

---

## AI Use Disclosure

This project was completed with AI assistance (Claude, Anthropic) for code scaffolding, analysis drafting, and document production. All strategic decisions, portal submissions, and discussion board posts were made by Srini Rajasekaran. Imperial College London confirmed via ticket 2821471 that AI use is permitted with a single disclosure statement at Module 25.

---

## Contact

Srini Rajasekaran  
Director, Head of Structured Credit and Real Estate Assurance, Citibank  
Imperial College London Professional Certificate in ML & AI, 2025–2026
