# Automated Model Risk Audit Toolkit with Quantitative Explainability Diagnostics

**Imperial College London — Professional Certificate in Machine Learning and Artificial Intelligence**
**Author:** Srini Rajasekaran


---

## NON-TECHNICAL EXPLANATION

Banks use machine learning to predict company bankruptcies, but these models are often opaque. Regulators increasingly require banks to not just perform well, but to explain why a model makes each decision and to demonstrate that those explanations are stable and consistent over time. This project builds an automated audit toolkit that checks whether an ML credit model's explanations are reliable, measuring their stability across samples, drift over time, and sensitivity to small input changes. The novel contribution is the Explanation Sensitivity Ratio (ESR): a new metric that detects when explanations are more fragile than predictions, a risk invisible to standard performance monitoring.

---

## DATA

**Dataset:** Polish Companies Bankruptcy Data
**Source:** UCI Machine Learning Repository, ID 365
**Citation:** Zieba, M., Tomczak, S.K., Tomczak, J.M. (2016). Ensemble Boosted Trees with Synthetic Features Generation in Application to Bankruptcy Prediction. Expert Systems with Applications, 58, 93-101.

The dataset contains 10,503 company-year observations with 64 financial ratio features (leverage, profitability, liquidity, coverage ratios) and a binary bankruptcy label. The class distribution is approximately 4.9% bankrupt and 95.1% solvent, reflecting a realistic wholesale credit portfolio. Approximately 10% of feature values are missing, consistent with real-world incomplete financial filings.

The dataset is a public academic resource available under UCI open-access terms. It is described and linked here: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

---

## MODEL

**Primary model:** XGBoost Gradient Boosted Trees (binary classification)

XGBoost was selected as the primary model under audit for three reasons. First, it is industry-standard for credit risk in financial services, making the audit findings directly relevant to real-world model governance. Second, gradient boosted trees naturally handle correlated, skewed financial ratios without parametric assumptions, confirmed by systematic comparison against Logistic Regression, Naive Bayes, SVM, KNN, and Random Forest, where XGBoost consistently achieved the highest AUC-ROC. Third, XGBoost is compatible with TreeSHAP, enabling exact SHAP value computation essential for the quantitative audit diagnostics.

A Logistic Regression model is also maintained as a SHAP calibration anchor. Material divergence between LR and XGBoost SHAP values signals non-linear behaviour requiring further examination.

---

## HYPERPARAMETER OPTIMISATION

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| n_estimators | 300 | Sufficient trees for stable convergence |
| max_depth | 5 | Balances complexity against overfitting |
| learning_rate | 0.05 | Low rate with higher n_estimators |
| subsample | 0.8 | Row subsampling reduces overfitting |
| colsample_bytree | 0.8 | Feature subsampling per tree |
| scale_pos_weight | 19.3 | Addresses 1:19 class imbalance |

SMOTE oversampling (k=5) applied to training fold only. RobustScaler used for feature scaling. Median imputation applied for missing values, fit on training data only to prevent data leakage.

---

## RESULTS

The XGBoost model achieves strong discrimination on the held-out test set. AUC-ROC and F1 on the minority class are the primary metrics. Accuracy is not reported as a naive classifier predicting all solvent achieves 95.1% accuracy, making it meaningless at this class prevalence.

**Audit Findings Scorecard**

| Diagnostic | Metric | Amber | Red |
|-----------|--------|-------|-----|
| A: Explanation Stability | Bootstrap variance (normalised) | >0.15 | >0.30 |
| B: Explanation Drift | Wasserstein distance | >0.10 | >0.25 |
| C: ESR (Novel contribution) | Explanation Sensitivity Ratio | >1.5 | >3.0 |
| D: Local Consistency | Behavioural cluster count | >2 clusters | N/A |

Full results and RAG scorecard are generated in notebooks/05_xgboost_shap_audit_toolkit.ipynb.

---

## STAGE 2 — BBO CHALLENGE

Black-box optimisation challenge across eight unknown functions of increasing dimensionality (2D to 8D). One query submitted per function per week from Module 12 to Module 24.

**Method:** Gaussian Process surrogate with UCB acquisition (beta=2.0). 8,000 candidates evaluated per function per week. Regression diagnostics (R², Shapiro-Wilk, skewness, kurtosis, Durbin-Watson, p-values per coefficient) used to validate linear structure before acting on beta coefficients. Only statistically significant inputs (p below 0.05) inform the regression query direction. Where regression fails any diagnostic gate, GP-UCB is used.

**Weekly progress, queries, evidence and reflections** are documented in stage2_bbo/.

| Function | Dims | Description |
|----------|------|-------------|
| F1 | 2D | Radiation field detection |
| F2 | 2D | Noisy ML log-likelihood |
| F3 | 3D | Drug discovery |
| F4 | 4D | Warehouse allocation |
| F5 | 4D | Chemical yield optimisation |
| F6 | 5D | Recipe scoring |
| F7 | 6D | ML hyperparameter tuning |
| F8 | 8D | Neural network tuning |

---

## REPOSITORY STRUCTURE

    imperial-ml-capstone/
        notebooks/              Stage 1 ML technique notebooks
        data/                   Polish bankruptcy dataset and feature map
        stage2_bbo/
            data/               Function data per module
            notebooks/          Weekly query notebooks
            reflections/        Weekly strategy reflections
            evidence/           Regression diagnostics and UCB analysis
        README.md
        data_sheet.md
        model_card.md
