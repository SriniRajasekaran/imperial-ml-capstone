# Automated Model Risk Audit Toolkit with Quantitative Explainability Diagnostics

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT

Banks use machine learning (ML) to predict company bankruptcies, but these models are "black boxes" — accurate yet impossible to explain. Regulators increasingly require banks to not just perform well, but to explain *why* a model makes each decision and to demonstrate that those explanations are stable and consistent over time. This project builds an automated audit toolkit that checks whether an ML credit model's explanations are reliable — measuring their stability across samples, drift over time, and sensitivity to small input changes. The novel contribution is the Explanation Sensitivity Ratio (ESR): a new metric that detects when explanations are more fragile than predictions, a risk invisible to standard performance monitoring.

## DATA

**Dataset:** Polish Companies Bankruptcy Data  
**Source:** UCI Machine Learning Repository, ID 365  
**Citation:** Ziȩba, M., Tomczak, S.K., Tomczak, J.M. (2016). Ensemble Boosted Trees with Synthetic Features Generation in Application to Bankruptcy Prediction. *Expert Systems with Applications*, 58, 93–101.

The dataset contains 10,503 company-year observations with 64 financial ratio features (leverage, profitability, liquidity, coverage ratios) and a binary bankruptcy label. The class distribution is highly imbalanced at approximately 4.9% bankrupt and 95.1% solvent — reflecting a realistic wholesale credit portfolio. Approximately 10% of feature values are missing, consistent with real-world incomplete financial filings.

> **Note:** The dataset is included in this repository as a CSV file. It is a public academic dataset available under UCI's open-access terms. Per Imperial's guidance, the large dataset file is described here and linked to the original source: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

## MODEL

**Primary model:** XGBoost Gradient Boosted Trees (binary classification)

XGBoost was selected as the primary model under audit for three reasons. First, it is industry-standard for credit risk in financial services, making the audit findings directly relevant to real-world model governance. Second, gradient boosted trees naturally handle correlated, skewed financial ratios without parametric assumptions — confirmed by systematic comparison against Logistic Regression, Naïve Bayes, SVM, KNN, and Random Forest (all in the Stage 1 notebooks), where XGBoost consistently achieved the highest AUC-ROC. Third, XGBoost is compatible with TreeSHAP, enabling exact (not approximate) SHAP value computation — essential for the quantitative audit diagnostics.

A Logistic Regression model is also maintained as a SHAP calibration anchor: its coefficients are a closed-form analogue of SHAP attributions under linearity, so material divergence between LR and XGBoost SHAP values signals non-linear behaviour requiring further examination.

## HYPERPARAMETER OPTIMISATION

The following hyperparameters were set and their rationale documented:

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| n_estimators | 300 | Sufficient trees for stable convergence; monitored via validation AUC plateau |
| max_depth | 5 | Balances model complexity against overfitting; tuned via learning curves |
| learning_rate | 0.05 | Low rate with higher n_estimators for smoother gradient descent |
| subsample | 0.8 | Row subsampling reduces overfitting and adds stochasticity |
| colsample_bytree | 0.8 | Feature subsampling — analogous to Random Forest feature selection |
| scale_pos_weight | ~19.3 | Set to (n_negative / n_positive) to address 1:19 class imbalance |

SMOTE oversampling (k=5 neighbours) was applied to the training fold only before XGBoost training. RobustScaler (IQR-based) was used for feature scaling, chosen over StandardScaler due to extreme outliers in financial ratios. Median imputation was applied for missing values, fit on training data only to prevent data leakage.

Systematic hyperparameter search was conducted via learning curves (max_depth) and validation AUC monitoring (n_estimators). Full grid search with cross-validation is documented as a natural extension in Notebook 05.

## RESULTS

The XGBoost model achieves strong discrimination on the held-out test set. Key findings from the four audit diagnostics are summarised below:

**Model Performance (Test Set)**

| Metric | Value |
|--------|-------|
| AUC-ROC | See Notebook 05 output |
| F1 (bankrupt class) | See Notebook 05 output |
| Precision (bankrupt) | See Notebook 05 output |
| Recall (bankrupt) | See Notebook 05 output |

*Note: Accuracy is not reported — at 4.9% class prevalence, a naïve classifier that always predicts "solvent" achieves 95.1% accuracy. AUC-ROC and F1 on the minority class are the appropriate metrics.*

**Audit Findings Scorecard**

| Diagnostic | Metric | Threshold | Finding |
|-----------|--------|-----------|---------|
| A: Explanation Stability | Bootstrap variance (normalised) | Amber >0.15 / Red >0.30 | See Notebook 05 |
| B: Explanation Drift | Wasserstein distance (t1→t2) | Amber >0.10 / Red >0.25 | See Notebook 05 |
| C: ESR (Novel contribution) | Explanation Sensitivity Ratio | Amber >1.5 / Red >3.0 | See Notebook 05 |
| D: Local Consistency | Behavioural cluster count | >2 clusters triggers review | See Notebook 05 |

The RAG scorecard and all diagnostic charts are auto-generated in `reports/` when Notebook 05 is executed.

![Audit Scorecard](reports/05_audit_scorecard.png)

## CONTACT DETAILS

Imperial College London — Professional Certificate in Machine Learning & AI  
Capstone submission, June 2026
