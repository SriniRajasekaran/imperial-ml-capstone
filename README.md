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

Stage 2 — BBO Challenge (Modules 12–24)
Current round: 4 of 13 complete. W4 queries submitted, results pending.
Running bests after Round 3
Function	Description	Initial best	Current best	Gain vs initial
F1 (2D)	Radiation field detection	0.000	0.000	No signal found
F2 (2D)	Noisy ML log-likelihood	0.611	0.611	Unrepeated initial spike
F3 (3D)	Drug discovery	-0.035	-0.022	+0.013
F4 (4D)	Warehouse allocation	-4.026	+0.390	+4.42
F5 (4D)	Chemical yield	1088.9	4128.3	+3039.4
F6 (5D)	Recipe scoring	-0.714	-0.546	+0.169
F7 (6D)	ML hyperparameter tuning	1.365	2.726	+1.361
F8 (8D)	Neural network tuning	9.598	9.846	+0.248
Week 4 (Module 15) portal strings
```
F1: 0.050000-0.950000
F2: 0.691130-0.898575
F3: 0.390888-0.247459-0.445221
F4: 0.436185-0.389492-0.378432-0.414376
F5: 0.966254-0.954711-0.973370-0.958107
F6: 0.050000-0.500000-0.500000-0.950000-0.050000
F7: 0.008484-0.134504-0.272871-0.262378-0.266104-0.759966
F8: 0.044068-0.389868-0.184727-0.160732-0.755472-0.375973-0.216312-0.508971
```
Methodology (as of Round 4)
Surrogate model: Gaussian Process with kernel selected per function by
leave-one-out Q2 cross-validation. Four kernels tested each round: RBF,
Matérn-3/2, Matérn-5/2, Mixture RBF. Current assignments: Mixture RBF for F4
(Q2=0.976), Matérn-5/2 for F5 (Q2=0.912) and F7 (Q2=0.888), RBF for F8 (Q2=0.915).
Acquisition functions: UCB primary. Expected Improvement and Probability of
Improvement tested alongside for exploitation-phase functions. UCB, EI, and PI
agree on identical query for F5 — strongest available confirmation of direction.
Candidate generation: Sobol quasi-random sequences (16,000 candidates).
Five-seed stability check applied; stable functions use 5-seed mean, unstable
use best individual seed.
Trust regions: Symmetric L2 ball constraints applied to F3 (r=0.20, first
positive predicted improvement) and F8 (r=0.40, UCB=EI agree within ball).
Regression: Applied to F6 this round — first clean four-gate pass (R2=0.738,
DW=1.917, Shapiro-Wilk p=0.400). Significant predictors: x1 negative, x4 positive,
x5 negative.
Gradient ascent: L-BFGS-B refinement of UCB argmax for exploitation-phase
functions. Improved F7 acquisition score from 3.117 to 3.423.
4-Corners (model-free): Applied to F1 where GP is ineligible (Q2=-0.198)
and all 14 observations return near-zero output. Top-left corner (x1=0.05, x2=0.95)
is the most unexplored region by L2 distance from all prior observations.
Evidence and reflections
All evidence workbooks, reflection documents, and query records are in `stage2_bbo/`.
Initial Imperial-provided datasets (n=10–40 per function) are in
`stage2_bbo/data/initial/`.