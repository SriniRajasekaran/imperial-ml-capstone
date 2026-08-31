# Model Card — Automated Credit Risk Audit Toolkit

*Credit Distress Classifier with Quantitative Explainability Diagnostics*
*Capstone Project, August 2026*
*Author: Srini Rajasekaran*

---

## Model Description

**Primary model:** Logistic Regression (L2 regularisation, C=0.1, class_weight='balanced')
**Challenger model:** XGBoost Gradient Boosted Trees (400 trees, depth 4, scale_pos_weight ~186)

Model selection was conducted across four independent test dimensions — asymmetric credit loss function, proper scoring rules (Brier Score and Log-Loss), Wilcoxon significance tests on 10-fold CV, and calibration analysis. LR was selected as primary on the basis of empirical evidence, not interpretability preference. See `docs/Model_Selection_Report.docx` for the full evidence base.

**Input:**
14 curated financial ratio features selected from the original 64-feature set, covering five credit categories: profitability (X1, X7), leverage (X2, X8), liquidity (X4, X5), coverage (X18, X48), and activity/working capital (X3, X6, X9, X28, X44, X29). Features were selected on the basis of economic interpretability in corporate credit analysis, non-redundancy, missingness below 40%, and absence of hard leakage signals (point-biserial |r| < 0.60 with the target).

For features with missingness exceeding 5% in the training set, binary MNAR (Missing Not At Random) indicator columns are appended alongside the imputed values. This preserves the structural signal that unsubmitted financial data carries — bankrupt firms exhibit materially higher missing rates than solvent firms.

Full feature definitions are in `data/feature_map.json`. Preprocessing: median imputation (training medians applied to OOT), RobustScaler (IQR-based, fitted on training only), MNAR indicators appended post-imputation.

**Output:**
A probability score in [0, 1] representing the estimated likelihood of company financial distress. Default decision threshold should be set from the institution's asymmetric cost structure. At cost ratio 2:1 the optimal threshold is 33.3%; at 5:1 it is 16.7%.

**Model Architecture — Primary (LR):**
Logistic Regression with L2 regularisation (C=0.1), balanced class weights. Coefficients are directly interpretable as marginal contributions to log-odds in the scaled feature space.

**Model Architecture — Challenger (XGBoost):**
Gradient Boosted Trees (binary:logistic). 400 trees of maximum depth 4. learning_rate 0.04, subsample 0.80, colsample_bytree 0.80, min_child_weight 5, scale_pos_weight ~186. eval_metric: AUC-PR.

Class imbalance is handled natively via balanced class weights (LR) and scale_pos_weight (XGBoost). SMOTE is not used — all training observations are real obligors, preserving the integrity of SHAP explanations.

---

## Central Finding: SHAP Agrees with LR

Spearman rank correlation between LR scaled coefficients and XGBoost mean |SHAP| values across core features: **rho = 0.93, p < 0.0001**.

Both independent methods rank the same five features first, in the same order:

| Rank | Feature | LR Rank | SHAP Rank |
|---|---|---|---|
| 1 | Working capital / Total assets (X3) | 1 | 1 |
| 2 | Current assets / ST liabilities (X4) | 2 | 4 |
| 3 | Equity / Total liabilities (X8) | 3 | 2 |
| 4 | EBIT / Total assets (X7) | 4 | 5 |
| 5 | Cash / Total assets (X5) | 5 | 3 |

This confirms the credit relationships in this dataset are near-linearly separable. XGBoost uses 400 trees to approximate a linear function that logistic regression expresses directly. This finding drives two governance conclusions: (1) LR is the correct primary model; (2) if XGBoost is deployed in an ML-first institution, SHAP controls are a necessary condition for regulatory compliance — not a supplementary feature.

---

## Parsimonious 7-Feature Specification

Derived from convergent evidence: LR coefficient ranking, XGBoost SHAP ranking, and ablation backtest across 8 feature configurations. All three methods agree on the same seven features.

| Rank | Code | Label | Direction |
|---|---|---|---|
| 1 | X3 | Working capital / Total assets | Lower → higher risk |
| 2 | X4 | Current assets / ST liabilities | Lower → higher risk |
| 3 | X8 | Equity / Total liabilities | Lower → higher risk |
| 4 | X7 | EBIT / Total assets | Lower → higher risk |
| 5 | X5 | Cash / Total assets | Lower → higher risk |
| 6 | X2 | Total liabilities / Total assets | Higher → higher risk |
| 7 | X6 | Retained earnings / Total assets | Lower → higher risk |

LR AUC = 0.9980 (99.9% of full 14-feature), FN = 5 vs 4 full at CR=2:1. Also include MNAR_X2 and MNAR_X6 — firms suppressing leverage or retained earnings reporting are disproportionately distressed.

The remaining 7 features (X1, X9, X18, X28, X44, X48, X29) individually add less than 0.001 AUC and do not recover the irreducible false negatives. The 4 firms both models miss all have missing X4 (current ratio) — a structural data gap, not a modelling failure.

See `results/Feature_Weight_Backtest.xlsx` for the full ablation evidence.

---

## Performance

**Evaluation data:**
Pseudo-OOT holdout constructed by composite financial distress score. Top-quartile distress observations form the OOT holdout (2,627 firms, 18.1% bankruptcy rate). Remaining 75% form the training pool (7,876 firms, 0.5% bankruptcy rate).

**Limitation on partitioning:** The repository contains only the UCI Year 3 cohort file. A true calendar-based temporal split requires the 5-year UCI cohort files. The distress-score OOT split is explicitly a proxy and is documented as a governance limitation.

**Head-to-head results (OOT holdout):**

| Metric | LR (Primary) | XGBoost (Challenger) | Winner |
|---|---|---|---|
| FN at CR=2:1 — missed defaults | **4** | 30 | **LR** |
| FN at CR=5:1 — regulatory standard | **2** | 22 | **LR** |
| OOT AUC-ROC | **0.9989** | 0.9978 | **LR** |
| OOT AUC-PR | **0.9956** | 0.9914 | **LR** |
| Brier Score (lower better) | **0.0121** | 0.0148 | **LR** |
| Log-Loss (lower better) | **0.044** | 0.053 | **LR** |
| CV Brier Score | 0.0042 | **0.0024** | XGBoost* |
| CV Log-Loss | 0.0177 | **0.0083** | XGBoost* |
| SHAP-LR coefficient rho | 0.93 | — | Confirms LR primacy |

\* XGBoost wins CV Brier/Log-Loss within training folds. This reflects scale_pos_weight optimisation for the 0.5% training imbalance, not generalisable OOT superiority. OOT results are the primary selection criterion.

**Note on near-perfect AUC:** AUC above 0.99 across all architectures reflects near-linear separability of the feature space — confirmed by the SHAP-LR rho=0.93 agreement. It is not leakage or overfitting.

---

## Audit Diagnostic RAG Scorecard

Applied to XGBoost — the architecture an institution would deploy in an ML-first context. The diagnostics constitute independent challenge of the challenger model.

| Diagnostic | RAG | Regulatory Basis | Finding |
|---|---|---|---|
| A: Explanation Stability | **AMBER** | SR 11-7, NIST AI RMF GOVERN 1.2 | Moderate bootstrap variance. Acceptable for ensemble. Monitor at re-validation. |
| B: Explanation Drift | **RED** | PRA SS1/23 §3.4 | SHAP distributions shifted materially between training and OOT. AUC degradation was only 0.0021 — drift was invisible to performance monitoring alone. |
| C: Explanation Sensitivity (ESR) | **RED** | SR 11-7 §IV | ESR elevated. Consistent with non-linear model applied to near-linear data. |
| D: Local Consistency | **AMBER** | EU AI Act Art. 13 | SHAP-LR rho=0.93 confirms global consistency. Local disagreement on borderline firms requires enhanced review. |

**Overall: GREEN / AMBER / RED / AMBER**

The two RED findings share a root cause: XGBoost models a near-linear problem non-linearly. If XGBoost is retained as challenger, Diagnostics B and C require continuous SHAP monitoring. Diagnostic B (Explanation Drift) is the primary governance output of this toolkit — it detected a regime shift that AUC monitoring would not have surfaced.

---

## SHAP as a Necessary Condition

If XGBoost or any non-linear ML model is deployed for credit decisioning, SHAP controls discharge three non-substitutable regulatory obligations:

| Governance Function | Regulatory Basis | Why SHAP is Required |
|---|---|---|
| Independent challenge | SR 11-7 §IV | The only tractable mechanism for auditing a tree ensemble at the individual decision level |
| Individual-level transparency | EU AI Act Art. 13 | Local SHAP is the only firm-level explanation artefact for a tree model |
| Behavioural monitoring | PRA SS1/23 §3.4 | Explanation drift detected regime shift before performance metrics degraded |

---

## Limitations

- **Temporal partitioning proxy:** The OOT split uses a composite distress score proxy. A true temporal partition requires the 5-year UCI cohort files.

- **Single dataset, single jurisdiction:** Trained on Polish companies under Polish GAAP. Applying to UK, US, or other markets without retraining and revalidation is a model risk finding under SR 11-7.

- **ESR epsilon dependency:** The Explanation Sensitivity Ratio is computed at epsilon = 0.10. Sensitivity analysis across epsilon = {0.01, 0.05, 0.10, 0.20} is required before regulatory submission.

- **Governance thresholds are illustrative:** Amber/Red cutoffs are set from first principles. Calibration to institution-specific historical model behaviour is required before operational use.

- **No fairness assessment:** The dataset contains no demographic variables. Fairness analysis is required before any real-world credit application.

- **Cost ratios are illustrative:** The institution must determine its own cost ratio from actual LGD estimates and operational review costs.

---

## Trade-offs

- **LR vs XGBoost:** LR is selected empirically. At cost ratios 2:1–10:1, LR produces materially fewer missed defaults (FN=4 vs 30 at CR=2:1). The SHAP-LR rho=0.93 agreement confirms XGBoost's non-linearity is not capturing structure LR misses.

- **7-feature vs 14-feature:** The parsimonious 7-feature specification sacrifices one additional missed default (FN=5 vs 4) for a model with half the input dimensions. Recommended production specification.

- **scale_pos_weight vs SMOTE:** Native imbalance weighting keeps training data real. SMOTE introduces synthetic observations, meaning SHAP explanations would partially reflect interpolated firms — not acceptable for an audit toolkit where SHAP evidence is the primary deliverable.

- **Recall vs precision:** At CR=5:1 the optimal threshold is 16.7%, yielding LR FN=2, FP=99. The 99 false positives are referred for enhanced manual review, not automatic adverse action. This is the correct governance response for uncertain signals.
