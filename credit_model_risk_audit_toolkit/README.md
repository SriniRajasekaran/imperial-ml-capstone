# Automated Credit Risk Audit Toolkit
### Quantitative Explainability Diagnostics for ML Credit Models

**Imperial College London | Professional Certificate in ML & AI | Capstone Project**
**Author: Srini Rajasekaran | August 2026**

---

## Overview

A modular Python-based audit framework that quantitatively evaluates the stability, robustness, and drift of ML model explanations using SHAP-based diagnostics. Built on the Polish Companies Bankruptcy Dataset (UCI ID 365, Year 3 cohort, 10,503 firms).

The primary contribution is not the classifier — it is the **explainability diagnostic framework**: quantitative, threshold-based, regulator-aligned audit findings that constitute evidence, not charts. The toolkit operationalises independent challenge for ML credit models as required under SR 11-7, PRA SS1/23, and EU AI Act Article 13.

---

## Central Finding

**SHAP agrees with LR (Spearman rho = 0.93, p < 0.0001).**

Both LR coefficients and XGBoost SHAP importances independently rank the same five features first, in the same order. This confirms the credit relationships in this dataset are near-linearly separable. XGBoost uses 400 trees to approximate a linear function that logistic regression expresses directly.

This finding drives the governance conclusion: **if XGBoost is deployed, SHAP controls are a necessary condition — not a supplementary feature.** The rho=0.93 agreement validates that the SHAP explanations are substantive. Without SHAP, an analyst cannot know whether the model uses features consistently across cohorts, and none of the three regulatory obligations below can be discharged.

---

## Model Selection — Evidence Summary

Both models evaluated identically: same features, same OOT holdout (2,627 firms, 18.1% bankrupt), same cost-ratio thresholds.

| Metric | LR (Primary) | XGBoost (Challenger) | Winner |
|---|---|---|---|
| FN at CR=2:1 — missed defaults | **4** | 30 | **LR** |
| FN at CR=5:1 — regulatory standard | **2** | 22 | **LR** |
| OOT AUC-ROC | **0.9989** | 0.9978 | **LR** |
| OOT AUC-PR | **0.9956** | 0.9914 | **LR** |
| OOT Brier Score | **0.0121** | 0.0148 | **LR** |
| OOT Log-Loss | **0.044** | 0.053 | **LR** |
| SHAP-LR coefficient correlation | rho = 0.93 | — | Confirms LR primacy |

XGBoost wins on CV Brier/Log-Loss within training folds — attributable to `scale_pos_weight` optimisation for the 0.5% training imbalance, not generalisable OOT superiority. Selection is data-driven across four independent test dimensions.

---

## 7-Feature Parsimonious Specification

Derived from convergent evidence: LR coefficient ranking, XGBoost SHAP ranking, and ablation backtest across 8 feature configurations. All three methods agree.

| Rank | Code | Label | Direction |
|---|---|---|---|
| 1 | X3 | Working capital / Total assets | Lower → higher risk |
| 2 | X4 | Current assets / ST liabilities | Lower → higher risk |
| 3 | X8 | Equity / Total liabilities | Lower → higher risk |
| 4 | X7 | EBIT / Total assets | Lower → higher risk |
| 5 | X5 | Cash / Total assets | Lower → higher risk |
| 6 | X2 | Total liabilities / Total assets | **Higher → higher risk** |
| 7 | X6 | Retained earnings / Total assets | Lower → higher risk |

LR AUC = 0.9980 (99.9% of full 14-feature) with FN = 5 vs 4 full. Also include MNAR_X2 and MNAR_X6 — firms that suppress leverage or retained earnings reporting are disproportionately distressed.

---

## Audit Diagnostic RAG Scorecard

Applied to XGBoost — the model an institution would deploy in an ML-first context.

| Diagnostic | RAG | Regulatory Basis | Finding |
|---|---|---|---|
| A: Explanation Stability | **AMBER** | SR 11-7, NIST GOVERN 1.2 | Moderate bootstrap variance. Acceptable for ensemble. |
| B: Explanation Drift | **RED** | PRA SS1/23 §3.4 | SHAP distributions shifted train→OOT. Performance metrics did not flag this. |
| C: Explanation Sensitivity (ESR) | **RED** | SR 11-7 model behaviour | ESR elevated. Consistent with non-linear model on near-linear data. |
| D: Local Consistency | **AMBER** | EU AI Act Art. 13 | SHAP-LR rho=0.93. Global agreement confirmed. |

**Overall: GREEN / AMBER / RED / AMBER**

The two RED findings share a root cause: XGBoost models a near-linear problem non-linearly. If XGBoost is retained as challenger, Diagnostics B and C require continuous SHAP monitoring.

---

## SHAP as a Necessary Condition for Non-Linear ML in Credit

SHAP controls discharge three non-substitutable regulatory obligations:

| Function | Regulatory Basis | Why SHAP is Required |
|---|---|---|
| Independent challenge | SR 11-7 §IV | Only tractable mechanism for auditing a tree ensemble at the decision level |
| Individual-level transparency | EU AI Act Art. 13 | Local SHAP is the only firm-level explanation artefact |
| Behavioural monitoring | PRA SS1/23 §3.4 | Explanation drift detected regime shift before performance degraded |

---

## Repository Structure

```
credit_model_risk_audit_toolkit/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_shap_explainability.ipynb
│   ├── 04_audit_diagnostics.ipynb
│   └── 05_xgboost_shap_audit_toolkit_v2.ipynb
├── data/
│   ├── polish_bankruptcy.csv
│   └── feature_map.json
├── reports/
│   └── [diagnostic charts]
├── docs/
│   ├── Model_Documentation_v3.docx
│   ├── Model_Selection_Report.docx
│   ├── Credit_Assessment_Three_Firms.docx
│   ├── model_card.md
│   └── data_sheet.md
├── results/
│   ├── Consolidated_Test_Results.xlsx
│   ├── Feature_Weight_Backtest.xlsx
│   └── Credit_Portfolio_Assessment.xlsx
└── README.md
```

---

## Technical Stack

- Python 3.12 | numpy 1.26.4 | scikit-learn 1.4.2
- xgboost 2.1.1 | shap 0.45.1 | imbalanced-learn 0.12.3
- Dataset: UCI ID 365, Polish Companies Bankruptcy, Year 3 cohort

---

## Regulatory Alignment

| Diagnostic | Framework | Specific Provision |
|---|---|---|
| Explanation drift (Wasserstein/KL/KS) | PRA SS1/23 | §3.4 ongoing monitoring |
| SHAP independent challenge + ESR | SR 11-7 | §IV independent challenge |
| Local SHAP attributions | EU AI Act | Art. 13 transparency |
| Stability + stress SHAP | NIST AI RMF | GOVERN 1.2 |

---

*This repository demonstrates evidence-based model selection, quantitative explainability diagnostics, and regulatory-aligned independent challenge for ML credit models.*
