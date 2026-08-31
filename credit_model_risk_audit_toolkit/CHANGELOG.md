# CHANGELOG

## August 2026 — Model Selection Reversal: XGBoost → Logistic Regression as Primary

### What changed

The primary model was switched from **XGBoost** to **Logistic Regression**. XGBoost is retained as the independent challenger model. All materials dated after 31 August 2026 reflect this change.

### Why this matters for earlier materials

Weekly reflections, interim submissions, and earlier notebook versions (v1 and early v2) position XGBoost as the primary model under audit. Those materials remain accurate for the period in which they were written — XGBoost was the working primary model during Weeks 1 through 4 of the capstone. The switch occurred in the final analysis phase and is documented here as the authoritative record.

### Evidence base for the switch

The switch was not made on interpretability grounds. It was driven by a formal model selection test suite run on the OOT holdout (2,627 firms, 18.1% bankrupt):

**1. Asymmetric credit loss function (primary criterion)**

At all cost ratios relevant to wholesale credit (2:1 through 10:1), LR produces fewer missed defaults:

| Cost Ratio | LR FN | XGBoost FN | LR Loss | XGBoost Loss |
|---|---|---|---|---|
| 2:1 | **4** | 30 | **63** | 73 |
| 5:1 | **2** | 22 | **109** | 128 |
| 10:1 | **1** | 17 | **162** | 192 |

In credit risk, a missed default is a loss event. A false alarm is an analyst review. These are not symmetric. LR's advantage on false negatives is the primary selection criterion.

**2. OOT proper scoring rules**

LR outperforms XGBoost on Brier Score (0.0121 vs 0.0148) and Log-Loss (0.044 vs 0.053) on the OOT holdout. XGBoost wins on CV Brier and Log-Loss within training folds — attributable to scale_pos_weight optimisation for the 0.5% training imbalance, not generalisable superiority.

**3. SHAP-LR agreement (the key analytical finding)**

Spearman rank correlation between LR scaled coefficients and XGBoost mean |SHAP| values: **rho = 0.93, p < 0.0001**.

Both independent methods rank the same five features first, in the same order. This confirms the credit relationships in this dataset are near-linearly separable. XGBoost uses 400 trees to approximate a linear function that logistic regression expresses directly. When SHAP and LR agree at rho=0.93, the non-linear model is not capturing structure the linear model misses — it is reproducing a linear function with unnecessary complexity.

**4. False negative anatomy**

Tracing which bankrupt firms each model misses: there is not a single firm that XGBoost catches and LR misses. LR catches 26 additional bankruptcies that XGBoost silently passes. XGBoost's higher precision in those 26 cases means it finds partial protective signals (e.g., positive interest coverage alongside high leverage) and routes firms to low-risk leaves. LR's linear accumulation of adverse signals catches the cumulative distress picture that XGBoost's tree routing misses.

### What XGBoost's role is now

XGBoost is retained as the independent challenger for three reasons:

1. **Disagreement signal:** Where LR flags and XGBoost clears, that disagreement is itself a governance finding requiring enhanced manual review (the Firm C case).
2. **Non-linear stress regime generalisation:** If production data develops non-linear credit dynamics not present in this dataset, XGBoost may detect them before LR. The SHAP diagnostics would surface this as a breakdown in the rho=0.93 agreement.
3. **Audit framework integrity:** The explainability diagnostics (ESR, drift, stability) are applied to XGBoost as the model an ML-first institution would deploy. The RED findings on Diagnostics B and C are the toolkit's primary governance output — they are only meaningful because XGBoost is the model under audit.

### 7-feature parsimonious specification

Separately from the model switch, a parsimonious 7-feature specification was derived and validated: X3, X8, X4, X5, X7, X2, X6. This achieves LR AUC = 0.9980 (99.9% of full 14-feature) with FN = 5 vs 4 full. Full ablation evidence in `results/Feature_Weight_Backtest.xlsx`.

### Files updated to reflect this change

| File | Change |
|---|---|
| `README.md` | Rewritten to reflect LR as primary, SHAP-LR finding as central result |
| `model_card.md` | LR as primary, XGBoost as challenger, SHAP-LR section added, 7-feature spec added |
| `data_sheet.md` | Model selection note added, near-linear separability documented as dataset property |
| `docs/Model_Documentation_v3.docx` | Full updated technical documentation |
| `docs/Model_Selection_Report.docx` | Formal SR 11-7 model selection evidence record |
| `results/Consolidated_Test_Results.xlsx` | Full head-to-head evidence across 6 sheets |
| `results/Feature_Weight_Backtest.xlsx` | SHAP-LR convergence analysis and ablation backtest |

### Files not updated (accurately reflect their period)

| File | Status |
|---|---|
| `notebooks/05_xgboost_shap_audit_toolkit_v2.ipynb` | Reflects XGBoost as primary during interim phase — remains valid as the audit diagnostic notebook applied to the challenger |
| Weekly reflections (external to this repo) | Accurately reflect the working model at the time of writing |

---

*This changelog entry constitutes the audit trail for the model selection decision. Any reviewer comparing earlier and later materials should refer to this record to understand the chronology.*
