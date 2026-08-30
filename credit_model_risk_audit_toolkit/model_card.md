# Model Card — XGBoost Credit Early Warning System

*Automated Model Risk Audit Toolkit*  
*Capstone Project, August 2026*

---

## Model Description

**Input:**  
14 curated financial ratio features selected from the original 64-feature set, covering five credit categories: profitability (X1, X7), leverage (X2, X8), liquidity (X4, X5), coverage (X18, X48), and activity/working capital (X3, X6, X9, X28, X44, X29). Features were selected on the basis of economic interpretability in corporate credit analysis, non-redundancy (no near-duplicate ratio pairs), missingness below 40%, and absence of hard leakage signals (point-biserial |r| < 0.60 with the target).

For features with missingness exceeding 5% in the training set, binary MNAR (Missing Not At Random) indicator columns are appended alongside the imputed values. This preserves the structural signal that unsubmitted financial data carries — particularly relevant for bankrupt firms, which exhibit materially higher missing rates than solvent firms.

Full feature definitions are in `data/feature_map.json`. Preprocessing: median imputation (training medians applied to OOT), RobustScaler (IQR-based, fitted on training only), MNAR indicators appended post-imputation.

**Output:**  
A probability score in [0, 1] representing the estimated likelihood of company bankruptcy within the 3-year forecast horizon. Default decision threshold 0.5; threshold should be calibrated to the institution's asymmetric cost structure (missed default vs false alarm) before operational use.

**Model Architecture:**  
XGBoost Gradient Boosted Trees (binary:logistic objective). 400 trees of maximum depth 4, trained sequentially. Key configuration: learning_rate 0.04, subsample 0.80, colsample_bytree 0.80, min_child_weight 5, scale_pos_weight ~186 (native imbalance correction — ratio of solvent to bankrupt firms in the training partition). eval_metric: AUC-PR (area under precision-recall curve, more informative than AUC-ROC at 5% event rates).

Class imbalance is handled natively via scale_pos_weight inside the XGBoost loss function. SMOTE synthetic oversampling is not used. All training observations are real obligors.

---

## Performance

**Evaluation data:**  
A pseudo-OOT (out-of-time) holdout partition constructed by composite financial distress score. Observations in the top quartile of a distress composite (high leverage + low profitability + low liquidity + low coverage) form the OOT holdout (~2,627 firms, ~18% bankruptcy rate). The remaining 75% form the training pool (~7,876 firms, ~0.5% bankruptcy rate). This proxies a stressed late-period cohort in the absence of a calendar year column in the available data file.

**Limitation on partitioning:** The repository contains only the UCI Year 3 cohort file (3-year ahead prediction, 10,503 rows). A true calendar-based temporal split requires the 5-year UCI cohort files. The distress-score OOT split is explicitly a proxy and is documented as a governance limitation.

**Primary metrics:** AUC-PR (primary for imbalanced credit data) and F1 on the bankrupt class. AUC-ROC is reported for reference but is not the primary governance metric at 5% event rates.

**Performance results (v2 notebook):**

| Metric | Training | OOT Holdout | Degradation |
|--------|----------|-------------|-------------|
| AUC-ROC | 1.0000 | 0.9979 | -0.0021 |
| AUC-PR | 0.9983 | 0.9916 | -0.0067 |
| F1 (bankrupt class) | 0.9767 | 0.9430 | -0.0338 |

OOT classification report (threshold = 0.5): Bankrupt precision 0.98, recall 0.91, F1 0.94 (n=475 bankrupt firms in OOT holdout). Solvent precision 0.98, recall 1.00, F1 0.99 (n=2,152 solvent firms).

**Note on near-perfect AUC:** AUC above 0.99 across all model architectures tested on this dataset reflects the near-linear separability of the Polish Bankruptcy feature space — not data leakage or overfitting. A naive classifier that always predicts solvent achieves 95% accuracy. The relevant discriminatory power measure is AUC-PR, which degrades more meaningfully between training and OOT and provides a cleaner signal of generalisation.

**Audit diagnostic results (v2 notebook):**

| Diagnostic | Metric | Result | Status | Regulatory Framework |
|-----------|--------|--------|--------|---------------------|
| A: Explanation Stability | Normalised bootstrap variance | 8 RED / 0 AMBER features | AMBER | NIST AI RMF GOVERN 1.2 |
| B: Explanation Drift | Wasserstein distance (train vs OOT) | 5 RED / 0 AMBER features | RED | PRA SS1/23 §3.4 |
| C: ESR (novel) | Explanation Sensitivity Ratio | 14 RED / 0 AMBER features | RED | SR 11-7 §IV |
| D: Local Consistency | Behavioural cluster heterogeneity | 4 distinct clusters | AMBER | EU AI Act Art. 13 |

The RED finding on Diagnostic B (Drift) is the most significant change from v1. The proper OOT holdout reveals that SHAP distributions shift materially between the training and stressed holdout populations, even though AUC-ROC degradation is only 0.21%. This is explanation drift invisible to performance monitoring — the central governance insight the toolkit is designed to surface.

---

## Limitations

- **Temporal partitioning proxy:** The OOT split uses a composite distress score as a cohort proxy. This is not equivalent to a calendar-based split. A true temporal partition requires the 5-year UCI cohort files (Year 1 training, Year 5 OOT), which are not available in this repository.

- **Single dataset, single jurisdiction:** Trained on Polish companies under Polish GAAP. Financial ratio norms, accounting standards, and insolvency law differ across jurisdictions. Applying this model to UK, US, or other markets without retraining and revalidation would constitute a model risk finding under SR 11-7.

- **ESR epsilon dependency:** The Explanation Sensitivity Ratio is computed at epsilon = 0.10. ESR magnitude is not scale-free — results vary with the perturbation size. Sensitivity analysis across epsilon = {0.01, 0.05, 0.10, 0.20} is required before citing ESR values in regulatory submissions.

- **Governance thresholds are illustrative:** Amber/Red cutoffs are set from first principles. Calibration to institution-specific historical model behaviour and peer benchmarks is required before operational use.

- **No fairness assessment:** The dataset contains no demographic variables. Fairness analysis against protected characteristics cannot be conducted and would be required before any real-world credit application. The four behavioural clusters identified in Diagnostic D warrant sectoral or geographic concentration analysis.

- **MNAR indicators are structurally correct but unvalidated:** The binary missing indicators capture the reporting behaviour signal. Their individual predictive contribution has not been isolated via ablation study.

---

## Trade-offs

- **14 features vs 64 features:** The curated feature set sacrifices marginal predictive coverage for economic legibility and audit tractability. The AUC-PR degradation from reducing features is immaterial (< 0.1%). The governance benefit — SHAP explanations that a credit analyst can read and challenge — is substantial.

- **scale_pos_weight vs SMOTE:** Native imbalance weighting keeps training data real. SMOTE achieves similar recall gains but introduces synthetic observations into the training set, which means SHAP explanations on SMOTE-augmented data partially reflect interpolated firms. For an audit toolkit where SHAP evidence is the primary deliverable, real-data-only training is the correct design choice.

- **Recall vs precision:** At threshold 0.5, bankrupt recall is 91% on the OOT holdout. Lowering the threshold increases recall at the cost of false alarms. The optimal threshold depends on the asymmetric cost structure of the specific portfolio.

- **Computational cost of ESR:** The ESR diagnostic runs SHAP inference for each feature perturbation, scaling as O(n_features × n_samples). At 14 features and 300 OOT samples this is fast. Extending to full production monitoring would require batching.
