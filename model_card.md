# Model Card — XGBoost Credit Early Warning System

*Automated Model Risk Audit Toolkit | Imperial College London — Professional Certificate in ML & AI*  
*Capstone Project, June 2026*

---

## Model Description

**Input:**  
64 financial ratio features derived from annual company accounts, covering profitability (e.g., net profit / total assets), leverage (e.g., total liabilities / total assets), liquidity (e.g., working capital / total assets), coverage (e.g., EBIT / total assets), and asset efficiency ratios. Features are labelled X1–X64 with full credit analyst terminology documented in `data/feature_map.json`. Inputs are preprocessed via median imputation (for missing values) and RobustScaler normalisation before model inference.

**Output:**  
A probability score in [0, 1] representing the estimated likelihood of company bankruptcy within the forecasting horizon. A decision threshold of 0.5 is used by default for binary classification; threshold optimisation is demonstrated in Notebook 04 (Logistic Regression baseline) to maximise F1 on the minority class.

**Model Architecture:**  
XGBoost Gradient Boosted Trees (binary:logistic objective). An ensemble of 300 decision trees, each of maximum depth 5, trained sequentially with each tree correcting the residual errors of the previous ensemble. Key configuration: learning rate 0.05, subsample 0.8, colsample_bytree 0.8, scale_pos_weight ~19.3 (to address 1:19 class imbalance). Full hyperparameter documentation in README.md.

The model is accompanied by a Logistic Regression calibration anchor: its coefficients serve as a closed-form SHAP reference. Material divergence between LR and XGBoost SHAP attributions for a given feature signals non-linear behaviour requiring further audit scrutiny under SR 11-7 §IV.

---

## Performance

**Evaluation data:** Held-out test set (20% of observations, stratified by class). Temporal drift analysis uses a proxy temporal split: first 60% of dataset as historical period (t1), remaining 40% as later period (t2).

**Primary metrics:** AUC-ROC and F1 score on the bankrupt class. Accuracy is explicitly excluded as a metric — at 4.9% class prevalence, a naïve classifier that always predicts "solvent" achieves 95.1% accuracy, making accuracy meaningless for credit risk evaluation.

**Performance results:** See Notebook 05 (`05_xgboost_shap_audit_toolkit.ipynb`) for full evaluation outputs including ROC curve, precision-recall curve, confusion matrix, and classification report. Charts are auto-saved to `reports/`.

**Model comparison:** XGBoost outperformed all five baseline models (Naïve Bayes, Logistic Regression, SVM with RBF kernel, KNN, Random Forest) on both AUC-ROC and F1 metrics. Full comparison is in Notebook 03 and Notebook 04.

**Audit diagnostic results:** Four SHAP-based explainability diagnostics were applied with RAG-coloured findings against calibrated governance thresholds:

| Diagnostic | Regulatory Framework | Threshold |
|-----------|---------------------|-----------|
| Explanation Stability (bootstrap variance) | NIST AI RMF GOVERN 1.2 | Amber >0.15 / Red >0.30 |
| Explanation Drift (Wasserstein distance) | PRA SS1/23 §3.4 | Amber >0.10 / Red >0.25 |
| Explanation Sensitivity Ratio — ESR | SR 11-7 §IV | Amber >1.5 / Red >3.0 |
| Local Consistency (behavioural clustering) | EU AI Act Art. 13 | >2 clusters triggers review |

Full findings scorecard is auto-generated in `reports/05_audit_scorecard.png`.

---

## Limitations

- **Geographic generalisation:** Trained on Polish companies under Polish GAAP and Polish insolvency law. Financial ratio norms differ materially across jurisdictions. Applying this model to UK, US, or other markets without retraining and validation would constitute a model risk finding.

- **Temporal coverage:** The economic cycle covered by the dataset is not disclosed. The model's behaviour during a full credit cycle (including severe downturn) has not been validated. Stress-scenario explainability tests in Notebook 05 use synthetic perturbations as a proxy — not real macroeconomic stress data.

- **Missing data assumption:** Median imputation assumes missing values are not systematically related to the outcome. This assumption is violated — bankrupt firms have higher missing rates. A production model should incorporate missingness indicator flags.

- **SMOTE synthetic samples:** Oversampling generates synthetic minority-class observations that may not reflect real company profiles. SMOTE parameters (k=5) have not been optimised via cross-validation.

- **Governance thresholds:** The ESR and drift thresholds (Amber/Red) are illustrative starting points based on professional judgement. Calibration to institution-specific historical model behaviour is required before operational use.

- **No fairness assessment:** The dataset contains no demographic variables. Fairness analysis against protected characteristics cannot be conducted with this data and would be required before any real-world credit application.

---

## Trade-offs

- **Interpretability vs performance:** XGBoost outperforms Logistic Regression on AUC but is less directly interpretable. SHAP mitigates this by providing post-hoc explanations, but SHAP values are an approximation of feature contributions — not the model's internal mechanism. The ESR diagnostic is designed precisely to test how robust these approximations are.

- **Recall vs precision:** At the default 0.5 threshold, the model prioritises balanced performance. Lowering the threshold increases recall (catching more bankruptcies) at the cost of precision (more false alarms). The optimal threshold depends on the asymmetric cost structure of a specific credit portfolio — flagging a healthy company as distressed is less costly than missing a true default.

- **Computational cost of SHAP diagnostics:** TreeSHAP is exact and fast for tree models. However, the ESR diagnostic requires running SHAP inference per-feature perturbation, scaling as O(n_features × n_samples). For 64 features and 200+ samples, this is manageable in development but would require batching or approximation in production monitoring.

- **SMOTE vs no SMOTE:** SMOTE improves minority class recall but introduces synthetic data into training. Models trained without SMOTE tend to under-predict bankruptcy. Models trained with SMOTE applied to the full dataset (not just the training fold) would overestimate generalisation — a data leakage risk documented explicitly in Notebook 02.
