# Credit Model Risk Audit Toolkit

**Imperial College London — Professional Certificate in ML & AI**  
**Stage 1 Capstone | Author: Srini Rajasekaran**

---

## What This Is

An automated model risk audit toolkit applied to the Polish Companies Bankruptcy dataset (UCI ID 365), framed as an independent challenge of an XGBoost credit early-warning model.

The toolkit goes beyond standard SHAP importance charts. It produces four quantitative diagnostics that generate RAG-rated findings against calibrated governance thresholds — structured as audit artefacts, not analytics dashboards.

---

## Folder Structure

```
credit_model_risk_audit_toolkit/
├── data/
│   ├── polish_bankruptcy.csv     10,503 firms, 64 financial ratios, binary bankruptcy target
│   └── feature_map.json          X1–X64 to credit analyst label mapping
├── notebooks/
│   ├── 01_data_eda_mathematical_foundations.ipynb
│   ├── 02_preprocessing_generalisation_evaluation.ipynb
│   ├── 03_knn_decision_trees_ensembles.ipynb
│   ├── 04_naive_bayes_logistic_svm.ipynb
│   └── 05_xgboost_shap_audit_toolkit.ipynb     ← core audit toolkit
└── reports/                      22 PNGs generated when notebooks are run
```

---

## Run Order

Notebooks must be run in sequence: **01 → 02 → 03 → 04 → 05**

Each notebook is self-contained (no pickle dependencies between them), but the pedagogical flow builds from EDA through preprocessing through model comparison to the final XGBoost audit.

Notebook 05 is the primary deliverable. It trains XGBoost on the Polish Bankruptcy dataset and runs all four audit diagnostics.

---

## Required Packages

```
pip install xgboost shap imbalanced-learn scikit-learn pandas numpy matplotlib seaborn scipy
```

All other dependencies (json, warnings, glob) are Python standard library.

---

## Audit Diagnostics

| # | Diagnostic | Metric | Amber | Red | Regulatory |
|---|---|---|---|---|---|
| A | Explanation Stability | Bootstrap variance (normalised) | >0.15 | >0.30 | NIST AI RMF GOVERN 1.2 |
| B | Explanation Drift | Wasserstein distance (t1→t2) | >0.10 | >0.25 | PRA SS1/23 §3.4 |
| C | Explanation Sensitivity Ratio | ESR per feature | >1.5 | >3.0 | SR 11-7 §IV |
| D | Local Consistency | Behavioural cluster count | >2 clusters | — | EU AI Act Art. 13 |

---

## Confirmed Results (August 15, 2026)

- XGBoost Test AUC: 0.9998 | F1: 0.9744
- Diagnostic A: GREEN (0 RED, 0 AMBER)
- Diagnostic B: AMBER (2 AMBER features — X8, X2)
- Diagnostic C: RED (all 64 features exceed ESR threshold — mean ESR ~115,000)
- Diagnostic D: AMBER (4 behavioural clusters; Cluster 0 = 82.4% bankruptcy rate)

The ESR RED finding is the central result: explanation sensitivity is orders of magnitude larger than prediction sensitivity across all features. The model passes conventional performance checks but fails SR 11-7 independent challenge on explanation robustness.

---

## How to Launch

Open Anaconda Prompt and navigate to the `notebooks/` folder:

```bash
cd path/to/credit_model_risk_audit_toolkit/notebooks
jupyter notebook
```

Then open each notebook and run: **Kernel → Restart & Run All**
