# Datasheet — Polish Companies Bankruptcy Dataset

*Automated Model Risk Audit Toolkit*  
*Capstone Project, August 2026*

---

## Motivation

- **For what purpose was the dataset created?**  
  To support research into corporate bankruptcy prediction using financial statement data. It enables development and benchmarking of ML models for predicting company insolvency within 1–5 year forecasting horizons, directly applicable to wholesale credit risk management in financial institutions.

- **Who created the dataset?**  
  Maciej Ziȩba, Sebastian K. Tomczak, and Jakub M. Tomczak at Wrocław University of Technology, Poland. Full citation: Ziȩba, M., Tomczak, S.K., Tomczak, J.M. (2016). *Ensemble Boosted Trees with Synthetic Features Generation in Application to Bankruptcy Prediction*. Expert Systems with Applications, 58, 93–101.

---

## Composition

- **What do the instances represent?**  
  Each instance is a company-year observation: one Polish company observed during one annual reporting period, labelled by whether the company filed for bankruptcy within the forecasting window.

- **Which cohort file is used in this project?**  
  The UCI repository provides five separate annual cohort files for 1-year through 5-year ahead prediction horizons. This project uses the **Year 3 cohort file only** (3-year ahead prediction). This file contains 10,503 observations and is the one available at `data/polish_bankruptcy.csv`.

  The Year 3 file was selected because it represents the most practically relevant credit early warning horizon for wholesale corporate credit portfolios — far enough ahead to be actionable, short enough to be predictable from observable financial ratios.

  The other four cohort files (Year 1: 7,027 rows; Year 2: 10,173 rows; Year 4: 9,792 rows; Year 5: 5,910 rows) are not included in this repository. Their absence means a true calendar-based temporal train/OOT split is not possible within this codebase. This is a documented limitation addressed in the v2 notebook via a composite distress score proxy partition.

- **How many instances?**  
  10,503 total: 517 bankrupt (4.92%), 9,986 solvent (95.08%). Reflects a realistic wholesale credit portfolio default rate.

- **What are the features?**  
  64 pre-computed financial ratios (X1–X64) derived from annual company accounts: profitability, leverage, liquidity, coverage, and asset efficiency ratios. Raw financial statements are not provided. Full feature definitions in `data/feature_map.json`.

  For this project, a curated subset of 14 features is used in the v2 audit notebook. Selection criteria: economic interpretability in corporate credit analysis, non-redundancy, missingness below 40%, and absence of hard leakage signals (point-biserial |r| < 0.60 with the binary target).

- **Is there missing data?**  
  Yes — approximately 10% of feature values are missing across the 64 features. Missingness is Not At Random (MNAR): bankrupt firms exhibit materially higher missing rates than solvent firms, violating the Missing At Random assumption. This is a model risk consideration, not merely a preprocessing convenience.

  The v2 preprocessing pipeline handles this by: (1) applying median imputation (training medians only, to prevent leakage) and (2) appending binary MNAR indicator columns for features exceeding 5% missingness in the training partition. The model can learn both the ratio level and the reporting behaviour signal independently.

- **Does the dataset contain confidential data?**  
  No. All data is derived from publicly available Polish company financial registry filings. No personal data, individual identifiers, or legally privileged information is present.

---

## Collection Process

- **How was the data acquired?**  
  From publicly available Polish court registers and financial disclosure databases. Financial ratios were computed from annual company accounts filed with public registries.

- **Sampling strategy?**  
  Not fully described in the original paper. Assumed to be a convenience sample of companies with available multi-year financial filings. Selection bias towards firms with complete filing histories is possible, which may understate missingness rates in the broader population.

- **Over what time frame?**  
  Specific calendar years are not disclosed in the UCI metadata. The 1–5 year forecasting structure implies the underlying data spans at least a 6-year period. The economic cycle covered is unknown, limiting stress-scenario generalisability. In particular, it is not known whether the dataset spans the 2008–2009 global financial crisis, which would materially affect the representativeness of the bankruptcy rate.

---

## Preprocessing Applied in This Project

The following preprocessing decisions are applied in `05_xgboost_shap_audit_toolkit_v2.ipynb`. The raw file (`data/polish_bankruptcy.csv`) is preserved unchanged as the canonical source.

| Step | Decision | Rationale |
|------|----------|-----------|
| Feature selection | 14 curated features from 64 | Economic interpretability; removes redundant algebraic transforms; eliminates leakage-prone features |
| Missingness screen | Exclude features > 40% missing | Above this threshold, imputed values are insufficiently grounded in observed data |
| Leakage screen | Exclude features with point-biserial \|r\| > 0.60 | Prevents near-perfect AUC driven by proxies rather than genuine credit signal |
| Imputation | Median, fitted on training partition only | Robust to financial ratio outliers; training-only fit prevents data leakage |
| MNAR indicators | Binary flags for features > 5% missing in training | Preserves the structural signal of non-reporting, which is itself a distress indicator |
| Scaling | RobustScaler (IQR-based), fitted on training only | Resistant to extreme outliers common in distressed firm financials |
| Class imbalance | scale_pos_weight = n_solvent / n_bankrupt (~186) | Native loss-function correction; avoids synthetic data in training set |
| Partitioning | Pseudo-OOT via composite distress score | Proxies stressed late-period cohort; true temporal split requires multi-year UCI files |
| Feature mapping | X1–X64 mapped to credit analyst labels | Supports SHAP output legibility for credit analysts and model risk reviewers |

---

## Uses

- **Intended uses:**  
  Model risk audit and independent challenge of ML credit models; SHAP explainability benchmarking; governance threshold calibration research; demonstration of explanation stability, drift, and sensitivity diagnostics.

- **Limitations affecting future uses:**  
  - **Geographic scope:** Polish companies only. Financial ratio norms, accounting standards (Polish GAAP vs IFRS), and insolvency law differ across jurisdictions. Direct application to UK, US, or other markets without revalidation is a model risk finding under SR 11-7.  
  - **Single cohort:** Only the Year 3 file is used. Multi-year analysis requires all five UCI cohort files.  
  - **Unknown economic cycle:** The covered period is undisclosed. Model performance under a full credit cycle downturn has not been validated.  
  - **No demographic data:** Fairness analysis against protected characteristics is not possible with this dataset.  
  - **Survivorship bias:** Companies that ceased operations without formal bankruptcy proceedings may be misclassified as solvent.

- **Tasks for which the dataset should not be used:**  
  Direct deployment as a production credit scoring model without jurisdiction-specific validation, regulatory approval, fairness assessment, and model risk sign-off. Automated individual credit decisions without human review would conflict with EU AI Act Article 13 requirements.

---

## Distribution

- **How is the dataset distributed?**  
  Publicly via the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data. Referenced in Ziȩba et al. (2016), Expert Systems with Applications, 58, 93–101.

- **Terms of use:**  
  Available for academic and research use. Users should cite the original paper when using this dataset.

---

## Maintenance

The UCI-hosted version is static — no updates or corrections have been published since the 2016 paper. The dataset is maintained as-is by the UCI repository. No ongoing active maintenance by the original authors is indicated.
