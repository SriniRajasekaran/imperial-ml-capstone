# Datasheet — Polish Companies Bankruptcy Dataset

*Completed for: Imperial College London — Professional Certificate in ML & AI*  
*Capstone Project: Automated Model Risk Audit Toolkit*

---

## Motivation

- **For what purpose was the dataset created?**  
  The dataset was created to support research into corporate bankruptcy prediction using financial statement data. It enables development and benchmarking of ML models for predicting company insolvency within 1–5 year forecasting horizons — directly applicable to wholesale credit risk management in financial institutions.

- **Who created the dataset and on behalf of which entity? Who funded it?**  
  Created by Maciej Ziȩba, Sebastian K. Tomczak, and Jakub M. Tomczak at Wrocław University of Technology, Poland. Published as part of academic research; no commercial funding is disclosed. Full citation: Ziȩba, M., Tomczak, S.K., Tomczak, J.M. (2016). *Ensemble Boosted Trees with Synthetic Features Generation in Application to Bankruptcy Prediction*. Expert Systems with Applications, 58, 93–101.

---

## Composition

- **What do the instances represent?**  
  Each instance is a company-year observation: one Polish company observed during one annual reporting period, labelled by whether the company filed for bankruptcy within the forecasting window (1–5 years).

- **How many instances of each type are there?**  
  10,503 total company-year observations: approximately 517 bankrupt (4.9%) and 9,986 solvent (95.1%). This reflects a realistic wholesale credit portfolio default rate and creates a significant class imbalance requiring explicit handling (SMOTE oversampling, class-weighted loss).

- **Is there any missing data?**  
  Yes — approximately 10% of feature values are missing across the 64 financial ratio features. Critically, the missingness is not random: bankrupt firms exhibit materially higher missing rates than solvent firms, violating the Missing At Random (MAR) assumption. This means imputation strategy is a model risk consideration, not merely a preprocessing convenience. Missingness indicator flags are recommended as additional features in production deployment.

- **Does the dataset contain confidential data?**  
  No. All data is derived from publicly available Polish company financial registry filings. No personal data, individual identifiers, or legally privileged information is present. The dataset contains only aggregated financial ratios — no raw financial statements.

---

## Collection Process

- **How was the data acquired?**  
  From publicly available Polish court registers and financial disclosure databases. Financial ratios were computed from annual company accounts filed with public registries.

- **Sampling strategy?**  
  Not fully described in the original paper. Assumed to be a convenience sample of companies with available multi-year financial filings in the Polish registry system. Selection bias towards firms with complete filing histories is possible.

- **Over what time frame was the data collected?**  
  Specific years are not disclosed in the UCI metadata. The multi-year forecasting structure (1–5 year horizons) implies the dataset spans at least a 6-year period. The economic cycle covered is unknown, which limits stress-scenario generalisability.

---

## Preprocessing/Cleaning/Labelling

- **Was any preprocessing done?**  
  The UCI dataset provides pre-computed financial ratios (not raw financial statements). For this capstone project, the following additional preprocessing was applied:
  - **Imputation:** Median imputation per feature, fit on training data only to prevent data leakage
  - **Scaling:** RobustScaler (IQR-based), chosen over StandardScaler due to extreme outliers in financial ratio distributions
  - **Class imbalance handling:** SMOTE oversampling (k=5 nearest neighbours) applied to the training fold only — never to validation or test sets
  - **Temporal split:** First 60% of observations treated as historical period (t1) for training; remaining 40% as later period (t2) for explanation drift analysis
  - **Feature mapping:** Raw feature codes (X1–X64) mapped to credit analyst terminology (e.g., X1 → net_profit_to_total_assets) documented in `data/feature_map.json`

- **Was the raw data saved?**  
  The original feature matrix (with missing values intact) is saved as `data/polish_bankruptcy.csv`. The preprocessing pipeline is applied in-code (Notebook 02) and fit only on training data, so the raw file is the canonical source.

---

## Uses

- **What other tasks could the dataset be used for?**  
  Multi-class financial distress prediction (early warning vs distress vs default); feature importance research in credit risk; explainability method benchmarking; fairness analysis in automated credit decisions; time-series extension for panel data modelling.

- **Anything about the composition that might impact future uses?**  
  Several important limitations apply:
  - **Geographic scope:** Polish companies only. Financial ratio norms, accounting standards (Polish GAAP vs IFRS), and insolvency law differ significantly across jurisdictions. Direct application to UK, US, or other markets without revalidation would be a model risk finding under SR 11-7.
  - **Temporal scope:** The economic cycle covered is unknown. A dataset spanning only benign economic conditions would understate stress-scenario bankruptcy rates.
  - **No demographic data:** Fairness analysis against protected characteristics (gender, ethnicity, geography) is not possible with this dataset. Any credit application using this model would require separate fairness assessment.
  - **Survivorship bias:** Companies that ceased operations without formal bankruptcy proceedings may be misclassified as solvent.

- **Tasks for which the dataset should not be used?**  
  Direct deployment as a production credit scoring model without jurisdiction-specific validation, regulatory approval (PRA/FCA in UK; OCC/Fed in US), fairness assessment, and model risk sign-off. Automated individual credit decisions without human review would conflict with EU AI Act Art. 13 requirements.

---

## Distribution

- **How has the dataset been distributed?**  
  Publicly available via the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data. Also referenced in the original journal paper (Expert Systems with Applications, 2016).

- **Copyright / IP / Terms of Use?**  
  UCI Machine Learning Repository datasets are available for academic and research use. No commercial restrictions are explicitly stated. Users should cite the original paper (Ziȩba et al., 2016) when using this dataset.

---

## Maintenance

- **Who maintains the dataset?**  
  The UCI Machine Learning Repository maintains the hosted version. The original authors (Wrocław University of Technology) created it; no ongoing active maintenance is indicated. The dataset is static — no updates or corrections have been published since the 2016 paper.
