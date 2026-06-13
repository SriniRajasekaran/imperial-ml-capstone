# Stage 2 — BBO Challenge

**Author:** Srini Rajasekaran  
**Programme:** Imperial College London — Professional Certificate in ML & AI

---

## Overview

Stage 2 involves optimising eight unknown black-box functions of increasing dimensionality (2D to 8D). One query is submitted per function per week (Modules 12 to 24). The goal is to find the maximum of each function through iterative, evidence-based querying.

---

## Query Strategy

Two methods are evaluated each week before selecting a final query:

**1. Multivariate OLS Regression**
Fitted with standardised inputs. Four diagnostic gates must pass before using regression:
- R² above 0.30
- Normal residuals (Shapiro-Wilk p above 0.05, skewness near zero, kurtosis near zero)
- No autocorrelation (Durbin-Watson between 1.5 and 2.5)
- Predicted y exceeds current best

Only statistically significant beta coefficients (p below 0.05) are used to set query direction. Insignificant inputs are set to 0.50.

**2. Gaussian Process with UCB Acquisition**
UCB(x) = GP mean(x) + beta x GP std(x)

Beta=2.0 (Week 1) corresponds to the 95th percentile of the GP predictive distribution. 8,000 random candidates are evaluated per function. Beta is adjusted as data accumulates.

---

## Repository Structure

```
stage2_bbo/
    data/
        module_12/          # Initial 10-40 data points per function
        module_13/          # Updated after Week 2 results
        ...
    notebooks/
        module_12_week1_queries.ipynb   # Full analysis and query generation
        module_13_week2_queries.ipynb   # Added each week
        ...
    reflections/
        module_12_reflection.md         # Posted to Imperial discussion board
        module_13_reflection.md
        ...
    evidence/
        module_12_submission_evidence.xlsx    # 8,000 candidate UCB analysis
        module_12_regression_diagnostics.xlsx # Full regression diagnostics
        module_12_regression_ready.xlsx       # Excel regression input layout
```

---

## Weekly Progress

| Module | Functions improved | Method | Notes |
|--------|-------------------|--------|-------|
| 12 | Pending results | Regression (F6, F8), GP-UCB (rest) | Week 1 baseline |

*Table updated each week as results are returned.*

---

## Function Reference

| Function | Dims | Description |
|----------|------|-------------|
| F1 | 2D | Radiation field detection |
| F2 | 2D | Noisy ML log-likelihood |
| F3 | 3D | Drug discovery (side effects) |
| F4 | 4D | Warehouse allocation |
| F5 | 4D | Chemical yield optimisation |
| F6 | 5D | Recipe scoring |
| F7 | 6D | ML hyperparameter tuning |
| F8 | 8D | Neural network tuning |
