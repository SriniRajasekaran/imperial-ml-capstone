# Stage 2 — BBO Challenge

**Author:** Srini Rajasekaran
**Programme:** Imperial College London — Professional Certificate in ML & AI

---

## Overview

Stage 2 involves optimising eight unknown black-box functions of increasing dimensionality (2D to 8D). One query is submitted per function per week (Modules 12 to 24). The goal is to find the maximum of each function through iterative, evidence-based querying.

---

## Query Strategy (current, updated through Week 5)

Two methods are evaluated each week before selecting a final query:

**1. Multivariate OLS Regression**
Fitted with standardised inputs. Four diagnostic gates must pass before using regression:
- R² above 0.30
- Normal residuals (Shapiro-Wilk p above 0.05, skewness near zero, kurtosis near zero)
- No autocorrelation (Durbin-Watson between 1.5 and 2.5)
- Predicted y exceeds current best

Only statistically significant beta coefficients (p below 0.05) are used to set query direction. Insignificant inputs are set to 0.50.

Three functions (F3, F4, F8) are permanently excluded from Regression regardless of gate outcome, following a Week 2 case where F8 passed all four gates and the submitted query still returned a worse result than before. This is the FORCE_GP list.

**2. Gaussian Process with UCB Acquisition**
UCB(x) = GP mean(x) + beta x GP std(x)

- Kernel: RBF by default, with length-scale bounds fixed at (0.15, 0.6) after an early degenerate-kernel failure. Matern-3/2, Matern-5/2, and a Mixture RBF are compared by leave-one-out Q^2 each week; the default is overridden where one of these shows a clear, repeated advantage.
- Candidate generation: Sobol sequences (replaced Latin Hypercube Sampling from Week 4 onward, which had itself replaced uniform random sampling after Week 1).
- Candidate grid size: 12,000 is the working default, set by an explicit convergence test (checking whether the UCB argmax stops moving as candidate count increases). This is re-verified per function per week rather than assumed, since some functions have needed a denser grid (up to 24,000) to fully converge.
- Seed convention: the final query is a 5-seed average, not a single draw. Where the five seeds genuinely disagree in direction, a median-of-5 or best-of-5-unanimous rule is used instead of naive averaging, and the reasoning is documented per function.
- Model-free fallback: for any function where both Regression and the GP fail their eligibility tests (currently F1), a 4-corners probe is used instead, checking GP mean and uncertainty directly at the four corners of the input space plus centre, prioritising corners that have never actually been queried.
- Ensemble checks: beyond single-kernel GPs, a kernel-averaged ensemble (blending predictions across RBF, Matern-3/2, Matern-5/2, Mixture RBF) and a bootstrap-bagged ensemble (multiple GPs fit on resampled data, to separate genuine between-model uncertainty from a single model's noise estimate) are run as cross-checks before finalising a query, particularly for functions with unstable or surprising results.

Beta is set per function from the actual prior-week outcome, not a fixed schedule; roughly 0.8-1.2 for a confirmed sustained improvement, up to 2.0-2.3 for a function with no usable signal yet.

---

## Repository Structure

**Current state (honest as of this update, not aspirational):**

```
stage2_bbo/
    data/
        module_12/          # Initial 10-40 Imperial-provided data points per function (Week 1)
                             # Weeks 2-5 do not yet have a corresponding data/ folder
    notebooks/
        module_12_week1_queries.ipynb   # Only notebook currently in the repo
                                         # Weeks 2-5 diagnostics were run but not yet committed
                                         # as executed notebooks - this is a known gap being fixed
    reflections/
        module_12_reflection.md
        module_13_week2_reflection.md
        module_15_1_reflection_v9_1.docx
        module_15_2_reflection_v2_1.docx
        # Reflections are posted to the Imperial discussion board, which is the actual
        # submission requirement. Keeping copies here is optional archival, not a
        # completion requirement - existing files are left as-is, no obligation to
        # backfill Module 14 or add future weeks here.
    evidence/
        module_12_regression_diagnostics.xlsx
        module_12_regression_ready.xlsx
        module_12_submission_evidence.xlsx
        week2/
            week2_regression_vs_gp_comparison.xlsx
            week2_results_and_week3_query_evidence.xlsx
        week4/
            bbo_week4_FINAL_v2_1.xlsx
        # Naming convention is inconsistent: Week 1 uses "module_12_*" directly under
        # evidence/, while Weeks 2 and 4 use "evidence/week{n}/" subfolders. There is
        # no week1/ or week3/ subfolder. Consolidating this onto a single convention
        # (evidence/week{n}/ for every week, including a backfilled week1/) is planned.
    queries/
        module_15_week4_queries.txt
        # Only Week 4's raw query strings are saved here; Weeks 1-3 and 5 are not
        # yet represented as standalone query files
```

**Planned fixes**, tracked so they don't get relitigated:
1. Consolidate all evidence files onto a single `evidence/week{n}/` naming convention, including a backfilled `week1/`.
2. Backfill executed notebooks for Weeks 2-5 (currently only Week 1's diagnostics notebook exists).
3. Add a `queries/` text file per week for Weeks 1-3 and 5, matching the format already used for Week 4.

Note: reflections are posted to the Imperial discussion board, which is the actual submission requirement - keeping copies in this repo's `reflections/` folder is optional archival only, not something that needs to be kept current or complete.

---

## Weekly Progress

| Module | Week | Functions improved | Notes |
|--------|------|----------------------|-------|
| 12 | 1 | F4, F5, F6, F7, F8 (marginal) | Baseline queries; F1, F2, F3 no improvement or worse |
| 13 | 2 | F3, F4, F5, F6, F7 | F8's regression query failed despite passing all four gates - this is what established the FORCE_GP rule |
| 14 | 3 | F5 (dramatic) | Five-seed averaging and Sobol candidate generation formalised this week |
| 15 | 4 | F2, F5, F8 (new bests) | F1 still at noise floor after 4 rounds; F3, F4, F6, F7 went backwards this round |
| 16 | 5 | Pending | Queries finalised, not yet submitted |

*Table updated each week as results are returned.*

---

## Function Reference

| Function | Dims | Description |
|----------|------|--------------|
| F1 | 2D | Radiation field detection |
| F2 | 2D | Noisy ML log-likelihood |
| F3 | 3D | Drug discovery (side effects) |
| F4 | 4D | Warehouse allocation |
| F5 | 4D | Chemical yield optimisation |
| F6 | 5D | Recipe scoring |
| F7 | 6D | ML hyperparameter tuning |
| F8 | 8D | Neural network tuning |
