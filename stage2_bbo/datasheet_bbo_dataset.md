# Datasheet: BBO Capstone Query Dataset

**Module 21.2 — Stage 2 Capstone, Imperial College London Professional Certificate in ML & AI**
**Prepared by:** Srini Rajasekaran | **Date:** August 2026

---

## Motivation

This dataset was created to support the Stage 2 Black-Box Optimisation (BBO) capstone challenge. The task is to maximise eight unknown functions of increasing dimensionality (2D through 8D) by submitting one query per function per week over thirteen rounds, receiving only a scalar output from the Imperial portal in return. The dataset accumulates every query submitted and every result received, forming the evidence base for surrogate modelling and strategy decisions each week.

The dataset serves two purposes: it is the training data for Gaussian Process surrogates fitted each week, and it is the audit record demonstrating that every submitted coordinate is traceable to a documented, reproducible computation.

---

## Composition

**Structure:** Eight functions, each with its own input-output history. Each row is one observation: a vector of input coordinates and a scalar output.

| Function | Dimensions | Initial observations | Weekly queries added | Total after 10 rounds |
|----------|-----------|---------------------|---------------------|----------------------|
| F1 — Radiation field detection | 2 | 10 | 10 | 20 |
| F2 — Noisy ML log-likelihood | 2 | 10 | 10 | 20 |
| F3 — Drug discovery | 3 | 15 | 10 | 25 |
| F4 — Warehouse allocation | 4 | 30 | 10 | 40 |
| F5 — Chemical yield | 4 | 20 | 10 | 30 |
| F6 — Recipe scoring | 5 | 20 | 10 | 30 |
| F7 — ML hyperparameter tuning | 6 | 30 | 10 | 40 |
| F8 — Neural network tuning | 8 | 40 | 10 | 50 |

**Format:** NumPy `.npy` arrays for initial data (provided by Imperial). Weekly additions held as arrays in the portal's cumulative `inputs.txt` / `outputs.txt` files and replicated in evidence workbooks.

**Input domain:** All inputs normalised to [0, 1] per dimension. True underlying variable meanings are not disclosed by Imperial.

**Outputs:** All functions are maximisation tasks. F3, F4, and F6 have all-negative outputs by design — the portal has negated the original minimisation objectives.

**Gaps:** The dataset is sparse relative to the input space — particularly in high dimensions. F8 (8D) has 50 observations covering approximately 10% of the domain by any reasonable coverage metric. F6 had a systematic coverage gap in x4 for the first nine rounds: the x4 coordinate sat between 0.597 and 0.963 exclusively, due to a Sobol artefact in the initial data that the GP surrogate then reinforced. This was identified and corrected in Round 10.

---

## Collection Process

**Initial data:** Provided by Imperial College London as part of the capstone challenge specification. Collected as part of the challenge design, not by the submitter.

**Weekly queries:** Generated using a documented two-method framework applied each week before submission:

1. OLS regression on standardised inputs (four eligibility gates: R² > 0.30, Shapiro-Wilk p > 0.05, Durbin-Watson 1.5–2.5, predicted y exceeds current best).
2. Gaussian Process surrogate with UCB acquisition. Kernel selected weekly by leave-one-out Q² comparison across RBF, Matern-3/2, and Matern-5/2. Candidate points generated via scrambled Sobol sequences (pool size 2^15 or 2^17 where convergence required escalation). Each function assigned a fixed seed (fn_index × 100) to ensure reproducibility.

**Time frame:** Rounds 1–10 completed between May and August 2026, one submission per function per week.

**Documented deviations:** In Round 10, the kernel selection step was not run before candidate generation for five of the eight functions, causing RBF to be used as a default. This is logged as Anomaly 8 in the evidence workbook anomaly register. The submitted coordinates stand as valid observations; the deviation does not affect the data values, only the strategy that produced them.

---

## Preprocessing and Uses

**Preprocessing applied:**
- Inputs standardised (zero mean, unit variance) before GP fitting. Standardisation fit on training data only; not applied to raw stored coordinates.
- Outputs normalised (zero mean, unit variance) before GP fitting to stabilise kernel hyperparameter optimisation.
- F5 outputs span approximately five orders of magnitude. A log transform was considered but not implemented; GP fitting used the normalised scale throughout.

**Intended uses:**
- Fitting GP surrogates each round to generate the next query.
- Retrospective analysis of search trajectories, kernel fit quality, and campaign performance.
- Educational demonstration of Bayesian optimisation methodology applied to unknown functions.

**Uses to avoid:**
- Do not treat the dataset as a characterisation of the true underlying functions. The true function definitions are not disclosed; inference about function structure is hypothesis, not ground truth.
- Do not use early-round data alone without the full cumulative history. GP quality degrades substantially when weekly submissions are loaded in isolation.
- Do not generalise performance metrics to other BBO problems without accounting for the specific challenge structure (one query per function per week, portal-mediated feedback, no gradient information).

---

## Distribution and Maintenance

**Availability:** Initial data provided by Imperial College London and subject to their challenge terms. Weekly submission data (inputs and outputs) is maintained in this GitHub repository under `stage2_bbo/evidence/`.

**Access:** Public repository. Initial `.npy` files are included for reproducibility. Portal output values are recorded in evidence workbooks.

**Maintenance:** Maintained by Srini Rajasekaran for the duration of the capstone challenge (Modules 12–25). No ongoing maintenance is planned after Module 25 submission.

**Terms of use:** This dataset documents a specific educational challenge submission. Reuse for academic comparison or methodology research is permitted with attribution. It should not be used to attempt to reverse-engineer the Imperial BBO challenge functions.
