# Model Card — BBO Surrogate Optimiser

**Imperial College London | Professional Certificate in ML & AI | Stage 2 Capstone**

---

## Model Description

**Model type:** Gaussian Process surrogate with Upper Confidence Bound (UCB) or Probability of Improvement (PI) acquisition, fitted weekly on the combined initial + weekly observation dataset.

**Developed by:** Srini Rajasekaran, with analytical scaffolding provided by Claude (Anthropic).

**Purpose:** To suggest the next query point for each of eight unknown black-box functions, with the objective of maximising the portal-returned output across 13 weekly cycles.

**Languages / Frameworks:** Python 3.x, scikit-learn (GaussianProcessRegressor), scipy (Sobol sequence generation, statistical tests), numpy, XGBoost + TreeExplainer (SHAP challenger analysis).

---

## Intended Use

**Primary use case:** Sequential black-box optimisation of unknown functions where the only signal is a scalar output from a query oracle, and the evaluation budget is extremely limited (one evaluation per function per week).

**Intended users:** The model was developed for and used solely within the Imperial College London Professional Certificate in ML & AI capstone assessment.

**Out-of-scope uses:** The surrogate is trained on point evaluations of specific functions provided by Imperial. It should not be applied to other optimisation problems without re-fitting on relevant data.

---

## Training Data

See `datasheet_bbo_dataset.md` for full details. The combined dataset (initial observations provided by Imperial + all weekly query-response pairs to date) was used for every weekly surrogate fit. The combined dataset rule is non-negotiable: weekly-only data produces critical signal reversals on F6, F7, and F8.

---

## Model Architecture and Parameters

**Kernel:** Selected weekly by C2 Kernel Challenger (leave-one-out cross-validation Q² comparison). Candidates: RBF, Matern-3/2, Matern-5/2. Final kernel assignments for Week 13:

| Fn | Kernel | LOO Q² |
|----|--------|--------|
| F1 | Matern32 | 0.117 |
| F2 | Matern32 | 0.255 |
| F3 | RBF | 0.096 |
| F4 | RBF | 0.567 |
| F5 | Matern32 | 0.816 |
| F6 | Matern32 | 0.320 |
| F7 | Matern32 | 0.658 |
| F8 | Matern32 | 0.182 |

**Length-scale bounds:** 2D (0.15, 0.6), 3-4D (0.12, 0.5), 5-6D (0.10, 0.45), 8D (0.08, 0.40). Bounds prevent kernel collapse (length scale → 0) and degenerate extrapolation (length scale → infinity).

**Acquisition function:** UCB(x) = GP\_mean(x) + beta × GP\_std(x) for F1–F7. PI (Probability of Improvement) for F8, permanently (7/7 historical backtest: PI > UCB).

**Beta schedule:** 2.0 (exploration, Weeks 1–3) → 1.5 (transition, Weeks 4–6) → 1.0 (exploitation, Weeks 7–10) → 0.5 (refinement, Weeks 11–13). Function-specific deviations applied throughout.

**Candidate generation:** Sobol low-discrepancy sequences, 2^14 = 16,384 candidates per session (default). Distinct seed per function. Convergence verified at 2^15 each week.

**Output standardisation:** GP fitted on standardised y (zero mean, unit variance within each weekly dataset) to prevent scale effects on kernel hyperparameter estimation.

---

## Evaluation

**Primary metric:** Portal-returned function value after each query (oracle evaluation). No access to the true function, gradient, or distribution.

**Surrogate quality metric:** Leave-one-out cross-validation Q² (LOO Q²). Threshold for GP eligibility: Q² > 0 (surrogate must beat the naive mean-prediction baseline). Functions with Q² < 0 (F1, F2) were treated as surrogate-ineligible and handled with model-free or low-beta strategies.

**Regression challenger metrics (C1):** R² > 0.30, Shapiro-Wilk p > 0.05, Durbin-Watson in [1.5, 2.5], predicted y at query > current best. All four gates required.

**Bootstrap ensemble (C3):** 30 resamples. Gap between ensemble mean and single GP mean used as a reliability check. A gap exceeding 0.5 × sigma triggered investigation before submission.

---

## Limitations

**Extreme data scarcity:** n/d ratios at the start of the campaign ranged from 5.0 (F1, F2) to 5.0 (F8). Most surrogate methods are unreliable at these ratios. GP Q² was negative for F1 and F2 throughout most of the campaign.

**Catastrophic surface roughness (F4):** Euclidean neighbours within distance 0.05 produce outputs ranging from -11.6 to +0.49. No smooth surrogate can model this accurately. The GP was used conservatively with restricted trust regions and explicit acknowledgment that the surrogate is unreliable.

**Narrow spikes (F1, F7):** F1's optimum has a confirmed width narrower than 0.014 Euclidean distance. F7's campaign best (3.149) exceeds the GP's predicted maximum (3.074), meaning the surrogate cannot represent the true optimum's full height. In both cases, the surrogate was used for direction only, with model-free probes or tight trust regions as the strategy.

**SHAP on small n:** Unregularised XGBoost memorises datasets of n = 22–53 observations (R² = 1.0 in-sample, LOO Q² as low as -10^13). Heavy regularisation was applied for all SHAP analysis. Apparent SHAP disagreements with Pearson r were traced to boundary concentration artefacts or near-zero magnitudes.

---

## Ethical Considerations

The model was used solely for an academic assessment exercise. The eight functions are mathematical optimisation targets; while they have real-world analogues (drug discovery, chemical yield, warehouse logistics), no actual decisions affecting real systems were made based on the surrogate's recommendations.

---

## Caveats and Recommendations

**Adaptive mesh refinement** would be the natural extension of the stratified topographic assessment used in Week 13. Rather than evaluating all hypercube sub-regions uniformly, AMR would allocate more Sobol points to sub-regions with high GP uncertainty and high GP mean — the same principle used in PDE-based numerical solvers to concentrate resolution where it matters.

**HEBO (Huawei Noah's Ark, NeurIPS 2020 BBO Challenge winner)** was identified as the most relevant reference for handling heteroskedasticity and non-stationarity — both confirmed properties of F4, F5, and F7. The campaign used heuristic beta scheduling; HEBO's approach to input warping and heteroskedastic modelling would address these issues more formally.

**Trust region management** mapped directly to domain-of-model-validity bounds in the model risk context: when GP uncertainty is high (surrogate extrapolating beyond observed data), the argmax should be constrained to the observed data range rather than trusted globally. This was operationalised via the within-data-range check in the pre-submission audit.
