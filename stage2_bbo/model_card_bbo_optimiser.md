# Model Card: BBO Optimisation Strategy

**Module 21.2 — Stage 2 Capstone, Imperial College London Professional Certificate in ML & AI**
**Prepared by:** Srini Rajasekaran | **Date:** August 2026

---

## Overview

**Name:** Gaussian Process UCB / PI Optimiser with Regression Challenger  
**Type:** Sequential model-based optimisation (Bayesian optimisation) with weekly human oversight  
**Version:** Week 11 (campaign round 11 of 13)  
**Framework:** scikit-learn `GaussianProcessRegressor`; acquisition functions implemented in Python with scrambled Sobol sampling

This is not a single deployed model. It is a weekly decision process: fit a surrogate, generate candidates, select a query, submit, observe. Each function has its own surrogate fitted independently on its own cumulative data.

---

## Intended Use

**Suitable for:**
- Optimising unknown scalar functions where only point evaluations are available (no gradient, no function definition)
- Settings where the query budget is severely restricted (one query per function per week)
- Educational demonstration of Bayesian optimisation workflow with documented audit trail

**Use cases to avoid:**
- Real-time or high-frequency optimisation (this approach requires a one-week feedback cycle)
- Functions with discontinuities or very high-dimensional inputs (beyond ~8 dimensions, GP surrogates become unreliable with the observation counts available here)
- Any setting where the kernel selection step cannot be run before candidate generation — the strategy degrades to arbitrary kernel defaults, as demonstrated in Round 10

---

## Details

### Strategy evolution across ten rounds

**Rounds 1–3 (exploration):** GP-UCB with beta=2.0 across all functions. Candidates via scrambled Sobol (2^15 points). Uniform application regardless of function response. Key finding: F1 produced no signal for six consecutive rounds, indicating a hotspot with an extremely narrow support region.

**Rounds 4–5 (diversification):** Introduction of multi-seed robustness (five seeds, output averaged or median-selected). Four-corner probes applied to F1 as a model-free alternative when GP eligibility checks failed consistently.

**Round 6 (inflection):** Methodological divergence by function. F5 and F6 moved to override/boundary-push strategies. F7 received its campaign best (3.008) from a trust-region query. F3 and F8 shifted toward interaction-informed and regression-supported approaches respectively. The core learning: functions require individual treatment by Round 6.

**Rounds 7–9 (exploitation):** Function-specific beta schedules applied (0.5 for F5 in deep exploitation, 1.2–1.5 for functions with uncertain surfaces). F8 switched from UCB to PI acquisition following a 7/7 hindsight backtest confirming PI superiority. F1 geometry confirmed as a narrow ridge via directional probes. F5 produced consecutive new bests through systematic boundary relaxation past a self-imposed ceiling.

**Round 10:** F5 returned 8561 (fourth consecutive new best). F6 returned −0.211 (new best, first query with x4 above 0.854). F4 reversed sharply from its Round 9 best, raising a surface-instability question. Kernel selection step skipped for five functions (Anomaly 8); corrected for Round 11.

**Round 11:** Full mandatory sequence formalised: C2 kernel challenger first, regression four-gate check, GP-UCB with C2 winner kernel, GP-UCB with RBF for audit comparison, bootstrap ensemble (30 resamples), convergence test (2^13/2^14/2^15/2^17), A1 seed stability (five seeds). F4, F6, F7, F8 required escalation to 2^17 candidates. F7 seed selection was a named override: seed 703 selected over the median-rule default (seed 702) on three corroborating grounds (highest UCB score, closest structural match to Round 6 best, ensemble direction agreement).

### Key technique choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Surrogate | Gaussian Process (scikit-learn) | Provides uncertainty estimates; UCB acquisition is natural |
| Kernel selection | LOO Q² comparison across RBF, Matern-3/2, Matern-5/2 | Data-driven; avoids kernel assumption baking in |
| Candidate sampling | Scrambled Sobol sequences | Low-discrepancy; more uniform coverage than pseudorandom |
| Acquisition (primary) | UCB (mu + beta × sigma) | Balances exploration and exploitation via beta |
| Acquisition (F8) | PI — Probability of Improvement | Switched after 7/7 hindsight backtest |
| Regression challenger | OLS with four eligibility gates | Provides a linear alternative; gates prevent spurious signals |
| Seed stability | Five seeds, median or best-UCB selection | Guards against single-seed artefacts in high-noise functions |
| Pool escalation | 2^15 default, escalate to 2^17 | Convergence test determines whether 32,768 candidates suffice |

---

## Performance

**Metric:** Portal output y (higher is better; all functions are maximisation tasks).

| Function | Initial best | Campaign best (after 10 rounds) | Best achieved at round | Total improvement |
|----------|-------------|--------------------------------|----------------------|-------------------|
| F1 — Radiation detection (2D) | 0.000 | 1.453 | Round 7 | +1.453 |
| F2 — ML log-likelihood (2D) | 0.611 | 0.746 | Round 6 | +0.135 |
| F3 — Drug discovery (3D) | −0.035 | −0.007 | Round 8 | +0.028 |
| F4 — Warehouse allocation (4D) | −4.026 | +0.492 | Round 9 | +4.518 |
| F5 — Chemical yield (4D) | 1089 | 8561 | Round 10 | +7472 |
| F6 — Recipe scoring (5D) | −0.714 | −0.211 | Round 10 | +0.503 |
| F7 — ML hyperparameter tuning (6D) | 1.365 | 3.008 | Round 6 | +1.643 |
| F8 — Neural network tuning (8D) | 9.598 | 9.971 | Round 9 | +0.373 |

All eight functions improved over the initial best. Six of eight improved over the initial best by more than 10% on a normalised scale. F5's improvement of 7472 units reflects both the scale of the function and the effectiveness of the boundary-relaxation strategy.

**Rounds where new campaign bests were achieved:** F5 set new bests in 9 of 10 rounds. F1 set one new best (Round 7) after six rounds at noise floor. F2 set one new best (Round 6). F3 set three (Rounds 2, 6, 8). F4 set three (Rounds 1, 2, 9). F6 set four (Rounds 1, 2, 7, 10). F7 set three (Rounds 1, 2, 6). F8 set five (Rounds 1, 3, 4, 5, 9).

---

## Assumptions and Limitations

**Stationarity assumption:** GP surrogates assume a single length scale characterises correlation across the entire domain. This is almost certainly violated for some functions. F4's non-stationarity diagnostic flag and its Round 10 sharp reversal are the clearest evidence.

**Beta as a heuristic:** The explore-exploit parameter beta is set by a phase-based schedule (2.0 in early rounds, reducing to 0.5 in deep exploitation) rather than formally derived. This is a documented simplification.

**Coverage constraint:** With 10–50 observations across 2D–8D spaces, the surrogate interpolates reliably only near sampled regions. GP predictions in unsampled areas are extrapolations with wide uncertainty bounds.

**One-query budget:** Each round allows exactly one query per function. Errors cannot be corrected within a round. Round 10's kernel default error illustrates the consequence: coordinates were generated from a weaker surrogate than the data supported, and the submissions could not be revised.

**Kernel default failure mode:** If the kernel selection step is skipped, the surrogate defaults to RBF. For five of the eight functions, RBF is not the best-fitting kernel. The LOO Q² gap reaches +0.086 for F8. This represents a known and documented failure mode.

---

## Ethical Considerations

**Transparency and reproducibility:** Every submitted coordinate is traceable to a named kernel, Sobol seed, candidate pool size, and search-bound derivation. Deviations from the documented process are logged in the anomaly register rather than absorbed silently. The methodology change log distinguishes deliberate decisions (GREEN) from corrections to errors (AMBER) from undocumented drift (RED).

**AI use disclosure:** AI tooling was used to support analysis, code generation, and drafting throughout the campaign, in line with Imperial's confirmed policy (ticket 2821471). A formal disclosure statement will be included at Module 25. No portal submissions were made by AI; all submissions are the direct action of Srini Rajasekaran.

**Real-world adaptation:** The campaign methodology — surrogate selection, convergence testing, seed stability checks, anomaly logging — is designed to be reproducible by a reviewer with access to the data and code. The same principles (document what was used, not just what should have been used) apply directly to model risk governance in financial services contexts, where independent replication of model outputs is a regulatory requirement.

**Limitations on generalisation:** Performance results reflect this specific challenge structure. The strategy is not claimed to generalise to other BBO benchmarks without re-validation.
