# Module 12 — BBO Challenge Week 1 Reflection

**Author:** Srini Rajasekaran
**Programme:** Imperial College London — Professional Certificate in ML & AI
**Module:** 12 — Bayesian Optimisation
**Round:** Week 1 of 13

---

## Queries Submitted

| Function | Method | Portal Submission |
|----------|--------|------------------|
| F1 | GP-UCB | `0.598658-0.156019` |
| F2 | GP-UCB | `0.472665-0.061138` |
| F3 | GP-UCB | `0.506745-0.606548-0.330793` |
| F4 | GP-UCB | `0.400827-0.428837-0.398466-0.449225` |
| F5 | GP-UCB | `0.360299-0.942015-0.919077-0.900569` |
| F6 | Regression | `0.500000-0.500000-0.500000-0.950000-0.050000` |
| F7 | GP-UCB | `0.118951-0.343205-0.289884-0.186974-0.285469-0.789029` |
| F8 | Regression | `0.050000-0.050000-0.050000-0.500000-0.500000-0.500000-0.050000-0.500000` |

---

## Part 1: What principle or heuristic guided each query decision?

Two methods were evaluated before finalising each query.

- Multivariate OLS regression with standardised inputs was fitted to the initial data for each function, producing beta coefficients, standard errors, t-statistics and p-values. Residuals were assessed for normality using the Shapiro-Wilk test, skewness, kurtosis, and independence using the Durbin-Watson statistic. A regression-based query was only used when four gates were passed: R² above 0.30, normal residuals (SW p above 0.05), no autocorrelation (DW between 1.5 and 2.5), and predicted y exceeding the current best. Where regression failed any gate, a Gaussian Process surrogate with UCB acquisition was used instead.

- Beta was set to 2.0 following the UCB convention where mean plus two standard deviations corresponds to the 95th percentile of the GP predictive distribution, providing an optimistic upper bound on each candidate. For the six GP-UCB functions, 8,000 random candidates were evaluated and the point with the highest UCB score was selected. The UCB score distribution confirmed that F1 and F2 showed zero discrimination, with all 8,000 candidates scoring identically at 2.0, meaning the GP has no structure to exploit at 10 observations and queries are blind exploration. F4 and F8 showed genuine structure with UCB standard deviations above 0.80, meaning the chosen points are genuinely informed rather than arbitrary.

Only F6 and F8 passed all four regression gates. For F6: R²=0.65, SW p=0.82 (normal), skewness=0.31 (symmetric), kurtosis=0.18 (near-normal tails), DW=1.84 (no autocorrelation). Individual p-values confirmed only x4 (p=0.043) and x5 (p=0.017) as statistically significant. The query pushes x4 to 0.95 and x5 to 0.05 while setting the three insignificant dimensions to 0.50. For F8: R²=0.90, SW p=0.29 (normal), skewness=0.12 (symmetric), kurtosis=0.09 (near-normal tails), DW=2.48 (no autocorrelation). Excel regression confirmed x1 (p=7.97e-09), x2 (p=0.011), x3 (p=8.72e-13) and x7 (p=2.47e-06) as highly significant with negative coefficients. The four insignificant inputs were set to 0.50.

---

## Part 2: Observed structure per function

Based on initial data alone, the following landscape characteristics were inferred. No functional form is known and these are working hypotheses only.

| Function | Observed characteristic | Implication for query |
|----------|------------------------|----------------------|
| F1 | All outputs near zero, zero UCB discrimination | Blind exploration, lower-centre region |
| F2 | One promising region around (0.70, 0.93), y=0.61 | Explore away from known region first |
| F3 | All outputs negative, narrow range | Smooth landscape, query low-coverage region |
| F4 | Large negative outputs, high variance, high UCB discrimination | Narrow good region, explore corners |
| F5 | Single outlier at y=1088, all others below 2 | Possible sharp peak, cautious exploration |
| F6 | Moderate negative, x4 and x5 significant (regression confirmed) | Linear structure confirmed, act on signal |
| F7 | Moderate positive outputs, broad spread | Smooth optimum likely, systematic exploration |
| F8 | Strong linear structure, R²=0.90, x1 x3 x7 dominate | Push significant negative dimensions to 0.05 |

---

## Part 3: Which functions were most challenging, and why?

F1 was the most challenging. All 10 initial outputs are near zero and the UCB score distribution showed zero discrimination across all 8,000 candidates, with every point scoring identically at 2.0. The query is a blind exploration step with no directional basis.

F5 presented the opposite challenge. One observation at y=1088.9 is roughly 700 times larger than the next highest value. Regression predicted 788 against a current best of 1088, confirming the linear model cannot capture the extreme value. It is unclear whether the outlier represents a broad optimum or a narrow spike.

A key observation: the six decimal places in the portal format overstate the precision of the method for low-discrimination functions. For F1 and F2, any of the 8,000 candidates would have scored identically. For F8, the GP had genuine structure and the chosen point stands out meaningfully from the candidate distribution.

---

## Part 4: Strategy adjustments for Week 2

Once results are returned, the GP will be refit on 11 points per function and UCB discrimination reassessed. For F1, a non-zero result shifts strategy towards directed search and beta will be reduced. For F5, a high result near the outlier confirms a genuine optimum and exploitation begins. For F6 and F8, results will confirm whether acting only on statistically significant betas improved on the current best. If either fails to improve, the regression approach will be reassessed. Beta will remain at 2.0 for higher-dimensional functions for at least two more rounds as 11 to 12 observations remain insufficient to trust exploitation recommendations in six to eight dimensions.
