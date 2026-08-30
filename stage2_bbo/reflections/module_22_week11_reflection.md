# Module 22 — Week 11 Reflection

**Week:** 11 of 13 | **Module:** 22 | **Date:** August 2026

---

## Queries Submitted

| Fn | Portal string | Method | Outcome |
|----|--------------|--------|---------|
| F1 | ~0 | GP-UCB Matern32 beta=2.0 | Noise floor (5th consecutive non-result) |
| F2 | 0.534920-0.839... | GP-UCB Matern32 beta=2.0 | 0.53492 — no new best |
| F3 | 0.999-0.45-0.42... | GP-UCB Matern52 beta=1.3 FORCE_GP | -0.02943 — no new best |
| F4 | 0.37-0.35-0.38-0.40... | GP-UCB Matern32 beta=1.2 FORCE_GP | +0.42895 — no new best |
| F5 | 0.999708-0.999805-0.999609-0.999549 | GP-UCB Matern52 beta=0.5 | **8,618.541 — new campaign best** |
| F6 | 0.39-0.31-0.62-0.901-0.09 | GP-UCB RBF beta=1.0 | -0.36972 — **REVERSAL from Wk10 best (-0.211)** |
| F7 | 0.126-0.189-0.427-0.197-0.314-0.686 | GP-UCB Matern32 beta=1.0 FORCE_GP | 2.96822 — no new best (Wk6 barrier at 3.008 held) |
| F8 | (PI 5-seed, seeds 800-804) | PI Matern32 FORCE_GP | **9.97950 — new campaign best** |

---

## Key Findings

**F5: fifth consecutive new best.** All four dimensions now confirmed as strongly positive monotone drivers (r = 0.664–0.753, all p < 0.0001). The initial dataset showed x1 as a negative driver (r = -0.28). The combined dataset reverses this to r = +0.75. Root cause: the initial design covered x1 only up to 0.84, leaving the high-yield corner entirely unsampled. Strategy based on initial data alone would have pointed in the wrong direction for x1 across the full campaign.

**F6 reversal: x4 non-monotone confirmed.** Week 10 achieved -0.211 at x4 = 0.854. Week 11 pushed x4 to 0.901 and returned -0.370 — a loss of 0.159 units. This confirmed that x4 has a ridge structure peaking between 0.73 and 0.85 that degrades sharply above 0.90. The overshoot was a deliberate test of the boundary; the result definitively maps the overshoot zone. Week 12 strategy corrected x4 back to [0.78, 0.87].

**F7: Wk6 barrier (3.008) still standing.** Six rounds of failed attempts to break the Week 6 best. The GP predicted 3.074 at the Week 12 query — the first positive surrogate signal in six rounds. This drove the Week 12 strategy.

**F8: PI acquisition continues to outperform.** Seventh consecutive round where PI beats UCB on this function. x1 and x3 confirmed co-dominant negative drivers (r = -0.759 and -0.742 respectively) across 51 observations.

---

## Diagnostic Highlights

- **IV9 clustering:** F6 high-y cluster centroid confirms x5 = 0.087 (low x5 is critical) and x4 = 0.823 (ridge region). This independently corroborates the overshoot diagnosis.
- **C3 bootstrap ensemble:** F4 ensemble gap = -0.370 (single GP mean = +0.271, ensemble mean = -0.098). Root cause: rough surface crash points are amplified through bootstrap resampling. Single GP Q² = 0.878 takes precedence for F4.
- **Combined dataset check (IV6):** Five functions showed material drift between last-6-week correlations and full-history correlations. Combined dataset confirmed essential for all strategy decisions.

---

## Week 12 Preview

Week 12 is the final query round before the last submission. The strategy going into Week 12:

- **F5:** Push all four dimensions to 0.9999 (tightest boundary yet)
- **F6:** x4 corrected to [0.78, 0.87], x5 as low as possible (<0.10)
- **F7:** GP predicted 3.074 at query — maintain same region with tighter trust bounds
- **F8:** PI 5-seed median, seeds 800–804, tighter trust region (±0.08 of Wk11 best)
- **F1:** Model-free probe, constrained Sobol within 0.014 Euclidean of Wk7 spike
- **F2:** Global probe to low-x2 region (only unexplored corner remaining)
- **F3/F4:** FORCE_GP, trust regions around Wk8 and Wk9 bests respectively
