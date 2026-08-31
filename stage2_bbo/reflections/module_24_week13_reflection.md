# Module 24 -- Week 13 Reflection (Final Round)

**Week:** 13 of 13 | **Module:** 24 | **August 2026**

---

## Queries Submitted

| Fn | Portal string | Method | Rationale |
|----|--------------|--------|-----------|
| F1 | 0.633407-0.631618 | GP-UCB Matern32 model-free, A1 median | Closest defensible probe to Wk7 spike (dist < 0.01). One final attempt. |
| F2 | 0.737498-0.890955 | GP-UCB Matern32 beta=2.0 | Low-x2 hypothesis eliminated Wk12. NW argmax at [0.732, 0.887]. Return to high-y region. |
| F3 | 0.999000-0.450000-0.440000 | GP-UCB RBF beta=1.3 FORCE_GP | x1 ceiling, x3=0.440 curvature optimum (dR2=0.346, p<0.001), x2 neutral reset. |
| F4 | 0.410600-0.385200-0.373400-0.426300 | Exact Wk9 best FORCE_GP | Rough surface (within 0.05: range -11.6 to +0.49). Only defensible final shot. |
| F5 | 0.999999-0.999999-0.999999-0.999999 | GP-UCB Matern32 beta=0.5 AGGRESSIVE | Marginal return rising across last 4 weeks. 6 methods confirm monotone corner. |
| F6 | 0.425301-0.319854-0.655696-0.800000-0.060000 | GP-UCB Matern32 beta=0.8 REVISED | x4 corrected to ridge centre 0.800. x5=0.060. X-trajectory pivot identified error. |
| F7 | 0.123775-0.167407-0.438382-0.216252-0.297972-0.681083 | GP-UCB Matern32 beta=0.8 REVISED FORCE_GP | x2 and x4 corrected DOWN per confirmed negative drivers. X-trajectory pivot. |
| F8 | 0.045000-0.173625-0.130000-0.122396-0.680561-0.518353-0.205000-0.657957 | PI Matern32 5-seed median REVISED FORCE_GP | Push active dims lower from Wk12 best. Noise dims held at Wk12 values. |

---

## Confirmed Results

| Fn | Submitted | Confirmed result | New best? | Notes |
|----|-----------|-----------------|-----------|-------|
| F1 | 0.633407-0.631618 | **1.846721** | **YES** | New campaign best. Final probe near Wk7 spike produced strongest result of the campaign. |
| F2 | 0.737498-0.890955 | 0.482389 | no | Below Wk6 best (0.746). High-variance surface confirmed. |
| F3 | 0.999000-0.450000-0.440000 | -0.014525 | no | Below Wk8 best (-0.007). Narrow optimum not reproduced. |
| F4 | 0.410600-0.385200-0.373400-0.426300 | **0.491925** | **YES** | Matched Wk9 best exactly. Exact-repeat strategy confirmed correct for rough surfaces. |
| F5 | 0.999999-0.999999-0.999999-0.999999 | **8,662.405** | **YES** | Seventh consecutive new best. Aggressive boundary push justified. |
| F6 | 0.425301-0.319854-0.655696-0.800000-0.060000 | **-0.058449** | **YES** | New campaign best. x5=0.060 and x4=0.800 confirmed as optimal combination. |
| F7 | 0.123775-0.167407-0.438382-0.216252-0.297972-0.681083 | 3.100521 | no | Below Wk12 best (3.149). Wk12 remains campaign best for F7. |
| F8 | 0.045000-0.173625-0.130000-0.122396-0.680561-0.518353-0.205000-0.657957 | **9.984774** | **YES** | New campaign best. PI acquisition continued to deliver. |

**Week 13 produced 5 new campaign bests (F1, F4, F5, F6, F8).**

F1 is the most notable result: the final probe near the Week 7 spike returned 1.847, surpassing the Week 7 result of 1.453 that had stood for six rounds. The exact-repeat strategy on F4 matched the Week 9 best precisely, confirming that on catastrophically rough surfaces, anchoring to the known best is the only defensible approach.

---

## Final Campaign Summary

| Fn | Description | Dims | Initial best | Final best | Best week | Total gain |
|----|------------|------|-------------|------------|-----------|-----------|
| F1 | Radiation field detection | 2 | 0.000000 | **1.846721** | Wk13 | Located and exceeded spike |
| F2 | Noisy ML log-likelihood | 2 | 0.611205 | 0.745754 | Wk6 | +22% |
| F3 | Drug discovery (side-effect min) | 3 | -0.034835 | -0.007182 | Wk8 | +79% |
| F4 | Warehouse allocation (cost min) | 4 | -4.025542 | +0.491925 | Wk9/13 | +112% |
| F5 | Chemical yield optimisation | 4 | 1,088.860 | **8,662.405** | Wk13 | +696% |
| F6 | Recipe scoring | 5 | -0.714265 | **-0.058449** | Wk13 | +92% |
| F7 | ML hyperparameter tuning | 6 | 1.364968 | 3.148939 | Wk12 | +131% |
| F8 | Neural network tuning | 8 | 9.598482 | **9.984774** | Wk13 | +4% |

Seven of eight functions improved materially over their initial best. Five functions achieved new bests in the final round. F2 and F3 represent genuine landscape constraints: F2 is a high-noise surface where the Week 6 result is likely a favourable noise draw, and F3's optimum is narrower than the surrogate can reliably locate with the available observation budget.

---

## Key Diagnostic Findings -- Final Session

**X-trajectory pivot analysis** caught directional errors in F6, F7, and F8 that no other test had flagged. For F6, x4 had been set to 0.732 (0.075 below the campaign best of 0.807). For F7, x2 and x4 were moving upward against confirmed negative drivers. Correcting these before submission directly contributed to the F6 and F8 new bests.

**Stratified topographic assessment** confirmed all eight queries were in the highest-ranked hypercube sub-region for their respective functions. No local maxima traps were identified.

**Thompson Sampling** (50 posterior draws per function) confirmed all eight queries without suggesting alternatives. F7's best Thompson Sampling draw had GP mean 1.03 versus the proposed query at 3.11, confirming the query was well above the surrogate's exploration frontier.

**Bootstrap signal reversal analysis** provided definitive confirmation that the combined dataset rule was correct throughout the campaign. Restricting to weekly-only data would have inverted strategy on F6, F7, and F8 -- the three functions that showed the most consistent improvement in the second half of the campaign.
