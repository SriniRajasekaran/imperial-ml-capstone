# Module 24 — Week 13 Reflection (Final Round)

**Week:** 13 of 13 | **Module:** 24 | **Date:** August 2026

**Status: PENDING PORTAL RESULTS** — Week 13 queries submitted. This file will be updated when results are confirmed.

---

## Queries Submitted

| Fn | Portal string | Method | Rationale |
|----|--------------|--------|-----------|
| F1 | 0.633407-0.631618 | GP-UCB Matern32 model-free, A1 median | Closest defensible probe to Wk7 spike (dist < 0.01). One final attempt. |
| F2 | 0.737498-0.890955 | GP-UCB Matern32 beta=2.0 | Low-x2 hypothesis eliminated Wk12. NW argmax at [0.732, 0.887] within 0.004. Return to high-y region. |
| F3 | 0.999000-0.450000-0.440000 | GP-UCB RBF beta=1.3 FORCE_GP | x1 ceiling, x3=0.440 curvature optimum (dR²=0.346, p<0.001), x2 neutral reset. |
| F4 | 0.410600-0.385200-0.373400-0.426300 | GP-UCB exact Wk9 best FORCE_GP | Rough surface (within 0.05: range -11.6 to +0.49). GP sigma=6.96 at global argmax. Only defensible final shot. |
| F5 | 0.999999-0.999999-0.999999-0.999999 | GP-UCB Matern32 beta=0.5 AGGRESSIVE | Marginal return ~77,000/unit and RISING across last 4 weeks. 6 methods confirm. |
| F6 | 0.425301-0.319854-0.655696-0.800000-0.060000 | GP-UCB Matern32 beta=0.8 REVISED | x4 corrected to ridge centre 0.800 (was 0.732 = 0.075 below best). x5=0.060 (lower). x2 lowered. |
| F7 | 0.123775-0.167407-0.438382-0.216252-0.297972-0.681083 | GP-UCB Matern32 beta=0.8 REVISED FORCE_GP | x2 corrected DOWN (r=-0.385, was moving up). x4 corrected DOWN (r=-0.449, same error). |
| F8 | 0.045000-0.173625-0.130000-0.122396-0.680561-0.518353-0.205000-0.657957 | PI Matern32 5-seed median FORCE_GP REVISED | Push active dims lower: x1→0.045, x3→0.130, x7→0.205. Noise dims (x5,x6,x8) held at Wk12 values. |

**F5 justification (aggressive step):** Marginal return per unit step has been RISING for four consecutive weeks (68,976 → 72,870 → 76,957 → 77,226). The function is on a steep linear slope at the boundary. A step to 0.999999 is 40% larger than the step to 0.9999, producing proportionally more expected gain. Pearson r, SHAP, PCA, NW, GP, and stratified topography all independently confirm the monotone corner structure with no local maxima in the approach direction.

**F6, F7, F8 revisions:** Identified by X-trajectory pivot analysis — checking coordinates at the campaign best vs the proposed query to verify all dimensions are moving in the driver direction. F6 had x4 set to 0.732 (0.075 below the best of 0.807). F7 had x2 and x4 moving upward despite both being confirmed negative drivers. F8 was repeating the observed point exactly, which cannot produce improvement.

---

## Results

*To be filled in when portal returns confirmed values.*

| Fn | Submitted | Confirmed result | New best? | Notes |
|----|-----------|-----------------|-----------|-------|
| F1 | 0.633407-0.631618 | PENDING | — | — |
| F2 | 0.737498-0.890955 | PENDING | — | — |
| F3 | 0.999000-0.450000-0.440000 | PENDING | — | — |
| F4 | 0.410600-0.385200-0.373400-0.426300 | PENDING | — | — |
| F5 | 0.999999-0.999999-0.999999-0.999999 | PENDING | — | — |
| F6 | 0.425301-0.319854-0.655696-0.800000-0.060000 | PENDING | — | — |
| F7 | 0.123775-0.167407-0.438382-0.216252-0.297972-0.681083 | PENDING | — | — |
| F8 | 0.045000-0.173625-0.130000-0.122396-0.680561-0.518353-0.205000-0.657957 | PENDING | — | — |

---

## Key Diagnostics Run in Week 13

The Week 13 session ran the most extensive diagnostic battery of the campaign, including four techniques introduced specifically for the final round:

**X-trajectory pivot analysis** — for each function, the coordinates at the campaign best were compared dimension-by-dimension against the proposed query, and each dimension was checked for consistency with its confirmed driver direction. This caught three directional errors (F6, F7, F8) that no other test had flagged.

**Nadaraya-Watson kernel regression** — non-parametric regression as an alternative surrogate to the GP. NW had higher Q² than the GP on 5 of 8 functions but produced overfit or structurally invalid argmax suggestions on several (F6 x4 pushed into the overshoot zone; F7 pointed to a different global region). The finding: higher fit quality does not imply a better query recommendation, particularly for functions with narrow peaks or non-monotone dimensions.

**PCA variance decomposition** — PC1 direction confirmed the query strategy for all 8 functions. F5 PC1 explains 62.9% of variance with equal positive loadings on all four dimensions, independently confirming the monotone corner strategy. F8 PC1 r = -0.923, with x1, x3, x7 dominating, confirming the push-low strategy for active dimensions.

**Stratified topographic assessment** — the unit hypercube was divided into 2^d sub-regions (e.g. 256 regions for F8), and the GP posterior mean was evaluated at 32,768 Sobol points to rank local peaks by stratum. The final query was in the top-ranked stratum for every function. No local maxima traps were identified.

**Bootstrap analysis (weekly-only vs combined)** — definitive confirmation that using weekly-only data would have inverted strategy on F6 (x4 and x5 both reverse), F7 (x1 reverses), and F8 (x2 reverses). This settled any remaining question about the mandatory combined-dataset rule.

---

## Campaign Summary

*(Provisional — to be finalised when Week 13 results are confirmed)*

| Fn | Initial best | Campaign best (through Wk12) | Wk13 result | Final best |
|----|-------------|------------------------------|-------------|------------|
| F1 | 0.000000 | 1.452581 (Wk7) | PENDING | — |
| F2 | 0.611205 | 0.745754 (Wk6) | PENDING | — |
| F3 | -0.034835 | -0.007182 (Wk8) | PENDING | — |
| F4 | -4.025542 | +0.491925 (Wk9) | PENDING | — |
| F5 | 1,088.860 | 8,636.766 (Wk12) | PENDING | — |
| F6 | -0.714265 | -0.094440 (Wk12) | PENDING | — |
| F7 | 1.364968 | 3.148939 (Wk12) | PENDING | — |
| F8 | 9.598482 | 9.983487 (Wk12) | PENDING | — |

*Update this table and the notes column when results arrive. Bold any new campaign bests.*
