# Module 23 — Week 12 Reflection

**Week:** 12 of 13 | **Module:** 23 | **Date:** August 2026

---

## Queries Submitted

| Fn | Portal string | Method | Outcome |
|----|--------------|--------|---------|
| F1 | 0.618518-0.649998 | GP-UCB Matern32 beta=2.0 model-free | 0.490674 — no new best (but not noise: spike confirmed real within dist=0.014) |
| F2 | 0.976978-0.001365 | GP-UCB Matern32 beta=2.0 global low-x2 probe | 0.017421 — near noise. Low-x2 hypothesis definitively rejected. |
| F3 | 0.998879-0.540986-0.416302 | GP-UCB Matern52 beta=1.3 FORCE_GP | -0.022733 — no new best (elevated x2=0.541 likely caused adverse interaction) |
| F4 | 0.370746-0.343230-0.403800-0.382278 | GP-UCB Matern32 beta=1.2 FORCE_GP | 0.350298 — no new best (0.079 from Wk9 best, lost 0.142 units) |
| F5 | 0.999708-0.999805-0.999609-0.999549 | GP-UCB Matern52 beta=0.5 | **8,636.766 — new campaign best (6th consecutive)** |
| F6 | 0.425301-0.349854-0.655696-0.806529-0.143673 | GP-UCB Matern32 beta=1.0 | **-0.094440 — new campaign best (+0.116 over prior best)** |
| F7 | 0.128775-0.172407-0.438382-0.226252-0.302972-0.676083 | GP-UCB Matern32 beta=1.0 FORCE_GP | **3.148939 — new campaign best (broke Wk6 barrier of 3.008 after 6 rounds)** |
| F8 | 0.061026-0.185625-0.155488-0.134396-0.680561-0.518353-0.233223-0.657957 | PI Matern32 FORCE_GP | **9.983487 — new campaign best** |

**Week 12 produced 4 new campaign bests (F5, F6, F7, F8) — the best single round since Week 1.**

Note: F8 portal string was corrected before submission. Session-generated string had PI = 0.393, which was not traceable to seeds 800–804 with the specified ±0.08 trust region bounds. Corrected to re-run median PI = 0.615. Logged as Anomaly 9 (AMBER — correction made before submission).

---

## Key Findings

**F6: x4 ridge confirmed and corrected.** Week 11 overshoot to x4 = 0.901 had produced a reversal (-0.211 to -0.370). Week 12 placed x4 at 0.807 (within the confirmed ridge band of 0.73–0.85) and produced a new best of -0.094, an improvement of 0.116 units. This is the function's best result in the campaign and validates the non-monotone x4 structure identified in the topographic assessment.

**F7: barrier broken after 6 rounds.** The Week 6 best of 3.008 stood for six weekly cycles. Week 12's query at [0.129, 0.172, 0.438, 0.226, 0.303, 0.676] returned 3.149. This was driven by the GP predicted mean of 3.074 — the first positive surrogate signal in six rounds — and corroborated by the high-y cluster centroid at [0.095, 0.249, 0.358, 0.241, 0.311, 0.709], which independently pointed to the same region.

**F3: x2 elevation caused regression.** The Week 12 query elevated x2 to 0.541 (noise dimension). This likely produced an adverse interaction with the x3 curvature structure. Week 13 reset x2 to 0.450 (neutral).

**F4: surface roughness confirmed catastrophic.** Moving 0.079 Euclidean from the Wk9 best lost 0.142 units. Observations within Euclidean distance 0.05 range from -11.6 to +0.49. Week 13 strategy: probe exactly at Wk9 best coordinates.

**Bootstrap analysis (final confirmation of combined-dataset rule):** Critical signal reversals confirmed for F6, F7, F8 when using weekly-only data. F6 x4 flips from r = -0.30 (weekly) to r = +0.617 (combined). This analysis settled any remaining uncertainty about the combined-dataset requirement.

---

## Diagnostic Highlights

- **C2 kernel reassignment:** F6 moved from RBF to Matern32 on full combined dataset. F5 and F3 updated to Matern32 and RBF respectively. All kernel changes were marginal in Q² terms but confirmed on updated n.
- **C3 bootstrap ensemble:** F6 ensemble gap = -0.073 (ensemble mean = -0.193, single GP = -0.120). Both above current best (-0.211). Genuine caution noted: expected improvement is real but magnitude is smaller than single GP implies.
- **F8 anomaly (Anomaly 9):** Session PI string PI = 0.393 not reproducible from seeds 800–804 with ±0.08 bounds. Corrected to re-run median PI = 0.615. All 8 dimensions verified within bounds before submission.

---

## Week 13 Preview

Week 13 is the final query cycle. Revised strategies for all 8 functions:

- **F1:** Exact Wk7 coordinates (0.628000-0.640000) — only remaining probe at the confirmed spike
- **F2:** Concede. Return to main x1-high, x2-mid region — low-x2 definitively closed
- **F3:** x1=0.999, x3=0.440 (curvature optimum), x2=0.450 (neutral reset)
- **F4:** Exact Wk9 best coordinates — rough surface means proximity to observed best is the only defensible approach
- **F5:** All four dimensions at 0.999999 — marginal return per unit is rising (not falling)
- **F6:** x4=0.800 (ridge centre), x5=0.060, x2 lowered — tighten around Wk12 best with x5 pushed lower
- **F7:** Tighten to ±0.02 of Wk12 best. x2 and x4 corrected downward (both were moving in wrong direction by 0.005–0.010)
- **F8:** PI 5-seed median, ±0.06 trust region. Push x1→0.045, x3→0.130, x7→0.205

Additional diagnostics introduced for Week 13: X-trajectory pivot analysis (checking coordinate directions against driver signs), Nadaraya-Watson kernel regression, PCA variance decomposition, and stratified topographic assessment (hypercube sub-regions ranked by local GP peak).
