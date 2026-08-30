# Module 24 -- Week 13 Reflection (Final Round)

**Week:** 13 of 13 | **Module:** 24 | **August 2026**

---

## Queries Submitted

| Fn | Portal string | Method | Rationale |
|----|--------------|--------|-----------|
| F1 | 0.633407-0.631618 | GP-UCB Matern32 model-free, A1 median | Closest defensible probe to Wk7 spike. One final attempt. |
| F2 | 0.737498-0.890955 | GP-UCB Matern32 beta=2.0 | Low-x2 eliminated Wk12. NW argmax at [0.732, 0.887] within 0.004. Return to high-y region. |
| F3 | 0.999000-0.450000-0.440000 | GP-UCB RBF beta=1.3 FORCE_GP | x1 ceiling, x3=0.440 curvature optimum (dR2=0.346, p<0.001), x2 neutral reset. |
| F4 | 0.410600-0.385200-0.373400-0.426300 | Exact Wk9 best FORCE_GP | Rough surface (within 0.05: range -11.6 to +0.49). GP sigma=6.96 at global argmax. Only defensible final shot. |
| F5 | 0.999999-0.999999-0.999999-0.999999 | GP-UCB Matern32 beta=0.5 AGGRESSIVE | Marginal return rising (68,976 to 77,226 per unit). 6 methods confirm monotone corner. |
| F6 | 0.425301-0.319854-0.655696-0.800000-0.060000 | GP-UCB Matern32 beta=0.8 REVISED | x4 corrected to ridge centre 0.800 (was 0.732). x5=0.060. x2 lowered. X-trajectory pivot identified error. |
| F7 | 0.123775-0.167407-0.438382-0.216252-0.297972-0.681083 | GP-UCB Matern32 beta=0.8 REVISED FORCE_GP | x2 corrected DOWN (r=-0.385, was moving up). x4 corrected DOWN (r=-0.449, same error). X-trajectory pivot. |
| F8 | 0.045000-0.173625-0.130000-0.122396-0.680561-0.518353-0.205000-0.657957 | PI Matern32 5-seed median REVISED FORCE_GP | Push active dims lower from Wk12 best. x1->0.045, x3->0.130, x7->0.205. Noise dims held. |

---

## 1. Exploration-Exploitation: How the Balance Evolved

Week 12 established three operating regimes simultaneously. The final round sharpened each. F5 and F8 were in deep exploitation: F5 had six consecutive new bests on a monotone all-high surface and received a final push to 0.999999 after marginal return analysis confirmed efficiency rising from 68,976 to 77,226 units per unit of movement; F8 ran PI acquisition permanently after a 7/7 backtest. F3, F4, F6, and F7 sat in evidence-constrained trust regions. F4 had a surface so rough that points within 0.05 Euclidean distance ranged from -11.6 to +0.49, making the Week 9 exact coordinates the only defensible final query. F6 and F7 required directional corrections after an X-trajectory pivot check found both had drifted away from their campaign-best coordinates, with F7 sending two dimensions upward against confirmed negative drivers. F1 and F2 remained model-free; GP surrogates were ineligible at negative LOO Q2, and both received targeted probes at the single most informative untested point.

The four-phase beta decay (2.0 to 0.5) was evidence-gated per function, not applied as a shared schedule. F1 and F2 held at 2.0 throughout. F5 dropped to 0.5 at Week 7 when the monotone corner structure was confirmed. F8 used PI acquisition where beta is not applicable.

---

## 2. Feedback, Belief Updating, and the Q-Value Parallel

Each portal return is a reward signal and each surrogate refit is a Q-table update: the GP revises its belief at every candidate point in proportion to the new observation, discounting distance through the kernel length-scale. Feedback drove permanent strategy changes across the campaign:

**F1, F2 (model-free):** Every return was treated as a signal about spatial coverage, not surrogate improvement. The Week 12 F2 probe returning 0.017 was a definitive negative update confirming the entire low-x2 region as infeasible, consistent with how a Q-table entry collapses to near-zero after a failed action.

**F3, F5, F6, F7 (evidence-constrained):** Key structural beliefs formed progressively. F3's x3 curvature (delta R2 = 0.490) first appeared at Week 5 and tightened each round until the search box was constrained permanently to x3 in [0.40, 0.55]. F5's x1 appeared to be a negative driver on the initial 20-point dataset (r = -0.28); adding the first six weekly returns reversed it to strongly positive (r = +0.75). F6's Week 11 overshoot to x4 = 0.901 produced a reversal to -0.370, fixing the ridge constraint [0.73, 0.85] as permanently non-negotiable. F7's Week 6 result (3.008) was not reproduced for six rounds despite nearby probes; each failed round updated the surrogate's belief that the peak was narrower than the GP could resolve.

**F4, F8 (high-variance functions):** Feedback updated method, not just direction. F4 crash points at Weeks 10 and 11 were hard negative updates signalling catastrophic roughness; the correct response was to anchor exactly to the known best. F8's Week 2 regression failure (all four gates passed, predicted y = 10.38, actual 9.65) triggered the permanent switch to PI acquisition.

The most consequential feedback finding was the experience replay result. Restricting analysis to the 12 weekly submissions only produced direction reversals on F6 (x4 and x5 both change sign), F7 (x1 reverses from r = +0.33 to r = -0.634), and F8 (x2 reverses from r = +0.40 to r = -0.446). For F1 through F5, no reversals were found. Using weekly-only data for F6 would have pushed x4 low and x5 high -- the opposite of the correct strategy.

---

## 3. AlphaGo Zero: Model-Based Planning and Self-Play

AlphaGo Zero improved by playing against itself with no human priors, updating its network from each game's outcome. The BBO campaign has the same closed loop: GP surrogate trained on all prior observations generates the next query; that query returns a new observation; the surrogate retrains. Whether each function's process resembled model-free or model-based planning depended on surrogate quality:

**F1 (model-free):** GP ineligible at LOO Q2 = 0.117 with no converging argmax. Every query was trial and error near the one confirmed non-zero observation.

**F2 (model-free to targeted probe):** GP ineligible. Five rounds near the Week 6 best produced no improvement. The Week 12 global probe was a deliberate trial of the only untested structural hypothesis; it returned noise floor, confirming the region as dead.

**F3, F6, F7 (model-based with structural override):** GP surrogate guided direction but structural constraints derived from 12 rounds of observation took precedence over surrogate argmaxima. For F6, the confirmed non-monotone x4 ridge rejected the NW kernel regression argmax at x4 = 0.899. For F7, the surrogate argmax was unstable at 2^17 candidates; the policy held near the observed best rather than following an unreliable model output.

**F4 (model-based with roughness constraint):** GP Q2 = 0.567 but the surface roughness was catastrophic. AlphaGo Zero learns to avoid locally promising moves that lead to worse positions; F4's policy did the same: the Week 9 best coordinates were the only safe anchor.

**F5, F8 (deeply model-based, full exploitation):** Both used the surrogate to evaluate thousands of candidates before committing. F5's marginal return analysis confirmed efficiency rising at the boundary, justifying 0.999999. F8 used PI across five Sobol seeds per round to select the candidate most likely to beat the current best.

---

## 4. RL Strategies: What Was Tried and What Remains

Thompson Sampling was run directly in the final session: 50 posterior draws per function were generated and evaluated. No draw produced a higher GP-mean candidate than the current query for any function, confirming the queries rather than changing them. For F7, the best Thompson Sampling draw had GP mean = 1.03 versus the current query at 3.11. The remaining strategies below require multi-episode infrastructure that cannot be applied within a 13-round constraint.

| RL Strategy | Applied in Week 13? | What it would add to future campaigns |
|-------------|--------------------|-----------------------------------------|
| Thompson Sampling | YES: 50 posterior draws per function. Confirmed queries, did not change them. | Would automate exploration-exploitation balance without manually tuned beta. |
| Reward shaping for information gain | Not applicable to a single query. | Would prevent exploratory probes from appearing wasteful; the Week 12 F2 probe (y = 0.017) had near-zero immediate reward but genuine informational value. |
| Contextual bandit over method selection | Not applicable to a single query. | Would learn per-function method preferences from outcomes rather than fixed thresholds. |
| Population-based beta training | Not applicable to a single query. | Would replace the manually calibrated four-phase schedule with one that adapts automatically to surface pattern. |

---

## Results

*To be updated when Week 13 portal results are confirmed.*

| Fn | Submitted | Confirmed result | New best? |
|----|-----------|-----------------|-----------|
| F1 | 0.633407-0.631618 | PENDING | -- |
| F2 | 0.737498-0.890955 | PENDING | -- |
| F3 | 0.999000-0.450000-0.440000 | PENDING | -- |
| F4 | 0.410600-0.385200-0.373400-0.426300 | PENDING | -- |
| F5 | 0.999999-0.999999-0.999999-0.999999 | PENDING | -- |
| F6 | 0.425301-0.319854-0.655696-0.800000-0.060000 | PENDING | -- |
| F7 | 0.123775-0.167407-0.438382-0.216252-0.297972-0.681083 | PENDING | -- |
| F8 | 0.045000-0.173625-0.130000-0.122396-0.680561-0.518353-0.205000-0.657957 | PENDING | -- |

---

## Campaign Summary (through Week 12)

| Fn | Initial best | Campaign best | Best week | Gain |
|----|-------------|--------------|-----------|------|
| F1 | 0.000000 | 1.452581 | Wk7 | Located spike |
| F2 | 0.611205 | 0.745754 | Wk6 | +22% |
| F3 | -0.034835 | -0.007182 | Wk8 | +79% |
| F4 | -4.025542 | +0.491925 | Wk9 | +112% |
| F5 | 1,088.860 | 8,636.766 | Wk12 | +694% |
| F6 | -0.714265 | -0.094440 | Wk12 | +87% |
| F7 | 1.364968 | 3.148939 | Wk12 | +130% |
| F8 | 9.598482 | 9.983487 | Wk12 | +4% |
