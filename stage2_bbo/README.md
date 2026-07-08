# PROJECT_CONTEXT.md

**Purpose of this file:** paste this entire document at the start of any new chat (in this project or otherwise) to restore working context immediately, without re-explaining the project from scratch. Update it after any session with a material decision, correction, or strategy change. This is a living log, not a one-time scope description.

---

## 1. Who and What

Srini Rajasekaran, ~25 years model risk / XVA / credit at NatWest, BoA, Citi. Currently Director, Head of Structured Credit & Real Estate Assurance at Citibank. Completing the Imperial College London Professional Certificate in ML & AI. Capstone has two stages:

- **Stage 1**: free-exploration bankruptcy early-warning audit toolkit (Modules 3-11), built around the Polish Companies Bankruptcy dataset (UCI ID 365), with SHAP-based explainability diagnostics, ESR (Explanation Sensitivity Ratio), and governance/regulatory alignment framing (SR 11-7, PRA SS1/23, EU AI Act, NIST AI RMF).
- **Stage 2**: fixed Black-Box Optimisation (BBO) challenge (Modules 12-24), one portal query per function per week across 8 unknown functions of dimensionality 2D-8D, scored only by the single y returned. Module 25 is final polish.

GitHub repo mirrors both stages. Stage 2 lives in `stage2_bbo/` with subfolders `notebooks/`, `data/module_XX/`, `reflections/`, `evidence/`.

## 2. Hard Rules - Always Apply, No Exceptions

- No AI self-reference anywhere in deliverables: GitHub commits, reflections, discussion board posts, README. These read as Srini's own analysis and judgment.
- No em dashes in any written output, anywhere, unless explicitly told otherwise for a one-off.
- Claude scaffolds and drafts; Srini directs strategy, decides overrides, runs the actual portal submissions, owns final judgment calls.
- Never call it "your project" in deliverable text; call it "the toolkit" / "the campaign."
- Reflections are posted to the Imperial discussion board - that is the actual submission requirement. They do not need to be pushed to GitHub. Existing reflection files already in the repo (`stage2_bbo/reflections/`) stay as-is; no obligation to backfill missing weeks or add future weeks there.
- **Imperial's actual Stage 2 requirements are only two things per week: the portal query submission, and the ~700-word discussion board reflection.** The evidence workbooks, diagnostics scripts, scenario/ensemble analyses, and convergence checks are Srini's own personal working assessment tooling, not something Imperial requires or grades. Useful to keep for rigor and as a personal record, and fine to reference in GitHub for portfolio purposes, but never to be treated as a capstone deliverable or checked against a submission requirement.
- Disclosure: Srini has queried Imperial directly about AI-use disclosure requirements for the capstone; status not yet confirmed in this record. Treat as unresolved, don't assume an answer either way.

## 3. Stage 2 BBO - Current Methodology (updated through Week 5)

**Two competing query-generation methods per function, decided fresh every week:**

- **GP-UCB**: Gaussian Process, UCB acquisition (`mu + beta*sigma`). Length-scale bounds fixed at (0.15, 0.6) since the Week 2 kernel-degeneracy fix.
- **Regression**: OLS on standardised inputs, significance-gated (only act on coefficients with p<0.05; insignificant inputs left at neutral 0.50). Eligibility requires **four** gates: R²>0.30, Shapiro-Wilk p>0.05 (residual normality), Durbin-Watson 1.5-2.5 (no autocorrelation), AND predicted y at the significance-gated query exceeds current best.

**FORCE_GP list (established after the Week 2 F8 regression failure, permanent): F3, F4, F8 cannot use Regression regardless of gate outcome.** F8 passed all four gates again in Week 5 (R²=0.92) and was still routed to GP-UCB purely on this rule - that is the rule doing its job, not an oversight.

**Candidate generation, current standard (superseded LHS as of Week 4):**
- **Sobol sequences** are now the primary candidate generator, replacing Latin Hypercube Sampling (which itself had replaced uniform random after Week 1).
- **Candidate grid size: 12,000 is the default**, established via an explicit convergence test in Week 4 (UCB score checked at 8k/12k/16k/24k; 12k was the first size where most functions' argmax stopped moving by more than 0.001). This replaced the Week 1 default of 8,000.
- **Convergence must be re-checked, not assumed, whenever a function's landscape has shifted meaningfully or a new best has just been confirmed.** In Week 5, re-running the 12k/16k/24k check found F5 and F7 had NOT fully converged at 12k (F5 needed 12k confirmed, F7 needed 24k). F7's final Week 5 query was taken from the 24k-converged result, not the 12k default, because the argmax genuinely moved between 16k and 24k. Always run this check per function per week rather than trusting the global default blindly.
- **Four-kernel comparison per round**: RBF, Matern-3/2, Matern-5/2, Mixture RBF (sum of two RBFs at different length scales), compared by LOO Q². RBF remains the working default unless a kernel shows a clear, repeated Q² advantage (Mixture RBF was assigned permanently to F4 earlier in the campaign on this basis, Q²=0.976 vs plain RBF).
- **Seed convention: 5-seed average is now the standard**, not a single-seed draw, per-function canonical seed scheme `2000+f` as the base seed, with 4 additional offset seeds for the stability check. Where the 5 seeds genuinely disagree in direction (this happened for F6, F7, F8 in Week 3), resolve via seed-level override rather than naive averaging: median-of-5 for cases like F6, best-of-5-under-unanimous-direction for cases like F7/F8, and document the specific reasoning per function in that week's evidence workbook (Submission Traceability sheet). This was formalised after a provenance question in Week 5 traced an apparent discrepancy back to an early single-seed draft that was later superseded by exactly this 5-seed process - the final Week 3 submission was correct and more rigorous than the early draft, not an error, but the convention itself was undocumented until now.
- **4-corners model-free probe**: reserved for functions where GP is ineligible (LOO Q² <= 0) AND no directional signal exists from regression either. Currently only F1 qualifies. Method: evaluate GP mean/std directly at the four corners of the input hypercube (0.05/0.05, 0.05/0.95, 0.95/0.05, 0.95/0.95) plus center, and check which corners have never actually been queried (by minimum distance to any historical observation), since a flat/degenerate GP fit can undersupply exploration credit to truly unvisited extremes versus using UCB alone.

**GP eligibility test**: leave-one-out cross-validation Q² > 0 (GP must beat naive mean-prediction baseline), n/dims ratio >= 3.0, flag (not auto-fail) if fitted length scale sits at a kernel bound.

**Decision sequence each week, per function:**
1. Re-fit regression on all data to date, test all 4 gates.
2. Re-fit GP (RBF default, check against Matern-3/2, Matern-5/2, Mixture RBF), generate UCB query via Sobol candidates with that function's own beta and 5-seed convention.
3. Run the 12k/16k/24k convergence check; if the argmax hasn't stabilised by 24k, use the converged value even if it's not the 12k default.
4. If regression passes AND the function isn't on FORCE_GP, check GP's own prediction at the regression-implied point; treat strong disagreement as an override candidate.
5. Decide final method, document why, including any override.
6. Track step size (Euclidean distance) between consecutive weekly queries per function - should track confidence (small step = exploitation/confident, large step = exploration/uncertain). A function breaking this pattern is worth investigating.

**Beta is function-specific, not uniform**, set from actual prior-week outcome (roughly: ~0.8-1.2 for confirmed sustained improvement, 1.5 for mixed/reversed results, 2.0 for no signal at all).

## 4. Function Status - Through Week 4 Confirmed, Week 5 Finalised (not yet submitted)

| Fn | Dims | Description | W4 result | Current best (post-W4) | W5 method | W5 beta | W5 query (proposed, not yet submitted) |
|----|------|--------------|-----------|--------------------------|------------|---------|------------------------------------------|
| F1 | 2 | Radiation field detection | ~4.4e-291 | 0.0 (initial data) | GP-UCB (bagged ensemble) | 2.3 | `0.749424-0.990000` |
| F2 | 2 | Noisy ML log-likelihood | 0.596529 | 0.611205 (initial best still stands) | GP-UCB | 1.7 | `0.775772-0.252390` |
| F3 | 3 | Drug discovery | -0.032780 | -0.021459 (W2) | GP-UCB (FORCE_GP, Matern32) | 1.3 | `0.540591-0.502380-0.520903` |
| F4 | 4 | Warehouse allocation | 0.311066 | 0.389822 (W2) | GP-UCB (FORCE_GP) | 1.2 | `0.399138-0.408810-0.412368-0.460663` |
| F5 | 4 | Chemical yield | 6172.329724 | 6172.329724 (W4, new best) | GP-UCB | 0.8 | `0.974099-0.980482-0.944372-0.936891` |
| F6 | 5 | Recipe scoring | -0.913811 | -0.545514 (W2) | GP-UCB (bagged ensemble) | 1.5 | `0.396883-0.061359-0.654724-0.963308-0.000542` |
| F7 | 6 | ML hyperparameter tuning | 2.374294 | 2.726212 (W2) | GP-UCB (5-seed median) | 1.3 | `0.131941-0.178922-0.387064-0.232804-0.344600-0.794261` |
| F8 | 8 | Neural network tuning | 9.909371 | 9.909371 (W4, new best) | GP-UCB (FORCE_GP, 5-seed median) | 1.4 | `0.128209-0.176123-0.177564-0.179740-0.536273-0.455703-0.251010-0.734363` |

**F5 is the standout trend**: four consecutive confirmed weekly improvements (1089 -> 2044 -> 2620 -> 4128 -> 6172), the only function where every week has been a new best. **F1 remains fully unsolved**: four rounds of near-zero output, GP eligibility strongly negative (LOO Q² = -0.85), regression uninformative (R² = 0.02). **F4 and F7 are volatile** rather than trending - up-down-up-down patterns across weeks, consistent with why regression keeps failing on both (no stable linear relationship for OLS to find).

## 5. Errors Found and Corrected - Keep This Section, Don't Delete Old Entries

- **Week 2, shared LHS seed**: F1 and F2 (both 2D) produced an identical query because both used `seed=42`. Fixed: distinct seed per function; length_scale_bounds=(0.15, 0.6) added.
- **Week 2, missing 4th regression gate**: live analysis had only tested 3 of the 4 documented gates. Corrected F6 from Regression to GP-UCB as a result.
- **Week 2, F8 regression passed all four gates and still failed live**: R²=0.90, predicted 10.38, actual portal return 9.651 (below prior best). First live disconfirmation of the four-gate framework. Directly motivated the permanent FORCE_GP list (F3, F4, F8).
- **Week 4, Sobol replaces LHS as primary candidate generator**, and the 12,000 candidate default was set via an explicit 8k/12k/16k/24k convergence test that week (see Section 3).
- **Week 5, candidate density non-convergence found for F5 and F7**: the initial Week 5 pass used 8,000 candidates (an unintentional regression back to the pre-Week-4 default). Re-running at the correct 12,000 default fixed F5 (query moved to a tighter, fully converged point, step size vs Week 4 dropped from 0.13 to 0.06). F7 required going to 24,000 candidates specifically - the argmax kept moving through 16k and only stabilised at 24k. Lesson: the 12k default is not a universal guarantee: run the convergence check per function per week rather than assuming.
- **Week 5, apparent Week 3 query discrepancy resolved (not an error)**: a search through past chats surfaced an early single-seed draft recommendation for Week 3 that differed meaningfully from what was actually submitted for F4-F8. Investigation found the actual Week 3 submission had gone through a later refinement pass the same session: candidate grid raised to 10k-12k, 5-seed averaging applied as the new standard, with seed-level override (median-of-5 for F6, best-of-5-under-unanimous-direction for F7 and F8) where the 5 seeds genuinely disagreed. The final submitted values match `bbo_week3_evidence_FINAL_v2.xlsx` exactly. This is now written up as the standing seed convention in Section 3 so it does not need to be reconstructed from an old chat again.

## 6. What's Pending / Not Yet Resolved

- Imperial's response to the AI-use disclosure query - status not confirmed in this record, don't assume resolved.
- Stage 1 bankruptcy model's ROC-AUC of 1.0000 on held-out test data - flagged previously, investigation status not confirmed in this record.
- **Week 5 - FINALISED, NOT yet submitted.** All eight queries below are the final recommendations (5-seed averaged, with F1 and F6 overridden to the bagged-ensemble answers). Do not treat any of them as confirmed until Srini actually submits via the portal.
- GitHub push status for Weeks 3-5 material (reflections, evidence workbooks, this context doc) - not confirmed current in this record; verify against the live repo before assuming anything is pushed.
- Module 13-15 discussion board post status - not confirmed in this record; verify directly.
- Module 16.1 and 16.2 both showed a due date of 25 June, already past as of this update - confirmed flexible/extendable per direct conversation with Imperial, not an active risk, but still open and not yet submitted.

## 7. Module 25 Prep Checklist

Module 25 is final polish and GitHub link submission only, not new analysis. Everything it needs has to already exist in the repo by then. Track readiness here every few weeks rather than figuring this out cold in Week 13.

- [ ] **Evidence workbooks kept organised** in `stage2_bbo/evidence/week{n}/` if you want them archived (optional - these are personal working diagnostics, not an Imperial requirement, so this is about your own tidiness/portfolio value, not a completion gate).
- [ ] **Function status table (Section 4 above) kept current**, method used, beta, query, verdict, per function, per week.
- [ ] **Error log (Section 5) never overwritten**, only appended to.
- [ ] **Stage 1 notebooks finalised and out of `.gitignore`** before Module 25, if not done earlier.
- [ ] **Top-level README updated** to reflect the finished Stage 2 BBO section once all 13 weeks are in.
- [ ] **A closing retrospective drafted for Module 25 itself**, once weekly data exists to draw from, covering how strategy evolved (beta schedule, regression vs GP-UCB balance, candidate grid convergence, seed conventions, error corrections) across the full campaign.
- [ ] **AI-use disclosure resolved** with Imperial before final submission, not left outstanding.

Revisit this checklist every 2 to 3 weeks rather than only at the end.

---

*Last updated: Week 5 queries finalised (5-seed averaged, F1/F6 overridden to bagged-ensemble answers), NOT yet submitted to the portal. Candidate grid, seed-averaging, ensemble, and 4-corner conventions formalised in Section 3; Week 3 provenance question investigated and resolved (Section 5); reflections confirmed as discussion-board-only, not a GitHub requirement, and evidence workbooks confirmed as personal assessment tooling, not an Imperial requirement (Section 2). Update the function table and Section 6 every week; append to Section 5 whenever a real error is found and fixed, don't overwrite past entries.*
