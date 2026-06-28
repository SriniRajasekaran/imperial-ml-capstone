# PROJECT_CONTEXT.md

**Purpose of this file:** paste this entire document at the start of any new chat (in this project or otherwise) to restore working context immediately, without re-explaining the project from scratch. Update it after any session with a material decision, correction, or strategy change. This is a living log, not a one-time scope description.

---

## 1. Who and What

Srini Rajasekaran, ~25 years model risk / XVA / credit at NatWest, BoA, Citi. Currently Director, Head of Structured Credit & Real Estate Assurance at Citibank. Completing the Imperial College London Professional Certificate in ML & AI. Capstone has two stages:

- **Stage 1**: free-exploration bankruptcy early-warning audit toolkit (Modules 3–11), built around the Polish Companies Bankruptcy dataset (UCI ID 365), with SHAP-based explainability diagnostics, ESR (Explanation Sensitivity Ratio), and governance/regulatory alignment framing (SR 11-7, PRA SS1/23, EU AI Act, NIST AI RMF).
- **Stage 2**: fixed Black-Box Optimisation (BBO) challenge (Modules 12–24), one portal query per function per week across 8 unknown functions of dimensionality 2D–8D, scored only by the single y returned. Module 25 is final polish.

GitHub repo mirrors both stages. Stage 2 lives in `stage2_bbo/` with subfolders `notebooks/`, `data/module_XX/`, `reflections/`, `evidence/`.

## 2. Hard Rules — Always Apply, No Exceptions

- No AI self-reference anywhere in deliverables: GitHub commits, reflections, discussion board posts, README. These read as Srini's own analysis and judgment.
- No em dashes in any written output, anywhere, unless explicitly told otherwise for a one-off.
- Claude scaffolds and drafts; Srini directs strategy, decides overrides, runs the actual portal submissions, owns final judgment calls.
- Never call it "your project" in deliverable text; call it "the toolkit" / "the campaign."
- Disclosure: Srini has queried Imperial directly about AI-use disclosure requirements for the capstone; awaiting their response as of last update. Treat this as unresolved, don't assume an answer.

## 3. Stage 2 BBO — Current Methodology (as of last update)

**Two competing query-generation methods per function, decided fresh every week:**

- **GP-UCB**: Gaussian Process with RBF kernel, UCB acquisition (`mu + beta*sigma`). Candidate points generated via Latin Hypercube Sampling (switched from uniform random after Week 1, confirmed LHS finds materially better candidates). Each function gets its own random seed (never shared — a shared seed=42 across two same-dimensional functions caused an identical, wrong query in Week 2; root-caused and fixed).
- **Regression**: OLS on standardised inputs, significance-gated (only act on coefficients with p<0.05; insignificant inputs left at neutral 0.50). Eligibility requires **four** gates: R²>0.30, Shapiro-Wilk p>0.05 (residual normality), Durbin-Watson 1.5–2.5 (no autocorrelation), AND predicted y at the significance-gated query exceeds current best.

**GP eligibility test**: leave-one-out cross-validation Q² > 0 (GP must beat naive mean-prediction baseline), n/dims ratio ≥ 3.0, flag (not auto-fail) if fitted length scale sits at a kernel bound.

**Decision sequence each week, per function:**
1. Re-fit regression on all data to date, test all 4 gates.
2. Re-fit GP, generate UCB query via LHS with that function's own beta and seed.
3. If regression passes, check GP's own prediction at the regression-implied point. If GP disagrees strongly (much worse mean, high sigma), treat as override candidate.
4. Decide final method, document why, including any override.
5. Track step size (Euclidean distance) between consecutive weekly queries per function — should track confidence (small step = exploitation/confident, large step = exploration/uncertain). A function breaking this pattern is worth investigating.

**Beta is function-specific, not uniform**, set from actual prior-week outcome.

## 4. Week 2 Results — Confirmed by Imperial Portal (NEW — this update)

All eight Week 2 queries returned and confirmed. Source: `bbo_week3_evidence_v2.xlsx`, sheet "Week 2 Results Assessment."

| Fn | Dims | W2 query submitted | W2 output | Prior best | New best? | Δ | Strategic implication |
|----|------|---------------------|-----------|------------|-----------|---|------------------------|
| F1 | 2 | `0.880188-0.778955` | −2.9109e-60 | 0.0 | No | ≈0 | Third round with no signal. Pure exploration probe continues, β stays at 2.0. |
| F2 | 2 | `0.810452-0.970929` | 0.0681 | 0.611 (initial) | No | −0.040 vs Wk1 | Went backwards again. High x1 confirmed wrong direction for the second time running — explore lower x1 next. |
| F3 | 3 | `0.380322-0.435481-0.509933` | −0.021459 | −0.034835 | **YES** | +0.013376 | First improvement. Direction confirmed, β reduced 1.8→1.5. |
| F4 | 4 | `0.435162-0.433070-0.367196-0.429856` | 0.38982 | −0.00553 | **YES** | +0.39535 | Second consecutive dramatic gain. Deepen exploitation, β reduced to 0.8. |
| F5 | 4 | `0.507993-0.974170-0.959359-0.875482` | 2619.616 | 2044.217 | **YES** | +575.40 | Strong continued gain, x2/x3/x4-high pattern reconfirmed. β reduced to 0.8. |
| F6 | 5 | `0.326894-0.261195-0.467702-0.596609-0.065132` | −0.545514 | −0.629393 | **YES** | +0.083879 | Consistent improvement on the corrected GP-UCB method. x4-high/x5-low confirmed. β 1.5→1.2. |
| F7 | 6 | `0.058449-0.198312-0.302879-0.285510-0.282202-0.724036` | 2.726212 | 2.089002 | **YES** | +0.637210 | Strong gain (+30.5%). x1-very-low/x6-high confirmed. Full exploitation, β 1.2→1.0. |
| F8 | 8 | `0.167273-0.137099-0.287499-0.005147-0.843863-0.006698-0.198576-0.592857` | 9.651064 | 9.7523 | **No** | −0.101236 | **Regression failed.** All 4 gates had passed and predicted y=10.38; actual result went backwards. Reset to GP-UCB, β raised to 2.0. |

**Net picture:** five of eight functions improved (F3, F4, F5, F6, F7). F2 worsened for the second consecutive week. F1 remains flat at noise-floor magnitude for a third round. F8 is the most consequential result this cycle — see Section 5.

## 5. Errors Found and Corrected — Keep This Section, Don't Delete Old Entries

- **Week 2, shared LHS seed**: F1 and F2 (both 2D) produced an identical query because both used `seed=42`. Root cause was twofold: shared seed AND F1's GP had a degenerate (collapsed to ~0) length scale, unconstrained. Fixed: distinct seed per function; length_scale_bounds=(0.15, 0.6) added.
- **Week 2, missing 4th regression gate**: live chat analysis across Weeks 1–2 only tested 3 of the documented 4 regression eligibility gates, omitting "predicted y exceeds current best." Corrected F6 from Regression to GP-UCB as a result. Always verify all 4 gates explicitly, every week, for every function being considered for Regression.
- **Week 2, stale derived artefacts**: after fixing a value (e.g. F2's query, F6's method), downstream sheets/cells built from the old value were not always automatically updated. Lesson: after any correction, explicitly re-scan every derived artefact, don't assume one fix propagates.
- **Week 2 result, F8 — regression passed all four gates and still failed live (NEW)**: F8 had R²=0.90, passed Shapiro-Wilk, Durbin-Watson, and gate 4 (predicted 10.38 against a current best of 9.752), and was the only function using Regression as the query method. The actual portal return was 9.651 — below the prior best, not above it. This is a genuine case of a statistically eligible model failing out-of-sample, not a process error: every gate was correctly tested and correctly passed, the model itself was simply wrong on this query point. Recorded here because it is the first live disconfirmation of the four-gate framework's reliability, and it directly motivates re-running the gate test cautiously rather than trusting a clean pass at face value going forward. F8 has been reset to GP-UCB at β=2.0 (full exploration) for Week 3, and flagged for SVM level-set cross-checking (Module 14) given only one historical observation sits above the current best, which makes that boundary thin.

## 6. What's Pending / Not Yet Resolved

- Imperial's response to the AI-use disclosure query (sent, awaiting reply).
- Stage 1 bankruptcy model showing a suspicious ROC-AUC of 1.0000 on held-out test data — not yet investigated, Srini said he'd look into it himself.
- F8's regression failure (Section 5) needs the gate test re-run on the Week 3 dataset before any future return to Regression is considered; for now it sits on GP-UCB.
- F8 seed-sensitivity re-check (flagged last week, most seed-sensitive function found, avg pairwise disagreement 0.477 across 5 seeds) — still outstanding, now compounded by the regression failure.
- UCB vs Expected Improvement comparison — still flagged for revisit; less urgent now that F8 (the function where EI diverged most) has moved off Regression entirely.
- **Module 13 discussion board post — not yet posted.** Reflection text is drafted and finalised (see `stage2_bbo/reflections/module_13_week2_reflection.md`), but Srini has not yet posted it to the Imperial discussion board.
- **GitHub repo — not yet updated.** Week 2 reflection, evidence workbook, and this context doc are all still local/in-chat only as of this update; none of it is pushed to the live repo yet.
- **Week 3 — NOT yet submitted.** The evidence workbook (`bbo_week3_evidence_v2.xlsx`) contains proposed Week 3 queries per function (gate re-test, GP eligibility re-test, expected step-size/direction), but these are draft recommendations only. No Week 3 portal submission has been made. Do not treat any "expected" column in that workbook as a confirmed outcome, and do not treat the proposed queries as final until Srini reviews and submits them.

## 7. Module 25 Prep Checklist (NEW — start tracking from Week 2 onward)

Module 25 is final polish and GitHub link submission only, not new analysis. Everything it needs has
to already exist in the repo by then. Track readiness here every few weeks rather than figuring this
out cold in Week 13.

- [ ] **Every week's reflection pushed** to `stage2_bbo/reflections/` as it's finalised, not batched up
      later. Missing weeks are the hardest thing to reconstruct retroactively.
- [ ] **Every week's evidence workbook pushed** to `stage2_bbo/evidence/week{n}/`, including gate
      re-tests, GP eligibility checks, and any comparison sheets built that week.
- [ ] **Submission log (Section 5 above) updated every week**, both the query submitted and the
      confirmed result once it returns.
- [ ] **Function table (Section 4 above) kept current**, method used, beta, verdict, per function,
      per week, not just the most recent snapshot.
- [ ] **Error log (Section 5, "Errors Found and Corrected") never overwritten**, only appended to, so
      Module 25's retrospective can show genuine methodological evolution rather than a clean story
      that skips the real mistakes.
- [ ] **Stage 1 notebooks finalised and out of `.gitignore`** before Module 25, if not done earlier
      (currently still staged individually per Section 3 conventions in the Imperial Capstone Project
      Context Document).
- [ ] **Top-level README updated** to reflect the finished Stage 2 BBO section once all 13 weeks are
      in, not left as a Week 1-era draft.
- [ ] **A closing retrospective drafted for Module 25 itself**, once weekly data exists to draw from,
      covering how strategy evolved (beta schedule, regression vs GP-UCB balance, error corrections)
      across the full campaign rather than any single week.
- [ ] **AI-use disclosure resolved** with Imperial before final submission, not left outstanding.

Revisit this checklist every 2 to 3 weeks rather than only at the end. If any box above is unchecked
going into the final few weeks, that is the moment to fix it, not Module 25 itself.

---

*Last updated: Week 2 results confirmed and logged. Week 3 queries proposed but not submitted.
Discussion board post and GitHub update both still pending. Module 25 prep checklist added. Update
the function table and Section 6 every week; append to Section 5 whenever a real error is found and
fixed, don't overwrite past entries.*
