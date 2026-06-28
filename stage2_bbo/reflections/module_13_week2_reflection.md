# Module 13: BBO Challenge Week 2 Reflection

**Module:** 13 (Logistic Regression)
**Round:** Week 2 of 13

---

## Queries Submitted

| Function | Method | Portal Submission |
|----------|--------|-------------------|
| F1 | GP-UCB | `0.880188-0.778955` |
| F2 | GP-UCB | `0.810452-0.970929` |
| F3 | GP-UCB | `0.380322-0.435481-0.509933` |
| F4 | GP-UCB | `0.435162-0.433070-0.367196-0.429856` |
| F5 | GP-UCB | `0.507993-0.974170-0.959359-0.875482` |
| F6 | GP-UCB | `0.326894-0.261195-0.467702-0.596609-0.065132` |
| F7 | GP-UCB | `0.058449-0.198312-0.302879-0.285510-0.282202-0.724036` |
| F8 | Regression | `0.167273-0.137099-0.287499-0.005147-0.843863-0.006698-0.198576-0.592857` |

---

## What was the main change in strategy this week, and what prompted it?

**Beta moved from uniform to function-specific**, set from the actual Week 1 outcome:

- F4, F5 dropped to 1.0 (improved 99.9% and nearly doubled)
- F6, F8 sit at 1.5
- F7 sits at 1.2
- F1 to F3 stayed at 1.8 to 2.0 (no improvement)

Lower beta shifts UCB toward the GP's mean, so F4 and F5 are weighted toward exploitation, pushing further into the region that already proved itself. F1 to F3 stay at high beta because nothing has earned that confidence yet, so the uncertainty term stays dominant and the search continues. Beta here reads directly how much the model has earned the right to trust its own prediction.

**Regression eligibility was re-tested from scratch.**

- *F4:* fails the fourth gate decisively, not narrowly. Regression predicted −4.631 against a current best of 0.3898, while R², normality, and Durbin-Watson all passed. This has now failed gate 4 in both Week 1 and Week 2, by a comparable margin each time, so this is a consistent property of the fit, not a one-off. Stays on GP-UCB, where the model is confident (Q²=0.936).
- *F3:* same pattern, more mildly, kept on GP-UCB.
- *F6:* the full four-gate standard (rather than the three I had been checking) shows it fails the predicted-improvement gate this week. Moves to GP-UCB.
- *F8:* passes all four gates cleanly, stays on Regression.

**Candidate generation switched to Latin Hypercube Sampling.** Uniform random sampling was leaving uneven coverage, more noticeable as dimensionality grows. Running both side by side on Week 1 data showed LHS finding a meaningfully higher UCB score for two higher-dimensional functions, which is what triggered the switch.

---

## Did you focus more on exploration or exploitation this week? Why? What trade-offs did you weigh?

Both, split by function. Decomposing each chosen point's UCB score into mean (exploitation) and beta-times-sigma (exploration): F1, F2 near 100% exploration (GP mean essentially zero), F4, F5 near 100% exploitation (GP confident after Week 1), F6 to F8 in between.

The clearest trade-off was on F1. Checking why F1 and F2 produced an identical query, I found F1's GP length scale had collapsed to near zero, treating the function as indistinguishable from noise, a different problem from genuinely lacking information. Fixed with a length-scale constraint, not a beta change.

---

## Have any participant discussions or recent outputs influenced how you approached this week's submission?

Mainly my own Week 1 evidence package, used as a repeatable check, which surfaced F3 and F4 newly clearing the gates this week. I also ran a structured UCB versus Expected Improvement comparison: five of eight functions picked the same point under both, three diverged, most notably F8 (strongest regression support, R²=0.90). UCB was kept for consistency with the taught method; flagged for revisit.

---

## If you were to fit a simple linear or logistic regression model to one of these functions, which assumptions would be most likely violated?

This was already part of the process from Week 1, when every function ran through the same four-gate test before any method was chosen; the larger Week 2 dataset sharpens the same violations rather than introducing new ones.

Logistic regression doesn't apply here, since every output is continuous with no class label to estimate.

Linear regression is the relevant comparison:

- *F1:* near-noise outputs, R²=0.044, Shapiro-Wilk rejects normality outright
- *F7:* Durbin-Watson at 1.42, just outside range, suggesting unmodelled non-linearity or interaction
- *F8:* eight inputs against forty-one observations, a thin ratio even though R²=0.90 looks reassuring

---

## Are there regions where the output appears roughly linear, or where a decision boundary might form? How might a logistic regression classifier perform?

**F5 is the clearest candidate.** A small high-yield cluster shares x2, x3, x4 all above roughly 0.90, separated from a larger lower-yield cluster nearer 0.50, with a visible gap rather than a gradient, a usable boundary more than for any other function.

**F4 shows a milder version,** three of four inputs moving the same direction between buckets, less sharply separated. **F1, F2, F7 show no clustering**, outputs changing too gradually, or with too little signal, for a binary framing to help.

---

## Did you find it useful to consider individual feature effects before deciding on your query point?

Yes. This is where regression earned its place even where its gates didn't justify it as the query method outright.

- *F8:* four of eight inputs (x1, x2, x3, x7) returned individually significant coefficients, three below p=0.0001, pushed toward their informative extreme; the rest left at the neutral midpoint.
- *F6:* the same exercise, smaller scale, revealed (once properly gated) that F6 should not be regression-led this round.

Without it there would be no principled way to decide which inputs deserved to move, since a GP gives a recommended point but doesn't explain which input drives it.
