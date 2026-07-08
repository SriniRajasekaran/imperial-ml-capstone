# Imperial Capstone — Project Context Document

**Purpose of this file:** Persistent reference document for the Claude Project. Upload
to the Project knowledge base. At the start of each new chat, Claude treats this as
ground truth rather than relying on conversation memory.

**Owner:** Srini Rajasekaran
**Programme:** Imperial College London — Professional Certificate in ML & AI
**Last updated:** July 2026 — current through Week 4 (Module 15)

---

## 1. Who I Am and What This Project Is

Srini Rajasekaran — approximately 25 years in investment banking spanning rates,
credit, XVA, and structured products. Senior model risk and independent assurance
roles at NatWest, Bank of America, and Citibank. Currently Director, Head of
Structured Credit and Real Estate Assurance at Citibank. Completing the Imperial
College London Professional Certificate in ML & AI.

**Stage 1 (Modules 3–11):** Free-exploration portfolio. Automated Model Risk Audit
Toolkit with Quantitative Explainability Diagnostics, using the Polish Companies
Bankruptcy dataset (UCI ID 365). Five technique notebooks. No fixed weekly deadline.

**Stage 2 (Modules 12–24):** Fixed BBO (Black-Box Optimisation) challenge. One query
per function per week via the Imperial capstone portal, across 8 unknown functions
(2D to 8D). Weekly reflection posted to the Imperial discussion board (~700 words).

**Module 25:** Final polish and GitHub link submission only.

---

## 2. Ethical Working Framework

- Claude provides scaffolding, code, analysis, and drafts. Srini directs strategy,
  runs notebooks, submits portal queries, and supplies genuine observations.
- Reflections must read as Srini's own genuine voice. No AI self-reference anywhere.
- No em dashes in any reflection or discussion board text — ever.
- Do not say "your project" — use "the campaign" or "the toolkit."
- AI-use disclosure: query sent to Imperial, response pending. Treat as unresolved.

---

## 3. Stage 1 — Current State

**GitHub repo:** https://github.com/SriniRajasekaran/imperial-ml-capstone

Core deliverable: Automated Model Risk Audit Toolkit. XGBoost as primary model,
Logistic Regression as SHAP calibration anchor. Hyperparameters: n_estimators=300,
max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
scale_pos_weight=19.3. SMOTE (k=5) on training fold only. RobustScaler. Median
imputation fit on training data only.

Four audit diagnostics: Explanation Stability (bootstrap SHAP variance), Explanation
Drift (Wasserstein/KL/KS), ESR novel contribution (ESR_i = |dSHAP_i| / (|S_i| + d)),
Local Consistency (k-means on SHAP vectors). Regulatory alignment: PRA SS1/23 §3.4,
SR 11-7 §IV, EU AI Act Art. 13, NIST AI RMF GOVERN 1.2.

---

## 4. Stage 2 — BBO Challenge: Method and Conventions

### 4.1 CRITICAL — Dataset

The full dataset per function is: **initial Imperial-provided observations + weekly
submissions.** Always load both. Never work from submitted queries alone.

| Fn | n initial | n after W4 |
|----|-----------|------------|
| F1 | 10 | 14 |
| F2 | 10 | 14 |
| F3 | 15 | 19 |
| F4 | 30 | 34 |
| F5 | 20 | 24 |
| F6 | 20 | 24 |
| F7 | 30 | 34 |
| F8 | 40 | 44 |

Initial data lives at: stage2_bbo/data/initial/function_{f}/initial_inputs.npy and
initial_outputs.npy. This error (working from n=3 instead of n=13–44) was caught in
the Week 4 session and corrected. Do not repeat it.

### 4.2 Two-method framework

**Method 1 — OLS Regression** (standardised inputs). All FOUR gates must pass:
1. R² > 0.30
2. Shapiro-Wilk p > 0.05 (residual normality)
3. Durbin-Watson 1.5–2.5 (no autocorrelation)
4. Regression-predicted y at significance-gated query exceeds current best

When gates pass, only significant coefficients (p < 0.05) set direction:
x_i to 0.95 if positive, 0.05 if negative, 0.50 if not significant.

**Method 2 — GP-UCB.** UCB(x) = GP_mean(x) + beta * GP_std(x). Candidate
generation via Sobol quasi-random sequences (switched from LHS in Week 4; Sobol
gives lower UCB variance especially in higher dimensions). Standard candidate
counts: 12k for 2D–3D, 16k for 4D–8D.

Five-seed stability check: if average pairwise distance across 5 seeds < 0.15,
use 5-seed mean; otherwise use best individual seed. Seeds: seed_base = 4000 + f
for Round 4; seed_base + k*100 for k=0..4.

**FORCE_GP list** (always GP-UCB, never regression):
- F3: historical regression instability (W1 went backward)
- F4: non-normal residuals; collapse behaviour
- F8: W2 regression passed all 4 gates but failed out-of-sample

### 4.3 Kernel selection

Test all four kernels each round; select by LOO Q2. Default initial ls =
max(0.10, 0.30 - 0.03*(d-2)). Bounds: (0.05, 1.00).

| Kernel | Best for | Current winner |
|--------|----------|----------------|
| RBF | Smooth, monotone surfaces | F8 (Q2=0.915) |
| Matern-3/2 | Once-differentiable, ridges | F1, F2, F6 (Q2=0.699) |
| Matern-5/2 | Twice-differentiable | F5 (Q2=0.912), F7 (Q2=0.888) |
| Mixture RBF | Multi-scale, narrow ridge | F4 (Q2=0.976) — campaign best |

Mixture RBF: C1*RBF(ls=0.10, bounds=(0.05,0.30)) + C2*RBF(ls=0.40, bounds=(0.20,1.00)).
Use when single-scale kernels give Q2 < 0.80 on a function with ridge structure.

### 4.4 Acquisition functions

Campaign is NOT limited to GP-UCB. Full family available:

- UCB: mu + beta*sigma. Default. Requires beta tuning.
- EI: E[max(f-f*,0)]. Analytically tractable. No beta. Best for exploitation (Q2>0.80).
- PI: P[f(x)>f*]. Simpler than EI. Best late campaign.
- Thompson Sampling: sample from GP posterior. Good for noisy functions (F2).

Run EI alongside UCB from Round 5 for functions with Q2 > 0.80. Where they agree:
strong confirmation. Where they disagree: pick by predicted improvement.

In Week 4: for F5, UCB=EI=PI all agree (identical query). For F8, UCB=EI agree
within trust region. These are the strongest confirmations available.

### 4.5 Trust regions

Constrain candidate search to neighbourhood around best observed point. Appropriate
when: GP eligible (Q2>0) AND direction confirmed AND exploitation phase.

Symmetric TR: L2 ball of radius r around best observed point. Generate Sobol
candidates, filter to those within radius, run UCB/EI within filtered set.

Key results from Week 4:
- F3: TR r=0.20 gives first positive predicted improvement (+0.002). Adopt.
- F4: TR marginal (+0.011). Step-floor constraint is equivalent.
- F7: TR makes things worse. GP+gradient ascent wins.
- F8: TR r=0.40, UCB=EI agree. Predicted +0.108. Adopt.
- In 8D, use r >= 0.40 (smaller radii have too few candidates).

### 4.6 4-Corners test (model-free)

Applicable ONLY when: GP ineligible (Q2<0) AND no directional signal.
Currently F1 only.

Method: generate all 2^d corners (each dim = 0.05 or 0.95). Compute minimum L2
distance from all prior observations (full dataset, not just submitted queries).
Select corner with maximum min-distance.

CRITICAL: Run on the full dataset. With n=3 (submitted queries only) the wrong
corner was selected in Week 4 (BL instead of LH). On all 13 obs for F1, the
top-left (0.05, 0.95) has max min-distance (0.328).

### 4.7 Gradient ascent refinement

After seed-pick selection, run L-BFGS-B minimisation of negative UCB on the GP
surface, multi-started from the Sobol argmax. Clips to [0,1]^d. Use for stable
functions in exploitation phase. Improved F7 UCB 3.117 to 3.423 in Week 4.

Do NOT use for structurally unstable functions as the sole search method.

### 4.8 GP eligibility gates

- LOO Q2 > 0 (GP beats naive mean-prediction baseline)
- n/dims >= 3.0 (coverage adequacy, flag not auto-fail)
- Length-scale bounds: ls in [0.05, 1.00]

### 4.9 Beta schedule

Function-specific. Updated weekly based on outcomes.

| Beta | Phase |
|------|-------|
| 2.0 | Exploration — no signal, GP ineligible, or reset after collapse |
| 1.5 | Transition — recent new best, beginning to shift |
| 1.2 | Balanced — known region, refining |
| 1.0 | Exploitation — confirmed direction, targeting known basin |
| 0.8 | Deep exploitation — Q2 > 0.90, confident direction |
| 0.5 | Refinement — late campaign, near confirmed optimum |

W4 betas: F1=2.0, F2=2.0, F3=1.5, F4=2.0, F5=0.8, F6=1.2, F7=1.0, F8=1.5.

### 4.10 Function reference table — current state through W4

| Fn | Dims | n (W4) | Running best | W4 portal string | Kernel | Q2 | Beta | Method |
|----|------|---------|--------------|------------------|--------|----|------|--------|
| F1 | 2 | 14 | 0.000 | 0.050000-0.950000 | M32 | -0.198 | 2.0 | 4-corners LH |
| F2 | 2 | 14 | 0.611 | 0.691130-0.898575 | M32 | -0.276 | 2.0 | GP-UCB |
| F3 | 3 | 19 | -0.0215 | 0.390888-0.247459-0.445221 | RBF | 0.011 | 1.5 | TR r=0.20 |
| F4 | 4 | 34 | 0.3898 | 0.436185-0.389492-0.378432-0.414376 | MixRBF | 0.976 | 2.0 | GP-UCB+grad FORCE |
| F5 | 4 | 24 | 4128.34 | 0.966254-0.954711-0.973370-0.958107 | M52 | 0.912 | 0.8 | GP-UCB (UCB=EI=PI) |
| F6 | 5 | 24 | -0.5455 | 0.050000-0.500000-0.500000-0.950000-0.050000 | M32 | 0.699 | 1.2 | Regression |
| F7 | 6 | 34 | 2.7262 | 0.008484-0.134504-0.272871-0.262378-0.266104-0.759966 | M52 | 0.888 | 1.0 | GP-UCB+grad |
| F8 | 8 | 44 | 9.8464 | 0.044068-0.389868-0.184727-0.160732-0.755472-0.375973-0.216312-0.508971 | RBF | 0.915 | 1.5 | TR r=0.40 FORCE |

Notes on W4 changes:
- F1: updated from GP-UCB (0.778313-0.788605) to 4-corners LH on full n=13 dataset
- F3: updated to TR r=0.20 — first positive predicted improvement for this function
- F8: updated to TR r=0.40 — UCB and EI agree; step reduced 0.845 to 0.394

### 4.11 Regression gate history — errors not to repeat

- W1: only 3 gates applied. Gate 4 was missing. Corrected.
- W2: F6 retroactively failed under correct 4-gate standard.
- W2: F8 passed all 4 gates (R2=0.90) but returned 9.651 vs predicted 10.38.
  First live disconfirmation. F8 permanently on FORCE_GP.
- W4: F6 passes all 4 gates cleanly for the first time (R2=0.738). Regression used.

### 4.12 Variable role inference (working hypotheses, not ground truth)

- F1: no signal on either input. Spatial coordinates.
- F2: x1 dominant driver (r=0.75). x2 secondary. Initial best 0.611 at (0.703, 0.927).
- F3: x3 dominant negative driver — toxicity analogue.
- F4: x1, x2, x4 negative drivers (allocation proportions). x3 neutral.
- F5: x2, x3, x4 positive drivers (reagent concentrations). x1 process condition.
- F6: x4 positive, x5 negative (key recipe ingredients). x1 negative moderately.
- F7: x1 negative, x6 positive. Other dims neutral or marginal.
- F8: x1 (learning rate), x3 (dropout) both dominant negative (r=-0.65 each).
  x7 secondary negative. x2, x4-x6, x8 neutral.

---

## 5. Submission Log

| Module | Week | Submitted | W results | Reflection | Notes |
|--------|------|-----------|-----------|------------|-------|
| 12 | 1 | F1-F8 | F4 +99.9%, F5 nearly doubled | Yes | No extension |
| 13 | 2 | F1-F8 | F3,F4,F5,F6,F7 improved. F8 regression failed live. | Pending | Date extension obtained |
| 14 | 3 | F1-F8 | F5 NEW BEST 4128 (+1509). F8 NEW BEST 9.846. F4 COLLAPSED -0.967. | Yes | Pushed to GitHub |
| 15 | 4 | F1-F8 | Pending — queries submitted | Drafted (15.1 v9, 15.2 v2) | 15.2 due date 18 Jun may need extension request |

W3 results in full:
F1: 7.495e-125, F2: 0.153, F3: -0.046, F4: -0.967, F5: 4128.34,
F6: -0.770, F7: 2.607, F8: 9.846

---

## 6. Reflection Conventions

- No em dashes, ever. First person, Srini's voice, no AI self-reference.
- Target under 700 words (up to ~800-1000 acceptable given prompt volume).
- Answer Imperial's specific prompts for that module — they change each week.
- Always include actual portal strings, R2, p-values, DW, LOO Q2 where relevant.
- Module 15.1 prompts: support vectors, gradients, classification framing, model
  choice (linear/SVM/NN), interpretability vs flexibility, backpropagation/boundary.
- Module 15.2 prompts: hyperparameter effects on convergence, discrete vs continuous,
  BBO applied to NN directly.
- Preferred structure: common methodology section (bullets), then thematic sections
  with bullets for specific cases and prose for cross-cutting points.

---

## 7. GitHub Conventions

- Author: "Srini Rajasekaran" via git config user.name / user.email.
- Branch: main.
- Local path: C:/Users/rsnee/Documents/Nivas HD/ML AI References/Capstone Project/Stage 2/Capstone_repo

Repo structure for Stage 2:
```
stage2_bbo/
    data/initial/function_{f}/    initial_inputs.npy, initial_outputs.npy
    data/module_{N}/function_{f}/ accumulated data through module N
    evidence/week{N}/             evidence workbooks per week
    reflections/                  .md and .docx reflection files
    queries/                      portal string records per week
```

W3 and W4 pushed in Week 4 session. Initial Imperial data (.npy files) now in repo.

---

## 8. Key Methodological Decisions — Do Not Re-Litigate

1. Sobol over LHS from W4: lower UCB variance, especially in higher dimensions.
2. Mixture RBF for F4 permanently: Q2=0.976 vs RBF 0.690. Ridge confirmed.
3. FORCE_GP for F3, F4, F8: historical failures documented in Section 4.11.
4. 4-corners only when GP ineligible AND no directional signal: F1 only currently.
5. Trust region for F3 (r=0.20) and F8 (r=0.40): UCB=EI agreement confirmed.
6. EI tested alongside UCB: F5 (three functions agree), F8 (UCB=EI within TR).
7. Always load full dataset: initial data + weekly submissions. Never n=3 only.
8. Gate 4 is never relaxed: regression must predict above current best.
9. F8 regression permanently blocked: one live failure is sufficient evidence.
10. Gradient ascent as refinement: post-Sobol L-BFGS-B for exploitation functions.

---

## 9. Module 25 Prep Checklist

- [x] W1 reflection pushed
- [ ] W2 reflection pushed (pending — date extension)
- [x] W3 reflection pushed
- [x] W4 reflection drafted and pushed
- [x] Evidence workbooks W3 and W4 pushed
- [x] Initial dataset (.npy files) added to repo
- [ ] Stage 1 notebooks finalised and out of .gitignore
- [ ] Top-level README updated with Stage 2 results through all rounds
- [ ] Closing retrospective drafted (do after Round 13)
- [ ] AI-use disclosure resolved with Imperial before final submission

---

## 10. How to Use This Project Effectively

- Load initial data (stage2_bbo/data/initial/) every session. Never work from
  submitted queries only.
- The function table in Section 4.10 is the single source of truth for current bests.
- Section 8 lists methodological decisions already made. Do not re-open without new evidence.
- Update Section 5 after W4 results come back from the portal.
- BBO_Master_Tracker.xlsx holds the raw numbers; this document holds methodology context.
