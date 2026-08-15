# Stage 2 — BBO Challenge

This folder contains all artefacts for the Stage 2 Black-Box Optimisation (BBO) capstone challenge (Modules 12–25, Imperial College London Professional Certificate in ML & AI).

## Structure

```
stage2_bbo/
├── reflections/          Weekly discussion board reflections (Modules 12–22)
├── evidence/             Evidence workbooks and master tracker
├── datasheet_bbo_dataset.md    Dataset documentation (Module 21.2)
├── model_card_bbo_optimiser.md  Approach documentation (Module 21.2)
└── README.md
```

## Key documents

| Document | Description |
|----------|-------------|
| [datasheet_bbo_dataset.md](datasheet_bbo_dataset.md) | Datasheet documenting the BBO query dataset (composition, collection, preprocessing, uses) |
| [model_card_bbo_optimiser.md](model_card_bbo_optimiser.md) | Model card for the GP-UCB optimisation strategy (overview, performance, assumptions, ethics) |

## Campaign summary (after 10 rounds)

| Function | Dims | Initial best | Campaign best | Round achieved |
|----------|------|-------------|--------------|----------------|
| F1 — Radiation detection | 2 | 0.000 | 1.453 | Round 7 |
| F2 — ML log-likelihood | 2 | 0.611 | 0.746 | Round 6 |
| F3 — Drug discovery | 3 | −0.035 | −0.007 | Round 8 |
| F4 — Warehouse allocation | 4 | −4.026 | +0.492 | Round 9 |
| F5 — Chemical yield | 4 | 1089 | 8561 | Round 10 |
| F6 — Recipe scoring | 5 | −0.714 | −0.211 | Round 10 |
| F7 — Hyperparameter tuning | 6 | 1.365 | 3.008 | Round 6 |
| F8 — Neural net tuning | 8 | 9.598 | 9.971 | Round 9 |

All eight functions improved. F5 set new bests in 9 of 10 rounds.

## Core methodology

- **Surrogate:** Gaussian Process (scikit-learn), kernel selected weekly by LOO Q² comparison (RBF vs Matern-3/2 vs Matern-5/2)
- **Acquisition:** UCB (mu + beta × sigma) for F1–F7; PI for F8 (switched Round 9 after 7/7 hindsight backtest)
- **Candidates:** Scrambled Sobol sequences, 2^15 default, escalated to 2^17 where not converged
- **Challenger:** OLS regression with four eligibility gates run weekly; used when all gates pass (except FORCE_GP functions)
- **Seed stability:** Five seeds per function per round (fn_index × 100 scheme); median or corroborated-best selection

## Reflections

All discussion board reflections are in the `reflections/` folder, one per module.
