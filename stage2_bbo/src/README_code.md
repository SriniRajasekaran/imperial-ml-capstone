# BBO Campaign — Code Reference

## Module structure

| File | Purpose |
|------|---------|
| `data_loader.py` | Load initial data + cumulative weekly submissions |
| `gp_core.py` | GP fitting, UCB/PI acquisition, Sobol candidates |
| `diagnostics.py` | Full battery: IV1-IV10, C1-C5, A1-A5 |
| `week_runner.py` | Master sequence — runs all steps in mandatory order |
| `requirements.txt` | Python dependencies |

## Mandatory weekly sequence

```
C2 → C1 → C1b → GP-UCB(C2 kernel) → GP-UCB(RBF) → C3 → A3 → A1 → A4 → [F8: PI]
```

**C2 must run first, every week, before any candidate generation.**
Skipping C2 causes the GP to default to RBF, which is suboptimal for 6 of 8 functions
(documented as Anomaly 8 in evidence workbooks).

## Usage

```bash
python week_runner.py \
    --initial_data data/initial_data/ \
    --inputs_txt   data/week_10/inputs.txt \
    --outputs_txt  data/week_10/outputs.txt \
    --n_weeks      10 \
    --output_dir   evidence/week_11/
```

## Key constants (update weekly)

In `week_runner.py`:
- `BETA`   — exploration/exploitation parameter per function
- `BOUNDS` — search box per function (None = full [0,1]^d)

In `gp_core.py`:
- `C2_KERNELS` — kernel assignments from most recent C2 run
- `SEEDS`      — fn_index × 100 (fixed across campaign)
- `F8_PI_SEEDS` — [800, 801, 802, 803, 804] (fixed)

## FORCE_GP functions

F3, F4, F7, F8 always use GP regardless of regression gate outcomes.
Controlled via `FORCE_GP = {3, 4, 7, 8}` in `gp_core.py`.

## Diagnostic tests

### IV tests (Input Variable)
| Test | Description |
|------|-------------|
| IV1 | Pearson r per dimension |
| IV2 | Pairwise interaction correlations |
| IV3 | Co-movement confound (OLS vs orthogonalised) |
| IV4 | Non-linearity / curvature (linear vs quadratic R2 gain) |
| IV5 | Tail outlier detection (F5 canonical) |
| IV6 | Local-global driver reversal |
| IV7 | Boundary corner probe |
| IV8 | Heteroskedasticity (Breusch-Pagan) + non-stationarity (split-half LS ratio) |
| IV9 | Ridge-walk topology (GP mean profile along gradient path) |
| IV10 | Length-scale bound sensitivity (rerun every 3-4 weeks) |

### C tests (Challenger Model)
| Test | Description |
|------|-------------|
| C1 | OLS regression 4-gate |
| C1b | Bayesian Ridge tie-breaker |
| C2 | Kernel challenger (LOO Q2) — **run first** |
| C3 | Bootstrap ensemble (30 resamples) |
| C5 | SHAP challenger (XGBoost + TreeExplainer) |

### A tests (Acquisition/Search)
| Test | Description |
|------|-------------|
| A1 | Seed stability (5-seed pairwise distance) |
| A3 | Candidate pool convergence (2^13/14/15/17) |
| A4 | Acquisition comparison (UCB vs EI vs PI) |
