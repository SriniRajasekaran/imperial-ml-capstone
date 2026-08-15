"""
week_runner.py
Master weekly sequence for BBO campaign.

Mandatory execution order (no step may be skipped):
    1.  C2  — kernel challenger (LOO Q2: RBF / Matern32 / Matern52)
    2.  C1  — regression 4-gate check
    3.  C1b — Bayesian Ridge tie-breaker (if C1 passes narrowly)
    4.  GP-UCB with C2 winner kernel
    5.  GP-UCB with RBF (audit comparison)
    6.  C3  — bootstrap ensemble GP (30 resamples)
    7.  A3  — candidate pool convergence test (2^13/14/15/17)
    8.  A1  — seed stability check (5 seeds)
    9.  A4  — acquisition comparison (UCB vs EI vs PI)
    10. F8  — PI 5-seed median (overrides all above for F8)
    11. Decision: select final candidate with documented rationale

Usage
-----
    python week_runner.py \\
        --initial_data  data/initial_data/ \\
        --inputs_txt    data/week_10/inputs.txt \\
        --outputs_txt   data/week_10/outputs.txt \\
        --n_weeks       10 \\
        --output_dir    evidence/week_11/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from data_loader import load_all
from gp_core import (C2_KERNELS, FORCE_GP, SEEDS, F8_PI_SEEDS,
                     ucb_query, pi_query_5seed)
from diagnostics import (
    c2_kernel_challenger, c1_regression_gates, c1b_bayesian_ridge,
    c3_bootstrap_ensemble, a1_seed_stability, a3_candidate_convergence,
    a4_acquisition_comparison,
    iv1_individual_sensitivity, iv2_interaction_effects,
    iv4_nonlinearity_curvature, iv5_tail_outlier,
    iv8_heteroskedasticity_nonstationarity, iv10_ls_bound_sensitivity,
)

# ── Beta schedule (update each week based on outcomes) ───────────────────────
BETA = {1: 2.0, 2: 1.5, 3: 1.3, 4: 1.2, 5: 0.5, 6: 1.0, 7: 1.2, 8: 1.0}

# ── Search bounds (None = full [0,1]^d; update per function as needed) ───────
BOUNDS = {fn: None for fn in range(1, 9)}
# Example overrides (Week 11):
# BOUNDS[5] = [(0.990, 0.9999)] * 4
# BOUNDS[6] = [(0.01,0.40),(0.01,0.40),(0.30,0.90),(0.90,0.99),(0.01,0.20)]


def run_weekly_battery(fn: int, X: np.ndarray, y: np.ndarray,
                        output_dir: str) -> dict:
    """
    Run the full diagnostic and candidate generation battery for one function.

    Parameters
    ----------
    fn         : function index (1-8)
    X, y       : combined dataset (initial + all weekly submissions)
    output_dir : where to write per-function JSON results

    Returns
    -------
    results : dict with all test outputs and final candidate recommendation
    """
    print(f"\n{'='*60}")
    print(f"F{fn} ({X.shape[1]}D, n={len(y)}, best={y.max():.6g})")
    print(f"{'='*60}")
    results = {'fn': fn, 'dims': int(X.shape[1]), 'n': int(len(y)),
               'current_best': float(y.max())}

    # ── Step 1: C2 kernel challenger ─────────────────────────────────────────
    print("\n[1] C2 Kernel Challenger...")
    c2 = c2_kernel_challenger(X, y)
    results['C2'] = c2
    kernel = c2['winner']
    print(f"    Winner: {kernel} (Q2={c2['scores'][kernel]['q2']})")

    # ── Step 2: C1 regression gates ──────────────────────────────────────────
    print("\n[2] C1 Regression Gates...")
    c1 = c1_regression_gates(X, y)
    results['C1'] = c1
    reg_eligible = c1['all_pass'] and fn not in FORCE_GP
    print(f"    All pass: {c1['all_pass']} | FORCE_GP: {fn in FORCE_GP} | Eligible: {reg_eligible}")

    # ── Step 3: C1b (if regression passed narrowly) ──────────────────────────
    if c1['all_pass']:
        print("\n[3] C1b Bayesian Ridge tie-breaker...")
        c1b = c1b_bayesian_ridge(X, y)
        results['C1b'] = c1b

    # ── Step 4: GP-UCB with C2 winner ────────────────────────────────────────
    beta   = BETA[fn]
    seed   = SEEDS[fn]
    bounds = BOUNDS[fn]
    m      = 15  # will be updated by convergence test

    print(f"\n[4] GP-UCB ({kernel}, beta={beta}, seed={seed})...")
    coord_c2, mu_c2, sig_c2, ucb_c2 = ucb_query(X, y, beta, seed, kernel, m, bounds)
    results['gp_c2'] = {'coord': coord_c2.tolist(), 'mu': round(mu_c2,4),
                          'sigma': round(sig_c2,4), 'ucb': round(ucb_c2,4)}
    print(f"    {coord_c2.round(4)} | mu={mu_c2:.4f}, sigma={sig_c2:.4f}")

    # ── Step 5: GP-UCB with RBF (audit comparison) ───────────────────────────
    print(f"\n[5] GP-UCB (RBF, audit comparison)...")
    coord_rbf, mu_rbf, sig_rbf, _ = ucb_query(X, y, beta, seed, 'RBF', m, bounds)
    dist_rbf = float(np.linalg.norm(coord_c2 - coord_rbf))
    results['gp_rbf_comparison'] = {'coord': coord_rbf.tolist(),
                                      'dist_from_c2': round(dist_rbf, 4)}
    print(f"    {coord_rbf.round(4)} | dist from C2: {dist_rbf:.4f}")

    # ── Step 6: C3 bootstrap ensemble ────────────────────────────────────────
    print(f"\n[6] C3 Bootstrap Ensemble (30 resamples)...")
    c3 = c3_bootstrap_ensemble(X, y, beta, seed, kernel, n_boots=30, m=m, bounds=bounds)
    dist_ens = float(np.linalg.norm(np.array(c3['best_coord']) - coord_c2))
    results['C3'] = c3
    results['C3']['dist_from_c2'] = round(dist_ens, 4)
    print(f"    {np.array(c3['best_coord']).round(4)} | dist from C2: {dist_ens:.4f}")

    # ── Step 7: A3 convergence test ──────────────────────────────────────────
    print(f"\n[7] A3 Candidate Pool Convergence...")
    a3 = a3_candidate_convergence(X, y, beta, seed, kernel, bounds)
    results['A3'] = a3
    m_final = 17 if not a3['converged_at_2_15'] else 15
    print(f"    Converged at 2^15: {a3['converged_at_2_15']} | Using 2^{m_final}")

    # Rerun GP-UCB at final pool size if escalated
    if m_final == 17:
        coord_final, mu_f, sig_f, _ = ucb_query(X, y, beta, seed, kernel, 17, bounds)
        print(f"    2^17 result: {coord_final.round(4)} | mu={mu_f:.4f}")
    else:
        coord_final = coord_c2

    # ── Step 8: A1 seed stability ─────────────────────────────────────────────
    print(f"\n[8] A1 Seed Stability (5 seeds)...")
    a1 = a1_seed_stability(X, y, seed, beta, kernel, m_final, bounds)
    results['A1'] = a1
    print(f"    Mean pairwise dist: {a1['mean_pairwise_dist']} | Flag: {a1['flag']}")

    # Apply seed rule
    if a1['flag'] in ('WARN >0.15', 'FLAG >0.25'):
        coord_seed_selected = np.array(a1['median_coord'])
        seed_note = f"Median rule applied (seed={a1['median_seed']})"
    else:
        coord_seed_selected = coord_final
        seed_note = "Primary seed used"
    print(f"    {seed_note}")

    # ── Step 9: A4 acquisition comparison ────────────────────────────────────
    print(f"\n[9] A4 Acquisition Comparison (UCB/EI/PI)...")
    a4 = a4_acquisition_comparison(X, y, beta, seed, kernel, m_final, bounds)
    results['A4'] = a4
    print(f"    UCB-PI dist: {a4['pairwise_distances']['UCB_vs_PI']}")

    # ── Step 10: F8 PI override ───────────────────────────────────────────────
    if fn == 8:
        print(f"\n[10] F8 PI 5-seed median (permanent switch)...")
        pi_coord, pi_score, all_pi = pi_query_5seed(X, y, F8_PI_SEEDS,
                                                     kernel, m_final, bounds)
        results['F8_PI'] = {'coord': pi_coord.tolist(), 'pi_score': round(pi_score, 4),
                              'all_pi': [round(p, 4) for p in all_pi]}
        coord_seed_selected = pi_coord
        seed_note = f"PI 5-seed median (seeds {F8_PI_SEEDS})"
        print(f"    {pi_coord.round(4)} | PI={pi_score:.4f}")

    # ── Final recommendation ──────────────────────────────────────────────────
    final = coord_seed_selected
    portal_str = '-'.join(f'{v:.6f}' for v in final)

    results['final_recommendation'] = {
        'coord': final.tolist(),
        'portal_string': portal_str,
        'kernel_used': kernel,
        'pool_size': f'2^{m_final}',
        'seed_note': seed_note,
        'regression_eligible': reg_eligible,
        'force_gp': fn in FORCE_GP,
        'method': 'PI_5seed' if fn == 8 else f'GP-UCB ({kernel})',
    }

    print(f"\nFINAL: {portal_str}")

    # Save to JSON
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"f{fn}_battery.json"
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    with open(out_path, 'w') as f_out:
        json.dump(results, f_out, indent=2, cls=NumpyEncoder)

    return results


def run_all_functions(initial_data_dir, inputs_txt, outputs_txt,
                      n_weeks, output_dir):
    """Run the full weekly battery for all 8 functions."""
    print(f"Loading data: {n_weeks} weeks")
    all_X, all_y = load_all(initial_data_dir, inputs_txt, outputs_txt, n_weeks)

    all_results = {}
    portal_strings = {}

    for fn in range(1, 9):
        res = run_weekly_battery(fn, all_X[fn], all_y[fn], output_dir)
        all_results[fn] = res
        portal_strings[fn] = res['final_recommendation']['portal_string']

    # Print summary
    print(f"\n{'='*60}")
    print("PORTAL SUBMISSION STRINGS")
    print(f"{'='*60}")
    for fn in range(1, 9):
        print(f"F{fn}: {portal_strings[fn]}")

    # Save summary
    summary_path = Path(output_dir) / "weekly_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({'portal_strings': portal_strings,
                   'n_weeks': n_weeks}, f, indent=2)

    return all_results, portal_strings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BBO Weekly Battery Runner')
    parser.add_argument('--initial_data', required=True,
                        help='Path to initial_data/ folder')
    parser.add_argument('--inputs_txt',   required=True,
                        help='Path to cumulative inputs.txt')
    parser.add_argument('--outputs_txt',  required=True,
                        help='Path to cumulative outputs.txt')
    parser.add_argument('--n_weeks',      type=int, required=True,
                        help='Number of completed weeks')
    parser.add_argument('--output_dir',   default='evidence/current_week/',
                        help='Where to write JSON results')
    args = parser.parse_args()

    run_all_functions(args.initial_data, args.inputs_txt,
                      args.outputs_txt, args.n_weeks, args.output_dir)
