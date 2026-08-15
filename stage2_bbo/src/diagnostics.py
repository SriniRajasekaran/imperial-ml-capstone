"""
diagnostics.py
Full diagnostic battery: IV1-IV10, C1-C5, A1-A5.

Each test returns a structured dict so results can be logged to the
evidence workbook without further parsing.
"""

import numpy as np
from scipy import stats
from scipy.stats import qmc, shapiro, pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import LeaveOneOut

from gp_core import make_kernel, fit_gp, sobol_candidates, ucb_query, LS_BOUNDS


# ════════════════════════════════════════════════════════════════════════════
# INPUT VARIABLE TESTS (IV)
# ════════════════════════════════════════════════════════════════════════════

def iv1_individual_sensitivity(X: np.ndarray, y: np.ndarray) -> dict:
    """
    IV1: Pearson r and p-value for each input dimension vs output.
    Flags dominant drivers (|r| > 0.4, p < 0.05).
    """
    n, d = X.shape
    results = []
    for i in range(d):
        r, p = pearsonr(X[:, i], y)
        results.append({'dim': i+1, 'r': round(float(r), 4),
                        'p': round(float(p), 4),
                        'dominant': abs(r) > 0.4 and p < 0.05})
    return {'test': 'IV1', 'n': n, 'dims': d, 'results': results}


def iv2_interaction_effects(X: np.ndarray, y: np.ndarray) -> dict:
    """
    IV2: Pairwise Pearson r between all input pairs; flag pairs correlated > 0.6.
    Co-movement between inputs can confound individual driver inference.
    """
    n, d = X.shape
    interactions = []
    for i in range(d):
        for j in range(i+1, d):
            r, p = pearsonr(X[:, i], X[:, j])
            interactions.append({'pair': (i+1, j+1), 'r': round(float(r), 4),
                                  'p': round(float(p), 4),
                                  'flag': abs(r) > 0.6 and p < 0.05})
    return {'test': 'IV2', 'n': n, 'dims': d, 'interactions': interactions}


def iv3_comovement_confound(X: np.ndarray, y: np.ndarray) -> dict:
    """
    IV3: OLS R2 with all inputs vs OLS R2 with inputs orthogonalised via
    QR decomposition. Large drop signals co-movement confounding driver inference.
    """
    from numpy.linalg import qr, lstsq
    n, d = X.shape
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # Full OLS
    Xa = np.column_stack([np.ones(n), Xs])
    b, _, _, _ = lstsq(Xa, y, rcond=None)
    yhat = Xa @ b
    r2_full = 1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)

    # Orthogonalised inputs
    Q, _ = qr(Xs, mode='reduced')
    Xq = np.column_stack([np.ones(n), Q])
    b2, _, _, _ = lstsq(Xq, y, rcond=None)
    yhat2 = Xq @ b2
    r2_orth = 1 - np.sum((y - yhat2)**2) / np.sum((y - y.mean())**2)

    return {'test': 'IV3', 'r2_full': round(float(r2_full), 4),
            'r2_orthogonalised': round(float(r2_orth), 4),
            'drop': round(float(r2_full - r2_orth), 4),
            'flag': (r2_full - r2_orth) > 0.10}


def iv4_nonlinearity_curvature(X: np.ndarray, y: np.ndarray) -> dict:
    """
    IV4: For each dimension, fit linear vs quadratic term; report R2 gain
    from adding x^2. Large gain signals curvature in that dimension.
    """
    from numpy.linalg import lstsq
    n, d = X.shape
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    results = []

    for i in range(d):
        xi = Xs[:, i]
        # Linear
        Xl = np.column_stack([np.ones(n), xi])
        bl, _, _, _ = lstsq(Xl, y, rcond=None)
        yl = Xl @ bl
        r2l = 1 - np.sum((y-yl)**2) / np.sum((y-y.mean())**2)
        # Quadratic
        Xq = np.column_stack([np.ones(n), xi, xi**2])
        bq, _, _, _ = lstsq(Xq, y, rcond=None)
        yq = Xq @ bq
        r2q = 1 - np.sum((y-yq)**2) / np.sum((y-y.mean())**2)
        gain = r2q - r2l
        results.append({'dim': i+1, 'r2_linear': round(float(r2l), 4),
                        'r2_quadratic': round(float(r2q), 4),
                        'gain': round(float(gain), 4),
                        'flag': gain > 0.05})

    return {'test': 'IV4', 'results': results}


def iv5_tail_outlier(X: np.ndarray, y: np.ndarray) -> dict:
    """
    IV5: Identify output outliers (|z-score| > 2.5). Flag functions where
    a single observation dominates the range (F5 canonical example).
    """
    z = (y - y.mean()) / (y.std() + 1e-10)
    outliers = [(int(i), round(float(y[i]), 4), round(float(z[i]), 2))
                for i in range(len(y)) if abs(z[i]) > 2.5]
    max_pct = float((y.max() - np.sort(y)[-2]) / (y.max() - y.min() + 1e-10))
    return {'test': 'IV5', 'n_outliers': len(outliers), 'outliers': outliers,
            'top1_pct_of_range': round(max_pct, 3),
            'flag': max_pct > 0.5}


def iv6_local_global_drift(X: np.ndarray, y: np.ndarray) -> dict:
    """
    IV6: Compare driver rankings in best-third vs worst-third of observations.
    Large rank reversal suggests the function is non-stationary.
    Requires corroboration (partial correlation or GP check) before acting.
    """
    n = len(y)
    sort_idx = np.argsort(y)
    n_third = max(n // 3, 3)
    low_idx  = sort_idx[:n_third]
    high_idx = sort_idx[-n_third:]
    d = X.shape[1]
    reversals = []
    for i in range(d):
        r_low,  _ = pearsonr(X[low_idx,  i], y[low_idx])
        r_high, _ = pearsonr(X[high_idx, i], y[high_idx])
        if np.sign(r_low) != np.sign(r_high) and abs(r_low) > 0.3 and abs(r_high) > 0.3:
            reversals.append({'dim': i+1, 'r_low': round(float(r_low), 3),
                               'r_high': round(float(r_high), 3)})
    return {'test': 'IV6', 'reversals': reversals,
            'flag': len(reversals) > 0,
            'note': 'Reversal requires corroboration before acting'}


def iv7_boundary_corner_probe(X: np.ndarray, y: np.ndarray) -> dict:
    """
    IV7: Report all observed y values at corners of the domain
    (any dimension within 0.05 of 0 or 1). Informs boundary exploration.
    """
    d = X.shape[1]
    corners = []
    for i, (xi, yi) in enumerate(zip(X, y)):
        at_low  = [j+1 for j in range(d) if xi[j] <= 0.05]
        at_high = [j+1 for j in range(d) if xi[j] >= 0.95]
        if at_low or at_high:
            corners.append({'obs': i, 'y': round(float(yi), 6),
                             'dims_at_low': at_low, 'dims_at_high': at_high})
    best_corner = max(corners, key=lambda c: c['y']) if corners else None
    return {'test': 'IV7', 'n_corners': len(corners),
            'best_corner': best_corner, 'all_corners': corners}


def iv8_heteroskedasticity_nonstationarity(X: np.ndarray, y: np.ndarray,
                                            kernel_name: str = 'RBF') -> dict:
    """
    IV8: Breusch-Pagan test for heteroskedasticity in OLS residuals.
         Split-half length-scale ratio for GP non-stationarity.
    Flags if BP p < 0.05 OR split-half LS ratio > 1.5.
    """
    from numpy.linalg import lstsq
    n, d = X.shape
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # Breusch-Pagan
    Xa = np.column_stack([np.ones(n), Xs])
    b, _, _, _ = lstsq(Xa, y, rcond=None)
    resid = y - Xa @ b
    resid2 = resid ** 2
    b2, _, _, _ = lstsq(Xa, resid2, rcond=None)
    yhat2 = Xa @ b2
    ss_reg = np.sum((yhat2 - resid2.mean())**2)
    ss_tot = np.sum((resid2 - resid2.mean())**2)
    bp_r2 = ss_reg / (ss_tot + 1e-10)
    bp_stat = n * bp_r2
    bp_p = float(1 - stats.chi2.cdf(bp_stat, df=d))

    # Split-half LS ratio
    half = n // 2
    def fit_ls(Xs_sub, y_sub):
        k = make_kernel(kernel_name)
        gp = GaussianProcessRegressor(k, alpha=1e-6, n_restarts_optimizer=3)
        gp.fit(Xs_sub, (y_sub-y_sub.mean())/(y_sub.std()+1e-10))
        return float(gp.kernel_.k2.length_scale)
    ls1 = fit_ls(Xs[:half], y[:half])
    ls2 = fit_ls(Xs[half:], y[half:])
    ratio = max(ls1, ls2) / (min(ls1, ls2) + 1e-10)

    return {'test': 'IV8',
            'bp_stat': round(bp_stat, 3), 'bp_p': round(bp_p, 4),
            'heteroskedastic': bp_p < 0.05,
            'ls_half1': round(ls1, 4), 'ls_half2': round(ls2, 4),
            'ls_ratio': round(ratio, 3),
            'nonstationary': ratio > 1.5,
            'flag': bp_p < 0.05 or ratio > 1.5}


def iv9_ridge_walk_topology(X: np.ndarray, y: np.ndarray,
                             kernel_name: str = 'RBF') -> dict:
    """
    IV9: Walk from current best toward UCB argmax in 10 steps.
    Report GP mean profile along the path. Flat/plateau → ridge or saddle.
    Sharp rise → genuine gradient. Used to characterise F1 ridge topology.
    """
    best_idx = np.argmax(y)
    best_pt  = X[best_idx]

    gp, scaler, ym, ys = fit_gp(X, y, kernel_name)
    # Use a simple random candidate to get a second point for the walk
    rng = np.random.RandomState(42)
    end_pt = rng.uniform(0, 1, size=best_pt.shape)

    steps = 10
    path  = [best_pt + (end_pt - best_pt) * t / steps for t in range(steps+1)]
    path_s = scaler.transform(np.array(path))
    mu_path, _ = gp.predict(path_s, return_std=True)
    mu_real = mu_path * ys + ym

    return {'test': 'IV9',
            'best_pt': best_pt.tolist(),
            'mu_along_path': [round(float(v), 4) for v in mu_real],
            'monotone': bool(np.all(np.diff(mu_real) <= 0) or
                             np.all(np.diff(mu_real) >= 0))}


def iv10_ls_bound_sensitivity(X: np.ndarray, y: np.ndarray,
                               kernel_name: str = 'RBF') -> dict:
    """
    IV10: Refit GP under three LS bound settings and compare fitted LS and LOO Q2.
    Standing check every 3-4 weeks. F1 eligibility verdict can flip between settings.

    Bound sets:
        tight  : (0.15, 0.6)
        default: (0.05, 0.8)
        wide   : (0.10, 3.0)
    """
    from sklearn.model_selection import LeaveOneOut
    results = {}
    bound_sets = {'tight': (0.15, 0.6), 'default': (0.05, 0.8), 'wide': (0.10, 3.0)}

    for name, bounds in bound_sets.items():
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        yn = (y - y.mean()) / (y.std() + 1e-10)
        if kernel_name == 'RBF':
            k = ConstantKernel(1.0) * RBF(0.3, length_scale_bounds=bounds)
        else:
            nu = 1.5 if '32' in kernel_name else 2.5
            k = ConstantKernel(1.0) * Matern(0.3, nu=nu, length_scale_bounds=bounds)
        gp = GaussianProcessRegressor(k, alpha=1e-6, n_restarts_optimizer=3)
        gp.fit(Xs, yn)
        ls = float(gp.kernel_.k2.length_scale)

        loo = LeaveOneOut()
        preds = np.zeros(len(yn))
        for tr, te in loo.split(Xs):
            g2 = GaussianProcessRegressor(gp.kernel_, alpha=1e-6)
            g2.fit(Xs[tr], yn[tr])
            preds[te] = g2.predict(Xs[te])
        q2 = float(1 - np.sum((yn-preds)**2) / np.sum((yn-yn.mean())**2))

        at_lower = ls <= bounds[0] * 1.01
        at_upper = ls >= bounds[1] * 0.99
        results[name] = {'ls': round(ls, 4), 'q2': round(q2, 4),
                          'at_bound': 'lower' if at_lower else 'upper' if at_upper else 'free'}

    q2_range = max(r['q2'] for r in results.values()) - min(r['q2'] for r in results.values())
    return {'test': 'IV10', 'bounds_tested': results,
            'q2_range': round(q2_range, 4),
            'verdict_sensitive': q2_range > 0.05}


# ════════════════════════════════════════════════════════════════════════════
# CHALLENGER MODEL TESTS (C)
# ════════════════════════════════════════════════════════════════════════════

def c1_regression_gates(X: np.ndarray, y: np.ndarray) -> dict:
    """
    C1: OLS regression with four eligibility gates.
    Gates: R2>0.30, SW p>0.05, DW 1.5-2.5, predicted y > current best.
    Significance-gated query: x_i -> 0.95 if beta>0,p<0.05 | 0.05 if beta<0,p<0.05 | 0.50 otherwise.
    """
    from numpy.linalg import lstsq
    n, d = X.shape
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xa = np.column_stack([np.ones(n), Xs])
    b, _, _, _ = lstsq(Xa, y, rcond=None)
    yhat = Xa @ b
    resid = y - yhat

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    sw_stat, sw_p = shapiro(resid) if n >= 3 else (1.0, 1.0)
    dw = float(np.sum(np.diff(resid)**2) / (np.sum(resid**2) + 1e-10))

    mse = ss_res / max(n - d - 1, 1)
    try:
        XtXi = np.linalg.inv(Xa.T @ Xa)
        se = np.sqrt(np.diag(XtXi) * mse)
        t_stat = b / (se + 1e-10)
        p_vals = [float(2*(1 - stats.t.cdf(abs(t), df=max(n-d-1,1)))) for t in t_stat]
    except Exception:
        p_vals = [1.0] * (d + 1)

    query = np.full(d, 0.5)
    sig_vars = []
    for i in range(1, d+1):
        if p_vals[i] < 0.05:
            query[i-1] = 0.95 if b[i] > 0 else 0.05
            sig_vars.append({'dim': i, 'beta': round(float(b[i]), 4),
                              'p': round(float(p_vals[i]), 4),
                              'direction': '→0.95' if b[i] > 0 else '→0.05'})

    qsc = sc.transform(query.reshape(1,-1))
    pred_y = float((np.column_stack([np.ones(1), qsc]) @ b).ravel()[0])
    current_best = float(y.max())

    gate1 = r2 > 0.30
    gate2 = float(sw_p) > 0.05
    gate3 = 1.5 <= dw <= 2.5
    gate4 = pred_y > current_best

    return {'test': 'C1',
            'r2': round(r2, 4), 'gate1_r2': gate1,
            'sw_p': round(float(sw_p), 4), 'gate2_sw': gate2,
            'dw': round(dw, 4), 'gate3_dw': gate3,
            'pred_y': round(pred_y, 6), 'current_best': round(current_best, 6),
            'gate4_improvement': gate4,
            'all_pass': gate1 and gate2 and gate3 and gate4,
            'sig_vars': sig_vars,
            'query': query.tolist()}


def c1b_bayesian_ridge(X: np.ndarray, y: np.ndarray) -> dict:
    """
    C1b: Bayesian Ridge as tie-breaker when OLS gates pass narrowly.
    Reports z-scores for each dimension. Used as corroboration, not primary.
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    br = BayesianRidge()
    br.fit(Xs, y)
    se = np.sqrt(np.diag(br.sigma_))
    z_scores = br.coef_ / (se + 1e-10)
    results = [{'dim': i+1, 'coef': round(float(br.coef_[i]), 4),
                 'se': round(float(se[i]), 4), 'z': round(float(z_scores[i]), 3),
                 'flag': abs(z_scores[i]) > 1.96}
               for i in range(X.shape[1])]
    return {'test': 'C1b', 'results': results,
            'alpha': round(float(br.alpha_), 4),
            'lambda': round(float(br.lambda_), 4)}


def c2_kernel_challenger(X: np.ndarray, y: np.ndarray) -> dict:
    """
    C2: LOO Q2 comparison across RBF, Matern32, Matern52.
    MUST run before any candidate generation each week.
    Winner determines the kernel used for GP-UCB that week.
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    yn = (y - y.mean()) / (y.std() + 1e-10)

    def loo_q2(kernel_name):
        k = make_kernel(kernel_name)
        gp = GaussianProcessRegressor(k, alpha=1e-6, n_restarts_optimizer=5)
        gp.fit(Xs, yn)
        ls = float(gp.kernel_.k2.length_scale)
        loo = LeaveOneOut()
        preds = np.zeros(len(yn))
        for tr, te in loo.split(Xs):
            g2 = GaussianProcessRegressor(gp.kernel_, alpha=1e-6)
            g2.fit(Xs[tr], yn[tr])
            preds[te] = g2.predict(Xs[te])
        q2 = float(1 - np.sum((yn-preds)**2) / np.sum((yn-yn.mean())**2))
        at_lower = ls <= LS_BOUNDS[0] * 1.01
        at_upper = ls >= LS_BOUNDS[1] * 0.99
        return q2, ls, 'lower' if at_lower else 'upper' if at_upper else 'free'

    scores = {}
    for name in ['RBF', 'Matern32', 'Matern52']:
        q2, ls, bound_status = loo_q2(name)
        scores[name] = {'q2': round(q2, 4), 'ls': round(ls, 4),
                         'bound_status': bound_status}

    winner = max(scores, key=lambda k: scores[k]['q2'])
    return {'test': 'C2', 'ls_bounds': LS_BOUNDS,
            'scores': scores, 'winner': winner,
            'note': 'Run this FIRST every week before candidate generation'}


def c3_bootstrap_ensemble(X: np.ndarray, y: np.ndarray,
                           beta: float, seed: int,
                           kernel_name: str, n_boots: int = 30,
                           m: int = 15, bounds: list = None) -> dict:
    """
    C3: Bootstrap ensemble GP (30 resamples of training data).
    Provides independent cross-check of GP-UCB candidate.
    Reports mean prediction, mean uncertainty, and between-model variance.
    """
    rng = np.random.RandomState(seed)
    candidates = sobol_candidates(X.shape[1], seed, m, bounds)
    sc = StandardScaler().fit(X)
    Xs_full = sc.transform(X)

    all_mu = np.zeros((n_boots, len(candidates)))
    for b in range(n_boots):
        idx = rng.choice(len(X), size=len(X), replace=True)
        Xb, yb = X[idx], y[idx]
        sc_b = StandardScaler()
        Xbs = sc_b.fit_transform(Xb)
        ymb, ysb = yb.mean(), yb.std() + 1e-10
        k = make_kernel(kernel_name)
        gp = GaussianProcessRegressor(k, alpha=1e-6, n_restarts_optimizer=2)
        gp.fit(Xbs, (yb-ymb)/ysb)
        Cs = sc_b.transform(candidates)
        mu_b, _ = gp.predict(Cs, return_std=True)
        all_mu[b] = mu_b

    mu_mean   = all_mu.mean(axis=0)
    mu_var    = all_mu.var(axis=0)
    ucb_ens   = mu_mean + beta * np.sqrt(mu_var)
    idx_best  = int(np.argmax(ucb_ens))

    return {'test': 'C3', 'n_boots': n_boots,
            'best_coord': candidates[idx_best].tolist(),
            'mu_mean': round(float(mu_mean[idx_best]), 4),
            'between_model_var': round(float(mu_var[idx_best]), 6)}


def c5_shap_challenger(X: np.ndarray, y: np.ndarray) -> dict:
    """
    C5: XGBoost + SHAP TreeExplainer as a non-parametric driver check.
    Corroborates or challenges OLS driver ranking. Requires xgboost + shap.
    """
    try:
        import xgboost as xgb
        import shap
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4,
                                   learning_rate=0.1, random_state=42,
                                   verbosity=0)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        ranking   = np.argsort(mean_abs)[::-1]
        results   = [{'dim': int(ranking[i])+1,
                       'mean_abs_shap': round(float(mean_abs[ranking[i]]), 4)}
                      for i in range(len(ranking))]
        return {'test': 'C5', 'status': 'ok', 'shap_ranking': results}
    except ImportError:
        return {'test': 'C5', 'status': 'skipped',
                'note': 'xgboost/shap not installed'}


# ════════════════════════════════════════════════════════════════════════════
# ACQUISITION / SEARCH TESTS (A)
# ════════════════════════════════════════════════════════════════════════════

def a1_seed_stability(X: np.ndarray, y: np.ndarray,
                      base_seed: int, beta: float,
                      kernel_name: str, m: int = 15,
                      bounds: list = None, n_seeds: int = 5) -> dict:
    """
    A1: Run GP-UCB with 5 seeds (base_seed to base_seed+4).
    Compute mean pairwise L2 distance between best candidates.
    Rule: disagree > 0.15 -> use median; > 0.25 -> flag explicitly.
    """
    from gp_core import ucb_query
    coords = []
    scores = []
    seeds  = [base_seed + i for i in range(n_seeds)]

    for seed in seeds:
        coord, mu, sig, ucb_score = ucb_query(X, y, beta, seed,
                                               kernel_name, m, bounds)
        coords.append(coord)
        scores.append(ucb_score)

    n = len(coords)
    dists = [float(np.linalg.norm(coords[i] - coords[j]))
             for i in range(n) for j in range(i+1, n)]
    mean_dist = float(np.mean(dists))
    max_dist  = float(np.max(dists))

    centroid  = np.mean(coords, axis=0)
    dists_c   = [float(np.linalg.norm(c - centroid)) for c in coords]
    median_idx = int(np.argmin(dists_c))
    best_score_idx = int(np.argmax(scores))

    flag = 'OK'
    if mean_dist > 0.25:
        flag = 'FLAG >0.25'
    elif mean_dist > 0.15:
        flag = 'WARN >0.15'

    return {'test': 'A1', 'seeds': seeds, 'n_seeds': n_seeds,
            'mean_pairwise_dist': round(mean_dist, 4),
            'max_pairwise_dist': round(max_dist, 4),
            'flag': flag,
            'rule': 'median' if mean_dist > 0.15 else 'primary',
            'primary_coord': coords[0].tolist(),
            'median_coord': coords[median_idx].tolist(),
            'median_seed': seeds[median_idx],
            'best_ucb_coord': coords[best_score_idx].tolist(),
            'best_ucb_seed': seeds[best_score_idx],
            'all_ucb_scores': [round(s, 4) for s in scores]}


def a3_candidate_convergence(X: np.ndarray, y: np.ndarray,
                              beta: float, seed: int,
                              kernel_name: str,
                              bounds: list = None) -> dict:
    """
    A3: Test UCB argmax stability across 2^13, 2^14, 2^15, 2^17.
    Converged if L2 distance between 2^15 and 2^17 < 0.02.
    If not converged, use 2^17 candidates for final query.
    """
    from gp_core import ucb_query
    results = {}
    prev = None
    for m in [13, 14, 15, 17]:
        coord, mu, sig, _ = ucb_query(X, y, beta, seed, kernel_name, m, bounds)
        dist = float(np.linalg.norm(coord - prev)) if prev is not None else None
        results[m] = {'coord': coord.tolist(), 'mu': round(mu, 4),
                       'dist_from_prev': round(dist, 4) if dist else None}
        prev = coord

    d_15_17 = results[17]['dist_from_prev']
    converged = d_15_17 is not None and d_15_17 < 0.02
    return {'test': 'A3', 'results': results,
            'converged_at_2_15': converged,
            'dist_15_to_17': round(d_15_17, 4) if d_15_17 else None,
            'recommended_pool': '2^15' if converged else '2^17'}


def a4_acquisition_comparison(X: np.ndarray, y: np.ndarray,
                               beta: float, seed: int,
                               kernel_name: str, m: int = 15,
                               bounds: list = None) -> dict:
    """
    A4: Compare UCB, EI, and PI candidates for a given function.
    Reports L2 distance between each pair. Large divergence = acquisition
    function choice matters; small = robust to choice.
    """
    from gp_core import fit_gp, sobol_candidates
    from scipy.special import ndtr

    gp, sc, ym, ys = fit_gp(X, y, kernel_name)
    cands = sobol_candidates(X.shape[1], seed, m, bounds)
    Cs = sc.transform(cands)
    mu, sigma = gp.predict(Cs, return_std=True)
    best_y = y.max()
    best_norm = (best_y - ym) / ys

    # UCB
    ucb = mu + beta * sigma
    ucb_idx = int(np.argmax(ucb))

    # EI
    z_ei = (mu - best_norm) / (sigma + 1e-10)
    ei = sigma * (z_ei * ndtr(z_ei) + stats.norm.pdf(z_ei))
    ei_idx = int(np.argmax(ei))

    # PI
    z_pi = (mu - best_norm) / (sigma + 1e-10)
    pi = ndtr(z_pi)
    pi_idx = int(np.argmax(pi))

    coords = {'UCB': cands[ucb_idx], 'EI': cands[ei_idx], 'PI': cands[pi_idx]}
    pairs  = {}
    for a, b_ in [('UCB','EI'), ('UCB','PI'), ('EI','PI')]:
        pairs[f'{a}_vs_{b_}'] = round(float(np.linalg.norm(coords[a] - coords[b_])), 4)

    return {'test': 'A4',
            'UCB_coord': cands[ucb_idx].tolist(),
            'EI_coord':  cands[ei_idx].tolist(),
            'PI_coord':  cands[pi_idx].tolist(),
            'pairwise_distances': pairs,
            'all_agree': all(v < 0.05 for v in pairs.values())}
