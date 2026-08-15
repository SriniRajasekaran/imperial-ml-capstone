"""
gp_core.py
Gaussian Process fitting, UCB and PI acquisition with Sobol candidate sampling.

Kernel assignments (C2 Week 11, confirmed on full dataset):
    F1: RBF       F2: Matern52   F3: Matern32   F4: Matern32
    F5: Matern32  F6: Matern32   F7: Matern32   F8: Matern32

Seeds (standing scheme: fn_index * 100):
    F1=100 F2=200 F3=300 F4=400 F5=500 F6=600 F7=700
    F8 PI seeds: 800, 801, 802, 803, 804
"""

import numpy as np
from scipy.stats import qmc
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler

LS_BOUNDS = (0.05, 0.8)

# Standing kernel assignments from C2 challenger (update after each weekly C2 run)
C2_KERNELS = {
    1: 'RBF', 2: 'Matern52', 3: 'Matern32', 4: 'Matern32',
    5: 'Matern32', 6: 'Matern32', 7: 'Matern32', 8: 'Matern32',
}

# FORCE_GP: use GP regardless of regression gate outcome
FORCE_GP = {3, 4, 7, 8}

# Seeds: fn_index * 100
SEEDS = {fn: fn * 100 for fn in range(1, 9)}
F8_PI_SEEDS = [800, 801, 802, 803, 804]


def make_kernel(kernel_name: str) -> object:
    """Return a scikit-learn kernel object by name."""
    ls = dict(length_scale=0.3, length_scale_bounds=LS_BOUNDS)
    if kernel_name == 'RBF':
        return ConstantKernel(1.0) * RBF(**ls)
    elif kernel_name == 'Matern32':
        return ConstantKernel(1.0) * Matern(nu=1.5, **ls)
    elif kernel_name == 'Matern52':
        return ConstantKernel(1.0) * Matern(nu=2.5, **ls)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


def fit_gp(X: np.ndarray, y: np.ndarray,
           kernel_name: str) -> tuple:
    """
    Fit a GP surrogate with normalised outputs.

    Returns
    -------
    gp      : fitted GaussianProcessRegressor
    scaler  : fitted StandardScaler (for inputs)
    y_mean  : float  (output mean, for denormalisation)
    y_std   : float  (output std,  for denormalisation)
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    y_mean, y_std = y.mean(), y.std() + 1e-10
    y_norm = (y - y_mean) / y_std

    kernel = make_kernel(kernel_name)
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6,
        n_restarts_optimizer=5, normalize_y=False
    )
    gp.fit(Xs, y_norm)
    return gp, scaler, y_mean, y_std


def sobol_candidates(dims: int, seed: int, m: int = 15,
                     bounds: list = None) -> np.ndarray:
    """
    Generate 2^m Sobol candidates in [0,1]^dims, optionally rescaled to bounds.

    Parameters
    ----------
    dims   : int
    seed   : int   scramble seed
    m      : int   pool size = 2^m (default 15 -> 32768; use 17 -> 131072 if not converged)
    bounds : list of (lb, ub) tuples per dimension, or None for [0,1]^dims
    """
    sampler = qmc.Sobol(d=dims, scramble=True, seed=seed)
    pts = sampler.random_base2(m=m)
    if bounds is not None:
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        pts = lb + pts * (ub - lb)
    return pts


def ucb_query(X: np.ndarray, y: np.ndarray,
              beta: float, seed: int, kernel_name: str,
              m: int = 15, bounds: list = None) -> tuple:
    """
    GP-UCB acquisition: return argmax of mu + beta*sigma over Sobol candidates.

    Returns
    -------
    best_coord  : np.ndarray  query point
    mu_best     : float       GP mean at query point (normalised scale)
    sigma_best  : float       GP std at query point
    ucb_best    : float       UCB score at query point
    """
    gp, scaler, ym, ys = fit_gp(X, y, kernel_name)
    candidates = sobol_candidates(X.shape[1], seed, m, bounds)
    Cs = scaler.transform(candidates)
    mu, sigma = gp.predict(Cs, return_std=True)
    ucb = mu + beta * sigma
    idx = np.argmax(ucb)
    return candidates[idx], float(mu[idx]), float(sigma[idx]), float(ucb[idx])


def pi_query_5seed(X: np.ndarray, y: np.ndarray,
                   seeds: list, kernel_name: str,
                   m: int = 15, bounds: list = None) -> tuple:
    """
    PI acquisition with 5-seed median selection (standing rule for F8).

    For each seed, find the candidate maximising P(f(x) > f_best).
    Select the candidate corresponding to the median PI score across seeds.

    Returns
    -------
    coord       : np.ndarray  selected query point
    pi_score    : float       PI score of selected point
    all_pi      : list        PI scores for all seeds (for seed disagreement check)
    """
    best_y = y.max()
    all_coords, all_pi_scores = [], []

    for seed in seeds:
        gp, scaler, ym, ys = fit_gp(X, y, kernel_name)
        candidates = sobol_candidates(X.shape[1], seed, m, bounds)
        Cs = scaler.transform(candidates)
        mu, sigma = gp.predict(Cs, return_std=True)
        z = (mu - (best_y - ym) / ys) / (sigma + 1e-10)
        pi = ndtr(z)
        idx = np.argmax(pi)
        all_coords.append(candidates[idx])
        all_pi_scores.append(float(pi[idx]))

    median_idx = np.argsort(all_pi_scores)[len(all_pi_scores) // 2]
    return all_coords[median_idx], all_pi_scores[median_idx], all_pi_scores
