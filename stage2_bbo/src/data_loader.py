"""
data_loader.py
Load initial data and cumulative weekly submissions for all 8 BBO functions.

CRITICAL: Always load initial data PLUS all weekly submissions together.
Never load weekly submissions in isolation — GP quality assessment is
fundamentally corrupted without the full combined dataset.
"""

import numpy as np
import re
from pathlib import Path


def load_initial_data(data_dir: str) -> tuple[dict, dict]:
    """
    Load Imperial-provided initial observations for all 8 functions.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing function_1/ ... function_8/ subfolders,
        each with initial_inputs.npy and initial_outputs.npy.

    Returns
    -------
    init_X : dict  {fn_index (1-8): np.ndarray shape (n, dims)}
    init_y : dict  {fn_index (1-8): np.ndarray shape (n,)}
    """
    init_X, init_y = {}, {}
    for fn in range(1, 9):
        fn_dir = Path(data_dir) / f"function_{fn}"
        init_X[fn] = np.load(fn_dir / "initial_inputs.npy")
        init_y[fn] = np.load(fn_dir / "initial_outputs.npy")
    return init_X, init_y


def load_weekly_submissions(inputs_txt: str, outputs_txt: str,
                            n_weeks: int) -> tuple[dict, dict]:
    """
    Parse cumulative inputs.txt and outputs.txt from the Imperial portal zip.

    The portal returns cumulative files: inputs.txt contains all submissions
    to date as array([...]) blocks, ordered by week then by function (8 per week).
    outputs.txt contains one row per week, 8 values per row.

    Parameters
    ----------
    inputs_txt : str   Path to inputs.txt (cumulative)
    outputs_txt : str  Path to outputs.txt (cumulative)
    n_weeks : int      Number of weeks to read (rows in outputs.txt)

    Returns
    -------
    weekly_X : dict  {fn_index: list of np.ndarray, one per week}
    weekly_y : dict  {fn_index: list of float, one per week}
    """
    # Parse inputs
    with open(inputs_txt) as f:
        content = f.read()
    arrays_raw = re.findall(r'array\(\[(.*?)\]\)', content, re.DOTALL)

    weekly_X = {fn: [] for fn in range(1, 9)}
    for week_idx in range(n_weeks):
        for fn_idx in range(8):
            arr = arrays_raw[week_idx * 8 + fn_idx]
            vals = [float(v.strip()) for v in arr.split(',') if v.strip()]
            weekly_X[fn_idx + 1].append(np.array(vals))

    # Parse outputs
    with open(outputs_txt) as f:
        lines = f.read().strip().split('\n')

    weekly_y = {fn: [] for fn in range(1, 9)}
    for line in lines[:n_weeks]:
        clean = re.findall(r'np\.float64\((.*?)\)', line)
        if len(clean) == 8:
            vals = [float(v) for v in clean]
        else:
            vals = [float(v) for v in
                    re.findall(r'[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?', line)][:8]
        for fn_idx, v in enumerate(vals[:8]):
            weekly_y[fn_idx + 1].append(v)

    return weekly_X, weekly_y


def build_combined_dataset(init_X: dict, init_y: dict,
                           weekly_X: dict, weekly_y: dict) -> tuple[dict, dict]:
    """
    Combine initial observations with all weekly submissions.

    Parameters
    ----------
    init_X, init_y     : from load_initial_data()
    weekly_X, weekly_y : from load_weekly_submissions()

    Returns
    -------
    all_X : dict  {fn: np.ndarray shape (n_total, dims)}
    all_y : dict  {fn: np.ndarray shape (n_total,)}
    """
    all_X, all_y = {}, {}
    for fn in range(1, 9):
        wX = np.vstack(weekly_X[fn]) if weekly_X[fn] else np.empty((0, init_X[fn].shape[1]))
        wy = np.array(weekly_y[fn])  if weekly_y[fn] else np.array([])
        all_X[fn] = np.vstack([init_X[fn], wX])
        all_y[fn] = np.concatenate([init_y[fn], wy])
    return all_X, all_y


def load_all(initial_data_dir: str, inputs_txt: str,
             outputs_txt: str, n_weeks: int) -> tuple[dict, dict]:
    """
    Convenience wrapper: load initial + weekly, return combined dataset.

    Usage
    -----
    all_X, all_y = load_all(
        initial_data_dir="data/initial_data",
        inputs_txt="data/week_10/inputs.txt",
        outputs_txt="data/week_10/outputs.txt",
        n_weeks=10
    )
    """
    init_X, init_y = load_initial_data(initial_data_dir)
    wX, wy = load_weekly_submissions(inputs_txt, outputs_txt, n_weeks)
    return build_combined_dataset(init_X, init_y, wX, wy)


# ── Colab / notebook convenience wrapper ─────────────────────────────────────

def parse_outputs_line(line: str) -> list:
    """
    Parse one line of outputs.txt from the Imperial portal.

    The portal uses Python repr format: [np.float64(0.123), np.float64(-4.5e-12), ...]
    This is NOT valid JSON. Never use json.loads() on portal files.

    This function handles both formats seen across the campaign:
      - Early weeks: [np.float64(...), np.float64(...), ...]
      - Later weeks: [0.021564, 0.543253, ...]  (bare floats, still not JSON)

    Parameters
    ----------
    line : str   One row from outputs.txt

    Returns
    -------
    values : list of float, length 8 (one per function)
    """
    import re

    # Try np.float64(...) pattern first
    np_matches = re.findall(r'np\.float64\((.*?)\)', line)
    if len(np_matches) == 8:
        return [float(v) for v in np_matches]

    # Fallback: bare float/int pattern
    bare = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', line)
    values = [float(v) for v in bare]
    if len(values) >= 8:
        return values[:8]

    raise ValueError(
        f"Could not parse 8 values from outputs.txt line.\n"
        f"Line: {line[:100]}\n"
        f"NOTE: Portal outputs.txt is Python repr, NOT JSON. "
        f"Never use json.loads() on portal files."
    )
