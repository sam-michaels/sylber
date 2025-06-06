# tuner.py
# ----------------------------------------------------------------------
# Grid / random search utilities for segmentation algorithms in cutters.py
# ----------------------------------------------------------------------

import itertools
import importlib
import inspect
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from evaluation_helpers import run_algo, frames_to_times, eval_boundaries
from params import SEARCH_SPACE   # your per-algorithm search-space dict


# ----------------------------------------------------------------------
# Helper: enumerate Cartesian product of a parameter grid
# ----------------------------------------------------------------------
def all_param_combos(space: Dict[str, List]):
    """
    Yield every possible combination in a (possibly nested) search-space
    dictionary as a plain dict that can be expanded with **kwargs.
    """
    if not space:                         # empty dict → one empty combo
        yield {}
        return

    keys, values = zip(*space.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ----------------------------------------------------------------------
# Core: evaluate a single algorithm over its parameter grid
# ----------------------------------------------------------------------
def evaluate_algo(
    name: str,
    fn,                       # the segmentation function from cutters.py
    param_grid: Dict[str, List],
    clips,                    # list of (wav_path, ref_times, wav, states)
    tol: float = 0.02,        # boundary-match tolerance in seconds
):
    """
    Run `fn` for every parameter combination on every clip and return a
    DataFrame sorted by mean F-score (higher is better).

    Each row contains:
        algo, <hyper-parameters>, f_mean, f_std, t_mean, mem_mean
    """
    rows = []

    for p in all_param_combos(param_grid):
        f_scores, times, mems = [], [], []

        for wav_path, ref_times, wav, states in clips:
            # Measure runtime & peak memory while running the algorithm
            segs, t, m = run_algo(lambda st: fn(st, **p), states)

            # Convert frame indices → seconds
            est_times = frames_to_times(segs)

            # Boundary F-score (if ground truth available)
            if ref_times is not None:
                _, _, f, *_ = eval_boundaries(ref_times, est_times)
            else:
                f = np.nan

            f_scores.append(f)
            times.append(t)
            mems.append(m)

        rows.append(
            dict(
                algo=name, **p,
                f_mean=np.nanmean(f_scores),
                f_std=np.nanstd(f_scores),
                t_mean=np.mean(times),
                mem_mean=np.mean(mems),
            )
        )

    return pd.DataFrame(rows).sort_values("f_mean", ascending=False)


# ----------------------------------------------------------------------
# Utility: discover all tunable functions in cutters.py
# ----------------------------------------------------------------------
def discover_cutters(cutters_module):
    """
    Return {name: function} for every function in cutters whose first
    positional argument is `states` and whose name appears in SEARCH_SPACE.
    """
    cutters_module = importlib.reload(cutters_module)

    return {
        name: fn
        for name, fn in inspect.getmembers(cutters_module, inspect.isfunction)
        if fn.__code__.co_varnames[:1] == ("states",) and name in SEARCH_SPACE
    }