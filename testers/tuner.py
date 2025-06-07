# testers/tuner.py
# ----------------------------------------------------------------------
# Grid‐search utilities for segmentation algorithms in testers/cutters.py
# ----------------------------------------------------------------------

import itertools
import importlib
import inspect
import warnings
from typing import Dict, List, Tuple
from pathlib import Path
import time

import numpy as np
import pandas as pd

# Helpers for evaluating & converting frames → times + comparing to ground truth
from testers.evaluation_helpers import run_algo, frames_to_times, eval_boundaries

# The “full” SEARCH_SPACE; in a notebook you’ll get it via params.get_space(...)
from testers.params import SEARCH_SPACE

# We need this module object in order to “discover” its functions
import testers.cutters as cutters_module


def all_param_combos(space: Dict[str, List]):
    """
    Yield every possible combination in a search‐space dictionary as a plain dict.
    Example:
        space = {"ker_win": [4, 8], "peak_thr": [1.0, 1.5]}
        => yields {"ker_win":4,"peak_thr":1.0}, {"ker_win":4,"peak_thr":1.5}, ...
    """
    if not space:
        yield {}
        return

    keys, values = zip(*space.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def evaluate_algo(
        name: str,
        fn,  # segmentation function (e.g. bocpd_offline, foote_novelty, etc.)
        param_grid: Dict[str, List],
        clips,  # list of (wav_path, ref_times, wav, states)
        tol: float = 0.02,  # boundary‐match tolerance (unused in eval_boundaries call)
):
    """
    For each (algo, each hyper‐parameter combo), run `fn` on every clip in `clips`.
    Return a DataFrame of (algo, hyper‐params, f_mean, f_std, t_mean, mem_mean).
    We’ll print per‐combo and per‐clip timing so you can see exactly where the time goes.
    """
    rows = []

    # Record when we started this algorithm (e.g. 'bocpd_offline')
    algo_start = time.perf_counter()

    # Precompute how many combos there are (so we can print “combo 1/4” etc.)
    all_combos = list(all_param_combos(param_grid))
    total_combos = len(all_combos)

    for combo_idx, p in enumerate(all_combos, start=1):
        print(f"→ Running grid for '{name}' combo {combo_idx}/{total_combos} with params: {p}")

        # Mark the time at the start of this combo
        combo_start = time.perf_counter()

        f_scores, times, mems = [], [], []

        for clip_idx, (wav_path, ref_times, wav, states) in enumerate(clips, start=1):
            # Start timing for just this one clip
            clip_start = time.perf_counter()

            # Run the cutter (e.g. bocpd_offline or foote_novelty) on states
            segs, t, m = run_algo(lambda st: fn(st, **p), states)

            # Time spent on this single call
            n_frames = states.shape[0]
            clip_elapsed = time.perf_counter() - clip_start

            # Convert the returned frame‐pairs into times in seconds
            est_times = frames_to_times(segs)

            # Compute boundary F1 (if we have ground truth)
            if ref_times is not None:
                _, _, f, *_ = eval_boundaries(ref_times, est_times)
            else:
                f = np.nan

            f_scores.append(f)
            times.append(t)
            mems.append(m)

            # Every 10 clips (or on the last clip), print a “liveness” update:
            if clip_idx % 10 == 0 or clip_idx == len(clips):
                total_since_combo = time.perf_counter() - combo_start
                print(
                    f"    • {name:15s} combo {combo_idx}/{total_combos} "
                    f"clip {clip_idx}/{len(clips)} → "
                    f"(frames: {n_frames}, "
                    f"this clip: {clip_elapsed:.2f}s, "
                    f"since combo start: {total_since_combo:.2f}s)"
                )

        # After running all clips for this combo, measure combo duration
        combo_elapsed = time.perf_counter() - combo_start
        avg_f = np.nanmean(f_scores)
        print(f"○ Finished combo {combo_idx}/{total_combos} in {combo_elapsed:.1f}s,  f_mean = {avg_f:.3f}\n")

        rows.append(
            dict(
                algo=name, **p,
                f_mean=avg_f,
                f_std=np.nanstd(f_scores),
                t_mean=np.mean(times),
                mem_mean=np.mean(mems),
            )
        )

    # After all combos for this algorithm
    algo_elapsed = time.perf_counter() - algo_start
    print(f"⇒ Done all combos for '{name}' in {algo_elapsed:.1f} seconds\n")

    return pd.DataFrame(rows).sort_values("f_mean", ascending=False)


def discover_cutters(cutters_module) -> Dict[str, callable]:
    """
    Return {name: function} for every function in cutters_module whose first
    positional argument is `states` and whose name appears in SEARCH_SPACE.

    Example:
        >>> import testers.cutters as cm
        >>> discover_cutters(cm)
        {"foote_novelty": <function foote_novelty>, "bocpd_offline": <function bocpd_offline>, ...}
    """
    cutters_module = importlib.reload(cutters_module)
    return {
        name: fn
        for name, fn in inspect.getmembers(cutters_module, inspect.isfunction)
        if fn.__code__.co_varnames[:1] == ("states",) and name in SEARCH_SPACE
    }


def tune_all(
    search_space: Dict[str, Dict[str, List]],
    clips: List[Tuple[Path, np.ndarray, np.ndarray, np.ndarray]],
    tol: float = 0.02
) -> pd.DataFrame:
    """
    Given a `search_space` dict of the form:
        {"foote_novelty": {"ker_win":[4,8], "peak_thr":[1.0,1.5]}, ...}
    and a list of `clips`, run `evaluate_algo(...)` for each cutter listed in `search_space`.

    Args:
      search_space: dict mapping algo_name → its param_grid (a dict of lists).
      clips:        list of (wav_path, ref_times, wav_signal, states_array).
      tol:          float, tolerance (seconds) for boundary matching.

    Returns:
      One big pandas.DataFrame combining rows from all algos, sorted by f_mean (desc) then t_mean (asc).
      Columns include: ["algo", <hyperparam keys…>, "f_mean", "f_std", "t_mean", "mem_mean"].
    """
    results = []
    cutters = discover_cutters(cutters_module)

    for algo_name, fn in cutters.items():
        if algo_name not in search_space:
            warnings.warn(f"Skipping '{algo_name}' (no entry in search_space).")
            continue

        grid = search_space[algo_name]
        print(f"→ Running grid for '{algo_name}' with params: {grid}")
        df_algo = evaluate_algo(algo_name, fn, grid, clips, tol=tol)
        results.append(df_algo)

    if not results:
        raise RuntimeError("No valid cutters found in provided search_space.")

    df_full = pd.concat(results, ignore_index=True)
    df_full = df_full.sort_values(["f_mean", "t_mean"], ascending=[False, True])
    return df_full
