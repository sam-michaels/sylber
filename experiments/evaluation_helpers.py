# --- evaluation_helpers.py -----------------------------------------------
import time, tracemalloc, pandas as pd, mir_eval
from pathlib import Path
import torchaudio, numpy as np
from sylber.utils.segment_utils import get_segment  # if you’ll extract wave snippets

TOL = 0.02  # boundary tolerance in seconds


def run_algo(fn, states):
    """Return segments + runtime + peak MB."""
    start = time.perf_counter()
    tracemalloc.start()
    segs = fn(states)  # returns list of (start_frame, end_frame)
    mem, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return segs, time.perf_counter() - start, mem / 1e6


def frames_to_times(segs, sr=16000, hop=320):  # ~20 ms hop
    return np.array([s * hop / sr for s, _ in segs])


def eval_boundaries(ref_times, est_times):
    """
    Compute boundary precision, recall, F1, and median deviation for matched boundaries.
    ref_times and est_times are 1D numpy arrays of boundary times in seconds.
    """
    # tolerance window
    tol = TOL  # assuming TOL is defined elsewhere in this file

    # track matched reference indices
    matched_ref = set()
    tp = 0
    deviations = []

    for e in est_times:
        if len(ref_times) == 0:
            break
        diffs = np.abs(ref_times - e)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tol and min_idx not in matched_ref:
            tp += 1
            matched_ref.add(min_idx)
            deviations.append(diffs[min_idx])

    prec = tp / len(est_times) if len(est_times) > 0 else 0.0
    rec = tp / len(ref_times) if len(ref_times) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # median deviation among all matched pairs
    med_dev = np.median(deviations) if deviations else np.nan

    # Return (precision, recall, f1, median deviation ref, median deviation est)
    # Here median deviations for reference and estimated are the same.
    return prec, rec, f1, med_dev, med_dev
