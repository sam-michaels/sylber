# ----------------------------Imports--------------------------------

import numpy as np
from scipy.signal.windows import gaussian
from scipy.spatial.distance import cdist
import librosa
from kneed import KneeLocator
import ruptures as rpt
from sklearn.cluster import DBSCAN
import librosa.feature as lf

import experiments.compat_scipy_misc
from experiments.bocd_helpers import constant_hazard, StudentT
from bayesian_changepoint_detection.offline_changepoint_detection import offline_changepoint_detection

from sylber.utils.segment_utils import get_segment


def energy_drop(states, threshold_db=-35, min_len=3):
    energy = 10 * np.log10(np.sum(states ** 2, axis=-1) + 1e-12)
    mask = (energy < threshold_db).astype(int)
    cuts = np.where(np.diff(mask) == 1)[0]
    segs, prev = [], 0

    for c in cuts:
        if c - prev >= min_len:
            segs.append((prev, c))
            prev = c
    segs.append((prev, len(states)))

    return segs


def foote_novelty(states, ker_win=12, peak_thr=2.5):

    # 1. self-similarity matrix on cosine distance
    S = 1 - cdist(states, states, "cosine")
    # 2. checkerboard Gaussian kernel
    L = ker_win
    g = np.outer(gaussian(L, L / 6),
                 gaussian(L, L / 6))
    k = np.block([[g, -g], [-g, g]])
    # 3. novelty curve = 2-D conv along diagonal
    nov = np.array([(S[i - L:i + L, i - L:i + L] * k).sum()
                    if i >= L and i + L < len(S) else 0
                    for i in range(len(S))])
    peaks = np.where(nov > peak_thr * nov.std())[0]
    segs = [(s, e) for s, e in zip(np.r_[0, peaks], np.r_[peaks, len(S)])]
    return segs


def bocpd_offline(states, lam=200, thresh=0.5):
    signal = states[:, 0]

    st = StudentT(0, 1, 0.1, 1)
    # wrapper that matches BOCPD’s expected signature
    obs_ll = lambda data, s, e: np.sum(np.log(st.pdf(data[s:e])))

    Q, P, Pcp = offline_changepoint_detection(
        signal,
        constant_hazard(lam),
        obs_ll,          # ← use the wrapper
        truncate=-40
    )

    cps = np.where(Pcp[-1] > thresh)[0]
    return [(s, e) for s, e in zip(np.r_[0, cps], np.r_[cps, len(signal)])]


def sylber_greedy(states, norm_thr=0.20, merge_thr=0.30):
    """Sylber default: frame-to-frame Δ-cosine + merge."""
    return get_segment(states, norm_thr, merge_thr)
