# ----------------------------Imports--------------------------------

import numpy as np
from scipy.signal.windows import gaussian
from scipy.spatial.distance import cdist
import librosa
from kneed import KneeLocator
import ruptures as rpt
from sklearn.cluster import DBSCAN
import librosa.feature as lf

import testers.compat_scipy_misc
from testers.bocd_helpers import constant_hazard, StudentT
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
    k = np.block([[g, -g], [-g, g]])    # type: ignore
    # 3. novelty curve = 2-D conv along diagonal
    nov = np.array([(S[i - L:i + L, i - L:i + L] * k).sum()
                    if i >= L and i + L < len(S) else 0
                    for i in range(len(S))])
    peaks = np.where(nov > peak_thr * nov.std())[0]
    segs = [(s, e) for s, e in zip(np.r_[0, peaks], np.r_[peaks, len(S)])]
    return segs


def bocpd_offline(states, lam=200, thresh=0.5):
    """
    Same as before, but precompute log‐pdfs for each frame so that
    obs_ll(s, e) = cumulative_log[e] - cumulative_log[s], instead of
    calling st.pdf(...) on each slice.
    """
    # 1) Extract the 1D “signal” from your 2D states (e.g. first dimension):
    signal = states[:, 0]
    n = len(signal)

    # 2) Make a Student‐T object once:
    st = StudentT(0, 1, 0.1, 1)

    # 3) Precompute log‐pdf for each individual sample in `signal`:
    #    log_probs[i] = log( p(signal[i] | StudentT) )
    log_probs = np.log(st.pdf(signal))

    # 4) Build a “prefix‐sum” array of those log‐probs:
    #    cumulative_log[k] = sum_{i=0..k-1} log_probs[i],
    #    so that sum_{i=s..e-1} log_probs[i] = cumulative_log[e] - cumulative_log[s].
    cumulative_log = np.concatenate(([0.0], np.cumsum(log_probs)))   # type: ignore

    # 5) Define obs_ll using prefix sums (O(1) per call):
    obs_ll = lambda data, s, e: float(cumulative_log[e] - cumulative_log[s])

    # 6) Call BOCPD with the truncated run‐length support as before:
    Q, P, Pcp = offline_changepoint_detection(
        signal,
        constant_hazard(lam),
        obs_ll,
        truncate=-40
    )

    # 7) Threshold the final Pcp to get change points
    cps = np.where(Pcp[-1] > thresh)[0]

    # 8) Package them into (start_frame, end_frame) pairs
    segments = [(int(s), int(e)) for s, e in zip(np.r_[0, cps], np.r_[cps, n])]
    return segments


def sylber_greedy(states, norm_thr=0.20, merge_thr=0.30):
    """Sylber default: frame-to-frame Δ-cosine + merge."""
    return get_segment(states, norm_thr, merge_thr)


def ruptures_pelt(states, model="rbf", penalty=3, min_size=1, jump=1):
    """
    Run Ruptures PELT on the 0th dimension of `states` (or on any 1D projection).
    Returns a list of (start_frame, end_frame) segments.

    Args:
      states: np.ndarray, shape = (n_frames, n_feats)
      model: cost model ("rbf", "l1", "l2", "linear", "normal", etc.)
      penalty: penalty value for PELT; larger → fewer change points
      min_size: minimum segment length (in frames)
      jump: subsampling factor (speedup); 1 = no skipping, 2 = skip every other frame, etc.
    """
    # Extract a 1D signal from `states`. Here, we simply take the first dimension:
    sig = states[:, 0]

    # Build the PELT `algo` object and fit:
    # NOTE: if you want to use multivariate, you could pass e.g. states instead of sig[:, None].
    algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(sig)

    # `penalty` controls the number of change points
    change_points = algo.predict(pen=penalty)
    # change_points is a list of frame‐indices where a change happens (including `len(sig)` at end)

    # Convert change points into segments: e.g., [cp1, cp2, cp3] → [(0,cp1),(cp1,cp2),(cp2,cp3)]
    cps = np.array(change_points)
    starts = np.concatenate(([0], cps[:-1]))  # type: ignore
    ends = cps
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def dbscan_segmentation(states, eps=0.5, min_samples=5, metric="cosine"):
    """
    Use DBSCAN on the high‐dim states to cluster frames; every time the cluster
    label changes, that’s a boundary. Finally, convert label‐runs into segments.
    """
    # states: shape = (n_frames, n_feats)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(states)
    labels = clustering.labels_  # type: ignore # shape = (n_frames,), -1 for noise, 0,1,2… for clusters

    # Force every “noise” point (label = -1) to point to its neighbor’s cluster (optional)
    # For simplicity, treat "-1" as its own cluster

    segments = []
    prev_label = labels[0]
    start_frame = 0
    for i, lab in enumerate(labels[1:], start=1):
        if lab != prev_label:
            segments.append((start_frame, i))
            start_frame = i
            prev_label = lab
    segments.append((start_frame, len(labels)))
    return segments


def spectral_flux_segmentation(states, sr=16000, flux_thr=0.1, min_dist=5, hop_length=512):
    """
    states: 1D numpy array of raw audio samples
    sr:     sampling rate
    flux_thr: threshold for normalized flux
    min_dist: minimum distance between peaks in frames
    """
    audio = states.astype(float)
    S = np.abs(librosa.stft(audio, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    flux = flux / (np.max(flux) + 1e-8)
    onset_frames = librosa.util.peak_pick(
        flux,
        pre_max=3, post_max=3, pre_avg=3, post_avg=3,
        delta=flux_thr, wait=min_dist
    )
    boundaries = np.sort(onset_frames)
    starts = np.concatenate(([0], boundaries))  # type: ignore
    ends = np.concatenate((boundaries, [len(flux)]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


# ────────────────────────────────────────────────────────────────────────────
# New Method #4: Energy Knee
def energy_knee_segmentation(states, smoothing=5):
    energy = np.sum(states ** 2, axis=1)
    kernel = np.ones(smoothing) / smoothing
    energy_smooth = np.convolve(energy, kernel, mode="same")
    x = np.arange(len(energy_smooth))
    y = energy_smooth
    kn = KneeLocator(x, y, curve="convex", direction="decreasing")
    knees = kn.knee
    if knees is None:
        return [(0, len(states))]
    if not isinstance(knees, (list, np.ndarray)):
        knees = [knees]
    boundaries = sorted([int(k) for k in knees if 0 < k < len(states)])
    starts = np.concatenate(([0], boundaries))  # type: ignore
    ends = np.concatenate((boundaries, [len(states)]))  # type: ignore
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


# ────────────────────────────────────────────────────────────────────────────
# New Method #5: Uniform (Fixed‐Length) Segmentation
def uniform_segmentation(states, seg_len=50):
    n = len(states)
    starts = list(range(0, n, seg_len))
    segments = []
    for s in starts:
        e = min(s + seg_len, n)
        segments.append((s, e))
    return segments


# ────────────────────────────────────────────────────────────────────────────
# Dispatcher: run_cut_method
def run_cut_method(algo: str, states: np.ndarray, **hyper) -> list[tuple[int, int]]:
    if algo == "foote_novelty":
        return foote_novelty(states, **hyper)
    elif algo == "bocpd_offline":
        return bocpd_offline(states, **hyper)
    elif algo == "sylber_greedy":
        return sylber_greedy(states, **hyper)
    elif algo == "energy_drop":
        return energy_drop(states, **hyper)
    elif algo == "ruptures_pelt":
        return ruptures_pelt(states, **hyper)
    elif algo == "dbscan_segmentation":
        return dbscan_segmentation(states, **hyper)
    elif algo == "spectral_flux":
        return spectral_flux_segmentation(states, **hyper)
    elif algo == "energy_knee":
        return energy_knee_segmentation(states, **hyper)
    elif algo == "uniform_segmentation":
        return uniform_segmentation(states, **hyper)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
