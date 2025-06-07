# testers/params.py

from sklearn.model_selection import ParameterGrid

# ── Tiny “smoke‐test” search space (very fast; just to verify plumbing) ──
TINY_SEARCH_SPACE = {
    "foote_novelty": dict(ker_win=[8, 16], peak_thr=[1.0, 1.5]),
    "bocpd_offline": dict(lam=[25, 100], thresh=[0.35, 0.5]),
    "ruptures_pelt": dict(model=["rbf"], penalty=[3], min_size=[1], jump=[5]),
    "dbscan_segmentation": dict(eps=[0.5], min_samples=[5], metric=["cosine"]),
    "spectral_flux": dict(sr=[16000], flux_thr=[0.1], min_dist=[5], hop_length=[512]),
    "energy_knee": dict(smoothing=[5]),
    "uniform_segmentation": dict(seg_len=[50]),
}


# ── Full search space for final sweep ──
SEARCH_SPACE = {
    "foote_novelty": dict(
        ker_win=[4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
        peak_thr=[0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    ),
    "bocpd_offline": dict(
        lam=[25, 50, 100, 200, 400],
        thresh=[0.2, 0.35, 0.5, 0.65, 0.8]
    ),
    "ruptures_pelt": dict(
        model=["rbf", "l2"],
        penalty=[1, 3, 5, 10],
        min_size=[1, 5, 10],
        jump=[1, 5, 10]
    ),
    "dbscan_segmentation": dict(
        eps=[0.3, 0.5, 0.7],
        min_samples=[3, 5, 10],
        metric=["cosine", "euclidean"]
    ),
    "spectral_flux": dict(
        sr=[16000],
        flux_thr=[0.05, 0.1, 0.2],
        min_dist=[3, 5, 10],
        hop_length=[256, 512]
    ),
    "energy_knee": dict(
        smoothing=[3, 5, 10]
    ),
    "uniform_segmentation": dict(
        seg_len=[25, 50, 100]
    ),
}


def get_space(tiny: bool = False) -> dict[str, dict]:
    """
    Return either the tiny search space (for quick debugging)
    or the full search space (for an exhaustive sweep).

    Example:
        >>> from testers.params import get_space
        >>> small = get_space(tiny=True)
        >>> full  = get_space(tiny=False)
    """
    return TINY_SEARCH_SPACE if tiny else SEARCH_SPACE
