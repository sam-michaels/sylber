SEARCH_SPACE = {
    "foote_novelty": dict(  # parameter name : iterable of values
        ker_win=[4, 6, 8, 10, 12, 16, 20, 24, 28, 32],  # window radius
        peak_thr=[0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    ),
    "bocpd_offline": dict(
        lam=[25, 50, 100, 200, 400],
        thresh=[0.2, 0.35, 0.5, 0.65, 0.8]
    ),
}
