# experiments/bocd_helpers.py
import numpy as np
from scipy.stats import t


def constant_hazard(lam):
    """Return a function h(r) = 1/lam (Adams & MacKay, Eq. 17)."""
    inv = 1.0 / lam
    return lambda r: inv + r * 0  # works on scalars or numpy arrays


class StudentT:
    """Predictive distribution used in the original BOCPD algorithm."""

    def __init__(self, mu0, kappa0, alpha0, beta0):
        self.mu0, self.kappa0 = mu0, kappa0
        self.alpha0, self.beta0 = alpha0, beta0

    def pdf(self, x):
        dof = 2 * self.alpha0
        scale = np.sqrt(self.beta0 * (self.kappa0 + 1) /
                        (self.alpha0 * self.kappa0))
        return t.pdf(x, dof, loc=self.mu0, scale=scale)
