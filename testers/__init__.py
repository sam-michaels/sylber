# Ensure SciPy compatibility shim is loaded before any Bayesian changepoint modules import SciPy
import testers.compat_scipy_misc

# Expose submodules for convenience
from testers import (
    cutters,
    bocd_helpers,
    evaluation_helpers,
    params,
    tuner,
)

# (Optional) define __all__ for explicit exports
__all__ = [
    "cutters",
    "bocd_helpers",
    "evaluation_helpers",
    "params",
    "tuner",
]
