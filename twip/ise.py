"""Integral Square Error criterion (paper Section 4, Tables 3-4)."""
from __future__ import annotations

import numpy as np


def ise(y, T0: float) -> float:
    """Discrete ISE = T0 * sum_k y[k]^2 (rectangular integral of y^2)."""
    y = np.asarray(y, float)
    return float(T0 * np.sum(y**2))
