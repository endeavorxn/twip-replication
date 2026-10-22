"""Closed-loop simulation of the discrete TWIP plant (paper Fig. 4 scheme).

The plant is the verified discrete model x[k+1] = Ad x[k] + Bd u[k] with
T0 = 10 ms.  A digital controller runs at the sampling rate (sampler),
its output is held constant over one period (zero-order hold) and passed
through a saturation block before reaching the plant.  States are
x = [theta, psi, theta_dot, psi_dot] and inputs u = [u_left, u_right].
"""
from __future__ import annotations

import numpy as np


def simulate(
    Ad: np.ndarray,
    Bd: np.ndarray,
    controller,
    x0: np.ndarray,
    n_steps: int,
    disturbance=None,
    u_sat: float | None = None,
):
    """Simulate the sampled closed loop.

    controller : callable(x, k) -> array-like of shape (2,) or scalar.
        Scalar is applied to both motors (u_left = u_right = u); this is how
        the paper's controllers effectively drive the vehicle.
    disturbance : callable(k) -> array-like of shape (4,), optional.
        Additive state perturbation injected at step k (used for the
        external-force-on-pitch scenario of Section 4.2).
    u_sat : float, optional.  Symmetric saturation limit |u| <= u_sat applied
        after the ZOH (paper Fig. 4 saturation block; the paper does not state
        the limit, so None = unsaturated by default).

    Returns a dict with the state trajectory X (n_steps+1, 4) and the input
    trajectory U (n_steps, 2).
    """
    Ad = np.asarray(Ad, float)
    Bd = np.asarray(Bd, float)
    x = np.asarray(x0, float).copy()
    X = np.zeros((n_steps + 1, 4))
    U = np.zeros((n_steps, 2))
    X[0] = x
    for k in range(n_steps):
        u = np.asarray(controller(x, k), float)
        if u.ndim == 0:
            u = np.array([float(u), float(u)])
        if u_sat is not None:
            u = np.clip(u, -u_sat, u_sat)
        U[k] = u
        x = Ad @ x + Bd @ u
        if disturbance is not None:
            dx = np.asarray(disturbance(k), float)
            x = x + dx
        X[k + 1] = x
    return {"X": X, "U": U}


def settling_time(t, y, tol=0.02):
    """Last time |y| leaves the +/- tol band around zero (relative to the
    initial offset for a decaying response), in seconds."""
    y = np.asarray(y, float)
    band = tol * max(abs(y[0]), 1e-12)
    outside = np.abs(y) > band
    if not outside.any():
        return 0.0
    return float(t[np.max(np.nonzero(outside)[0]) + 1]) if np.max(np.nonzero(outside)[0]) + 1 < len(t) else float("inf")
