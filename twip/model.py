"""Continuous and discrete TWIP pitch model (Villacrés et al. 2016).

State  x1 = [theta, psi, theta_dot, psi_dot]^T  (wheel angle, body pitch angle)
Input   u = [u_left, u_right]^T                 (motor voltages)

The linearized model is x1_dot = A1 x1 + B1 u (Eqs. 1, 4-5), discretized with
zero-order hold at T0 = 10 ms (Eqs. 23-24).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .params import Params


def continuous_matrices(params: Params | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build the continuous pitch model (A1, B1) from Table 1 parameters.

    Coefficient formulas are Eqs. (6)-(20) of the paper; the assembled
    matrices are the structures of Eqs. (4)-(5). With the paper's Table 1
    values this reproduces the published Eqs. (21)-(22).

    Returns
    -------
    (A1, B1) : (4, 4), (4, 2) ndarrays
    """
    p = Params() if params is None else params

    e11 = (2 * p.m + p.M) * p.R**2 + 2 * p.Jw + 2 * p.n**2 * p.Jm        # Eq. (14)
    e12 = p.M * p.L * p.R - 2 * p.n**2 * p.Jm                            # Eq. (15)
    e22 = p.M * p.L**2 + p.Jpsi + 2 * p.n**2 * p.Jm                      # Eq. (16)
    det = e11 * e22 - e12**2                                             # Eq. (17)

    alpha = p.n * p.Kt / p.Rm                                            # Eq. (18)
    beta = p.n * p.Kt * p.Kb / p.Rm + p.fm                               # Eq. (19)
    sigma = beta + p.fw                                                  # Eq. (20)

    a32 = -p.g * p.M * p.L * e12 / det                                   # Eq. (6)
    a42 = p.g * p.M * p.L * e11 / det                                    # Eq. (7)
    a33 = -2 * (sigma * e22 + beta * e12) / det                          # Eq. (8)
    a43 = 2 * (sigma * e12 + beta * e11) / det                           # Eq. (9)
    a34 = 2 * beta * (e22 + e12) / det                                   # Eq. (10)
    a44 = -2 * beta * (e11 + e12) / det                                  # Eq. (11)
    b3 = alpha * (e22 + e12) / det                                       # Eq. (12)
    b4 = -alpha * (e11 + e12) / det                                      # Eq. (13)

    A1 = np.array([                                                      # Eq. (4)
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, a32, a33, a34],
        [0.0, a42, a43, a44],
    ])
    B1 = np.array([                                                      # Eq. (5)
        [0.0, 0.0],
        [0.0, 0.0],
        [b3, b3],
        [b4, b4],
    ])
    return A1, B1


def discretize(Ac: np.ndarray, Bc: np.ndarray, T0: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretization, Eqs. (23)-(24).

    Eq. (23): Ad = expm(Ac*T0).  Eq. (24) as printed, Bd = Ac^-1 (Ad - I) Bc,
    cannot be evaluated because Ac is singular (its first column is zero), so
    Bd is taken from the standard augmented-matrix identity

        expm([[Ac, Bc], [0, 0]] * T0) = [[Ad, Bd], [0, I]]

    which is algebraically identical to the printed formula.

    Parameters
    ----------
    Ac : (n, n) continuous state matrix
    Bc : (n, m) continuous input matrix
    T0 : sampling time [s], 10 ms in the paper

    Returns
    -------
    (Ad, Bd) : (n, n), (n, m) ndarrays
    """
    n, m = Bc.shape
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = Ac
    aug[:n, n:] = Bc
    F = expm(aug * T0)
    return F[:n, :n].copy(), F[:n, n:].copy()


def open_loop_poles(Ad: np.ndarray) -> np.ndarray:
    """Open-loop discrete poles, eigenvalues of Ad sorted by descending real
    part (then descending imaginary part), cf. Eqs. (27)-(30)."""
    ev = np.linalg.eigvals(np.asarray(Ad))
    order = np.lexsort((-ev.imag, -ev.real))
    return ev[order]
