"""LQR design by Riccati backward iteration (paper Section 3.2, Eqs. 36-45).

The infinite-horizon discrete LQR gain is obtained exactly as the paper
describes: the backward Riccati recursion (Eqs. 40-41) is iterated from
P(N) = Q, K(N) = 0 until the gain sequence satisfies the paper's convergence
criterion (Eq. 42),  ||K(i) - K(i-1)|| < gamma ||K(i)||.

Input reduction: the pitch plant has Bd = [b b] with IDENTICAL columns, and
the paper drives both motors with a single signal (every published controller
gain is one row applied to both motors).  Restricting the input to the
achievable diagonal  u_left = u_right = u  turns the plant into
x[k+1] = Ad x[k] + (2b) u[k] while the penalty R = 0.00017 I2 (Eq. 44)
contributes r u^2 + r u^2 = (2r) u^2.  The design is therefore carried out on
the reduced single-input problem (Ad, 2b, Q, 2r), whose per-motor gain
reproduces the published Eq. (45) to all printed digits.  This reduction is
lossless: the unconstrained two-input Riccati solution has identical rows
equal to the reduced gain (verified in scripts/run_lqr.py).
"""
from __future__ import annotations

import numpy as np

from .model import continuous_matrices, discretize
from .params import Params

Q_EQ43 = np.diag([0.38, 0.43, 0.09, 0.09])   # Eq. (43), state weightings
R_EQ44 = 0.00017                             # Eq. (44) as printed: R = 0.00017 I2


def riccati_backward(
    Ad: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    gamma: float = 1e-9,
    max_iter: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve the DARE by the paper's backward iteration, Eqs. (40)-(42).

    Eq. (40):  K(i) = (R + B' P(i+1) B)^-1 B' P(i+1) A
    Eq. (41):  P(i) = Q + K(i)' R K(i) + (A - B K(i))' P(i+1) (A - B K(i))
    Eq. (42):  stop when ||K(i) - K(i-1)|| < gamma ||K(i)||

    Started from P(N) = Q, K(N) = 0, as the paper specifies.

    Returns
    -------
    (K, P, n_iter) : gain (m, n), converged Riccati matrix (n, n), and the
        number of iterations needed to satisfy Eq. (42).
    """
    Ad = np.asarray(Ad, float)
    B = np.atleast_2d(np.asarray(B, float))
    if B.shape[0] != Ad.shape[0]:
        B = B.T                                   # allow a 1-D column input
    Q = np.asarray(Q, float)
    R = np.atleast_2d(np.asarray(R, float))

    P = Q.copy()                                  # P(N) = Q
    K_prev = np.zeros((R.shape[0], Ad.shape[1]))  # K(N) = 0
    for i in range(1, max_iter + 1):
        S = R + B.T @ P @ B                       # Eq. (40) bracket
        K = np.linalg.solve(S, B.T @ P @ Ad)
        Acl = Ad - B @ K
        P = Q + K.T @ R @ K + Acl.T @ P @ Acl     # Eq. (41)
        if np.linalg.norm(K - K_prev) < gamma * np.linalg.norm(K):   # Eq. (42)
            return K, (P + P.T) / 2, i
        K_prev = K
    raise RuntimeError(f"Riccati iteration did not converge in {max_iter} iterations")


def reduced_plant(Ad: np.ndarray, Bd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the twin-input plant to its single effective motor input.

    With Bd = [b b] and u_left = u_right = u the state equation is
    x+ = Ad x + (2b) u; the matching input penalty is R_eff = 2 * r
    (two motors each spending r u^2).  Returns (Ad, 2b).
    """
    Bd = np.asarray(Bd, float)
    if not np.allclose(Bd[:, 0], Bd[:, 1]):
        raise ValueError("reduction assumes the two input columns are identical")
    return np.asarray(Ad, float), 2.0 * Bd[:, :1]


def design_lqr(params: Params | None = None, gamma: float = 1e-9) -> tuple[np.ndarray, dict]:
    """Design the paper's LQR controller (Eqs. 36-45).

    Returns
    -------
    K : (4,) ndarray, the per-motor gain row such that u = -K x with
        u_left = u_right = u; reproduces published Eq. (45).
    info : dict with the convergence data (iterations, gamma), the converged
        Riccati matrix P, the closed-loop poles, and the weightings used.
    """
    p = Params() if params is None else params
    Ac, Bc = continuous_matrices(p)
    Ad, Bd = discretize(Ac, Bc, p.T0)

    A, B = reduced_plant(Ad, Bd)
    R = 2.0 * R_EQ44 * np.eye(1)                  # r u^2 + r u^2 on the diagonal input
    K2, P, n_iter = riccati_backward(A, B, Q_EQ43, R, gamma=gamma)

    poles = np.linalg.eigvals(A - B @ K2)
    info = {
        "gamma": gamma,
        "iterations": n_iter,
        "Q": Q_EQ43.copy(),                       # Eq. (43)
        "R_printed": R_EQ44,                      # Eq. (44)
        "R_effective_scalar": 2.0 * R_EQ44,       # both motors on the diagonal input
        "P": P,
        "poles": poles,
        "reduction": ("Bd = [b b]; constrained to u_l = u_r = u the plant is "
                      "(Ad, 2b) with penalty 2r; see module docstring"),
    }
    return K2.ravel().copy(), info


class LQRController:
    """State feedback u = -K x (Eq. 45).

    Returns a scalar so :func:`twip.simulate.simulate` applies it to both
    motors (u_left = u_right = u), which is how the paper's single-row gains
    drive the vehicle.
    """

    def __init__(self, K):
        self.K = np.asarray(K, float).ravel()

    def __call__(self, x, k: int = 0) -> float:
        return float(-self.K @ np.asarray(x, float))
