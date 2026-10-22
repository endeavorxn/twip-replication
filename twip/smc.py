"""Optimal Sliding Mode Control for the TWIP (paper Section 3.3, Eqs. 46-64).

The paper's "Optimal SMC" chooses the sliding surface so that the sliding
motion has exactly the LQR closed-loop poles (Eqs. 60-61).  Because the
one-step discrete reaching law contributes a further deadbeat pole at the
origin, the ideal equivalent control (Eq. 57) coincides with the LQR
feedback up to published rounding -- verified in scripts/run_smc.py.

Construction (single effective input; the paper's Eq. (26) Bd has two
identical columns, so u_l = u_r = u throughout)
  1. Reduce to SISO: b = Bd @ [1, 1] ("sum" wiring) or b = Bd[:, 0]
     ("column" wiring; half the input authority, twice the gain).
  2. Phase-variable (controllable canonical form) transformation
     T = ctrb(Ad, b) @ W with ctrb = [b, Ab, A^2 b, A^3 b] and W the upper
     triangular Toeplitz matrix built from the characteristic coefficients
     lambda^4 + c3 lambda^3 + c2 lambda^2 + c1 lambda + c0 of (Ad, b):

             [1  c3  c2  c1]
         W = [0  1   c3  c2]
             [0  0   1   c3]
             [0  0   0   1 ]

     so that bbar = T^-1 b = e1 and Abar = T^-1 Ad T is the first-row
     companion form (subdiagonal ones).  This is the e1-companion reading
     of the paper's e4-companion description (Sbar = [a3, a2, a1, 1] with
     the reversed controllability matrix); the pulled-back surface S is
     identical either way.
  3. Desired sliding polynomial  zeta^3 + a1 zeta^2 + a2 zeta + a3  from
     the sliding poles p1,2 = 0.9318 +/- 0.0042j and p3 = 0.9797
     (Eqs. 60-61).  In phase variables the surface is Sbar = [1, a1, a2, a3]
     (equivalently [a3, a2, a1, 1] in the reversed coordinates); on s = 0
     the constrained dynamics are the companion of exactly this cubic.
  4. Pull back:  S = Sbar T^-1.

A useful structural property (verified numerically): this construction
yields S b = Sbar (T^-1 b) = Sbar e1 = 1 exactly -- the surface comes out
normalized.  The published Eq. (62) surface satisfies S_pub @ Bd[:,0] =
1.0046 ~= 1, i.e. the paper evidently used the same construction with the
single-column input; and with Sb = 1 the equivalent control Eq. (63) is
simply u_eq = -S Ad x.

Control law (Eqs. 57-58, 64)
    u_eq(k) = -(S b)^-1 S Ad x(k)              (equivalent control)
    u(k)    = u_eq(k) + v(k),
    v(k)    = +cte          if s(k) <  -L
            = -cte s(k)/L   if |s(k)| <= L     (boundary layer)
            = -cte          if s(k) >  +L
with s(k) = S x(k).  (The paper does not state cte or L; the middle branch
is written -cte s/L so that v is continuous at |s| = L for every cte -- at
the paper-neutral default cte = 1 it coincides numerically with a literal
"-s/L" reading.)

Reaching dynamics: s(k+1) = S(Ad x + b_wiring u) = Sb_wiring v(k), where
Sb_wiring = S @ (input column actually driving the plant).  With the
boundary layer this gives s(k+1) = -(Sb_wiring cte / L) s(k): the layer is
contractive (no chattering) iff 0 < Sb_wiring cte/L < 1 and the state
reaches it in one step iff Sb_wiring cte <= L.  scripts/run_smc.py sweeps
(cte, L) around this condition; the paper's Fig. 5-12 "no chattering"
claim implicitly requires it.
"""
from __future__ import annotations

import numpy as np

# --- published quantities of Section 3.3 -------------------------------------
S_PUB = np.array([-1.924, -147.918, -1.669, -21.065])       # Eq. (62)
UEQ_GAIN_PUB = np.array([1.924, 160.270, 2.717, 21.581])    # Eq. (63)
SLIDING_POLES = np.array([0.9318 + 0.0042j,                 # Eqs. (60)-(61)
                          0.9318 - 0.0042j,
                          0.9797])
LQR_ROW_PUB = np.array([-1.099, -81.44, -1.368, -10.86])    # Eq. (45), u = -K x


def controllability_matrix(Ad: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ctrb(Ad, b) = [b, Ab, A^2 b, A^3 b] (n x n for the SISO pair)."""
    Ad = np.asarray(Ad, float)
    b = np.asarray(b, float)
    return np.column_stack([np.linalg.matrix_power(Ad, i) @ b
                            for i in range(Ad.shape[0])])


def ccf_transform(Ad: np.ndarray, b: np.ndarray):
    """Phase-variable transformation T = ctrb @ W (module docstring, step 2).

    Returns (T, Abar, bbar) with bbar = T^-1 b = e1 and Abar = T^-1 Ad T the
    first-row companion form (subdiagonal ones, coefficients -c0..-c3 in the
    first row).
    """
    Ad = np.asarray(Ad, float)
    n = Ad.shape[0]
    coeffs = np.poly(Ad)[:n]                    # [1, c3, c2, c1]
    W = np.array([[coeffs[j - i] if j >= i else 0.0
                   for j in range(n)] for i in range(n)])
    T = controllability_matrix(Ad, b) @ W
    Abar = np.linalg.solve(T, Ad @ T)
    bbar = np.linalg.solve(T, np.asarray(b, float))
    return T, Abar, bbar


def sliding_polynomial(poles=SLIDING_POLES):
    """(a1, a2, a3) of zeta^3 + a1 zeta^2 + a2 zeta + a3 (Eqs. 60-61)."""
    coeffs = np.real_if_close(np.poly(np.asarray(poles)))
    return float(coeffs[1]), float(coeffs[2]), float(coeffs[3])


def sliding_surface(Ad: np.ndarray, b: np.ndarray,
                    poles=SLIDING_POLES) -> tuple[np.ndarray, dict]:
    """Sliding surface S with S @ b == 1 (module docstring, steps 2-4).

    Returns (S, info); info carries the transformation, the polynomial
    coefficients and the sliding poles actually realized on the constrained
    dynamics (eigenvalues of the on-manifold restriction).
    """
    Ad = np.asarray(Ad, float)
    T, Abar, bbar = ccf_transform(Ad, b)
    a1, a2, a3 = sliding_polynomial(poles)
    Sbar = np.array([1.0, a1, a2, a3])
    S = Sbar @ np.linalg.inv(T)
    info = {"T": T, "Abar": Abar, "bbar": bbar, "Sbar": Sbar,
            "poly": (a1, a2, a3), "poles": np.asarray(poles)}
    return S, info


def on_manifold_poles(Ad: np.ndarray, b: np.ndarray,
                      S: np.ndarray) -> np.ndarray:
    """Eigenvalues of the closed loop restricted to the manifold S x = 0.

    Uses the exact deadbeat equivalent control u = -(S b)^-1 S Ad x, whose
    closed loop leaves the manifold invariant; the restriction carries the
    sliding poles, the remaining eigenvalue is the reaching mode (0 for the
    exact CCF surface).
    """
    b = np.asarray(b, float)
    Keq = equivalent_control_gain(Ad, b, S)
    Acl = Ad + np.outer(b, Keq)                 # x+ = Ad x + b u, u = Keq x
    from scipy.linalg import null_space
    N = null_space(np.asarray(S, float).reshape(1, -1))     # S N = 0, 4x3
    Ared = np.linalg.pinv(N) @ Acl @ N
    return np.linalg.eigvals(Ared)


def equivalent_control_gain(Ad: np.ndarray, b: np.ndarray,
                            S: np.ndarray) -> np.ndarray:
    """Feedback row K_eq with u_eq = K_eq x, Eq. (57):

    u_eq(k) = -(S b)^-1 S Ad x(k)   (Sb = 1 for the CCF surface).
    """
    Ad = np.asarray(Ad, float)
    S = np.asarray(S, float)
    return -(S @ Ad) / float(S @ np.asarray(b, float))


def boundary_layer(s: float, cte: float, L: float) -> float:
    """Reaching control v of Eq. (64) (module docstring for the convention).

    v = +cte if s < -L;  -cte s/L if |s| <= L;  -cte if s > L.
    """
    if s < -L:
        return cte
    if s > L:
        return -cte
    return -cte * s / L


class SMCController:
    """Discrete Optimal SMC law, Eqs. (57)-(58) with the Eq. (64) layer.

    Returns a scalar u applied to both motors (u_l = u_r = u); the plant is
    therefore driven through b_wiring = Bd @ [1, 1] regardless of the column
    the surface was designed from -- see ``sb_wiring`` for the resulting
    effective Sb that sets the reaching condition.
    """

    def __init__(self, Ad: np.ndarray, b: np.ndarray, S: np.ndarray,
                 cte: float = 1.0, L: float = 4.0, ideal: bool = False,
                 b_wiring: np.ndarray | None = None):
        self.Ad = np.asarray(Ad, float)
        self.b = np.asarray(b, float)           # design input column
        self.S = np.asarray(S, float)
        self.cte = float(cte)
        self.L = float(L)
        self.ideal = bool(ideal)                # True: u = u_eq only (v = 0)
        # input column actually driving the plant (u_l = u_r = u -> Bd@[1,1]);
        # defaults to the design column itself.
        self.b_wiring = (self.b if b_wiring is None
                         else np.asarray(b_wiring, float))
        self.sb_design = float(self.S @ self.b)
        self.Keq = equivalent_control_gain(self.Ad, self.b, self.S)

    @property
    def sb_wiring(self) -> float:
        """S @ b_wiring -- the Sb that governs s(k+1) = Sb v."""
        return float(self.S @ self.b_wiring)

    def s(self, x: np.ndarray) -> float:
        """Sliding variable s = S x (Eq. 59)."""
        return float(self.S @ np.asarray(x, float))

    def u_eq(self, x: np.ndarray) -> float:
        """Equivalent control of Eq. (57)."""
        return float(self.Keq @ np.asarray(x, float))

    def __call__(self, x: np.ndarray, k: int) -> float:
        u = self.u_eq(x)
        if not self.ideal:
            u += boundary_layer(self.s(x), self.cte, self.L)
        return u
