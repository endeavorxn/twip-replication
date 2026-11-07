"""Shared TWIP replication core: model, controllers, simulation.

Rebuilt from Villacrés et al. (IJCSR 2016), Eqs. (4)-(64) and Table 1, with the
two corrections documented in the CTRL_6 report: g = 9.81 (Table 1 prints 9.8,
only 9.81 reproduces Eq. (21)) and B_d from the augmented matrix exponential
(Eq. (24) divides by the singular A_c; identical ZOH algebra).
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are

T0 = 0.01          # 10 ms sampling (paper Table: 100 Hz)
SIM_STEPS = 1000   # 10 s
KICK_T = 5.0
KICK_IDX = int(KICK_T / T0)
X0 = np.array([0.0, 0.1, 0.0, 0.0])  # theta, psi, theta_dot, psi_dot

PID_PSI = (-77.97, -0.01, -8.79)  # Table 2, pitch loop
PID_TH = (-1.07, -0.01, -1.36)    # Table 2, wheel loop
Q_DIAG = np.array([0.38, 0.43, 0.09, 0.09])  # Eq. (43)
R_PRINT = 0.00017                            # Eq. (44), per motor
S_ADOPTED = np.array([-2.197, -150.49, -1.685, -21.18])  # report's adopted surface
S_PRINT = np.array([-1.924, -147.918, -1.669, -21.065])  # paper Eq. (62) as printed


# ------------------------------------------------------------------ model
def build_model():
    g, m, M, R = 9.81, 0.03, 0.6, 0.021
    Jw = m * R**2 / 2
    L = 0.26 / 2
    Jpsi = M * L**2 / 3
    Jm = 1e-5
    Rm, Kb, Kt, n = 6.69, 0.468, 0.317, 1
    fm, fw = 0.0022, 0.0

    e11 = (2 * m + M) * R**2 + 2 * Jw + 2 * n**2 * Jm
    e12 = M * L * R - 2 * n**2 * Jm
    e22 = M * L**2 + Jpsi + 2 * n**2 * Jm
    detE = e11 * e22 - e12**2
    alpha = n * Kt / Rm
    beta = n * Kt * Kb / Rm + fm
    sigma = beta + fw

    Ac = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, -g * M * L * e12 / detE, -2 * (sigma * e22 + beta * e12) / detE,
         2 * beta * (e22 + e12) / detE],
        [0, g * M * L * e11 / detE, 2 * (sigma * e12 + beta * e11) / detE,
         -2 * beta * (e11 + e12) / detE],
    ])
    b3 = alpha * (e22 + e12) / detE
    b4 = -alpha * (e11 + e12) / detE
    Bc = np.array([[0, 0], [0, 0], [b3, b3], [b4, b4]])

    # ZOH discretisation via the augmented matrix exponential (A_c singular).
    aug = np.zeros((6, 6))
    aug[:4, :4] = Ac
    aug[:4, 4:] = Bc
    Mexp = expm(aug * T0)
    return Ac, Bc, Mexp[:4, :4], Mexp[:4, 4:]


AC, BC, AD, BD = build_model()
B_EFF = BD[:, 0] + BD[:, 1]  # u_l = u_r = u


# ------------------------------------------------------------------ designs
def lqr_gain():
    """R_eff = 2r (both motors spend r u^2 on the diagonal input)."""
    b2d = B_EFF.reshape(-1, 1)
    P = solve_discrete_are(AD, b2d, np.diag(Q_DIAG), 2 * R_PRINT)
    r = 2 * R_PRINT + float(B_EFF @ P @ B_EFF)
    return ((B_EFF @ P @ AD) / r).ravel()


K_LQR = lqr_gain()


def ueq_from_surface(S):
    """u_eq(x) = -(S b)^-1 (S A) x, scale-invariant in S."""
    return -S @ AD / (S @ B_EFF)


UEQ_ADOPTED = ueq_from_surface(S_ADOPTED)


# ------------------------------------------------------------------ controllers
class PidController:
    """Eq. (34), one loop per output, loop outputs summed, E = -x.

    mode 'clamped': the shared saturated u(k-1) feeds the integrators
    (anti-windup). mode 'raw': per-loop unsaturated accumulators.
    """

    def __init__(self, mode="clamped"):
        self.mode = mode
        self.p_psi = [0.0, 0.0, 0.0]  # u(k-1), E(k-1), E(k-2)
        self.p_th = [0.0, 0.0, 0.0]
        self.last_u = 0.0

    @staticmethod
    def _c(gains):
        KP, KI, KD = gains
        return (KP + KI * T0 / 2 + KD / T0,
                -KP + KI * T0 / 2 - 2 * KD / T0,
                KD / T0)

    def __call__(self, x):
        e_psi, e_th = -x[1], -x[0]
        u = 0.0
        for gains, hist, e in ((PID_PSI, self.p_psi, e_psi),
                               (PID_TH, self.p_th, e_th)):
            c1, c2, c3 = self._c(gains)
            u += hist[0] + c1 * e + c2 * hist[1] + c3 * hist[2]
        self.last_u = u
        return u

    def advance(self, u_sat, x):
        """Update the loop histories; 'clamped' feeds the saturated output
        back into the integrators (anti-windup), 'raw' feeds the raw one."""
        u_fb = self.last_u if self.mode == "raw" else u_sat
        self.p_psi = [u_fb / 2, -x[1], self.p_psi[1]]
        self.p_th = [u_fb / 2, -x[0], self.p_th[1]]


class LqrController:
    def __call__(self, x):
        return float(-K_LQR @ x)


class SmcController:
    """u = u_eq + v; v per Eq. (64) boundary layer, v = clip(-s/L, +/-cte).

    Default u_eq follows the paper's Eq. (63) construction,
    u_eq = -(S b)^{-1} S A with the single-motor b, which equals
    -2 K_LQR for the adopted surface — the primary design per
    results/smc.json. The b-sum design-identity reference (u_eq = -K)
    is obtained by passing ueq_vec=-K_LQR explicitly.
    """

    def __init__(self, cte=1.0, L=4.0, surface=S_ADOPTED, ueq_vec=None):
        self.cte, self.L, self.S = cte, L, surface
        if ueq_vec is None:
            ueq_vec = -(surface @ AD) / (surface @ BD[:, 0])
        self.ueq_vec = ueq_vec

    def __call__(self, x):
        ueq = float(self.ueq_vec @ x)
        s = float(self.S @ x)
        return ueq + float(np.clip(-s / self.L, -self.cte, self.cte))


# ------------------------------------------------------------------ simulation
def simulate(controller, rail=None, kick_amp=None):
    """Discrete plant loop, ZOH control at 10 ms. rail=None -> unsaturated.

    kick_amp: one-sample additive kick on psi at t = 5 s (applied at the
    control sample, so the controller sees the kicked state).
    """
    x = X0.copy()
    xs = np.zeros((SIM_STEPS, 4))
    us = np.zeros(SIM_STEPS)
    for k in range(SIM_STEPS):
        if kick_amp is not None and k == KICK_IDX:
            x[1] += kick_amp
        u = controller(x)
        if rail:
            u = float(np.clip(u, -rail, rail))
        us[k] = u
        xs[k] = x
        if isinstance(controller, PidController):
            controller.advance(u, x)
        x = AD @ x + BD[:, 0] * u + BD[:, 1] * u
    return xs, us


def ise_psi(xs):
    return T0 * np.sum(xs[:, 1] ** 2) * 1e3


def ise_theta(xs):
    return T0 * np.sum(xs[:, 0] ** 2) * 1e3


def startup_energy(us, n=100):
    """E = T0 sum(ul^2+ur^2) over the first second; ul = ur = u."""
    return T0 * np.sum(2 * us[:n] ** 2)


def recovery_time(xs, kick_amp=0.05):
    """Time after the kick until psi stays inside the +/-2%-of-kick band
    (last exit from the band), per the team metric (B.2). Returns None if
    psi is still outside the band at the end of the window (no recovery)."""
    band = 0.02 * kick_amp
    outside = np.where(np.abs(xs[KICK_IDX:, 1]) > band)[0]
    last = xs.shape[0] - 1 - KICK_IDX
    if not len(outside):
        return 0.0
    if outside[-1] == last:
        return None
    return (outside[-1] + 1) * T0


def overshoot(xs):
    """Max |psi| strictly after the kick sample, per the team metric (B.2)."""
    return float(np.max(np.abs(xs[KICK_IDX + 1:, 1])))
