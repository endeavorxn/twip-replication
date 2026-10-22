"""Digital PID control of the TWIP (paper Section 3.1, Eqs. 31-35, Table 2).

The paper regulates x_ref = 0 (Eq. 35) with two independent loops whose
errors are E_psi = -psi and E_theta = -theta, gains from Table 2
(psi loop KP = -77.97, KI = -0.01, KD = -8.79; theta loop KP = -1.07,
KI = -0.01, KD = -1.36; tuned by trial and error).  Both loop outputs are
summed into a single drive signal that is applied to both motors
(u_left = u_right = u_psi + u_theta): the paper never states the wiring,
but Bd's two columns are identical (verified) so a common-mode drive is
the only structure the published model can distinguish, and this wiring is
the only sign convention of Table 2 that closes stably (see
scripts/run_pid.py, which reports the alternatives and their closed-loop
spectra).
"""
from __future__ import annotations

from .params import Params


class PIDLoop:
    """One incremental discrete PID loop, Eq. (34).

    Eq. (31) is the parallel continuous form; Eqs. (32)-(33) discretize it
    with a trapezoidal (Tustin) integral and a first-difference derivative,
    which the paper rearranges into the incremental form of Eq. (34):

        u(k) = u(k-1)
             + E(k)  * (KP + KI*T0/2 +  KD/T0)
             + E(k-1)* (-KP + KI*T0/2 - 2*KD/T0)
             + E(k-2)* (KD/T0)

    The instance stores the accumulated output and the two previous errors,
    so ``update`` must be called exactly once per sampling instant, in time
    order (the simulator guarantees this).

    Parameters
    ----------
    kp, ki, kd : float.  Loop gains, Table 2 (negative for both loops).
    T0 : float.  Sampling period [s], 10 ms in the paper.
    """

    def __init__(self, kp: float, ki: float, kd: float, T0: float):
        self.kp, self.ki, self.kd, self.T0 = kp, ki, kd, T0
        # Eq. (34) coefficients
        self.a0 = kp + ki * T0 / 2 + kd / T0
        self.a1 = -kp + ki * T0 / 2 - 2 * kd / T0
        self.a2 = kd / T0
        self._e1 = 0.0   # E(k-1)
        self._e2 = 0.0   # E(k-2)
        self._u = 0.0    # u(k-1), accumulated output

    def update(self, e: float) -> float:
        """Advance the recursion one sample with error ``e`` = E(k)."""
        u = self._u + self.a0 * e + self.a1 * self._e1 + self.a2 * self._e2
        self._e2, self._e1, self._u = self._e1, e, u
        return u

    def reset(self) -> None:
        """Zero the accumulator and error history."""
        self._e1 = self._e2 = self._u = 0.0


def make_pid_controller(
    psi_pid: PIDLoop, theta_pid: PIDLoop
):
    """Combine two PIDLoops into a controller(x, k) -> [u, u] closure.

    Wiring assumption (documented, paper silent): each loop sees its own
    error E = x_ref - x = -x (Eq. 35, x_ref = 0), and the two loop outputs
    are summed and applied to both motors, u_left = u_right = u_psi+u_theta.
    The closure holds internal PID states; call it once per step in order.
    """

    def controller(x, k):
        u = psi_pid.update(-x[1]) + theta_pid.update(-x[0])
        return [u, u]

    return controller


def table2_pid(params: Params | None = None) -> tuple[PIDLoop, PIDLoop]:
    """Both Table 2 loops with the paper's published gains.

    Returns (psi_pid, theta_pid).
    """
    T0 = (Params() if params is None else params).T0
    psi_pid = PIDLoop(kp=-77.97, ki=-0.01, kd=-8.79, T0=T0)   # Table 2, psi loop
    theta_pid = PIDLoop(kp=-1.07, ki=-0.01, kd=-1.36, T0=T0)  # Table 2, theta loop
    return psi_pid, theta_pid
