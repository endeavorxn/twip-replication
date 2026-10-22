"""Table 1 physical parameters of the TWIP (Villacrés et al. 2016).

All quantities SI. The paper's Table 1 lists g = 9.8 m/s^2, but the published
matrices of Eqs. (21)-(22) are reproduced an order of magnitude better with
g = 9.81 (see scripts/verify_model.py), so 9.81 is the library default; pass
``Params(g=9.8)`` to use the literal Table 1 value.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """TWIP physical constants, Table 1 of Villacrés et al. (2016)."""

    g: float = 9.81          # gravity [m/s^2]; Table 1 prints 9.8, see module docstring
    m: float = 0.03          # wheel mass [kg]
    R: float = 0.021         # wheel radius [m]
    M: float = 0.6           # body mass [kg]
    W: float = 0.09          # body width [m]
    D: float = 0.05          # body depth [m]
    H: float = 0.26          # body height [m]
    L: float = 0.13          # CoM distance from wheel axle [m] = H/2
    Jm: float = 1e-5         # DC motor rotor inertia [kg m^2]
    Rm: float = 6.69         # DC motor resistance [ohm]
    Kb: float = 0.468        # back-e.m.f. constant [V s/rad]
    Kt: float = 0.317        # torque constant [N m/A]
    n: float = 1.0           # gear ratio
    fm: float = 0.0022       # body-DC motor friction coefficient
    fw: float = 0.0          # body-surface friction coefficient

    # Derived inertias (Table 1 formulas)
    @property
    def Jw(self) -> float:
        """Wheel inertia moment Jw = m*R^2/2 [kg m^2]."""
        return self.m * self.R**2 / 2

    @property
    def Jpsi(self) -> float:
        """Body pitch inertia Jpsi = M*L^2/3 [kg m^2]."""
        return self.M * self.L**2 / 3

    @property
    def Jphi(self) -> float:
        """Body yaw inertia Jphi = M*(W^2 + D^2)/12 [kg m^2] (unused in the pitch model)."""
        return self.M * (self.W**2 + self.D**2) / 12

    @property
    def T0(self) -> float:
        """Sampling time T0 = 10 ms used throughout the paper."""
        return 0.01
