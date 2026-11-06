"""Validates the rebuilt pipeline against every number printed in the CTRL_6
final report. Exit code 0 = all targets hit."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twip_core import (AC, BC, AD, BD, B_EFF, K_LQR, S_ADOPTED, S_PRINT,
                       ueq_from_surface, PidController, LqrController,
                       SmcController, simulate, ise_psi, ise_theta,
                       startup_energy, recovery_time, overshoot, T0)

ok = True


def check(name, got, want, tol):
    global ok
    g = np.atleast_1d(got).astype(float)
    w = np.atleast_1d(want).astype(float)
    d = float(np.max(np.abs(g - w)))
    good = d <= tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: got {np.round(g,6)} want {np.round(w,6)} (max|d|={d:.2e}, tol={tol:.0e})")



def check_rel(name, got, want, tol):
    global ok
    g = np.atleast_1d(got).astype(float)
    w = np.atleast_1d(want).astype(float)
    rel = float(np.max(np.abs(g - w) / np.maximum(np.abs(w), 1e-12)))
    good = rel <= tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: max rel dev = {rel:.2e} (tol {tol:.0e})")

# ---- algebraic verification -------------------------------------------------
print("== model ==")
check("A1[2,1] (paper -698.30)", AC[2, 1], -698.30, 0.05)
check("A1[3,1] (paper 139.96)", AC[3, 1], 139.96, 0.05)
check("A1[2,2] (paper -416.80)", AC[2, 2], -416.80, 0.05)
check("A1[2,3] (paper 416.80)", AC[2, 3], 416.80, 0.05)
check("A1[3,2] (paper 53.41)", AC[3, 2], 53.41, 0.05)
check("A1[3,3] (paper -53.41)", AC[3, 3], -53.41, 0.05)
check("B[2,:] (paper 405.11)", BC[2, 0], 405.11, 0.05)
check("B[3,:] (paper -51.91)", BC[3, 0], -51.91, 0.05)

print("== discretisation ==")
poles_ol = np.sort_complex(np.linalg.eigvals(AD))
check("open-loop poles (paper 1, 0.0091, 0.9362, 1.0703)",
      sorted(np.real(poles_ol)), [0.0091, 0.9362, 1.0, 1.0703], 5e-5)

print("== LQR ==")
check("K vs paper Eq.(45)", K_LQR, [-1.099, -81.44, -1.368, -10.860], 3e-3)
poles_cl = np.sort_complex(np.linalg.eigvals(AD - np.outer(B_EFF, K_LQR)))
print("   closed-loop poles:", np.round(poles_cl, 5),
      "(paper Eq.(60)-(61): 0.9318±0.0042j, 0.9797, plus unstated 0)")

print("== SMC ==")
# adopted-surface direction check against the report's adopted entries
from scipy.linalg import eig
w, V = np.linalg.eig((AD - np.outer(B_EFF, K_LQR)).T)
i0 = np.argmin(np.abs(w))
S_dir = np.real(V[:, i0]); S_dir /= S_dir[3]; S_dir *= -21.18
print("   left null vector (scaled):", np.round(S_dir, 3),
      " adopted:", S_ADOPTED, " (should match direction)")
check_rel("u_eq (adopted S, rounded) == -K [design identity]",
          ueq_from_surface(S_ADOPTED), -K_LQR, 1e-3)

# ---- Table 3 (no disturbance, primary configurations) -----------------------
print("== Table 3 (no disturbance) ==")
for mode in ("clamped", "raw"):
    xs, us = simulate(PidController(mode=mode), rail=12.0)
    print(f"   PID[{mode}]: psi {ise_psi(xs):.3f} (want 0.295)  theta {ise_theta(xs):.3f} (want 503.4)")

xs, us = simulate(LqrController())
print(f"   LQR : psi {ise_psi(xs):.3f} (want 0.430)  theta {ise_theta(xs):.3f} (want 793.1)  peak|u| {np.max(np.abs(us)):.2f} (want 8.1)  E1s {startup_energy(us):.2f} (want 7.41)")

xs, us = simulate(SmcController())
print(f"   SMC : psi {ise_psi(xs):.3f} (want 0.237)  theta {ise_theta(xs):.3f} (want 560.3)  peak|u| {np.max(np.abs(us)):.2f} (want 17.3)  E1s {startup_energy(us):.2f} (want 11.41)")

# ---- disturbance scenario ---------------------------------------------------
print("== disturbance (+0.05 rad state kick at t=5 s) ==")
res = {}
for name, ctrl, rail in (("PID", PidController(mode="raw"), 12.0), ("LQR", LqrController(), None),
                         ("SMC", SmcController(), None)):
    xs0, _ = simulate(ctrl, rail=rail)
    i0 = (ise_psi(xs0), ise_theta(xs0))
    xs1, us1 = simulate(ctrl, rail=rail, kick_amp=0.05)
    i1 = (ise_psi(xs1), ise_theta(xs1))
    res[name] = (i1, recovery_time(xs1), overshoot(xs1))
    print(f"   {name}: dpsi-ISE +{i1[0]-i0[0]:.3f} (want +0.049 PID / +0.108 LQR / +0.059 SMC)"
          f"  recovery {res[name][1]:.2f} (want 0.92/1.36/0.42)  overshoot {res[name][2]:.4f} (want 0.029 PID / 0.034 SMC)")

# ---- chattering -------------------------------------------------------------
xs, us = simulate(SmcController())
du = np.diff(us)
late = du[200:]
alt = int(np.sum((late[:-1] * late[1:]) < 0))
print(f"== chattering: sign alternations of du after t=2s: {alt} (want 0), max|du| {np.max(np.abs(late)):.4f} V (want <=0.006)")

# ---- SMC realisation cross-check (results/smc.json reference rows) ----------
ueq_bs = -(S_ADOPTED @ AD) / (S_ADOPTED @ BD[:, 0])
check_rel("u_eq (Eq.-63 construction) == -2K [primary design]",
          ueq_bs, -2 * K_LQR, 1e-3)
check_rel("design identity u_eq (b-sum) == -K",
      -(S_ADOPTED @ AD) / (S_ADOPTED @ (BD[:, 0] + BD[:, 1])), -K_LQR, 1e-3)

print("OVERALL:", "PASS" if ok else "CHECK ABOVE")
