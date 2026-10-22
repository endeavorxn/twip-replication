"""Verify twip.model against the published quantities of Villacrés et al. (2016).

Checks (Eqs. 21-30):
  * A1, B1        recomputed from Table 1 vs printed Eqs. (21)-(22)
  * g investigation  Table 1 says g = 9.8; compare it against g = 9.81
  * Ac singularity   Eq. (24) needs Ac^-1, but Ac is singular -> augmented expm
  * Ad, Bd        ZOH discretization at T0 = 10 ms vs printed Eqs. (25)-(26),
                  cross-checked against scipy.signal.cont2discrete
  * poles         open-loop eigenvalues of Ad vs Eqs. (27)-(30)

Writes results/model_verification.json and exits nonzero on tolerance failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import cont2discrete

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twip.model import continuous_matrices, discretize, open_loop_poles  # noqa: E402
from twip.params import Params  # noqa: E402

# --- published values (Eqs. 21-22, 25-30) ---------------------------------
A1_PUB = np.array([[0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, -698.30, -416.80, 416.80],
                   [0, 139.96, 53.41, -53.41]])                       # Eq. (21)
B1_PUB = np.array([[0, 0], [0, 0], [405.11, 405.11], [-51.91, -51.91]])  # Eq. (22)
AD_PUB = np.array([[1, -0.010, 0.003, 0.007],
                   [0, 1.004, 0.0009, 0.0091],
                   [0, -1.122, 0.1207, 0.8690],
                   [0, 0.649, 0.1129, 0.8910]])                       # Eq. (25)
BD_PUB = np.array([[0.0068, 0.0068],
                   [-0.0009, -0.0009],
                   [0.8546, 0.8546],
                   [-0.1097, -0.1097]])                               # Eq. (26)
POLES_PUB = np.array([1.0, 0.0091, 1.0703, 0.9362])                   # Eqs. (27)-(30)

# --- tolerances ------------------------------------------------------------
TOL_A1B1 = 1.0     # entries are O(700) -> < 0.2 % relative
TOL_ADBD = 0.01
TOL_POLE = 0.005
TOL_C2D = 1e-9     # augmented expm vs scipy cont2discrete must agree to ~1e-12


def max_dev(C: np.ndarray, P: np.ndarray) -> float:
    return float(np.abs(np.asarray(C) - np.asarray(P)).max())


def show(name: str, C: np.ndarray, P: np.ndarray, tol: float) -> float:
    """Print computed vs published vs deviation entry-by-entry."""
    C, P = np.asarray(C), np.asarray(P)
    dev = C - P
    print(f"\n{name}  computed:")
    print(np.array2string(C, precision=6, suppress_small=False))
    print(f"{name}  published:")
    print(np.array2string(P, precision=4, suppress_small=False))
    print(f"{name}  deviation (computed - published):")
    print(np.array2string(dev, precision=6, suppress_small=False))
    d = float(np.abs(dev).max())
    status = "OK " if d <= tol else "FAIL"
    print(f"{name} max abs dev = {d:.6g}  (tol {tol:g})  [{status}]")
    return d


def main() -> int:
    results: dict = {"paper": "Villacrés et al. 2016, IJCSR Vol.1, Eqs. (21)-(30)"}
    failures: list[str] = []

    # --- 1. g investigation: Table 1 value vs the better-fitting 9.81 -----
    print("=" * 72)
    print("g investigation (Table 1 prints g = 9.8)")
    print("=" * 72)
    g_devs = {}
    for g in (9.8, 9.81):
        A1g, B1g = continuous_matrices(Params(g=g))
        da, db = max_dev(A1g, A1_PUB), max_dev(B1g, B1_PUB)
        g_devs[repr(g)] = {"max_abs_dev_A1_vs_eq21": da, "max_abs_dev_B1_vs_eq22": db}
        print(f"  g = {g:5.2f}:  max|A1 - Eq21| = {da:.4f}   max|B1 - Eq22| = {db:.4f}")
    g_best = min((9.8, 9.81), key=lambda g: g_devs[repr(g)]["max_abs_dev_A1_vs_eq21"])
    print(f"  -> g = {g_best} reproduces Eqs. (21)-(22) more closely; "
          f"library default Params(g={g_best}) (overridable).")

    # --- 2. continuous model with the selected g ---------------------------
    params = Params()  # default g selected above
    A1, B1 = continuous_matrices(params)
    print("\n" + "=" * 72)
    print(f"Continuous model, Params(g={params.g})   [Eqs. (6)-(20) recomputed]")
    print("=" * 72)
    dev_a1 = show("A1", A1, A1_PUB, TOL_A1B1)
    dev_b1 = show("B1", B1, B1_PUB, TOL_A1B1)
    if dev_a1 > TOL_A1B1:
        failures.append(f"A1 dev {dev_a1:.4g} > {TOL_A1B1}")
    if dev_b1 > TOL_A1B1:
        failures.append(f"B1 dev {dev_b1:.4g} > {TOL_A1B1}")

    # --- 3. singularity of Ac (Eq. 24 as printed is not evaluable) --------
    rank = int(np.linalg.matrix_rank(A1))
    cond = float(np.linalg.cond(A1))
    singular = bool(not np.isfinite(cond) or rank < A1.shape[0])
    print("\n" + "=" * 72)
    print("Ac singularity check (Eq. (24) requires Ac^-1)")
    print("=" * 72)
    print(f"  rank(Ac) = {rank} of {A1.shape[0]}, cond(Ac) = {cond:.3g} "
          f"-> Ac is {'SINGULAR' if singular else 'regular'}; "
          "Eq. (24) as written cannot be evaluated -> augmented expm used.")

    # --- 4. discretization at T0 = 10 ms -----------------------------------
    Ad, Bd = discretize(A1, B1, params.T0)
    print("\n" + "=" * 72)
    print(f"Discretization (augmented expm), T0 = {params.T0*1e3:.0f} ms   [Eqs. (23)-(24)]")
    print("=" * 72)
    dev_ad = show("Ad", Ad, AD_PUB, TOL_ADBD)
    dev_bd = show("Bd", Bd, BD_PUB, TOL_ADBD)
    if dev_ad > TOL_ADBD:
        failures.append(f"Ad dev {dev_ad:.4g} > {TOL_ADBD}")
    if dev_bd > TOL_ADBD:
        failures.append(f"Bd dev {dev_bd:.4g} > {TOL_ADBD}")

    Ad2, Bd2, *_ = cont2discrete((A1, B1, np.eye(4), np.zeros((4, 4))), params.T0, "zoh")
    dev_c2d = max(max_dev(Ad, Ad2), max_dev(Bd, Bd2))
    print(f"\nCross-check vs scipy.signal.cont2discrete: "
          f"max abs dev = {dev_c2d:.3g}  (tol {TOL_C2D:g})")
    if dev_c2d > TOL_C2D:
        failures.append(f"cont2discrete cross-check dev {dev_c2d:.3g} > {TOL_C2D}")

    # --- 5. open-loop poles -------------------------------------------------
    poles = open_loop_poles(Ad)
    print("\n" + "=" * 72)
    print("Open-loop discrete poles   [Eqs. (27)-(30)]")
    print("=" * 72)
    print("  computed  " + "  ".join(f"{p:+.5f}" for p in poles))
    print("  published " + "  ".join(f"{p:+.5f}" for p in sorted(POLES_PUB, reverse=True)))
    dev_poles = max(float(np.min(np.abs(p - POLES_PUB))) for p in poles)
    status = "OK " if dev_poles <= TOL_POLE else "FAIL"
    print(f"poles max abs dev = {dev_poles:.6g}  (tol {TOL_POLE:g})  [{status}]")
    if dev_poles > TOL_POLE:
        failures.append(f"pole dev {dev_poles:.4g} > {TOL_POLE}")

    # --- 6. machine-readable summary ----------------------------------------
    results.update({
        "g_investigation": {
            "table1_value": 9.8,
            "candidates": g_devs,
            "selected": g_best,
            "note": ("Table 1 prints g = 9.8 but Eqs. (21)-(22) are reproduced to "
                     "published rounding accuracy only with g = 9.81; 9.81 is the "
                     "Params default and can be overridden with Params(g=9.8)."),
        },
        "parameters": {"g": params.g, "m": params.m, "R": params.R, "M": params.M,
                       "W": params.W, "D": params.D, "H": params.H, "L": params.L,
                       "Jw": params.Jw, "Jpsi": params.Jpsi, "Jphi": params.Jphi,
                       "Jm": params.Jm, "Rm": params.Rm, "Kb": params.Kb,
                       "Kt": params.Kt, "n": params.n, "fm": params.fm, "fw": params.fw,
                       "T0": params.T0},
        "continuous": {"A1": A1.tolist(), "B1": B1.tolist(),
                       "max_abs_dev_A1_vs_eq21": dev_a1,
                       "max_abs_dev_B1_vs_eq22": dev_b1},
        "ac_singularity": {"rank": rank, "condition_number": ("inf" if singular else cond),
                           "singular": singular,
                           "method": ("Eq. (24) needs Ac^-1 but Ac has a zero first "
                                      "column; Bd taken from augmented expm "
                                      "expm([[Ac,Bc],[0,0]]*T0) = [[Ad,Bd],[0,I]].")},
        "discrete": {"T0": params.T0, "Ad": Ad.tolist(), "Bd": Bd.tolist(),
                     "max_abs_dev_Ad_vs_eq25": dev_ad,
                     "max_abs_dev_Bd_vs_eq26": dev_bd,
                     "cont2discrete_max_abs_dev": dev_c2d,
                     "discretization_method": "augmented-matrix expm (ZOH)"},
        "poles": {"computed": [float(p) for p in poles],
                  "published": POLES_PUB.tolist(),
                  "max_abs_dev": dev_poles},
        "tolerances": {"A1_B1": TOL_A1B1, "Ad_Bd": TOL_ADBD, "poles": TOL_POLE,
                       "cont2discrete": TOL_C2D},
        "all_passed": not failures,
        "failures": failures,
        "method_notes": ("All entries recomputed from Table 1 via Eqs. (6)-(20); "
                         "published matrices used only for verification. Bd by "
                         "augmented expm because Ac is singular."),
    })
    out = Path(__file__).resolve().parents[1] / "results" / "model_verification.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else "FAILED: " + "; ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
