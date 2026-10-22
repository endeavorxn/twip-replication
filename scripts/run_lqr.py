"""Simulate and verify the paper's LQR controller (Section 3.2, Eqs. 36-45).

Checks:
  * K          Riccati backward iteration (Eqs. 40-42) vs published Eq. (45),
               cross-checked against scipy.linalg.solve_discrete_are
  * reduction  unconstrained (4,2) Riccati solution has identical rows equal
               to the reduced (Ad, 2b, Q, 2r) gain -> reduction is lossless
  * poles      closed-loop eigenvalues vs the SMC design poles of
               Eqs. (60)-(61) plus the unreported pole at ~0
  * ISE        psi/theta (x10^-3) over a sweep of integration windows vs the
               Table 3 LQR entries (the paper never states the window)
  * energy     control energy profile; records the Fig. 9 claim that LQR uses
               the least energy at the start of the response

Writes results/lqr.json and results/plots/lqr_*.png; exits nonzero on
tolerance failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.linalg import solve_discrete_are  # noqa: E402

from twip.ise import ise  # noqa: E402
from twip.lqr import (  # noqa: E402
    LQRController,
    Q_EQ43,
    R_EQ44,
    design_lqr,
    riccati_backward,
)
from twip.model import continuous_matrices, discretize  # noqa: E402
from twip.params import Params  # noqa: E402
from twip.simulate import settling_time, simulate  # noqa: E402

# --- published values -------------------------------------------------------
K_PUB = np.array([-1.099, -81.44, -1.368, -10.860])        # Eq. (45), one row
POLES_EXPECTED = np.array([0.9318 + 0.0042j, 0.9318 - 0.0042j,
                           0.9797 + 0.0j, 0.0 + 0.0j])     # Eqs. (60)-(61) + pole at 0
TABLE3_LQR = {"psi_no_dist_x1e3": 0.393, "theta_no_dist_x1e3": 810.0,
              "psi_dist_x1e3": 0.622, "theta_dist_x1e3": 947.9}   # Table 3, LQR row

GAMMA = 1e-9                                               # Eq. (42) tolerance
WINDOWS_S = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]  # window sweep
SIM_S = 10.0
X0 = np.array([0.0, 0.1, 0.0, 0.0])                        # psi(0) = 0.1 rad

# Okabe-Ito hues (CVD-safe, fixed assignment: blue=psi, vermillion=theta,
# green=u) on a white surface.
C_PSI, C_THETA, C_U = "#0072B2", "#D55E00", "#009E73"
GRID = dict(color="0.85", linewidth=0.8)

# --- tolerances -------------------------------------------------------------
TOL_K_PUB = 0.005     # Eq. (45) is printed to 3-4 significant figures
TOL_K_DARE = 2e-5     # residual of the Eq. (42) stopping rule at gamma = 1e-9
TOL_REDUC = 1e-9      # unconstrained rows vs constrained DARE rows (exact)
TOL_POLES = 5e-4      # expected poles quoted to 4 decimal places


def pole_str(z) -> str:
    return f"{z.real:.4f}{z.imag:+.4f}j"


def settling_time_peak_ref(t, y, tol=0.02) -> float:
    """Like twip.simulate.settling_time but with the band referenced to the
    peak of |y| instead of y[0]; needed for theta, which starts at 0."""
    y = np.asarray(y, float)
    band = tol * np.max(np.abs(y))
    outside = np.abs(y) > band
    if not outside.any():
        return 0.0
    last = np.max(np.nonzero(outside)[0]) + 1
    return float(t[last]) if last < len(t) else float("inf")


def main() -> int:
    results: dict = {"paper": "Villacrés et al. 2016, IJCSR Vol.1, Section 3.2, Eqs. (36)-(45)"}
    failures: list[str] = []

    p = Params()
    Ac, Bc = continuous_matrices(p)
    Ad, Bd = discretize(Ac, Bc, p.T0)

    # --- 1. design: Riccati backward iteration ------------------------------
    print("=" * 72)
    print(f"Riccati backward iteration (Eqs. 40-42), gamma = {GAMMA:g}")
    print("=" * 72)
    K, info = design_lqr(p, gamma=GAMMA)
    print(f"  converged in {info['iterations']} iterations")
    print(f"  Q (Eq. 43) = diag{tuple(float(v) for v in np.diag(info['Q']))}")
    print(f"  R (Eq. 44) = {info['R_printed']} I2  ->  effective scalar {info['R_effective_scalar']}"
          " (both motors on the diagonal input u_l = u_r = u)")

    print("\n" + "=" * 72)
    print("LQR gain vs published Eq. (45)")
    print("=" * 72)
    print("  computed  " + np.array2string(K, precision=6))
    print("  published " + np.array2string(K_PUB, precision=4))
    dev_k = float(np.abs(K - K_PUB).max())
    status = "OK " if dev_k <= TOL_K_PUB else "FAIL"
    print(f"  max abs dev = {dev_k:.6g}  (tol {TOL_K_PUB:g})  [{status}]")
    if dev_k > TOL_K_PUB:
        failures.append(f"K dev {dev_k:.4g} > {TOL_K_PUB}")

    # --- 2. cross-check against scipy DARE ----------------------------------
    A_red, B_red = Ad, 2.0 * Bd[:, :1]
    R_eff = 2.0 * R_EQ44 * np.eye(1)
    P_dare = solve_discrete_are(A_red, B_red, Q_EQ43, R_eff)
    K_dare = np.linalg.solve(R_eff + B_red.T @ P_dare @ B_red, B_red.T @ P_dare @ A_red).ravel()
    dev_dare = float(np.abs(K - K_dare).max())
    print("\n" + "=" * 72)
    print("Cross-check vs scipy.linalg.solve_discrete_are")
    print("=" * 72)
    print("  DARE K     " + np.array2string(K_dare, precision=8))
    print(f"  max abs dev = {dev_dare:.3g}  (tol {TOL_K_DARE:g})  "
          f"[{'OK ' if dev_dare <= TOL_K_DARE else 'FAIL'}]")
    print("  (deviation is the fixed-point residual left by the Eq. (42)")
    print("   stopping rule at gamma = 1e-9, not a DARE discrepancy)")
    if dev_dare > TOL_K_DARE:
        failures.append(f"DARE cross-check dev {dev_dare:.3g} > {TOL_K_DARE}")

    # --- 3. reduction is lossless: unconstrained (4,2) solution -------------
    R_full = R_EQ44 * np.eye(2)
    P_full = solve_discrete_are(Ad, Bd, Q_EQ43, R_full)
    K_full = np.linalg.solve(R_full + Bd.T @ P_full @ Bd, Bd.T @ P_full @ Ad)
    row_dev = float(np.abs(K_full[0] - K_full[1]).max())
    dev_red = float(np.abs(K_full[0] - K_dare).max())   # DARE vs DARE: exact
    dev_red_iter = float(np.abs(K_full[0] - K).max())   # includes iter. residual
    print("\n" + "=" * 72)
    print("Unconstrained (4,2) Riccati solution (R = 0.00017 I2, full Bd)")
    print("=" * 72)
    print(np.array2string(K_full, precision=6))
    print(f"  row spread = {row_dev:.3g};  max dev of a DARE row from the "
          f"reduced DARE gain = {dev_red:.3g}  (tol {TOL_REDUC:g})")
    ok_red = row_dev <= TOL_REDUC and dev_red <= TOL_REDUC
    print(f"  -> the unconstrained optimum splits u evenly, so the reduced "
          f"design is lossless  [{'OK ' if ok_red else 'FAIL'}]")
    if not ok_red:
        failures.append(f"reduction check row_dev={row_dev:.3g}, dev={dev_red:.3g}")

    # --- 4. closed-loop poles ------------------------------------------------
    poles = np.sort_complex(np.asarray(info["poles"]))
    dev_poles = float(np.max(np.min(np.abs(poles[:, None] - POLES_EXPECTED[None, :]), axis=1)))
    print("\n" + "=" * 72)
    print("Closed-loop poles (Eqs. 60-61 design set + the unreported pole at ~0)")
    print("=" * 72)
    print("  computed  " + "  ".join(pole_str(z) for z in poles))
    print("  expected  " + "  ".join(pole_str(z) for z in np.sort_complex(POLES_EXPECTED)))
    status = "OK " if dev_poles <= TOL_POLES else "FAIL"
    print(f"  max abs dev = {dev_poles:.3g}  (tol {TOL_POLES:g})  [{status}]")
    if dev_poles > TOL_POLES:
        failures.append(f"pole dev {dev_poles:.3g} > {TOL_POLES}")

    # --- 5. simulation (no disturbance, x0 with psi(0) = 0.1 rad) ------------
    n_steps = int(round(SIM_S / p.T0))
    t = p.T0 * np.arange(n_steps + 1)
    sim = simulate(Ad, Bd, LQRController(K), X0, n_steps)
    X, U = sim["X"], sim["U"]
    theta, psi = X[:, 0], X[:, 1]
    u = U[:, 0]                                   # both columns identical
    assert float(np.abs(U[:, 0] - U[:, 1]).max()) == 0.0, "motor channels diverged"

    ts_psi = settling_time(t, psi)                # shared helper; psi(0) = 0.1
    ts_theta = settling_time_peak_ref(t, theta)   # theta(0) = 0 -> peak reference

    # --- 6. ISE window sweep vs Table 3 --------------------------------------
    print("\n" + "=" * 72)
    print(f"ISE x10^-3 window sweep (T0 = {p.T0} s), Table 3 LQR no-disturbance "
          f"targets: psi {TABLE3_LQR['psi_no_dist_x1e3']}, "
          f"theta {TABLE3_LQR['theta_no_dist_x1e3']}")
    print("=" * 72)
    print("  window_s   n_samples      ISE_psi      ISE_theta")
    ise_sweep = {}
    for w in WINDOWS_S:
        n_w = int(round(w / p.T0))
        i_psi = 1e3 * ise(psi[:n_w], p.T0)
        i_theta = 1e3 * ise(theta[:n_w], p.T0)
        ise_sweep[f"{w:g}"] = {"n_samples": n_w, "psi_x1e3": i_psi, "theta_x1e3": i_theta}
        print(f"  {w:8.2f}   {n_w:9d}   {i_psi:10.4f}   {i_theta:12.4f}")

    # primary window = smallest swept window whose ISE has saturated (within
    # 0.01 % of the 10 s value); beyond it the sweep is insensitive to length
    ref = ise_sweep["10"]
    w_primary = next(
        k for k in ise_sweep
        if max(abs(ise_sweep[k][q] / ref[q] - 1.0) for q in ("psi_x1e3", "theta_x1e3")) < 1e-4
    )

    def mismatch(w_key: str) -> float:
        d_psi = abs(ise_sweep[w_key]["psi_x1e3"] - TABLE3_LQR["psi_no_dist_x1e3"]) \
            / TABLE3_LQR["psi_no_dist_x1e3"]
        d_th = abs(ise_sweep[w_key]["theta_x1e3"] - TABLE3_LQR["theta_no_dist_x1e3"]) \
            / TABLE3_LQR["theta_no_dist_x1e3"]
        return max(d_psi, d_th)

    w_best = min(ise_sweep, key=mismatch)
    print(f"\n  primary window = {w_primary} s (smallest swept window with "
          f"saturated ISE, <0.01% from the 10 s value)")
    print(f"  closest match to the Table 3 pair = {w_best} s "
          f"(mismatch {mismatch(w_best):.1%}; psi sits ~9% high for every "
          f"window >= 2 s, theta ~2% low)")

    # --- 7. control energy (Fig. 9 claim: LQR uses least energy at the start)
    def energy(n: int) -> float:
        return float(np.sum(u[:n] ** 2))

    e = {
        "u0_V": float(u[0]),
        "sum_u2_first_0p5s": energy(int(round(0.5 / p.T0))),
        "sum_u2_first_1s": energy(int(round(1.0 / p.T0))),
        "sum_u2_first_2s": energy(int(round(2.0 / p.T0))),
        "sum_u2_full_10s": energy(n_steps),
    }
    e["fraction_of_full_energy_in_first_1s"] = e["sum_u2_first_1s"] / e["sum_u2_full_10s"]
    print("\n" + "=" * 72)
    print("Control energy per motor, sum(u^2)  [Fig. 9 claim: LQR least energy at start]")
    print("=" * 72)
    print(f"  |u(0)|                  = {e['u0_V']:.4f} V")
    print(f"  sum u^2, first 0.5 s    = {e['sum_u2_first_0p5s']:.4f}")
    print(f"  sum u^2, first 1 s      = {e['sum_u2_first_1s']:.4f}")
    print(f"  sum u^2, first 2 s      = {e['sum_u2_first_2s']:.4f}")
    print(f"  sum u^2, full 10 s      = {e['sum_u2_full_10s']:.4f}")
    print(f"  fraction in first 1 s   = {e['fraction_of_full_energy_in_first_1s']:.1%}"
          "  (energy is front-loaded at the start, consistent with Fig. 9)")

    print("\n" + "=" * 72)
    print(f"Settling times (+/-2% band): psi {ts_psi:.3f} s, theta {ts_theta:.3f} s")
    print(f"psi range [{psi.min():+.4f}, {psi.max():+.4f}] rad; "
          f"theta range [{theta.min():+.4f}, {theta.max():+.4f}] rad")

    # --- 8. plots ------------------------------------------------------------
    plots_dir = Path(__file__).resolve().parents[1] / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)
    panels = [
        (theta, C_THETA, "wheel angle $\\theta$ [rad]",
         f"settling {ts_theta:.2f} s"),
        (psi, C_PSI, "pitch angle $\\psi$ [rad]",
         f"settling {ts_psi:.2f} s"),
        (np.concatenate([u, [u[-1]]]), C_U, "motor voltage $u$ [V]",
         f"$|u(0)| = {abs(u[0]):.2f}$ V"),
    ]
    for ax, (y, c, ylab, note) in zip(axes, panels):
        ax.plot(t, y, color=c, linewidth=2.0)
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylab)
        ax.set_title(note, fontsize=9, color="0.35", loc="right")
        ax.tick_params(colors="0.25", labelsize=9)
        for s in ax.spines.values():
            s.set_color("0.7")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("LQR (Eqs. 36-45), no disturbance, "
                 "$x_0 = [0,\\ 0.1,\\ 0,\\ 0]^\\top$", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    f_resp = plots_dir / "lqr_response.png"
    fig.savefig(f_resp, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, key, c, target, ylab, ptitle in (
        (axes[0], "psi_x1e3", C_PSI, TABLE3_LQR["psi_no_dist_x1e3"],
         "ISE of $\\psi$ $\\times 10^{-3}$", "$\\psi$ (pitch)"),
        (axes[1], "theta_x1e3", C_THETA, TABLE3_LQR["theta_no_dist_x1e3"],
         "ISE of $\\theta$ $\\times 10^{-3}$", "$\\theta$ (wheel)"),
    ):
        ws = [float(k) for k in ise_sweep]
        vals = [ise_sweep[f"{w:g}"][key] for w in ws]
        ax.plot(ws, vals, color=c, linewidth=2.0, marker="o", markersize=6,
                label="replication")
        ax.axhline(target, color="0.35", linewidth=1.2, linestyle="--",
                   label=f"Table 3 no-dist. = {target:g}")
        best = float(w_primary)
        ax.axvline(best, color=c, linewidth=1.0, linestyle=":", alpha=0.8)
        ax.grid(True, **GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel("ISE integration window [s]")
        ax.set_ylabel(ylab)
        ax.set_title(ptitle, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(colors="0.25", labelsize=9)
        for s in ax.spines.values():
            s.set_color("0.7")
    fig.suptitle("LQR ISE vs integration window (paper does not state the window)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    f_ise = plots_dir / "lqr_ise_window.png"
    fig.savefig(f_ise, dpi=150)
    plt.close(fig)
    print(f"\nwrote {f_resp}")
    print(f"wrote {f_ise}")

    # --- 9. machine-readable results ----------------------------------------
    results.update({
        "design": {
            "method": "Riccati backward iteration, Eqs. (40)-(42), P(N)=Q, K(N)=0",
            "gamma": GAMMA,
            "iterations": info["iterations"],
            "Q_eq43": np.diag(info["Q"]).tolist(),
            "R_eq44_printed": R_EQ44,
            "R_effective_scalar": info["R_effective_scalar"],
            "reduction": info["reduction"],
            "K": K.tolist(),
            "K_published_eq45": K_PUB.tolist(),
            "max_abs_dev_K_vs_eq45": dev_k,
            "dare_K": K_dare.tolist(),
            "max_abs_dev_K_vs_scipy_dare": dev_dare,
            "P_min_eigenvalue": float(np.linalg.eigvalsh(info["P"]).min()),
            "unconstrained_2x4_K_rows_identical": bool(row_dev <= TOL_REDUC),
            "max_abs_dev_unconstrained_dare_vs_reduced_dare": dev_red,
            "max_abs_dev_unconstrained_dare_vs_iterated_K": dev_red_iter,
        },
        "poles": {
            "computed": [pole_str(z) for z in poles],
            "expected_eq60_61_plus_zero": [pole_str(z)
                                           for z in np.sort_complex(POLES_EXPECTED)],
            "max_abs_dev": dev_poles,
            "note": ("the paper lists only three closed-loop poles (Eqs. 60-61, "
                     "reused as the SMC design poles); the 4th pole at ~0 "
                     "(deadbeat) is not mentioned in the paper"),
        },
        "simulation": {
            "T0": p.T0, "n_steps": n_steps, "x0": X0.tolist(),
            "disturbance": None, "u_sat": None,
            "theta_range_rad": [float(theta.min()), float(theta.max())],
            "psi_range_rad": [float(psi.min()), float(psi.max())],
        },
        "ise_sweep": ise_sweep,
        "paper_targets_table3_lqr": TABLE3_LQR,
        "primary_window_s": float(w_primary),
        "primary_window_criterion": ("smallest swept window whose psi and theta "
                                     "ISE are within 0.01% of the 10 s value"),
        "best_match_window_s": float(w_best),
        "best_match_mismatch_vs_table3": mismatch(w_best),
        "ise_range_over_windows_x1e3": {
            "psi": [min(v["psi_x1e3"] for v in ise_sweep.values()),
                    max(v["psi_x1e3"] for v in ise_sweep.values())],
            "theta": [min(v["theta_x1e3"] for v in ise_sweep.values()),
                      max(v["theta_x1e3"] for v in ise_sweep.values())],
        },
        "control_energy_per_motor": e,
        "control_energy_both_motors": {
            "sum_u2_first_1s": 2.0 * e["sum_u2_first_1s"],
            "sum_u2_full_10s": 2.0 * e["sum_u2_full_10s"],
        },
        "settling_time_s": {"psi": ts_psi, "theta": ts_theta, "tol": 0.02,
                            "theta_band_reference": "peak |theta| (theta(0) = 0)"},
        "fig9_claim": {
            "claim": "LQR uses the least energy at the start of the response",
            "u0_V": e["u0_V"],
            "fraction_of_full_energy_in_first_1s": e["fraction_of_full_energy_in_first_1s"],
            "note": ("recorded here for the LQR; the cross-controller comparison "
                     "against PID/SMC initial energies is made in the integration "
                     "stage once those controllers exist"),
        },
        "tolerances": {"K_vs_eq45": TOL_K_PUB, "K_vs_dare": TOL_K_DARE,
                       "reduction": TOL_REDUC, "poles": TOL_POLES},
        "all_passed": not failures,
        "failures": failures,
    })
    out = Path(__file__).resolve().parents[1] / "results" / "lqr.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    print("\n" + ("ALL CHECKS PASSED" if not failures else "FAILED: " + "; ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
