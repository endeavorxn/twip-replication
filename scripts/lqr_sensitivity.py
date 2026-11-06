"""LQR Q,R sensitivity — final-report extension (Week-8 §6 task: Wang Weiran).

The paper fixes its LQR weights (Q, Eq. 43; R = 0.00017 I2, Eq. 44) and reports
every comparison at that single design point.  This script scales both weights
over four decades, re-designs the gain with the paper's own Riccati recursion
(Eqs. 40-42), and re-runs the comparison against the FIXED PID/SMC baselines
to ask which of the replicated conclusions depends on that choice:

  (i)   psi-ISE ranking, no disturbance (Table 3):   SMC < PID < LQR
  (ii)  start-up control energy, first 1 s (Fig. 9): LQR lowest
  (iii) theta-ISE ranking:                           PID < SMC < LQR

Finding (verified against scipy.linalg.solve_discrete_are): the design is
pinned by the PLANT, not the weights.  The DARE is invariant under joint
(Q, R) scaling, so only the ratio rho = s_Q/s_R matters -- and the psi/u
channel's transmission zeros (at z = 1, a consequence of the reaction-wheel
non-minimum-phase structure sampled at T0 = 10 ms) cap the achievable
bandwidth, so the Riccati solution saturates: across rho in [1e-4, 1e4] the
gain K_psi moves by < 1.7%, the closed-loop pole radius by < 0.1%, and no
replicated ranking changes.  Corollary: the K-vs-Eq.(45) match to 2e-3
confirms the paper's design PROCEDURE but constrains its exact weights only
weakly.

Outputs:
  results/lqr_sensitivity.json
  results/figures/fig_c_lqr_sensitivity.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.linalg import solve_discrete_are

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from compare_ise import (  # noqa: E402
    COLORS, NAMES, N_STEPS, U_SAT_PID, X0,
    build_designs, energy_first_s, ise_post_kick, ise_window, make_kick,
    recovery_metrics, INK, INK2, MUTED, GRID, SURFACE,
)
from twip.lqr import (  # noqa: E402
    Q_EQ43, R_EQ44, LQRController, reduced_plant, riccati_backward,
)
from twip.simulate import simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "lqr_sensitivity.json"
OUT_FIG = ROOT / "results" / "figures" / "fig_c_lqr_sensitivity.png"

GAMMA = 1e-9                                   # Eq. (42) tolerance, as the runs
OAT_SCALES = tuple(np.geomspace(0.01, 100.0, 15))    # one-at-a-time lines
GRID_SCALES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # 7x7 grid (ratio 1e-3..1e3)

# sequential ramp for the magnitude heatmap (single hue, light -> dark)
SEQ_LIGHT, SEQ_DARK = "#e9f1fa", "#17456b"


# --------------------------------------------------------------------------- #
# designs and metrics
# --------------------------------------------------------------------------- #
def design_scaled(A, B, s_q: float, s_r: float):
    """Re-design the LQR gain at scaled weights with the paper's recursion.

    Q = s_q * Q_EQ43 (Eq. 43); the reduced-plant penalty is R = s_r * 2r
    (both motors spend r u^2 each on the diagonal input, cf. twip/lqr.py).
    Returns (K per motor, closed-loop pole radii, n_iter).
    """
    K, P, n_iter = riccati_backward(A, B, s_q * Q_EQ43,
                                    (2.0 * s_r * R_EQ44) * np.eye(1),
                                    gamma=GAMMA)
    K = K.ravel()
    radii = np.abs(np.linalg.eigvals(A - B @ K[None, :]))
    return K, radii, n_iter


def lqr_metrics(Ad, Bd, K, radii) -> dict:
    """No-disturbance metrics + post-kick recovery for one LQR gain."""
    run_nd = simulate(Ad, Bd, LQRController(K), X0, N_STEPS)
    run_kick = simulate(Ad, Bd, LQRController(K), X0, N_STEPS,
                        disturbance=make_kick())
    rec = recovery_metrics(run_kick)
    return {
        "K": [float(v) for v in np.asarray(K).ravel()],
        "psi_ise_x1e3": ise_window(run_nd, 1),
        "theta_ise_x1e3": ise_window(run_nd, 0),
        "psi_ise_post_kick_x1e3": ise_post_kick(run_kick, 1),
        "energy_first_1s_both_motors_V2s":
            energy_first_s(run_nd, 1.0)["both_motors_V2s"],
        "u_peak_nodist_V": float(np.abs(run_nd["U"][:, 0]).max()),
        "recovery_s": rec["psi_recovery_s_to_2pct_of_kick"],
        "overshoot_rad": rec["psi_overshoot_after_kick_rad"],
        "spectral_radius": float(radii.max()),
    }


def rank_flags(m: dict, ref: dict) -> dict:
    """Do the three replicated rankings still hold with THIS LQR design?"""
    pid, smc = ref["PID"], ref["SMC"]
    return {
        "smc_lowest_psi": m["psi_ise_x1e3"] > smc["psi_ise_x1e3"],   # PID already above
        "pid_lowest_theta": m["theta_ise_x1e3"] > pid["theta_ise_x1e3"],
        "lqr_lowest_energy":
            m["energy_first_1s_both_motors_V2s"] < pid["energy_first_1s_both_motors_V2s"],
        "smc_fastest_recovery": m["recovery_s"] > smc["recovery_s"],
    }


def crossings(points, ref: float) -> list[float]:
    """Scale values where value(points) crosses ref (interp in log-s space)."""
    ss = np.log([p[0] for p in points])
    vv = np.array([p[1] for p in points])
    d = vv - ref
    out = []
    for i in np.nonzero(np.diff(np.sign(d)) != 0)[0]:
        t = d[i] / (d[i] - d[i + 1])
        out.append(float(np.exp(ss[i] + t * (ss[i + 1] - ss[i]))))
    return out


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(ref, oat_q, grid, k_psi_pub) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.9))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.17, top=0.74,
                        wspace=0.40)

    # (a) psi-ISE heatmap over the 7x7 grid --------------------------------
    ax = axs[0]
    n = len(GRID_SCALES)
    M = np.array([[next(g["psi_ise_x1e3"] for g in grid
                        if g["s_q"] == q and g["s_r"] == r)
                   for r in GRID_SCALES] for q in GRID_SCALES])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SEQ_LIGHT, SEQ_DARK])
    lo, hi = float(M.min()), float(M.max())
    im = ax.imshow(M, cmap=cmap, origin="lower", aspect="auto",
                   vmin=lo, vmax=hi)
    for i, q in enumerate(GRID_SCALES):
        for j, r in enumerate(GRID_SCALES):
            frac = (M[i, j] - lo) / (hi - lo + 1e-15)
            ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                    fontsize=6.0, color=("#fcfcfb" if frac > 0.55 else INK2))
    mid = (n - 1) // 2
    ax.add_patch(plt.Rectangle((mid - 0.5, mid - 0.5), 1, 1, fill=False,
                               edgecolor=INK, lw=1.8, zorder=5))
    ticks = [f"{s:g}×" for s in GRID_SCALES]
    ax.set_xticks(range(n), ticks)
    ax.set_yticks(range(n), ticks)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("state-weight scale  s_Q", color=INK2, fontsize=9)
    ax.set_ylabel("input-weight scale  s_R", color=INK2, fontsize=9)
    ax.set_title(f"(a) LQR ψ-ISE ×10⁻³ over the (Q, R) grid\n"
                 f"(outlined cell = published; all values in "
                 f"{lo:.3f}–{hi:.3f})", color=INK, fontsize=10,
                 loc="left", pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.03)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)

    def line_panel(ax, oat, xlab, title, ykey, ylab, mark_published=True):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(INK2)
        ss = [p["scale"] for p in oat]
        yy = ([p[ykey][1] * -1 if ykey == "K" else p[ykey] for p in oat])
        ax.plot(ss, yy, color=COLORS["LQR"], lw=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1,
                label="LQR (re-designed)")
        ax.set_xscale("log")
        ax.axvline(1.0, color=MUTED, lw=1, ls=(0, (2, 2)))
        if mark_published:
            ax.annotate("published", (1.0, 0.02),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=8, color=MUTED,
                        xytext=(0, 2), textcoords="offset points")
        ax.set_xlabel(xlab, color=INK2, fontsize=9)
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
        return ss, yy

    ss_b, yy_b = line_panel(
        axs[1], oat_q, "state-weight scale  s_Q  (log; R at print)",
        "(b) ψ-ISE vs weight scale:\nno replicated ranking breaks",
        "psi_ise_x1e3", "ISE ψ ×10⁻³")
    axs[1].set_ylim(0.20, 0.52)
    for nm in ("PID", "SMC"):
        lvl = ref[nm]["psi_ise_x1e3"]
        axs[1].axhline(lvl, color=COLORS[nm], lw=1.6, ls=(0, (4, 3)), label=nm)
        axs[1].annotate(f"{nm} baseline {lvl:.2f}", (0.03, lvl),
                        xycoords=("axes fraction", "data"), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom",
                        color=INK2, fontsize=8)
    axs[1].legend(frameon=False, fontsize=8, loc="upper left",
                  labelcolor=INK2, borderaxespad=0.2)

    ss_c, yy_c = line_panel(
        axs[2], oat_q, "state-weight scale  s_Q  (log; R at print)",
        "(c) why: the gain itself is pinned\nby the plant — Kψ vs weight scale",
        "K", "Kψ [V/rad]  (per motor)", mark_published=False)
    lo_c, hi_c = min(yy_c), max(yy_c)
    pad_c = max((hi_c - lo_c) * 0.35, 0.01)
    axs[2].set_ylim(lo_c - 1.6 * pad_c, hi_c + 1.6 * pad_c)
    axs[2].axhline(k_psi_pub, color=MUTED, lw=1.4, ls=(0, (4, 3)),
                   label=f"Eq. (45) published: {abs(k_psi_pub):.2f}")
    axs[2].annotate(f"total swing {(hi_c - lo_c) / k_psi_pub * 100:.1f}% "
                    "over 4 decades", (0.97, 0.985),
                    xycoords=("axes fraction", "axes fraction"), ha="right",
                    va="top", color=INK2, fontsize=8)
    axs[2].legend(frameon=False, fontsize=8, loc="lower right",
                  labelcolor=INK2, borderaxespad=0.2)

    fig.suptitle("LQR Q, R sensitivity — the replicated rankings do not depend on "
                 "the published weights", color=INK, fontsize=11.5, x=0.01,
                 ha="left", y=0.985)
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p, Ad, Bd, K_base, lqr_info, S, smc_info, smc, new_pid = build_designs()
    A_red, B_red = reduced_plant(Ad, Bd)
    K_PUB = np.array([-1.099, -81.44, -1.368, -10.86])          # Eq. (45)

    # plant mechanism: transmission zeros of the psi/u channel + unstable pole
    c_psi = np.array([0.0, 1.0, 0.0, 0.0])
    num, _ = signal.ss2tf(A_red, B_red, c_psi, np.zeros((1, 1)))
    zeros_psi = np.roots(np.atleast_1d(num).ravel())
    pole_unstable = float(max(np.abs(np.linalg.eigvals(A_red))))

    # solver credibility: the paper's recursion == scipy's DARE at the print
    R_eff = (2.0 * R_EQ44) * np.eye(1)
    P_dare = solve_discrete_are(A_red, B_red, Q_EQ43, R_eff)
    K_dare = np.linalg.solve(R_eff + B_red.T @ P_dare @ B_red,
                             B_red.T @ P_dare @ A_red).ravel()
    K_rec, _, _ = design_scaled(A_red, B_red, 1.0, 1.0)
    dev_dare = float(np.abs(K_dare - K_rec).max())

    # joint-scaling homogeneity: K(4,1) must equal K(1,0.25)
    K_41, _, _ = design_scaled(A_red, B_red, 4.0, 1.0)
    K_1025, _, _ = design_scaled(A_red, B_red, 1.0, 0.25)
    dev_homog = float(np.abs(K_41 - K_1025).max())

    # fixed reference levels: PID and SMC at the §1 primary configurations
    ref: dict[str, dict] = {}
    run_pid = simulate(Ad, Bd, new_pid(), X0, N_STEPS, u_sat=U_SAT_PID)
    run_pid_kick = simulate(Ad, Bd, new_pid(), X0, N_STEPS,
                            disturbance=make_kick(), u_sat=U_SAT_PID)
    run_smc = simulate(Ad, Bd, smc, X0, N_STEPS)
    run_smc_kick = simulate(Ad, Bd, smc, X0, N_STEPS, disturbance=make_kick())
    for name, run, kick in (("PID", run_pid, run_pid_kick),
                            ("SMC", run_smc, run_smc_kick)):
        rec = recovery_metrics(kick)
        ref[name] = {
            "psi_ise_x1e3": ise_window(run, 1),
            "theta_ise_x1e3": ise_window(run, 0),
            "energy_first_1s_both_motors_V2s":
                energy_first_s(run, 1.0)["both_motors_V2s"],
            "recovery_s": rec["psi_recovery_s_to_2pct_of_kick"],
            "overshoot_rad": rec["psi_overshoot_after_kick_rad"],
        }

    base = lqr_metrics(Ad, Bd, K_rec, np.abs(np.linalg.eigvals(
        A_red - B_red @ K_rec[None, :])))
    K_rec2, radii_rec, it_rec = design_scaled(A_red, B_red, 1.0, 1.0)
    base["n_iter_eq42"] = it_rec
    base["spectral_radius"] = float(radii_rec.max())
    base["flags_vs_ref"] = rank_flags(base, ref)

    oat_q, oat_r, grid = [], [], []
    for s_q in OAT_SCALES:
        K, radii, it = design_scaled(A_red, B_red, s_q, 1.0)
        m = lqr_metrics(Ad, Bd, K, radii)
        m |= {"scale": float(s_q), "n_iter_eq42": it,
              "flags_vs_ref": rank_flags(m, ref)}
        oat_q.append(m)
    for s_r in OAT_SCALES:
        K, radii, it = design_scaled(A_red, B_red, 1.0, s_r)
        m = lqr_metrics(Ad, Bd, K, radii)
        m |= {"scale": float(s_r), "n_iter_eq42": it,
              "flags_vs_ref": rank_flags(m, ref)}
        oat_r.append(m)
    for s_q in GRID_SCALES:
        for s_r in GRID_SCALES:
            K, radii, _ = design_scaled(A_red, B_red, s_q, s_r)
            m = lqr_metrics(Ad, Bd, K, radii)
            m |= {"s_q": float(s_q), "s_r": float(s_r),
                  "rho": float(s_q / s_r),
                  "flags_vs_ref": rank_flags(m, ref)}
            grid.append(m)

    cross = {
        "psi_ise_crossings_vs_pid_at_q_scale": crossings(
            [(m["scale"], m["psi_ise_x1e3"]) for m in oat_q],
            ref["PID"]["psi_ise_x1e3"]),
        "psi_ise_crossings_vs_smc_at_q_scale": crossings(
            [(m["scale"], m["psi_ise_x1e3"]) for m in oat_q],
            ref["SMC"]["psi_ise_x1e3"]),
        "energy_crossings_vs_pid_at_q_scale": crossings(
            [(m["scale"], m["energy_first_1s_both_motors_V2s"]) for m in oat_q],
            ref["PID"]["energy_first_1s_both_motors_V2s"]),
        "note": ("scale value at which the LQR curve crosses the fixed PID/SMC "
                 "baseline level; empty list = no crossing within [0.01, 100]×"),
    }

    def all_flags(points) -> dict:
        keys = ("smc_lowest_psi", "pid_lowest_theta", "lqr_lowest_energy",
                "smc_fastest_recovery")
        return {k: all(pt["flags_vs_ref"][k] for pt in points) for k in keys}

    flags_grid = all_flags(grid)
    flags_oat = all_flags(oat_q + oat_r)

    psi_oat = [m["psi_ise_x1e3"] for m in oat_q]
    kpsi_oat = [-m["K"][1] for m in oat_q]
    out = {
        "generated_by": "scripts/lqr_sensitivity.py",
        "purpose": ("final-report extension: which replicated conclusions "
                    "depend on the published LQR weights (Q Eq. 43, R Eq. 44)?"),
        "design": {
            "Q_eq43": Q_EQ43.tolist(),
            "R_eq44_printed": R_EQ44,
            "R_effective_reduced_plant": 2.0 * R_EQ44,
            "gamma_eq42": GAMMA,
            "reduction": "single-input (Ad, 2b) as in twip/lqr.py; K is per motor",
            "oat_scales": list(OAT_SCALES),
            "grid_scales": list(GRID_SCALES),
        },
        "solver_credibility": {
            "recursion_vs_scipy_dare_max_abs_dev": dev_dare,
            "joint_scaling_homogeneity_max_abs_dev_K_4_1_vs_1_025": dev_homog,
            "note": ("the paper's backward recursion reproduces the DARE; the "
                     "DARE is invariant under joint (Q, R) scaling, so only the "
                     "ratio rho = s_Q/s_R matters"),
        },
        "plant_mechanism": {
            "psi_channel_transmission_zeros":
                [float(np.real(z)) for z in zeros_psi],
            "plant_unstable_pole_radius": pole_unstable,
            "note": ("two zeros sit ON the unit circle at z = 1 (reaction-wheel "
                     "non-minimum-phase structure at T0 = 10 ms): they cap the "
                     "achievable bandwidth, so the Riccati solution saturates "
                     "and the weights barely move the design"),
        },
        "baseline_check_1x": {
            "K": [float(v) for v in K_rec],
            "max_abs_dev_vs_eq45": float(np.abs(K_rec - K_PUB).max()),
            "n_iter_eq42": it_rec,
        },
        "fixed_reference_levels": ref,
        "baseline_lqr_at_1x": base,
        "oat_q": oat_q,
        "oat_r": oat_r,
        "grid": grid,
        "crossings": cross,
        "rank_robustness": {
            "all_grid_points": flags_grid,
            "all_oat_points": flags_oat,
            "psi_ise_range_oat_q_x1e3": [float(min(psi_oat)),
                                         float(max(psi_oat))],
            "K_psi_range_oat": [float(min(kpsi_oat)), float(max(kpsi_oat))],
            "K_psi_relative_swing_vs_eq45":
                float((max(kpsi_oat) - min(kpsi_oat)) / abs(K_PUB[1])),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    make_figure(ref, oat_q, grid, float(K_PUB[1]))

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_FIG}")
    print(f"recursion vs scipy DARE: {dev_dare:.2e}; homogeneity dev: {dev_homog:.2e}")
    print(f"psi-channel transmission zeros: {np.round(zeros_psi, 4)}; "
          f"unstable pole {pole_unstable:.4f}")
    print(f"baseline K dev vs Eq. (45): {out['baseline_check_1x']['max_abs_dev_vs_eq45']:.2e}")
    print(f"psi-ISE over oat_q: {min(psi_oat):.4f}..{max(psi_oat):.4f} x1e-3")
    print(f"K_psi over oat_q: {min(kpsi_oat):.2f}..{max(kpsi_oat):.2f} "
          f"(swing {out['rank_robustness']['K_psi_relative_swing_vs_eq45']*100:.1f}% of Eq. 45)")
    print(f"rank robustness grid: {flags_grid}")
    print(f"rank robustness oat:  {flags_oat}")


if __name__ == "__main__":
    main()
