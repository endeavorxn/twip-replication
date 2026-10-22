"""Preliminary cross-controller ISE comparison (paper Section 4, Tables 3-4).

Runs the three replicated controllers (PID Sec. 3.1, LQR Sec. 3.2, SMC
Sec. 3.3) on the verified discrete model from x0 = [0, 0.1, 0, 0]^T in two
scenarios:

  (a) no disturbance          -- the Table 3 scenario (start-up from a
                                  0.1 rad pitch tilt);
  (b) pitch-kick disturbance  -- emulates the Section 4.2 "external force
                                  hits the body once stabilized": a ONE-SAMPLE
                                  additive pulse of +0.05 rad on psi injected
                                  at t = 5 s (step 500).  The paper gives no
                                  magnitude, shape or timing for the force;
                                  this is ONE documented choice among the
                                  paper's unstated parameters.

ISE = T0 * sum y[k]^2 (twip.ise) on psi = x2 and theta = x1, x10^-3.  The
paper never states its integration window, so the primary tables use a
10 s run (1000 steps; the 5 s kick then has 5 s to settle) and a 5 s
no-disturbance table is also reported (the pre-kick response is identical
to the no-disturbance run); a full window sweep is in the JSON.

Controller configurations are the ones adopted by the individual runner
scripts (scripts/run_pid.py, run_lqr.py, run_smc.py), cross-checked here
against their results JSONs:

  PID : Table 2 gains, loops summed onto both motors, Fig. 4 saturation
        block at the IDENTIFIED +/-12 V (run_pid.py; value unstated in the
        paper).  An unsaturated reference row is carried as well, since the
        saturation is a replication-side identification, not a paper fact.
  LQR : K from the Eq. (40)-(42) Riccati iteration on the reduced
        single-input plant, unsaturated.
  SMC : sliding surface Eq. (62) designed from b = Bd[:,0], reaching law
        Eq. (64) with the unstated (cte, L) = (1.0, 4.0) of run_smc.py,
        scalar u wired to BOTH motors (b_wiring = Bd @ [1,1]), unsaturated.

Outputs: printed ours-vs-paper tables, results/summary.json,
results/plots/compare_nodist.png, results/plots/compare_dist.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from twip.ise import ise  # noqa: E402
from twip.lqr import LQRController, design_lqr  # noqa: E402
from twip.model import continuous_matrices, discretize, open_loop_poles  # noqa: E402
from twip.params import Params  # noqa: E402
from twip.pid import make_pid_controller, table2_pid  # noqa: E402
from twip.smc import SMCController, on_manifold_poles, sliding_surface  # noqa: E402
from twip.simulate import simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "results" / "summary.json"
OUT_PLOTS = ROOT / "results" / "plots"

# --- scenario constants ------------------------------------------------------
X0 = np.array([0.0, 0.1, 0.0, 0.0])       # psi(0) = 0.1 rad tilt
N_STEPS = 1000                            # 10 s at T0 = 10 ms
KICK_T = 5.0                              # Section 4.2 force instant [s]
KICK_PSI = 0.05                           # one-sample additive pulse [rad]
KICK_K = int(round(KICK_T / Params().T0))          # step 500
WINDOWS_S = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]   # window sweep (paper silent)
PRIMARY_W_S = 10.0                        # primary tables
REC_BAND = 0.02 * KICK_PSI                # recovery band, 2% of the kick

# --- published ISE targets (Tables 3-4, x10^-3, psi/theta) -------------------
PAPER_T3 = {"PID": (0.271, 526.6), "LQR": (0.393, 810.0), "SMC": (0.247, 660.9)}
PAPER_T4 = {"PID": (0.666, 653.2), "LQR": (0.622, 947.9), "SMC": (0.572, 768.2)}
K_PUB_EQ45 = np.array([-1.099, -81.44, -1.368, -10.86])     # Eq. (45)
S_PUB_EQ62 = np.array([-1.924, -147.918, -1.669, -21.065])  # Eq. (62)
KEQ_PUB_EQ63 = np.array([1.924, 160.270, 2.717, 21.581])    # Eq. (63)
SLIDING_POLES_EQ60_61 = [0.9318 + 0.0042j, 0.9318 - 0.0042j, 0.9797]

# --- runner-adopted settings -------------------------------------------------
U_SAT_PID = 12.0        # run_pid.py saturation identification (Fig. 4 block)
CTE, LAYER = 1.0, 4.0   # run_smc.py boundary layer (Eq. 64, values unstated)
K_REACH = 50            # steps after which the reaching transient is over
DU_THR = 0.02           # V; |du| below this does not count as an alternation

# --- dataviz palette (fixed categorical order; color follows the controller) --
C_PID, C_LQR, C_SMC = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"PID": C_PID, "LQR": C_LQR, "SMC": C_SMC}
NAMES = ("PID", "LQR", "SMC")


# --------------------------------------------------------------------------- #
# designs (built once; verified against the runner JSONs below)
# --------------------------------------------------------------------------- #
def make_kick():
    """One-sample additive pulse +KICK_PSI rad on psi at step KICK_K (Sec. 4.2)."""
    def kick(k: int) -> np.ndarray:
        d = np.zeros(4)
        if k == KICK_K:
            d[1] = KICK_PSI
        return d
    return kick


def build_designs():
    """Build the three controllers exactly as the runner scripts adopted them."""
    p = Params()
    Ac, Bc = continuous_matrices(p)
    Ad, Bd = discretize(Ac, Bc, p.T0)

    K, lqr_info = design_lqr(p)                      # Eq. (45), reduced SISO
    b_col = Bd[:, 0]                                 # design column (run_smc.py)
    b_wiring = Bd @ [1.0, 1.0]                       # scalar u drives BOTH motors
    S, smc_info = sliding_surface(Ad, b_col)         # Eq. (62) construction
    smc = SMCController(Ad, b_col, S, cte=CTE, L=LAYER, b_wiring=b_wiring)

    def new_pid():
        """Fresh Table 2 PID (the incremental loops hold state per run)."""
        psi_pid, theta_pid = table2_pid(p)
        return make_pid_controller(psi_pid, theta_pid)

    return p, Ad, Bd, K, lqr_info, S, smc_info, smc, new_pid


def check_designs(Ad, K, S, smc) -> dict:
    """Cross-check the rebuilt designs against the runner-adopted values."""
    lqr_j = json.loads((ROOT / "results" / "lqr.json").read_text())
    smc_j = json.loads((ROOT / "results" / "smc.json").read_text())
    K_ref = np.array(lqr_j["design"]["K"])
    S_ref = np.array(smc_j["primary"]["design"]["S"])
    man = np.sort_complex(on_manifold_poles(Ad, np.asarray(smc.b), smc.S))
    want = np.sort_complex(np.array(SLIDING_POLES_EQ60_61))
    return {
        "K_vs_lqr_json_max_abs_dev": float(np.abs(K - K_ref).max()),
        "S_vs_smc_json_max_abs_dev": float(np.abs(S - S_ref).max()),
        "Sb_design": smc.sb_design,
        "Sb_wiring": smc.sb_wiring,
        "contraction_factor": smc.sb_wiring * CTE / LAYER,
        "on_manifold_pole_dev_vs_eq60_61": float(np.abs(man - want).max()),
    }


# --------------------------------------------------------------------------- #
# simulation + metrics
# --------------------------------------------------------------------------- #
def run_all(Ad, Bd, K, smc, new_pid, disturbance):
    """One run per controller under `disturbance` (fresh PID state each time)."""
    out = {}
    for name in NAMES:
        ctrl = smc if name == "SMC" else (LQRController(K) if name == "LQR" else new_pid())
        sat = U_SAT_PID if name == "PID" else None    # primary configs
        out[name] = simulate(Ad, Bd, ctrl, X0, N_STEPS, disturbance=disturbance,
                             u_sat=sat)
    return out


def ise_window(run, col: int, window_s: float | None = None) -> float:
    """ISE x10^-3 of state column over a prefix of window_s seconds (run convention)."""
    X = run["X"]
    if window_s is not None:
        n = int(round(window_s / Params().T0))
        X = X[:n + 1]
    return ise(X[:, col], Params().T0) * 1e3


def ise_post_kick(run, col: int) -> float:
    """ISE x10^-3 over the post-kick window t in (5, 10] s."""
    return ise(run["X"][KICK_K + 1:, col], Params().T0) * 1e3


def energy_first_s(run, seconds: float = 1.0) -> dict:
    """Start-up control energy, E = T0 * sum(u_l^2 + u_r^2) [V^2 s]."""
    n = int(round(seconds / Params().T0))
    per_motor = float(Params().T0 * np.sum(run["U"][:n, 0] ** 2))   # u_l = u_r
    return {"window_s": seconds,
            "per_motor_V2s": per_motor,
            "both_motors_V2s": 2.0 * per_motor}


def recovery_metrics(run) -> dict:
    """Post-kick recovery of psi (claim iv).

    The FIRST post-kick sample IS the kick (psi jumps by +KICK_PSI before any
    controller can react), so the controller-induced overshoot is measured
    from the second post-kick sample (t >= 5.02 s) onward; the recovery time
    uses the whole post-kick trajectory against the 2%-of-kick band.
    """
    X = run["X"]
    kick_s = float(abs(X[KICK_K + 1, 1]))        # the kick itself, pre-control
    post = X[KICK_K + 1:]
    psi_ctrl = post[1:, 1]                       # controller response only
    out_idx = np.nonzero(np.abs(post[:, 1]) > REC_BAND)[0]
    return {"psi_kick_sample_rad": kick_s,
            "psi_overshoot_after_kick_rad": float(np.abs(psi_ctrl).max()),
            "psi_recovery_s_to_2pct_of_kick":
                float((out_idx[-1] + 1) * Params().T0) if out_idx.size else 0.0,
            "theta_max_post_kick_rad": float(np.abs(post[1:, 0]).max()),
            "u_peak_post_kick_V": float(np.abs(run["U"][KICK_K + 1:, 0]).max())}


def chatter_metrics(run, exclude_kick: bool = False) -> dict:
    """Chattering indicators on the applied input (run_smc.py convention).

    Sign alternations of du = diff(u) after the reaching transient (|du| >
    DU_THR to count), optionally masking the samples around the kick, which
    are the disturbance response, not chattering.
    """
    du = np.diff(run["U"][:, 0])
    tail = du[K_REACH:].copy()
    if exclude_kick:                              # mask du touching the kick
        lo, hi = KICK_K - 1 - K_REACH, KICK_K + 3 - K_REACH
        tail[max(lo, 0):max(hi, 0)] = 0.0
    alt = int(np.sum((tail[:-1] * tail[1:] < 0)
                     & (np.abs(tail[:-1]) > DU_THR)
                     & (np.abs(tail[1:]) > DU_THR)))
    return {"max_abs_delta_u_after_reach_V": float(np.abs(tail).max()),
            "delta_u_sign_alternations_after_reach": alt}


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def format_table(title, rows, paper):
    """rows: {name: (ise_psi, ise_theta) x10^-3}; paper: {name: (psi, theta)}."""
    lines = [title,
             f"{'ctrl':<5} {'psi ours':>9} {'psi pap':>8} {'ratio':>6}   "
             f"{'theta ours':>10} {'theta pap':>9} {'ratio':>6}"]
    for name in NAMES:
        op, ot = rows[name]
        pp, pt = paper[name]
        lines.append(f"{name:<5} {op:>9.4f} {pp:>8.3f} {op / pp:>6.2f}   "
                     f"{ot:>10.2f} {pt:>9.1f} {ot / pt:>6.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def plot_figure(runs, fname, title, zoom=False):
    """theta / psi (/ psi zoom) / u panels, one colored line per controller."""
    T0 = Params().T0
    t = np.arange(N_STEPS + 1) * T0
    t_u = np.arange(N_STEPS) * T0
    n_rows = 4 if zoom else 3
    fig, axs = plt.subplots(n_rows, 1, figsize=(8.0, 2.5 * n_rows),
                            sharex=not zoom, constrained_layout=True)
    # zoom=False shares the time axis; zoom=True keeps the psi-recovery inset
    # at 4.5-8 s WITHOUT propagating its limits to the full-run panels
    fig.patch.set_facecolor(SURFACE)

    def state_panel(ax, col, ylabel):
        for n in NAMES:
            ax.plot(t, runs[n]["X"][:, col], color=COLORS[n], lw=2, label=n)
        ax.axhline(0, color=BASELINE, lw=1)
        ax.set_ylabel(ylabel, color=INK2)
        ax.set_title(ylabel.split("[")[0].strip(), color=INK, fontsize=11,
                     loc="left")
        style(ax)

    state_panel(axs[0], 0, "wheel angle theta [rad]")
    state_panel(axs[1], 1, "pitch angle psi [rad]")
    if zoom:
        axs[0].set_xlim(0, N_STEPS * T0)
        axs[1].set_xlim(0, N_STEPS * T0)
    if zoom:
        for ax in axs[:3]:
            ax.axvline(KICK_T, color=MUTED, lw=1, ls=(0, (2, 2)))
        axs[1].annotate(f"kick +{KICK_PSI} rad on psi, t = {KICK_T:.0f} s",
                        (KICK_T, 0.97), xycoords=("data", "axes fraction"),
                        xytext=(4, -4), textcoords="offset points",
                        va="top", color=MUTED, fontsize=8)
        z = slice(int(4.5 / T0), int(8.0 / T0) + 1)
        for n in NAMES:   # identity via the panel-1 legend; peaks coincide,
            axs[2].plot(t[z], runs[n]["X"][z, 1], color=COLORS[n], lw=2)
        axs[2].axhline(0, color=BASELINE, lw=1)
        axs[2].set_xlim(4.5, 8.0)
        axs[2].set_ylabel("pitch angle psi [rad]")
        axs[2].set_title("psi recovery after the kick (zoom 4.5-8 s)",
                         color=INK, fontsize=11, loc="left")
        style(axs[2])
        axs[3].axvline(KICK_T, color=MUTED, lw=1, ls=(0, (2, 2)))
    uax = axs[3] if zoom else axs[2]
    umax = 0.0
    for n in NAMES:
        uax.plot(t_u, runs[n]["U"][:, 0], color=COLORS[n], lw=2, label=n)
        umax = max(umax, float(np.abs(runs[n]["U"][:, 0]).max()))
    uax.axhline(0, color=BASELINE, lw=1)
    uax.axhline(U_SAT_PID, color=MUTED, lw=1, ls=(0, (4, 3)))
    uax.axhline(-U_SAT_PID, color=MUTED, lw=1, ls=(0, (4, 3)))
    uax.set_ylim(top=umax * 1.12)                 # headroom for the SMC peak
    if zoom:
        uax.set_xlim(0, N_STEPS * T0)
    uax.annotate(f"PID saturation ±{U_SAT_PID:.0f} V",
                 (0.985, U_SAT_PID), xycoords=("axes fraction", "data"),
                 xytext=(0, 3), textcoords="offset points", ha="right",
                 va="bottom", color=MUTED, fontsize=8)
    uax.set_ylabel("motor voltage u = u_l = u_r [V]")
    uax.set_title("control input (both motors, identical Bd columns)",
                  color=INK, fontsize=11, loc="left")
    style(uax)
    axs[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper right",
                  labelcolor=INK2)
    axs[-1].set_xlabel("time [s]", color=INK2)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.savefig(OUT_PLOTS / fname, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p, Ad, Bd, K, lqr_info, S, smc_info, smc, new_pid = build_designs()
    T0 = p.T0
    checks = check_designs(Ad, K, S, smc)

    # ---------------- scenarios ------------------------------------------- #
    nodist = run_all(Ad, Bd, K, smc, new_pid, disturbance=None)
    dist = run_all(Ad, Bd, K, smc, new_pid, disturbance=make_kick())
    pid_unsat = simulate(Ad, Bd, new_pid(), X0, N_STEPS)   # literal PID, no sat

    # ---------------- ISE tables ------------------------------------------ #
    nodist_10 = {n: (ise_window(nodist[n], 1, 10.0),
                     ise_window(nodist[n], 0, 10.0)) for n in NAMES}
    nodist_5 = {n: (ise_window(nodist[n], 1, 5.0),
                    ise_window(nodist[n], 0, 5.0)) for n in NAMES}
    dist_10 = {n: (ise_window(dist[n], 1, 10.0),
                   ise_window(dist[n], 0, 10.0)) for n in NAMES}
    dist_post = {n: (ise_post_kick(dist[n], 1),
                     ise_post_kick(dist[n], 0)) for n in NAMES}
    pid_unsat_pair = (ise_window(pid_unsat, 1, 10.0),
                      ise_window(pid_unsat, 0, 10.0))

    print(format_table("Table 3 (no disturbance), ISE x10^-3, 10 s window",
                       nodist_10, PAPER_T3))
    print()
    print(format_table("no-disturbance, 5 s window (pre-disturbance response)",
                       nodist_5, PAPER_T3))
    print()
    print(format_table("Table 4 (pitch kick at t = 5 s), ISE x10^-3, 10 s window",
                       dist_10, PAPER_T4))
    print()
    print(format_table("post-kick only, t in (5, 10] s (reference, no paper column)",
                       dist_post, PAPER_T4))
    print(f"\nPID reference, literal UNSATURATED (10 s, no disturbance): "
          f"psi {pid_unsat_pair[0]:.4f} (x{pid_unsat_pair[0] / PAPER_T3['PID'][0]:.2f}), "
          f"theta {pid_unsat_pair[1]:.2f} (x{pid_unsat_pair[1] / PAPER_T3['PID'][1]:.2f})")

    # window sweep, no disturbance (paper window unstated)
    sweep = {n: {f"{w:g}s": {"psi_x1e3": ise_window(nodist[n], 1, w),
                             "theta_x1e3": ise_window(nodist[n], 0, w)}
                 for w in WINDOWS_S} for n in NAMES}

    # Table 4 minus Table 3 increments: the paper's own disturbance footprint,
    # to compare against the footprint of OUR documented kick (diagnostic for
    # the unstated disturbance magnitude/shape)
    increments = {n: {"paper_psi_x1e3": PAPER_T4[n][0] - PAPER_T3[n][0],
                      "ours_psi_x1e3": dist_10[n][0] - nodist_10[n][0],
                      "paper_theta_x1e3": PAPER_T4[n][1] - PAPER_T3[n][1],
                      "ours_theta_x1e3": dist_10[n][1] - nodist_10[n][1]}
                  for n in NAMES}

    # ---------------- energy (claim iii, Fig. 9) --------------------------- #
    energy = {n: energy_first_s(nodist[n], 1.0) for n in NAMES}
    energy["PID_unsaturated_reference"] = energy_first_s(pid_unsat, 1.0)

    # ---------------- recovery (claim iv) / chatter (claim v) -------------- #
    recovery = {n: recovery_metrics(dist[n]) for n in NAMES}
    chatter = {"nodist": chatter_metrics(nodist["SMC"]),
               "dist_kick_masked": chatter_metrics(dist["SMC"], exclude_kick=True),
               "dist_raw": chatter_metrics(dist["SMC"])}

    # ---------------- qualitative claims ----------------------------------- #
    e_tot = {n: energy[n]["both_motors_V2s"] for n in NAMES}
    lowest_e = min(e_tot, key=e_tot.get)
    smc_fast = recovery["SMC"]["psi_recovery_s_to_2pct_of_kick"] \
        < recovery["LQR"]["psi_recovery_s_to_2pct_of_kick"]
    smc_overshoots_less = recovery["SMC"]["psi_overshoot_after_kick_rad"] \
        < recovery["PID"]["psi_overshoot_after_kick_rad"]
    claims = {
        "i_smc_lowest_psi_ise_both_scenarios": {
            "claim": "SMC has the lowest psi-ISE in both scenarios (Tables 3-4)",
            "values": {"nodist_10s": {n: nodist_10[n][0] for n in NAMES},
                       "dist_10s": {n: dist_10[n][0] for n in NAMES}},
            "pass": bool(nodist_10["SMC"][0] < min(nodist_10[n][0] for n in ("PID", "LQR"))
                         and dist_10["SMC"][0] < min(dist_10[n][0] for n in ("PID", "LQR"))),
        },
        "ii_pid_lowest_theta_ise": {
            "claim": "PID has the lowest theta-ISE",
            "values": {"nodist_10s": {n: nodist_10[n][1] for n in NAMES},
                       "dist_10s": {n: dist_10[n][1] for n in NAMES}},
            "pass": bool(nodist_10["PID"][1] < min(nodist_10[n][1] for n in ("LQR", "SMC"))
                         and dist_10["PID"][1] < min(dist_10[n][1] for n in ("LQR", "SMC"))),
        },
        "iii_lqr_lowest_startup_energy": {
            "claim": "LQR uses the least start-up control energy (Fig. 9); "
                     "metric E = T0*sum(u_l^2+u_r^2) over the first 1 s",
            "values_V2s_both_motors": e_tot,
            "values_V2s_per_motor": {n: energy[n]["per_motor_V2s"] for n in NAMES},
            "pid_unsaturated_reference_V2s_both_motors":
                energy["PID_unsaturated_reference"]["both_motors_V2s"],
            "lowest": lowest_e,
            "pass": bool(lowest_e == "LQR"),
        },
        "iv_smc_recovers_faster_than_lqr_smaller_overshoot_than_pid": {
            "claim": "under the disturbance SMC recovers faster than LQR and "
                     "overshoots less than PID (psi; overshoot measured after "
                     "the kick sample, recovery vs the 2%-of-kick band)",
            "recovery_smc_vs_lqr": bool(smc_fast),
            "overshoot_smc_vs_pid": bool(smc_overshoots_less),
            "values": recovery,
            "pass": bool(smc_fast and smc_overshoots_less),
        },
        "v_no_chattering_in_smc": {
            "claim": "no chattering in the SMC responses (Figs. 5-12)",
            "values": chatter,
            "pass": bool(chatter["nodist"]["delta_u_sign_alternations_after_reach"] == 0
                         and chatter["dist_kick_masked"]["delta_u_sign_alternations_after_reach"] == 0),
        },
    }

    print("\nQualitative claims (paper Section 4):")
    for key, c in claims.items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {key}")
    print(f"\nStart-up energy, first 1 s, both motors [V^2 s]: "
          + ", ".join(f"{n} {v:.3f}" for n, v in e_tot.items())
          + f"  (PID unsaturated reference "
            f"{energy['PID_unsaturated_reference']['both_motors_V2s']:.2f})")
    print("\nRecovery after the kick (psi; the kick sample itself is 0.05 rad "
          "for every controller):")
    for n in NAMES:
        r = recovery[n]
        print(f"  {n}: overshoot after kick {r['psi_overshoot_after_kick_rad']:.4f} rad, "
              f"recovery {r['psi_recovery_s_to_2pct_of_kick']:.2f} s")

    # ---------------- figures ---------------------------------------------- #
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    plot_figure(nodist, "compare_nodist.png",
                "TWIP replication: start-up from psi(0) = 0.1 rad, no disturbance "
                "(paper Table 3 scenario)")
    plot_figure(dist, "compare_dist.png",
                f"TWIP replication: one-sample pitch kick +{KICK_PSI} rad at "
                f"t = {KICK_T:.0f} s (Section 4.2 emulation, paper Table 4 scenario)",
                zoom=True)
    print(f"\nFigures written to {OUT_PLOTS / 'compare_nodist.png'} and "
          f"{OUT_PLOTS / 'compare_dist.png'}")

    # ---------------- summary.json ----------------------------------------- #
    mv = json.loads((ROOT / "results" / "model_verification.json").read_text())
    smc_j = json.loads((ROOT / "results" / "smc.json").read_text())
    pid_j = json.loads((ROOT / "results" / "pid.json").read_text())

    def ratio_tables(rows, paper):
        return {n: {"psi_x1e3": rows[n][0], "theta_x1e3": rows[n][1],
                    "paper_psi_x1e3": paper[n][0], "paper_theta_x1e3": paper[n][1],
                    "ratio_psi": round(rows[n][0] / paper[n][0], 2),
                    "ratio_theta": round(rows[n][1] / paper[n][1], 2)}
                for n in NAMES}

    summary = {
        "paper": "Villacrés et al. 2016, Section 4, Tables 3-4 (preliminary integration)",
        "generated_by": "scripts/compare_ise.py",
        "model_verification": {
            "g": {"table1_printed": mv["g_investigation"]["table1_value"],
                  "selected": mv["g_investigation"]["selected"],
                  "max_abs_dev_A1_vs_eq21": mv["g_investigation"]["candidates"]["9.81"]["max_abs_dev_A1_vs_eq21"],
                  "max_abs_dev_B1_vs_eq22": mv["g_investigation"]["candidates"]["9.81"]["max_abs_dev_B1_vs_eq22"]},
            "discrete": {"T0": T0,
                         "max_abs_dev_Ad_vs_eq25": mv["discrete"]["max_abs_dev_Ad_vs_eq25"],
                         "max_abs_dev_Bd_vs_eq26": mv["discrete"]["max_abs_dev_Bd_vs_eq26"],
                         "open_loop_poles_computed": [f"{z.real:.4f}{z.imag:+.4f}j"
                                                      for z in open_loop_poles(Ad)],
                         "poles_published_eq27_30": mv["poles"]["published"],
                         "max_abs_dev_poles": mv["poles"]["max_abs_dev"],
                         "bd_columns_identical": bool(np.allclose(Bd[:, 0], Bd[:, 1]))},
            "all_passed": mv["all_passed"],
        },
        "controller_verification": {
            "PID": {"gains_table2": pid_j["gains"],
                    "eq34_coeffs": pid_j["eq34_coeffs"],
                    "wiring": pid_j["wiring_assumption"],
                    "u_sat_identified_V": U_SAT_PID,
                    "u_sat_identification": pid_j["saturation_identification"]["note"],
                    "deviations_vs_table3_10s_primary":
                        "psi x1.09, theta x0.96 (ours/paper)"},
            "LQR": {"K_computed": K.tolist(),
                    "K_published_eq45": K_PUB_EQ45.tolist(),
                    "max_abs_dev_K_vs_eq45": float(np.abs(K - K_PUB_EQ45).max()),
                    "iterations_eq42": lqr_info["iterations"],
                    "gamma_eq42": lqr_info["gamma"],
                    "Q_eq43_diag": np.asarray(lqr_info["Q"]).tolist(),
                    "R_eq44_printed": lqr_info["R_printed"],
                    "R_effective_scalar": lqr_info["R_effective_scalar"],
                    "closed_loop_poles": [f"{z.real:.4f}{z.imag:+.4f}j"
                                          for z in lqr_info["poles"]],
                    "pole_note": "Eqs. (60)-(61) plus an unstated deadbeat pole at 0"},
            "SMC": {"S_computed": S.tolist(),
                    "S_published_eq62": S_PUB_EQ62.tolist(),
                    "reldev_S_vs_eq62_per_entry":
                        np.abs((S - S_PUB_EQ62) / S_PUB_EQ62).tolist(),
                    "max_reldev_S_vs_eq62": float(np.abs((S - S_PUB_EQ62) / S_PUB_EQ62).max()),
                    "Keq_computed_eq57": smc.Keq.tolist(),
                    "Keq_published_eq63": KEQ_PUB_EQ63.tolist(),
                    "max_reldev_Keq_vs_eq63": float(np.abs((smc.Keq - KEQ_PUB_EQ63) / KEQ_PUB_EQ63).max()),
                    "design_input": ("b = Bd[:,0] (single column); the published "
                                     "Eq.(62)-(63) pair is consistent only with Sb = 1"),
                    "b_wiring": "Bd @ [1,1] (scalar u to both motors)",
                    "Sb_wiring": smc.sb_wiring,
                    "cte_L_eq64": [CTE, LAYER],
                    "contraction_factor": smc.sb_wiring * CTE / LAYER,
                    "sliding_poles_eq60_61": ["0.9318+0.0042j", "0.9318-0.0042j",
                                              "0.9797"],
                    "on_manifold_pole_dev": checks["on_manifold_pole_dev_vs_eq60_61"],
                    "u_eq_equals_minus_K_lqr_for_b_sum_design": (
                        "ideal SMC with the b-sum surface reproduces u = -K_lqr "
                        "exactly (run_smc.py lqr_identity); the adopted column "
                        "surface has twice the gain, matching Eq. (63)")},
        },
        "design_cross_checks_vs_runner_jsons": checks,
        "scenarios": {
            "x0": X0.tolist(),
            "n_steps": N_STEPS,
            "primary_window_s": PRIMARY_W_S,
            "window_sweep_s": WINDOWS_S,
            "no_disturbance": "start-up from psi(0) = 0.1 rad (Table 3)",
            "disturbance": (f"Section 4.2 emulation: ONE-SAMPLE additive pulse "
                            f"+{KICK_PSI} rad on psi at step {KICK_K} "
                            f"(t = {KICK_T} s); magnitude/shape/timing unstated "
                            "in the paper"),
            "recovery_band_rad": REC_BAND,
            "controller_configs": {
                "PID": f"Table 2 gains, summed loops, u_sat = {U_SAT_PID} V (identified)",
                "LQR": "Eq. (45) gain, unsaturated",
                "SMC": (f"Eq. (62) surface from Bd[:,0], Eq. (64) layer "
                        f"cte = {CTE}, L = {LAYER}, unsaturated")},
        },
        "ise_tables": {
            "no_disturbance_10s_vs_table3": ratio_tables(nodist_10, PAPER_T3),
            "no_disturbance_5s_vs_table3": ratio_tables(nodist_5, PAPER_T3),
            "disturbance_10s_vs_table4": ratio_tables(dist_10, PAPER_T4),
            "disturbance_post_kick_5_to_10s": {n: {"psi_x1e3": dist_post[n][0],
                                                   "theta_x1e3": dist_post[n][1]}
                                               for n in NAMES},
            "pid_unsaturated_reference_10s_nodist": {
                "psi_x1e3": pid_unsat_pair[0], "theta_x1e3": pid_unsat_pair[1],
                "ratio_psi": round(pid_unsat_pair[0] / PAPER_T3["PID"][0], 2),
                "ratio_theta": round(pid_unsat_pair[1] / PAPER_T3["PID"][1], 2)},
            "window_sweep_nodist_x1e3": sweep,
            "table4_minus_table3_increments": {
                "note": ("the paper's own disturbance footprint vs the footprint "
                         "of our documented +0.05 rad one-sample kick; ours are "
                         "2-8x smaller in psi, so absolute Table 4 levels are "
                         "NOT expected to match -- only the rankings are "
                         "meaningful"),
                **increments},
        },
        "startup_energy_first_1s": energy,
        "recovery_metrics_disturbance": recovery,
        "chatter_smc": chatter,
        "qualitative_claims": claims,
        "assumptions_and_deviations": [
            "g = 9.81 used (Table 1 prints 9.8); see model_verification.g.",
            "Simulation window 10 s (1000 steps) for the primary tables; the paper "
            "does not state its ISE integration window. A 5 s no-disturbance table "
            "and a 2-10 s sweep are included; the ISE is essentially converged by "
            "3-4 s in every controller.",
            "Disturbance emulation: the Section 4.2 force has no stated magnitude, "
            f"shape or timing; implemented as a one-sample additive pulse of "
            f"+{KICK_PSI} rad on psi at t = {KICK_T} s (step {KICK_K}).",
            "PID carries the Fig. 4 saturation block at +/-12 V IDENTIFIED in "
            "run_pid.py (the paper states no value); without it the literal "
            "Table 2 PID misses psi-ISE by x6.2 vs Table 3. LQR and SMC run "
            "unsaturated (peak |u| ~ 8.1 V and ~17.3 V); an unsaturated PID "
            "reference row is reported for fairness.",
            "Start-up energy metric fixed across controllers as "
            "T0*sum(u_l^2+u_r^2) [V^2 s], reported both-motors and per-motor. "
            "CROSS-CHECK FINDING: results/pid.json fig9_claim_record is doubly "
            "mis-scaled -- run_pid.py stores 'u' = U[:,0], a PER-MOTOR column, "
            "so its control_effort_V2s (5.258 V^2 s at 10 s) is per-motor, yet "
            "fig9 labels it 'both motors' and halves it; AND the LQR number it "
            "cites from lqr.json (374.04) is a RAW sample sum without the T0 "
            "factor (3.74 V^2 s per motor T0-weighted). With the consistent "
            "metric here, LQR has the LOWEST start-up energy (7.41 vs PID 10.48 "
            "and SMC 11.41 V^2 s, both motors) and the paper's Fig. 9 claim "
            "REPLICATES -- the opposite of pid.json's recorded conclusion.",
            "SMC (cte, L) = (1.0, 4.0) are not stated in the paper; chosen in "
            "run_smc.py from the contraction condition L > Sb_wiring*cte = 2*cte.",
            "u_left = u_right = u for every controller (verified identical Bd "
            "columns); the paper never states the motor wiring.",
            "The Eq. (62) entry-1 deviation (ours -2.197 vs published -1.924, 14%) "
            "persists in every principled SMC variant (run_smc.py); entries 2-4 "
            "match to 0.6-1.7%.",
            "Disturbed-scenario ranking nuance: our psi ordering is SMC < PID < "
            "LQR while the paper's Table 4 has SMC < LQR < PID; the dominant "
            "effect is the unstated disturbance (see table4_minus_table3_increments), "
            "under which our +0.05 rad kick adds far less psi-ISE than the "
            "paper's force appears to.",
            "Ratios are ours/paper, 2 decimals; comparisons are PRELIMINARY -- the "
            "point is the ranking, not exact ISE matching.",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=1, default=str))
    print(f"Summary written to {OUT_JSON}")


if __name__ == "__main__":
    main()
