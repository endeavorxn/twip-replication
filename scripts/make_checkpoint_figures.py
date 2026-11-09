"""Checkpoint evidence figures (Week 8): Fig. A and Fig. B, plus sweep data.

Generates the two figures referenced by the Week-8 checkpoint and embeds the
sweep numbers behind its Section 5, so the submission carries its central
evidence instead of placeholders:

  Fig. A  results/figures/fig_a_pitch_phase.png
          Body pitch response psi(t), PID/LQR/SMC, no disturbance (the
          paper's Fig. 6 analogue) alongside the phase portrait (psi, psi_dot)
          (the paper's Fig. 10 analogue).  Same runs as the Section-1 ISE
          comparison (primary controller configurations).

  Fig. B  results/figures/fig_b_sweep.png
          Robustness sweep (checkpoint Sec. 5): post-kick psi-ISE vs kick
          magnitude on log-log axes with fitted power-law exponents, and
          post-kick recovery time per magnitude.  Protocol: one-sample
          pitch kicks at t = 5 s under the Section-1 configurations
          (PID +/-12 V identified, LQR/SMC unsaturated); a second variant
          with the Fig. 4 block at +/-9 V for every controller is stored
          in the JSON and reported in the checkpoint text.

  Data    results/sweep.json
          Both magnitude sweeps (ISE increments, fitted exponents, recovery
          times, recovery-loss margins), the Table 4 - Table 3 increment
          targets, the state-kick amplitude matching the paper's PID psi
          footprint, and the disturbance-shape sweep (state kick vs
          one-sample input kick vs 0.1 s input pulse) with the psi-increments
          each shape produces against the paper's own
          +0.395/+0.229/+0.325 x10^-3.

Run:  python scripts/make_checkpoint_figures.py
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

from compare_ise import (  # noqa: E402
    COLORS, GRID, INK, INK2, MUTED, NAMES, N_STEPS, X0,
    build_designs, ise, style,
)
from twip.lqr import LQRController  # noqa: E402
from twip.params import Params  # noqa: E402
from twip.simulate import simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_FIGS = ROOT / "results" / "figures"
OUT_JSON = ROOT / "results" / "sweep.json"

T0 = Params().T0
KICK_K = 500                       # t = 5 s (checkpoint Sec. 5 protocol)
SAT_V = 9.0                        # Fig. 4 saturation block, NXT supply
U_SAT_PRIMARY = {"PID": 12.0, "LQR": None, "SMC": None}   # Sec.-1 configs
U_SAT_ALL9 = {n: SAT_V for n in NAMES}                    # hardware-rail variant
MAGNITUDES = [0.02, 0.05, 0.10, 0.15, 0.20]              # fitted exponents
MAGNITUDES_EXT = MAGNITUDES + [0.25, 0.30, 0.35, 0.40]   # recovery margins
PAPER_INCREMENT = {"PID": 0.395e-3, "LQR": 0.229e-3, "SMC": 0.325e-3}
INPUT_KICK_V = 1.0                 # one-sample input (torque-level) kick
INPUT_PULSE_S = 0.1                # sustained input pulse width

# paper Fig. 10 axis ranges (angle [rad], angular speed [rad/s])
PHASE_X = (-0.045, 0.105)
PHASE_Y = (-4.10, 0.55)


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #
def make_kick(dpsi: float):
    """One-sample additive pulse +dpsi rad on psi at step KICK_K."""
    def kick(k: int) -> np.ndarray:
        d = np.zeros(4)
        if k == KICK_K:
            d[1] = dpsi
        return d
    return kick


def with_input_pulse(ctrl, volts: float, k0: int, k1: int):
    """Wrap a controller so its output carries +volts over steps [k0, k1)."""
    def wrapped(x, k):
        u = np.asarray(ctrl(x, k), float)
        if k0 <= k < k1:
            u = u + volts
        return u
    return wrapped


def run_controller(name, ctrl, disturbance, u_sat=None, n_steps=N_STEPS):
    """One run; fresh PID state per run (the incremental loops hold state)."""
    return simulate(DESIGN.Ad, DESIGN.Bd, ctrl, X0, n_steps,
                    disturbance=disturbance, u_sat=u_sat)


def run_all(disturbance, sat_map=U_SAT_PRIMARY, n_steps=N_STEPS):
    """One run per controller, saturation per the given per-controller map."""
    return {n: run_controller(
                n,
                DESIGN.smc if n == "SMC" else
                (DESIGN.lqr_ctrl if n == "LQR" else DESIGN.new_pid()),
                disturbance, sat_map[n], n_steps)
            for n in NAMES}


def recovery_time(run, kick: float):
    """Post-kick psi recovery vs the 2%-of-kick band.

    Returns (seconds, recovered).  `recovered` is False when the trajectory
    is still outside the band at the end of the window (the checkpoint calls
    this 'loses recovery'); seconds is then the full post-kick length.
    """
    post = run["X"][KICK_K + 1:, 1]
    band = 0.02 * kick
    out = np.nonzero(np.abs(post) > band)[0]
    if out.size == 0:
        return 0.0, True
    recovered = out[-1] < len(post) - 1
    return float((out[-1] + 1) * T0), bool(recovered)


def ise_post_kick_psi(run):
    """psi-ISE x10^-3 over the post-kick window t in (5, 10] s."""
    return ise(run["X"][KICK_K + 1:, 1], T0) * 1e3


def ise_increment_10s(run_d, run_n):
    """psi-ISE increment x10^-3, full window (Table 4 minus Table 3 style)."""
    return (ise(run_d["X"][:, 1], T0) - ise(run_n["X"][:, 1], T0)) * 1e3


def psi_overshoot(run):
    """Controller-induced psi overshoot after the kick (kick sample excluded)."""
    return float(np.abs(run["X"][KICK_K + 2:, 1]).max())


# --------------------------------------------------------------------------- #
# Fig. A - pitch response + phase portrait (paper Figs. 6 and 10 analogues)
# --------------------------------------------------------------------------- #
def label_line(ax, x, y, text, dx=6, dy=0):
    """Direct label at a point on a series (relief for low-contrast marks)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=INK2, fontsize=8.5, fontweight="bold")


def fine_phase_paths(runs, sub: int = 20):
    """Exact continuous plant path between the 10 ms control updates -- the
    paper's Fig. 10 drawing convention. The controller applies each sample's
    voltage for the whole sample (ZOH); the continuous plant moves smoothly
    inside the sample and changes direction at every update, which is what
    gives Fig. 10 its stepped texture. 20 sub-steps per 10 ms sample.
    """
    from twip.model import continuous_matrices
    from scipy.linalg import expm
    Ac, Bc = continuous_matrices(Params())
    aug = np.zeros((6, 6))
    aug[:4, :4] = Ac
    aug[:4, 4:] = Bc
    Msub = expm(aug * (T0 / sub))
    out = {}
    for n in NAMES:
        X, U = runs[n]["X"], runs[n]["U"]
        pts = np.empty((len(U) * sub + 1, 4))
        pts[0] = X[0]
        x = X[0].copy()
        i = 1
        for k in range(len(U)):
            st = np.concatenate([x, U[k]])
            for _ in range(sub):
                st = Msub @ st
                pts[i] = st[:4]
                i += 1
            x = st[:4]
        out[n] = pts
    return out


def fig_a() -> None:
    runs = run_all(disturbance=None)                      # Sec.-1 configs
    t = np.arange(N_STEPS + 1) * T0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                   constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")

    # -- pitch response (paper Fig. 6 analogue), 0-3.5 s like the paper -- --
    n_show = int(3.5 / T0)
    for n in NAMES:
        ax1.plot(t[:n_show + 1], runs[n]["X"][:n_show + 1, 1],
                 color=COLORS[n], lw=2, label=n)
    ax1.axhline(0, color="#c3c2b7", lw=1)
    ax1.set_xlim(0, 3.5)
    ax1.set_xlabel("time [s]", color=INK2)
    ax1.set_ylabel("pitch angle ψ [rad]", color=INK2)
    ax1.set_title("body pitch response, start-up from ψ(0) = 0.1 rad\n"
                  "(paper Fig. 6 analogue)", color=INK, fontsize=10.5,
                  loc="left")
    style(ax1)
    label_line(ax1, 0.62, runs["LQR"]["X"][62, 1] + 0.004, "LQR", dy=4)
    label_line(ax1, 0.36, runs["SMC"]["X"][36, 1] - 0.002, "SMC", dy=-12)
    label_line(ax1, 1.05, runs["PID"]["X"][105, 1] - 0.002, "PID", dy=-12)
    ax1.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)

    # -- phase portrait (paper Fig. 10 analogue) -------------------------- --
    # drawn the paper's way: continuous plant path under zero-order-hold
    # control, so every 10 ms update appears as a corner in the curve
    paths = fine_phase_paths(runs)
    for n in NAMES:
        pth = paths[n]
        ax2.plot(pth[:, 1], pth[:, 3], color=COLORS[n], lw=1.3, label=n)
    ax2.axhline(0, color="#c3c2b7", lw=1)
    ax2.set_xlim(*PHASE_X)
    ax2.set_ylim(*PHASE_Y)
    ax2.set_xlabel("pitch angle ψ [rad]", color=INK2)
    ax2.set_ylabel("pitch rate ψ̇ [rad/s]", color=INK2)
    ax2.set_title("phase portrait, spiralling into the origin\n"
                  "(paper Fig. 10 analogue — continuous path)", color=INK, fontsize=10.5,
                  loc="left")
    style(ax2)
    ax2.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGS / "fig_a_pitch_phase.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. B - magnitude sweep: log-log ISE + recovery bars (checkpoint Sec. 5)
# --------------------------------------------------------------------------- #
def magnitude_sweep(sat_map=U_SAT_PRIMARY) -> dict:
    nodist = run_all(disturbance=None, sat_map=sat_map)
    out = {}
    for m in MAGNITUDES_EXT:
        runs = run_all(disturbance=make_kick(m), sat_map=sat_map)
        row = {}
        for n in NAMES:
            rec_s, rec_ok = recovery_time(runs[n], m)
            row[n] = {
                "psi_ise_post_kick_increment_x1e3":
                    ise_post_kick_psi(runs[n]) - ise_post_kick_psi(nodist[n]),
                "recovery_s_to_2pct_of_kick": rec_s,
                "recovered": rec_ok,
                "psi_overshoot_after_kick_rad": psi_overshoot(runs[n]),
            }
        out[f"{m:g}"] = row
    return out


def fit_exponents(sweep: dict) -> dict:
    """Power-law exponent of psi-ISE increment vs kick over the primary grid."""
    xs = np.log10(np.asarray(MAGNITUDES, float))
    fits = {}
    for n in NAMES:
        ys = np.log10([sweep[f"{m:g}"][n]["psi_ise_post_kick_increment_x1e3"]
                       for m in MAGNITUDES])
        slope = float(np.polyfit(xs, ys, 1)[0])
        fits[n] = {"exponent": round(slope, 2),
                   "amplitude_x1e3": round(float(10 ** np.polyfit(xs, ys, 1)[1]
                                                 ), 4)}
    return fits


def fig_b(sweep: dict, fits: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                   constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")

    # -- log-log psi-ISE vs magnitude with fitted slopes ------------------ --
    # label offset per controller: LQR/SMC label above their fit line, PID
    # below its own, so the three mid-line labels cannot collide
    lab_dy = {"LQR": 9, "SMC": 9, "PID": -13}
    for n in NAMES:
        ys = [sweep[f"{m:g}"][n]["psi_ise_post_kick_increment_x1e3"]
              for m in MAGNITUDES]
        ax1.loglog(MAGNITUDES, ys, "o", color=COLORS[n], ms=8, label=n,
                   markerfacecolor=COLORS[n], markeredgecolor="#fcfcfb",
                   markeredgewidth=1.5, zorder=3)
        e = fits[n]["exponent"]
        xs = np.geomspace(MAGNITUDES[0], MAGNITUDES[-1], 32)
        c0 = fits[n]["amplitude_x1e3"]
        ax1.loglog(xs, c0 * xs ** e, color=COLORS[n], lw=1.6, ls=(0, (4, 3)),
                   zorder=2)
        # direct slope label at the middle of each fit line (contrast relief)
        xmid = float(np.sqrt(MAGNITUDES[0] * MAGNITUDES[-1]))
        ax1.annotate(f"{n}  slope {e:.2f}", (xmid, c0 * xmid ** e),
                     xytext=(4, lab_dy[n]), textcoords="offset points",
                     color=INK2, fontsize=8.5, fontweight="bold",
                     va="center", ha="left",
                     bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb",
                               ec="none", alpha=0.85))
    ax1.set_xlim(MAGNITUDES[0] * 0.8, MAGNITUDES[-1] * 1.6)
    ax1.set_xlabel("kick magnitude Δψ [rad]", color=INK2)
    ax1.set_ylabel("post-kick ψ-ISE increment [×10⁻³]", color=INK2)
    ax1.set_title("ψ-ISE vs kick magnitude, log–log\n"
                  "one-sample kicks, §1 configurations", color=INK,
                  fontsize=10.5, loc="left")
    style(ax1)

    # -- recovery-time bars per magnitude --------------------------------- --
    mags_bar = MAGNITUDES_EXT
    n_grp = len(NAMES)
    w = 0.27
    x = np.arange(len(mags_bar))
    t_max = max(sweep[f"{m:g}"][n]["recovery_s_to_2pct_of_kick"]
                for m in mags_bar for n in NAMES if sweep[f"{m:g}"][n]["recovered"])
    ceil = t_max * 1.18
    for j, n in enumerate(NAMES):
        vals, lost = [], []
        for m in mags_bar:
            row = sweep[f"{m:g}"][n]
            vals.append(row["recovery_s_to_2pct_of_kick"])
            lost.append(not row["recovered"])
        vals = [min(v, ceil) for v in vals]      # lost bars draw to the cap
        bars = ax2.bar(x + (j - 1) * w, vals, w * 0.92, color=COLORS[n],
                       label=n, zorder=3)
        for b, is_lost in zip(bars, lost):
            if is_lost:
                b.set_hatch("////")
                b.set_edgecolor("#fcfcfb")
        # 'x' marker above any lost bar
        for xi, b, is_lost in zip(x + (j - 1) * w, bars, lost):
            if is_lost:
                ax2.annotate("×", (xi, b.get_height() + ceil * 0.02),
                             ha="center", va="bottom", color=INK,
                             fontsize=11, fontweight="bold")
    ax2.set_xticks(x, [f"{m:g}" for m in mags_bar])
    ax2.set_ylim(0, ceil * 1.12)
    ax2.set_xlabel("kick magnitude Δψ [rad]", color=INK2)
    ax2.set_ylabel("recovery time to ±2% of kick [s]", color=INK2)
    ax2.set_title("post-kick recovery time per magnitude\n"
                  "hatched ×  = no recovery within the 5 s window",
                  color=INK, fontsize=10.5, loc="left")
    style(ax2)
    ax2.legend(frameon=False, fontsize=9, ncol=3, loc="upper left",
               labelcolor=INK2)

    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGS / "fig_b_sweep.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# shape sweep: state kick vs input kick vs input pulse (checkpoint Sec. 5)
# --------------------------------------------------------------------------- #
def shape_sweep(sat_map=U_SAT_PRIMARY) -> dict:
    """psi-ISE increments and SMC/PID overshoot per disturbance shape.

    All at t = 5 s, under the Sec.-1 saturation map.  The psi increments are
    compared with the paper's own Table 4 - Table 3 footprint, the diagnostic
    behind the input-channel hypothesis; nothing here establishes what the
    original Simulink model actually applied.
    """
    pulse_k = int(round(INPUT_PULSE_S / T0))
    nodist = run_all(disturbance=None, sat_map=sat_map)
    # state kicks go through the normal disturbance hook; input-channel shapes
    # need wrapped controllers, so they run individually
    shapes = {"state_kick_0.05rad": run_all(disturbance=make_kick(0.05),
                                             sat_map=sat_map)}
    for label, (v, k0, k1) in {
        "input_kick_1V_1sample": (INPUT_KICK_V, KICK_K, KICK_K + 1),
        "input_pulse_1V_0.1s": (INPUT_KICK_V, KICK_K, KICK_K + pulse_k),
    }.items():
        runs = {}
        for n in NAMES:
            ctrl = (DESIGN.smc if n == "SMC" else
                    (DESIGN.lqr_ctrl if n == "LQR" else DESIGN.new_pid()))
            runs[n] = run_controller(n, with_input_pulse(ctrl, v, k0, k1),
                                     None, u_sat=sat_map[n])
        shapes[label] = runs

    out = {}
    for label, runs in shapes.items():
        rows = {}
        for n in NAMES:
            inc = ise_increment_10s(runs[n], nodist[n])
            over = psi_overshoot(runs[n])
            rows[n] = {"psi_increment_x1e3": round(inc, 4),
                       "vs_paper_increment":
                           round(inc / PAPER_INCREMENT[n], 3),
                       "psi_overshoot_after_kick_rad": round(over, 5)}
        out[label] = rows
    return out


def state_kick_match_amplitude(target: float, controller: str,
                               sat_map=U_SAT_PRIMARY) -> float:
    """State-kick amplitude whose PID psi-increment matches a target footprint.

    Uses the fitted quadratic growth of the increment (log-log slope over the
    primary grid) to invert amplitude(footprint); a cross-check that the
    paper's Table 4 - Table 3 psi footprint corresponds to a state-channel
    kick of roughly 0.1-0.15 rad in our pipeline.
    """
    sweep = magnitude_sweep(sat_map)
    xs = np.log10(np.asarray(MAGNITUDES, float))
    ys = np.log10([sweep[f"{m:g}"][controller]["psi_ise_post_kick_increment_x1e3"]
                   for m in MAGNITUDES])
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(10 ** ((np.log10(target * 1e3) - intercept) / slope))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
class _Design:
    """Holder so run_controller/run_all can reach the shared designs."""


DESIGN = _Design()


def main() -> None:
    p, Ad, Bd, K, lqr_info, S, smc_info, smc, new_pid = build_designs()
    DESIGN.Ad, DESIGN.Bd = Ad, Bd
    DESIGN.smc, DESIGN.new_pid = smc, new_pid
    DESIGN.lqr_ctrl = LQRController(K)

    print("Fig. A (pitch response + phase portrait) ...")
    fig_a()

    print("Magnitude sweep, Sec.-1 configurations (PID +/-12 V, LQR/SMC "
          "unsaturated) ...")
    sweep = magnitude_sweep(U_SAT_PRIMARY)
    fits = fit_exponents(sweep)
    for n in NAMES:
        print(f"  {n}: fitted exponent {fits[n]['exponent']:.2f}")
    margins = {}
    for n in NAMES:
        lost = [float(m) for m in MAGNITUDES_EXT
                if not sweep[f"{m:g}"][n]["recovered"]]
        margins[n] = {"loses_recovery_beyond_rad": lost[0] if lost else None,
                      "recovered_at_sweep_ceiling":
                          not lost or lost[0] > MAGNITUDES_EXT[-1]}
    for n in NAMES:
        print(f"  {n}: recovery lost beyond "
              f"{margins[n]['loses_recovery_beyond_rad']} rad")

    print("Variant sweep, Fig. 4 block at +/-9 V for every controller ...")
    sweep9 = magnitude_sweep(U_SAT_ALL9)
    margins9 = {}
    for n in NAMES:
        lost = [float(m) for m in MAGNITUDES_EXT
                if not sweep9[f"{m:g}"][n]["recovered"]]
        margins9[n] = {"loses_recovery_beyond_rad": lost[0] if lost else None,
                       "recovered_at_sweep_ceiling":
                           not lost or lost[0] > MAGNITUDES_EXT[-1]}
    for n in NAMES:
        print(f"  {n}: recovery lost beyond "
              f"{margins9[n]['loses_recovery_beyond_rad']} rad")

    print("Fig. B (log-log sweep + recovery bars) ...")
    fig_b(sweep, fits)

    print("Shape sweep (state kick vs input kick vs input pulse) ...")
    shapes = shape_sweep(U_SAT_PRIMARY)
    for label, rows in shapes.items():
        inc = {n: rows[n]["psi_increment_x1e3"] for n in NAMES}
        over_ratio = rows["SMC"]["psi_overshoot_after_kick_rad"] / \
            rows["PID"]["psi_overshoot_after_kick_rad"]
        print(f"  {label}: psi increments {inc} (paper +0.395/+0.229/+0.325), "
              f"SMC/PID overshoot {over_ratio:.2f}")

    print("Cross-check: state-kick amplitude matching the paper's PID "
          "psi-increment footprint ...")
    amp = state_kick_match_amplitude(PAPER_INCREMENT["PID"], "PID")
    print(f"  ~{amp:.3f} rad")

    payload = {
        "generated_by": "scripts/make_checkpoint_figures.py",
        "protocol": {
            "kick": "one-sample additive pulse on psi at t = 5 s (step 500)",
            "primary_saturation": "PID +/-12 V (identified, run_pid.py); "
                                  "LQR and SMC unsaturated - the Section-1 "
                                  "configurations",
            "variant_saturation": "+/-9 V for every controller (the Fig. 4 "
                                  "block at the NXT supply rail)",
            "x0": X0.tolist(),
            "n_steps": N_STEPS,
            "fit_grid_rad": MAGNITUDES,
            "margin_grid_rad": MAGNITUDES_EXT,
            "recovery_band": "2% of the kick, within the post-kick window",
            "note": ("checkpoint Sec. 5 protocol. The regression-to-print "
                     "hypotheses below are supported as CONSISTENCIES, not "
                     "established as the original authors' simulation "
                     "choices; the original Simulink model is unavailable."),
        },
        "paper_increment_targets_x1e3": {k: v * 1e3 for k, v in PAPER_INCREMENT.items()},
        "fitted_exponents": fits,
        "recovery_loss_margins": margins,
        "fitted_exponents_9v_variant": fit_exponents(sweep9),
        "recovery_loss_margins_9v_variant": margins9,
        "state_kick_amplitude_matching_paper_PID_footprint_rad":
            round(amp, 4),
        "magnitude_sweep": sweep,
        "magnitude_sweep_9v_variant": sweep9,
        "shape_sweep": shapes,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    print(f"\nWrote {OUT_FIGS / 'fig_a_pitch_phase.png'}")
    print(f"Wrote {OUT_FIGS / 'fig_b_sweep.png'}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
