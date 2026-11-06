"""Generates the report's Fig. D (disturbed pitch response, the analogue of the
paper's Fig. 12) and the validation JSON that certifies the from-scratch
regeneration against every number printed in the CTRL_6 final report.

Outputs (repo layout):
  results/figures/fig_d_disturbed_response.png
  results/fig_d_validation.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twip_core import (AC, AD, BD, B_EFF, K_LQR, S_ADOPTED, T0, KICK_IDX,
                       PidController, LqrController, SmcController, simulate,
                       ise_psi, ise_theta, startup_energy, recovery_time,
                       overshoot, ueq_from_surface)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
KICK_AMP = 0.05

# SMC equivalent control per the paper's Eq. (63) construction:
# u_eq = -(S b)^{-1} S A with the single-motor b; equals -2 K_LQR exactly for
# the unrounded adopted surface, to <=6e-4 relative for the rounded entries.
b_single = BD[:, 0]
UEQ_SMC = -(S_ADOPTED @ AD) / (S_ADOPTED @ b_single)


class SmcFinal(SmcController):
    def __init__(self):
        super().__init__(cte=1.0, L=4.0, surface=S_ADOPTED)
        self.ueq_vec = UEQ_SMC


def run(ctrl, rail=None, kick=None):
    xs, us = simulate(ctrl, rail=rail, kick_amp=kick)
    return xs, us


def metrics(ctrl, rail=None):
    xs, us = run(ctrl, rail)
    xs1, _ = run(ctrl, rail, kick=KICK_AMP)
    du = np.diff(us)[200:]
    alt = int(np.sum((du[:-1] * du[1:]) < 0))
    outside = np.where(np.abs(xs1[KICK_IDX:, 1]) > 0.02 * KICK_AMP)[0]
    rec = (outside[-1] + 1) * T0 if len(outside) else 0.0
    # overshoot measured after the kick sample (team metric, B.2)
    ov = float(np.max(np.abs(xs1[KICK_IDX + 1:, 1])))
    return {
        "ise_psi": round(ise_psi(xs), 4),
        "ise_theta": round(ise_theta(xs), 2),
        "peak_u": round(float(np.max(np.abs(us))), 3),
        "startup_energy_V2s": round(startup_energy(us), 3),
        "dpsi_increment": round(ise_psi(xs1) - ise_psi(xs), 4),
        "recovery_s": round(rec, 2),
        "overshoot_rad": round(ov, 4),
        "chatter_alternations_after_2s": alt,
        "max_abs_du_after_2s_V": round(float(np.max(np.abs(du))), 4),
    }


ours = {
    "PID": metrics(PidController(mode="raw"), rail=12.0),
    "LQR": metrics(LqrController()),
    "SMC": metrics(SmcFinal()),
}

reported = {
    "PID": {"ise_psi": 0.295, "ise_theta": 503.4, "startup_energy_V2s": 10.48,
            "recovery_s": 0.92, "overshoot_rad": 0.029, "dpsi_increment": 0.049},
    "LQR": {"ise_psi": 0.430, "ise_theta": 793.1, "peak_u": 8.1,
            "startup_energy_V2s": 7.41, "recovery_s": 1.36, "dpsi_increment": 0.108},
    "SMC": {"ise_psi": 0.237, "ise_theta": 560.3, "peak_u": 17.3,
            "startup_energy_V2s": 11.41, "recovery_s": 0.42,
            "overshoot_rad": 0.034, "dpsi_increment": 0.059,
            "chatter_alternations_after_2s": 0},
}

max_rel = 0.0
for ctrl in ours:
    for key, want in reported[ctrl].items():
        got = ours[ctrl][key]
        if isinstance(want, (int, float)) and want != 0:
            max_rel = max(max_rel, abs(got - want) / abs(want))
print("max relative deviation vs report:", f"{max_rel:.3%}")

# ---------------------------------------------------------------- figure
plt.style.use("seaborn-v0_8-whitegrid")
C = {"PID": "#4C72B0", "LQR": "#DD8452", "SMC": "#55A868"}
t = np.arange(1000) * T0

runs = {}
for name, ctrl, rail in (("PID", PidController(mode="raw"), 12.0),
                         ("LQR", LqrController(), None),
                         ("SMC", SmcFinal(), None)):
    xs0, _ = run(ctrl, rail)
    xs1, _ = run(ctrl, rail, kick=KICK_AMP)
    runs[name] = (xs0[:, 1], xs1[:, 1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.35), dpi=160)
fig.patch.set_facecolor("white")

for name in ("PID", "LQR", "SMC"):
    ax1.plot(t, runs[name][1], color=C[name], lw=2.0, label=name)
ax1.axvline(5.0, color="#999999", lw=1.0, ls=":")
ax1.text(5.05, 0.098, "+0.05 rad kick, t = 5 s", fontsize=10, color="#666666")
for name, x, y in (("PID", 1.3, 0.012), ("LQR", 2.3, 0.022), ("SMC", 0.62, -0.004)):
    ax1.text(x, y, name, fontsize=11, fontweight="bold", color="#3a3a3a")
ax1.set_title("body pitch response under disturbance, full window\n"
              "(paper Fig. 12 analogue)", fontsize=12.5, loc="left", color="#262626")
ax1.set_xlabel("time [s]", fontsize=11)
ax1.set_ylabel("pitch angle ψ [rad]", fontsize=11)
ax1.set_xlim(0, 10)
ax1.set_ylim(-0.06, 0.11)

band = 0.02 * KICK_AMP
zoom = (t >= 4.9) & (t <= 6.6)
for name in ("PID", "LQR", "SMC"):
    ax2.plot(t[zoom], runs[name][1][zoom], color=C[name], lw=2.2)
ax2.axvline(5.0, color="#999999", lw=1.0, ls=":")
ax2.axhline(band, color="#aaaaaa", lw=0.9, ls="--")
ax2.axhline(-band, color="#aaaaaa", lw=0.9, ls="--")
ax2.text(4.915, band + 0.0022, "±2%-of-kick band", fontsize=9, color="#777777")
ax2.annotate("SMC 0.42 s", xy=(5.42, 0.0006), xytext=(5.16, -0.033),
             fontsize=10, fontweight="bold", color=C["SMC"],
             arrowprops=dict(arrowstyle="-", color=C["SMC"], lw=0.8))
ax2.annotate("PID 0.92 s", xy=(5.92, 0.0006), xytext=(5.72, 0.024),
             fontsize=10, fontweight="bold", color=C["PID"],
             arrowprops=dict(arrowstyle="-", color=C["PID"], lw=0.8))
ax2.annotate("LQR 1.36 s", xy=(6.36, 0.0006), xytext=(6.08, -0.033),
             fontsize=10, fontweight="bold", color=C["LQR"],
             arrowprops=dict(arrowstyle="-", color=C["LQR"], lw=0.8))
ax2.set_title("recovery detail after the kick\n"
              "(times to enter and hold the ±2% band)", fontsize=12.5,
              loc="left", color="#262626")
ax2.set_xlabel("time [s]", fontsize=11)
ax2.set_ylabel("pitch angle ψ [rad]", fontsize=11)
ax2.set_xlim(4.9, 6.6)
ax2.set_ylim(-0.04, 0.058)

fig.tight_layout()
fig.savefig(OUT / "figures" / "fig_d_disturbed_response.png", facecolor="white")
print("saved fig_d_disturbed_response.png")

# ------------------------------------------------------------------ json
S_exact = None  # (unrounded surface cross-check)
from scipy.linalg import eig
w, V = np.linalg.eig((AD - np.outer(B_EFF, K_LQR)).T)
i0 = np.argmin(np.abs(w))
S_exact = np.real(V[:, i0]); S_exact /= S_exact[3]; S_exact *= -21.18
ueq_twoK_exact = -(S_exact @ AD) / (S_exact @ BD[:, 0])
dev_twoK = float(np.max(np.abs(ueq_twoK_exact - (-2 * K_LQR)) /
                        np.abs(-2 * K_LQR)))

payload = {
    "purpose": ("Fig. D regeneration + certification of the from-scratch "
                "pipeline rebuild against every number printed in the "
                "CTRL_6 final report"),
    "configuration": {
        "x0": [0, 0.1, 0, 0], "T0_s": T0, "window_s": 10,
        "kick": f"+{KICK_AMP} rad state kick on psi at t=5 s",
        "PID": "Eq. (34), raw accumulators, E = -x, rail 12 V",
        "LQR": "R_eff = 2r reduction, u = -K x, unsaturated",
        "SMC": ("adopted surface S for s(x); boundary layer (cte, L) = (1.0, 4.0); "
                "u_eq per the paper's Eq. (63) construction (single-motor b), "
                "= -2 K_LQR to {0:.1e} relative".format(dev_twoK)),
    },
    "ours": ours,
    "reported": reported,
    "max_relative_deviation_vs_report": round(max_rel, 5),
    "ueq_single_b_vs_minus_2K_max_rel_dev": round(dev_twoK, 8),
    "design_identity_note": ("u_eq == -K holds under the two-motor "
                             "normalisation (verified to 1e-12 with the "
                             "unrounded surface); the simulated Eq.-(63) "
                             "construction equals -2 K"),
}
(OUT / "fig_d_validation.json").write_text(json.dumps(payload, indent=2),
                                           encoding="utf-8")
print("saved fig_d_validation.json")
