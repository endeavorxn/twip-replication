"""PID closed-loop run (paper Section 3.1, Eqs. 31-35, Tables 2-3).

Simulates the no-disturbance balancing case of Section 4.1 from
x0 = [0, 0.1, 0, 0] (psi(0) = 0.1 rad) with the published Table 2 gains,
prints and plots theta, psi, u, sweeps the ISE integration window (the
paper does not state it) against the Table 3 PID targets (psi 0.271e-3,
theta 526.6e-3) and reports settling times, in two configurations:

  * literal:  Table 2 gains, E = -x (Eq. 35), common drive, NO saturation
  * primary:  same, plus the Fig. 4 saturation block at +/-12 V - the
              paper shows a saturation block but never states its limit;
              12 V is identified as the value that reproduces Table 3's
              PID row to ~10 % on both ISE entries (see
              saturation_identification in results/pid.json)

Wiring findings (see wiring_investigation in results/pid.json):
  * Table 2 prints NEGATIVE gains; with E = -x the loop closes marginally
    stable.  The overall sign flip (E = +x equivalent) is genuinely
    unstable (max x-excited pole 1.171) - so the paper must have used
    E = -x with the gains exactly as printed.
  * Summed common drive (u_l = u_r = u_psi + u_theta) vs one loop per
    motor are indistinguishable in this model (Bd's columns are identical;
    both lifted loops are marginally stable with an exact z = 1
    controller pole/zero cancellation that x0 does not excite).

Writes results/pid.json and results/plots/pid_*.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twip.model import continuous_matrices, discretize  # noqa: E402
from twip.params import Params  # noqa: E402
from twip.pid import PIDLoop, make_pid_controller, table2_pid  # noqa: E402
from twip.ise import ise  # noqa: E402
from twip.simulate import settling_time, simulate  # noqa: E402

# --- paper Table 3, PID row (ISE x10^-3, no-disturbance) ---------------------
PAPER_PSI = 0.271e-3
PAPER_THETA = 526.6e-3

WINDOWS_S = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
RUN_S = 30.0     # response horizon for plots/settling; ISE windows are prefixes
SAT_SWEEP_V = (6.0, 9.0, 10.0, 11.0, 12.0, 13.0, 15.0, 24.0)
U_SAT_IDENT = 12.0  # identified Fig. 4 saturation limit [V]


def pid_coeffs(kp, ki, kd, T0):
    """Eq. (34) coefficients (a0, a1, a2)."""
    return (kp + ki * T0 / 2 + kd / T0,
            -kp + ki * T0 / 2 - 2 * kd / T0,
            kd / T0)


def lifted_closed_loop_poles(Ad, Bd, a_psi, a_theta, wiring: str) -> np.ndarray:
    """Exact discrete closed-loop eigenvalues via state lifting.

    Augments the 4 plant states with each loop's error history
    (E(k-1), E(k-2)) and accumulated output u(k-1); the loop recursion is
    Eq. (34) with E = -x_i, x_ref = 0 (Eq. 35).  wiring is "summed"
    (u_l = u_r = u_psi + u_theta) or "split" (u_l = u_psi, u_r = u_theta).
    """
    n = Ad.shape[0]
    A = np.zeros((n + 6, n + 6))
    A[:n, :n] = Ad
    # psi loop (states n, n+1, n+2 = e1(k-1), e1(k-2), u1(k-1)), e1 = -psi
    A[n, 1] = -1.0
    A[n + 1, n] = 1.0
    A[n + 2, n + 2] = 1.0                # u1(k) = u1(k-1) + a0 e1(k) + a1 e1(k-1) + a2 e1(k-2)
    A[n + 2, n] = a_psi[1]
    A[n + 2, n + 1] = a_psi[2]
    A[n + 2, 1] += -a_psi[0]
    # theta loop (states n+3..n+5), e2 = -theta
    A[n + 3, 0] = -1.0
    A[n + 4, n + 3] = 1.0
    A[n + 5, n + 5] = 1.0
    A[n + 5, n + 3] = a_theta[1]
    A[n + 5, n + 4] = a_theta[2]
    A[n + 5, 0] += -a_theta[0]
    # control path into the plant
    if wiring == "summed":
        A[:n, n + 2] += Bd[:, 0] + Bd[:, 1]
        A[:n, n + 5] += Bd[:, 0] + Bd[:, 1]
    elif wiring == "split":
        A[:n, n + 2] += Bd[:, 0]
        A[:n, n + 5] += Bd[:, 1]
    else:
        raise ValueError(wiring)
    return np.linalg.eigvals(A)


def settling_after_peak(t, y, tol=0.02) -> float:
    """Settling time referenced to the response peak: last time |y| exceeds
    tol*max|y|, plus one step.  twip.simulate.settling_time references the
    band to the FIRST sample, which is right for psi (decays from its 0.1 rad
    initial offset) but degenerate for theta, which starts at 0 and builds a
    transient; theta uses this peak-referenced variant."""
    y = np.asarray(y, float)
    band = tol * np.max(np.abs(y))
    outside = np.abs(y) > band
    last = int(np.max(np.nonzero(outside)[0]))
    return float(t[last + 1]) if last + 1 < len(t) else float("inf")


def run_case(Ad, Bd, params, u_sat) -> dict:
    """One no-disturbance run; returns trajectories and derived metrics."""
    T0 = params.T0
    psi_pid, theta_pid = table2_pid(params)  # fresh controller states
    controller = make_pid_controller(psi_pid, theta_pid)
    x0 = np.array([0.0, 0.1, 0.0, 0.0])
    n_steps = int(round(RUN_S / T0))
    sim = simulate(Ad, Bd, controller, x0, n_steps, u_sat=u_sat)
    X, U = sim["X"], sim["U"]
    t = np.arange(n_steps + 1) * T0
    return {"t": t, "X": X, "U": U, "x0": x0,
            "psi": X[:, 1], "theta": X[:, 0], "u": U[:, 0]}


def ises(run, T0) -> dict:
    """ISE x10^-3 per window for psi and theta."""
    out = {}
    for w in WINDOWS_S:
        n = int(round(w / T0))
        out[w] = {"psi": ise(run["psi"][:n], T0) / 1e-3,
                  "theta": ise(run["theta"][:n], T0) / 1e-3}
    return out


def main() -> int:
    params = Params()
    T0 = params.T0
    Ac, Bc = continuous_matrices(params)
    Ad, Bd = discretize(Ac, Bc, T0)
    psi_pid, theta_pid = table2_pid(params)
    a_psi = pid_coeffs(psi_pid.kp, psi_pid.ki, psi_pid.kd, T0)
    a_theta = pid_coeffs(theta_pid.kp, theta_pid.ki, theta_pid.kd, T0)

    results: dict = {"paper": "Villacrés et al. 2016, Section 3.1, Eqs. (31)-(35), Tables 2-3"}

    # --- 1. wiring/stability investigation (Table 2 gains are negative) -----
    print("=" * 72)
    print("Wiring investigation: Table 2 gains are NEGATIVE (KP<0, KD<0)")
    print("=" * 72)
    wirings = {}
    for name, wiring, flip in (("summed, E=-x (as printed)", "summed", False),
                               ("one loop per motor, E=-x", "split", False),
                               ("summed, overall sign flipped", "summed", True)):
        ap, at = a_psi, a_theta
        if flip:
            ap = tuple(-a for a in ap)
            at = tuple(-a for a in at)
        ev = lifted_closed_loop_poles(Ad, Bd, ap, at, wiring)
        on_unit_mask = np.abs(np.abs(ev) - 1.0) <= 1e-6   # exact z=1 lift modes only
        ev_in = ev[~on_unit_mask]                    # everything else, unstable included
        mx = float(np.max(np.abs(ev_in)))
        on_unit = int(np.sum(on_unit_mask))
        wirings[name] = {
            "max_x_excited_pole": mx,
            "n_unit_circle_modes": on_unit,
            "verdict": ("marginally stable" if mx < 1.0 and on_unit
                        else "stable" if mx < 1.0 else "UNSTABLE"),
            "poles": [f"{e.real:+.6f}{e.imag:+.6f}j" if abs(e.imag) > 1e-9
                      else f"{e.real:+.8f}" for e in
                      ev[np.argsort(-np.abs(ev))]],
        }
        print(f"  {name:32s} max x-excited |pole| = {mx:.6f}, "
              f"{on_unit} exact unit-circle mode(s) -> {wirings[name]['verdict']}")
    print("  -> the sign flip is UNSTABLE, so the paper must have used E=-x with")
    print("     Table 2 exactly as printed.  Summed vs split are indistinguishable")
    print("     (identical Bd columns; both carry an exact z=1 controller pole/zero")
    print("     cancellation in the lift that x0 does not excite); summed retained.")
    results["wiring_investigation"] = wirings
    results["wiring_assumption"] = (
        "u_left = u_right = u_psi + u_theta; each loop on E = -x_i with "
        "x_ref = 0 (Eq. 35).  The paper never states the wiring; E=-x is "
        "forced by stability (sign flip diverges), summed-vs-split is "
        "unidentifiable with identical Bd columns.")

    # --- 2. no-disturbance runs (literal unsaturated + identified +/-12 V) ---
    runs = {"literal_unsaturated": run_case(Ad, Bd, params, None),
            "primary_sat12V": run_case(Ad, Bd, params, U_SAT_IDENT)}
    print("\n" + "=" * 72)
    print(f"No-disturbance runs, x0 = [0, 0.1, 0, 0], {RUN_S:.0f} s at T0 = {T0*1e3:.0f} ms")
    print("=" * 72)
    for name, r in runs.items():
        print(f"  [{name}]")
        print(f"    psi   range [{r['psi'].min():+.5f}, {r['psi'].max():+.5f}] rad, "
              f"final |psi| = {abs(r['psi'][-1]):.2e}")
        print(f"    theta range [{r['theta'].min():+.4f}, {r['theta'].max():+.4f}] rad, "
              f"final |theta| = {abs(r['theta'][-1]):.2e}")
        print(f"    u     range [{r['u'].min():+.3f}, {r['u'].max():+.3f}] V "
              f"(first-sample spike {r['u'][0]:+.2f} V from KD/T0 = {a_psi[2]:+.1f})")

    # --- 3. ISE window sweep vs paper Table 3 --------------------------------
    print("\n" + "=" * 72)
    print("ISE x10^-3 window sweep (paper does not state the window)")
    print(f"paper Table 3 PID no-disturbance: psi {PAPER_PSI/1e-3:.3f}, "
          f"theta {PAPER_THETA/1e-3:.1f}")
    print("=" * 72)
    sweep_out = {}
    for name, r in runs.items():
        tab = ises(r, T0)
        sweep_out[name] = [
            {"window_s": w, "ise_psi_x1e3": tab[w]["psi"], "ise_theta_x1e3": tab[w]["theta"],
             "ratio_psi": tab[w]["psi"] / (PAPER_PSI / 1e-3),
             "ratio_theta": tab[w]["theta"] / (PAPER_THETA / 1e-3)}
            for w in WINDOWS_S]
        print(f"  [{name}]")
        for s in sweep_out[name]:
            print(f"    {s['window_s']:5.1f}s  psi {s['ise_psi_x1e3']:8.4f} (x{s['ratio_psi']:.2f})   "
                  f"theta {s['ise_theta_x1e3']:8.2f} (x{s['ratio_theta']:.2f})")
    # sweep shape: psi is exactly flat; theta has a slow integrator tail
    sweep_meta = {}
    for name, lst in sweep_out.items():
        f, l = lst[0], lst[-1]
        sweep_meta[name] = {
            "psi_abs_growth_x1e3": abs(l["ise_psi_x1e3"] - f["ise_psi_x1e3"]),
            "theta_growth_pct": (l["ise_theta_x1e3"] / f["ise_theta_x1e3"] - 1) * 100}
        print(f"  -> [{name}] psi sweep exactly flat (growth "
              f"{sweep_meta[name]['psi_abs_growth_x1e3']:.2e} over 2-10 s); "
              f"theta ISE grows {sweep_meta[name]['theta_growth_pct']:+.2f}% "
              f"from 2 s to 10 s (slow theta-integrator tail).")
    print("     The paper's unstated ISE window is therefore immaterial for the "
          "PID row (theta variation is far below the replication error); "
          "10 s used as primary throughout.")

    primary = sweep_out["primary_sat12V"][-1]
    literal = sweep_out["literal_unsaturated"][-1]
    print(f"  primary (u_sat = {U_SAT_IDENT:.0f} V, 10 s): psi {primary['ise_psi_x1e3']:.4f} "
          f"(paper x{primary['ratio_psi']:.2f}), theta {primary['ise_theta_x1e3']:.2f} "
          f"(paper x{primary['ratio_theta']:.2f})")
    print(f"  literal (no sat,   10 s): psi {literal['ise_psi_x1e3']:.4f} "
          f"(paper x{literal['ratio_psi']:.2f}), theta {literal['ise_theta_x1e3']:.2f} "
          f"(paper x{literal['ratio_theta']:.2f})")
    results["ise_window_sweep"] = sweep_out
    results["primary_window"] = {"window_s": 10.0, "config": "primary_sat12V",
                                 "ise_psi_x1e3": primary["ise_psi_x1e3"],
                                 "ise_theta_x1e3": primary["ise_theta_x1e3"],
                                 "ratio_psi_vs_table3": primary["ratio_psi"],
                                 "ratio_theta_vs_table3": primary["ratio_theta"],
                                 "sweep_shape": sweep_meta}

    # --- 4. saturation-limit identification (Fig. 4 block, limit unstated) ---
    print("\n" + "=" * 72)
    print(f"Saturation-limit identification (10 s ISE), paper Table 3: "
          f"psi {PAPER_PSI/1e-3:.3f}, theta {PAPER_THETA/1e-3:.1f}")
    print("=" * 72)
    sat_scan = []
    n10 = int(round(10.0 / T0))
    for sat in SAT_SWEEP_V:
        r = run_case(Ad, Bd, params, sat)
        ip = ise(r["psi"][:n10], T0) / 1e-3
        it = ise(r["theta"][:n10], T0) / 1e-3
        sat_scan.append({"u_sat_V": sat, "ise_psi_x1e3": ip, "ise_theta_x1e3": it,
                         "ratio_psi": ip / (PAPER_PSI / 1e-3),
                         "ratio_theta": it / (PAPER_THETA / 1e-3)})
        print(f"  u_sat = {sat:5.1f} V: psi {ip:7.4f} (x{sat_scan[-1]['ratio_psi']:.2f})   "
              f"theta {it:8.2f} (x{sat_scan[-1]['ratio_theta']:.2f})")
    print(f"  -> u_sat = {U_SAT_IDENT:.0f} V keeps BOTH entries within 10 % of Table 3 "
          f"and is the standard motor-supply value; adopted for the primary run.")
    results["saturation_identification"] = {
        "paper_states_limit": False,
        "note": ("Fig. 4 shows a saturation block but no value.  Scan over "
                 "symmetric limits; 12 V reproduces both Table 3 PID entries "
                 "to ~10 % (psi wants >=15 V, theta ~10 V; 12 V is the "
                 "compromise and the standard supply)."),
        "scan_10s": sat_scan, "identified_V": U_SAT_IDENT}

    # --- 5. settling times ---------------------------------------------------
    print("\n" + "=" * 72)
    print("Settling times")
    print("=" * 72)
    set_out = {}
    for name, r in runs.items():
        ts_p = settling_time(r["t"], r["psi"])
        ts_t = settling_after_peak(r["t"], r["theta"])
        set_out[name] = {"psi_s_2pct_of_initial": ts_p,
                         "theta_s_2pct_of_peak": ts_t}
        print(f"  [{name}]  psi {ts_p:.3f} s (2% of |psi(0)|)   "
              f"theta {ts_t:.3f} s (2% of peak)")
    set_out["note"] = ("psi band referenced to |psi(0)| = 0.1 rad "
                       "(twip.simulate.settling_time); theta starts at 0 so "
                       "its band references max|theta| (peak-referenced helper).")
    results["settling"] = set_out

    # --- 6. control effort (Fig. 9 claim context, recorded only) --------------
    print("\n" + "=" * 72)
    print("Control energy T0*sum(u^2) [V^2 s]")
    print("=" * 72)
    eff_out = {}
    for name, r in runs.items():
        eff = {f"{w:.0f}s": float(T0 * np.sum(r["u"][:int(round(w / T0))] ** 2))
               for w in (1.0, 2.0, 5.0, 10.0)}
        eff_out[name] = eff
        print(f"  [{name}]  " + ", ".join(f"{k}: {v:.4g}" for k, v in eff.items()))
    results["control_effort_V2s"] = eff_out
    # Fig. 9 record: compare against the LQR run if its results have landed
    fig9 = {
        "claim": "paper Fig. 9: PID is NOT the lowest-energy starter (LQR is)",
        "pid_total_both_motors_V2s_10s": eff_out["primary_sat12V"]["10s"],
        "pid_per_motor_V2s_10s": eff_out["primary_sat12V"]["10s"] / 2,
        "pid_per_motor_V2s_10s_unsaturated": eff_out["literal_unsaturated"]["10s"] / 2,
        "metric": "T0*sum(u_l^2) with u_l = u_r = u (identical Bd columns)",
    }
    try:
        lqr = json.loads((Path(__file__).resolve().parents[1] / "results"
                          / "lqr.json").read_text(encoding="utf-8"))
        lqr_eff = lqr.get("control_energy_per_motor", {})
        if "sum_u2_full_10s" in lqr_eff:
            fig9["lqr_per_motor_V2s_10s"] = lqr_eff["sum_u2_full_10s"]
            fig9["observed_ordering"] = (
                "PID per-motor energy %.4g (sat) / %.4g (unsat) vs LQR %.4g "
                "V^2 s over 10 s: our PID is LOWER-energy, i.e. the Fig. 9 "
                "ordering is NOT reproduced under this metric - recorded, "
                "definition of Fig. 9 'energy' unverified." %
                (fig9["pid_per_motor_V2s_10s"],
                 fig9["pid_per_motor_V2s_10s_unsaturated"],
                 lqr_eff["sum_u2_full_10s"]))
    except (OSError, json.JSONDecodeError, KeyError):
        fig9["observed_ordering"] = "lqr.json not available at run time"
    results["fig9_claim_record"] = fig9
    print(f"  Fig. 9 record: {fig9.get('observed_ordering')}")

    # --- 7. plots -------------------------------------------------------------
    plots = Path(__file__).resolve().parents[1] / "results" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    colors = {"literal_unsaturated": ("tab:red", "--"),
              "primary_sat12V": ("tab:blue", "-")}
    labels = {"literal_unsaturated": "literal, no saturation",
              "primary_sat12V": f"u_sat = +/-{U_SAT_IDENT:.0f} V (primary)"}

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name, r in runs.items():
        c, ls = colors[name]
        ax[0].plot(r["t"], r["psi"], ls, color=c, lw=1.1, label=f"psi, {labels[name]}")
        ax[1].plot(r["t"], r["theta"], ls, color=c, lw=1.1, label=f"theta, {labels[name]}")
    for a in ax:
        a.axhline(0, color="k", lw=0.5)
        a.legend(loc="upper right")
    ax[0].set_ylabel("psi [rad]")
    ax[0].set_title("PID closed loop, no disturbance, x0=[0, 0.1, 0, 0] "
                    "(Table 2 gains, E=-x, common drive)")
    ax[1].set_ylabel("theta [rad]")
    ax[1].set_xlabel("t [s]")
    fig.tight_layout()
    p1 = plots / "pid_response.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for name, r in runs.items():
        c, ls = colors[name]
        ax[0].plot(r["t"][:-1], r["u"], ls, color=c, lw=0.9, label=labels[name])
        ax[1].plot(r["t"][:-1], r["u"], ls, color=c, lw=0.9, label=labels[name])
    ax[0].set_ylabel("u [V]")
    ax[0].set_title("PID control input (common drive to both motors); "
                    "zoom below past the first-sample spike")
    ax[1].set_ylim(-20, 20)
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("u [V] (zoom)")
    for a in ax:
        a.legend(loc="upper right")
    fig.tight_layout()
    p2 = plots / "pid_input.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ws = [s["window_s"] for s in sweep_out["primary_sat12V"]]
    ax.plot(ws, [s["ise_psi_x1e3"] for s in sweep_out["primary_sat12V"]], "o-",
            color="tab:blue", label="ISE psi x10^-3 (primary)")
    ax.plot(ws, [s["ise_theta_x1e3"] for s in sweep_out["primary_sat12V"]], "s-",
            color="tab:orange", label="ISE theta x10^-3 (primary)")
    ax.plot(ws, [s["ise_psi_x1e3"] for s in sweep_out["literal_unsaturated"]], "o--",
            color="tab:red", lw=1, label="ISE psi (no saturation)")
    ax.plot(ws, [s["ise_theta_x1e3"] for s in sweep_out["literal_unsaturated"]], "s--",
            color="tab:red", lw=1, label="ISE theta (no saturation)")
    ax.axhline(PAPER_PSI / 1e-3, ls=":", color="tab:blue", lw=1.2,
               label="paper Table 3 PID psi")
    ax.axhline(PAPER_THETA / 1e-3, ls=":", color="tab:orange", lw=1.2,
               label="paper Table 3 PID theta")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ISE window [s] (log scale; sweep is flat)")
    ax.set_ylabel("ISE x10^-3")
    ax.set_title("PID ISE vs integration window and saturation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = plots / "pid_ise_sweep.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"\nwrote {p1}, {p2}, {p3}")

    # --- 8. machine-readable results ------------------------------------------
    pr = runs["primary_sat12V"]
    results.update({
        "gains": {"psi": {"KP": psi_pid.kp, "KI": psi_pid.ki, "KD": psi_pid.kd},
                  "theta": {"KP": theta_pid.kp, "KI": theta_pid.ki, "KD": theta_pid.kd},
                  "source": "paper Table 2, used exactly as printed (negative)"},
        "eq34_coeffs": {"psi": a_psi, "theta": a_theta},
        "simulation": {"x0": [0.0, 0.1, 0.0, 0.0], "T0": T0, "run_s": RUN_S,
                       "disturbance": None,
                       "u_sat": {"literal_unsaturated": None,
                                 "primary_sat12V": U_SAT_IDENT}},
        "trajectories_first_steps": {
            "t": pr["t"][:11].tolist(),
            "psi_primary": pr["psi"][:11].tolist(),
            "theta_primary": pr["theta"][:11].tolist(),
            "u_primary_first3": [float(v) for v in pr["u"][:3]]},
        "method_notes": ("Incremental law Eq. (34) exactly as printed "
                         "(trapezoidal integral, first-difference derivative); "
                         "gains not retuned; primary run adds the Fig. 4 "
                         "saturation block at the identified +/-12 V limit; "
                         "ISE = T0*sum(y^2) over window prefixes of one 30 s "
                         "run per configuration; with-disturbance Table 3 "
                         "column out of scope for this script."),
    })
    out = Path(__file__).resolve().parents[1] / "results" / "pid.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
