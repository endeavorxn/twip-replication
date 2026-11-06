"""A.6(d) data: SMC realisation, printed-vs-adopted surface, and boundary-layer
sensitivity of the Table-3 ISE levels. Outputs results/bl_sensitivity.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twip_core import (AD, BD, S_ADOPTED, S_PRINT, T0, K_LQR, SmcController,
                       simulate, ise_psi, ise_theta)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
b_single = BD[:, 0]


def ueq_for(S):
    """Paper's Eq. (63) construction: u_eq = -(S b)^{-1} S A, single-motor b."""
    return -(S @ AD) / (S @ b_single)


class Smc(SmcController):
    def __init__(self, ueq_vec, cte, L, surface):
        self.cte, self.L, self.S = cte, L, surface
        self.ueq_vec = ueq_vec


def table3(ueq_vec, cte, L, surface):
    xs, _ = simulate(Smc(ueq_vec, cte, L, surface))
    return round(ise_psi(xs), 4), round(ise_theta(xs), 2)


res = {}

# 1. realisation: Eq.-63 construction (= -2 K) vs design identity (- K)
res["realisation"] = {
    "eq63_construction_minus_2K": dict(zip(("psi", "theta"),
                                           table3(ueq_for(S_ADOPTED), 1.0, 4.0, S_ADOPTED))),
    "design_identity_minus_K": dict(zip(("psi", "theta"),
                                        table3(-K_LQR, 1.0, 4.0, S_ADOPTED))),
    "note": ("-K realisation: SMC psi-ISE 0.42 > PID 0.295 -> the paper's "
             "SMC-lowest psi ranking would NOT reproduce; the simulated "
             "Eq.-(63) realisation is therefore load-bearing and is the "
             "configuration used everywhere in this report"),
}

# 2. printed vs adopted surface, both under the Eq.-63 construction
res["surface"] = {
    "adopted_S": dict(zip(("psi", "theta"),
                          table3(ueq_for(S_ADOPTED), 1.0, 4.0, S_ADOPTED))),
    "printed_S_eq62": dict(zip(("psi", "theta"),
                               table3(ueq_for(S_PRINT), 1.0, 4.0, S_PRINT))),
    "printed_eq63_vector": dict(zip(("psi", "theta"),
                                    table3(np.array([1.924, 160.270, 2.717, 21.581]),
                                           1.0, 4.0, S_ADOPTED))),
}

# 3. boundary-layer grid (valid: L > 2 cte), Eq.-63 construction, adopted S
grid = {}
for cte in (0.5, 1.0, 2.0):
    for L in (2.0, 4.0, 8.0, 16.0):
        if L > 2 * cte:
            grid[f"cte={cte}_L={L}"] = dict(zip(("psi", "theta"),
                                                table3(ueq_for(S_ADOPTED), cte, L, S_ADOPTED)))
res["boundary_layer_grid"] = grid
psis = [v["psi"] for v in grid.values()]
thetas = [v["theta"] for v in grid.values()]
res["boundary_layer_range"] = {
    "psi_min": min(psis), "psi_max": max(psis),
    "theta_min": min(thetas), "theta_max": max(thetas),
    "rankings_intact": all(
        v["psi"] < 0.295 and 503.4 < v["theta"] < 793.1
        for v in grid.values()),
}

(OUT / "bl_sensitivity.json").write_text(json.dumps(res, indent=2),
                                         encoding="utf-8")
print(json.dumps(res, indent=2))
