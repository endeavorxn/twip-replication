# TWIP Controller Replication (CTRL_6)

Replication of **Villacrés, Viscaíno, Herrera, Camacho (2016), "Controllers Comparison to
Stabilize a Two-Wheeled Inverted Pendulum: PID, LQR and Sliding Mode Control",
*International Journal of Control Systems and Robotics*, Vol. 1, pp. 29–36.**

Team CTRL_6: LyuKaiyuan 23219909, WangWeiran 23219961, PengZhangyi 23219928,
ZhangZhuoyuan 23219926, CaoYubo 23219927, YuXuan 23219950.

## Goal

Reproduce the paper's discrete-time TWIP model and its three controllers (PID, LQR, SMC),
then the ISE comparison of Tables 3–4 (with and without disturbance), starting from the
physical parameters of Table 1 — i.e. the model is **rebuilt from first principles**, not
copied from the paper's printed matrices.

## Layout

```
twip/            reusable library
  params.py        Table 1 physical parameters
  model.py         continuous A1,B1 (Eqs. 4-22), discretization (Eqs. 23-24), poles
  pid.py           discrete PID (Eq. 34), Table 2 gains
  lqr.py           Riccati iteration (Eqs. 40-41), Q/R (Eqs. 43-44), K (Eq. 45)
  smc.py           sliding surface from LQR poles, u_eq (Eqs. 57-63), boundary layer (Eq. 64)
  simulate.py      closed-loop simulation (sampler + ZOH + saturation, Fig. 4 scheme)
  ise.py           integral square error of psi and theta
scripts/         runnable entry points
  verify_model.py  model vs published matrices/poles (Eqs. 21-22, 25-30)
  run_*.py         controller simulations
  compare_ise.py   preliminary ISE table vs paper Tables 3-4
results/         JSON/CSV outputs and plots (committed: summary; ignored: bulk plots)
```

## Usage

```
python scripts/verify_model.py
python scripts/run_pid.py
python scripts/run_lqr.py
python scripts/run_smc.py
python scripts/compare_ise.py
```

Requires Python 3.11+ with numpy, scipy, matplotlib (no MATLAB needed).

## Status

See `results/summary.json` for the latest verified quantities and ISE comparison.
