#!/usr/bin/env python3
"""Smoothing the Waddington-OT drift on the bimodal merging flow.

On EB the smoothed Waddington-OT field is the most stable under resampling
(exp6_eb_bootstrap_v2.py) but its accuracy cannot be checked against a
true drift.  Here the same Nadaraya-Watson smoothing is applied to the
Waddington-OT displacements of Experiment 3, where the true drift
u* = -2 tanh(2(1-t)x) is known, so accuracy and stability can be read off
together.

For each snapshot count M in {5, 10, 20, 50}, the couplings are the ones of
exp3_baselines.py (N_sample = 200 per snapshot, eps = dt).  The per-particle
velocity v = (T(x) - x) / dt is turned into a field on [-4, 4] by
  - linear interpolation in x (the rule of Table exp4-baselines), and
  - Nadaraya-Watson smoothing with bandwidth h = c * h_med, where h_med is
    the median pairwise distance among the source particles, for
    c in {0.1, 0.25, 0.5, 1}.
Drift grid MSE against u* is computed on the grid of exp3_baselines.py.
Stability is the across-seed relative dispersion of the field on the grid,
sqrt(mean Var) / sqrt(mean of squared mean), over five sampling seeds.

Output
------
- output/exp3/wot_smoothing.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alltime_ot.baselines import _sinkhorn_log
from experiments.exp3_baselines import (
    sample_bimodal, T_END, t_grid, x_grid, TT, U_STAR,
)

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp3")
os.makedirs(OUT, exist_ok=True)

M_LIST = [5, 10, 20, 50]
C_LIST = [0.1, 0.25, 0.5, 1.0]
SEEDS = [42, 43, 44, 45, 46]
N_SAMPLE = 200
EPS_FACTOR = 1.0


def couplings(M, seed):
    """Source particles and velocities per interval, as in sinkhorn_wot_drift."""
    rng = np.random.default_rng(seed)
    t_pts = np.linspace(0.0, T_END, M + 1)
    dt = T_END / M
    eps = EPS_FACTOR * dt
    samples = [sample_bimodal(t_pts[k], N_SAMPLE, rng) for k in range(M + 1)]
    out = []
    for k in range(M):
        Xs, Xt = samples[k], samples[k + 1]
        C = (Xs[:, None] - Xt[None, :]) ** 2
        a = np.full(N_SAMPLE, 1.0 / N_SAMPLE)
        gamma = _sinkhorn_log(a, a, C, reg=eps)
        Tm = (gamma @ Xt) / (gamma.sum(axis=1) + 1e-30)
        out.append((Xs, (Tm - Xs) / dt))
    return out, dt


def field_linear(Xs, v, x):
    order = np.argsort(Xs)
    xs, vs = Xs[order], v[order]
    _, idx = np.unique(xs, return_index=True)
    xs_u, vs_u = xs[np.sort(idx)], vs[np.sort(idx)]
    return interp1d(xs_u, vs_u, kind="linear", fill_value=(vs_u[0], vs_u[-1]),
                    bounds_error=False)(x)


def field_nw(Xs, v, x, h):
    logw = -(x[:, None] - Xs[None, :]) ** 2 / (2.0 * h * h)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return (w @ v) / w.sum(axis=1)


def grid_field(cpl, dt, rule, c=None):
    M = len(cpl)
    U = np.zeros_like(TT)
    for i_t, t in enumerate(t_grid):
        k = min(max(int(t / dt), 0), M - 1)
        Xs, v = cpl[k]
        if rule == "linear":
            U[i_t] = field_linear(Xs, v, x_grid)
        else:
            h_med = float(np.median(np.abs(Xs[:, None] - Xs[None, :])))
            U[i_t] = field_nw(Xs, v, x_grid, c * h_med)
    return U


def main():
    res = {}
    print(f"{'M':>3} {'rule':<8} {'MSE mean':>9} {'MSE sd':>7} {'rel disp':>9}")
    for M in M_LIST:
        rules = [("linear", None)] + [("nw", c) for c in C_LIST]
        fields = {r: [] for r in rules}
        for seed in SEEDS:
            cpl, dt = couplings(M, seed)
            for r in rules:
                fields[r].append(grid_field(cpl, dt, *r))
        for r in rules:
            F = np.stack(fields[r])                       # (S, T, X)
            mse = ((F - U_STAR) ** 2).mean(axis=(1, 2))   # per seed
            var = F.var(axis=0, ddof=1); mu = F.mean(axis=0)
            rel = float(np.sqrt(var.mean()) / np.sqrt((mu ** 2).mean()))
            name = "linear" if r[0] == "linear" else f"nw_c{r[1]:g}"
            res[f"M{M}_{name}"] = {"M": M, "rule": name,
                                   "mse_mean": float(mse.mean()),
                                   "mse_sd": float(mse.std(ddof=1)),
                                   "rel_dispersion": rel,
                                   "mean_abs_field": float(np.abs(mu).mean())}
            print(f"{M:>3} {name:<8} {mse.mean():>9.3f} {mse.std(ddof=1):>7.3f} "
                  f"{rel:>9.3f}", flush=True)
    zero_mse = float((U_STAR ** 2).mean())
    print(f"zero-drift MSE {zero_mse:.3f}")
    path = os.path.join(OUT, "wot_smoothing.json")
    with open(path, "w") as f:
        json.dump({"seeds": SEEDS, "N_sample": N_SAMPLE, "eps_factor": EPS_FACTOR,
                   "zero_drift_mse": zero_mse, "results": res}, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
