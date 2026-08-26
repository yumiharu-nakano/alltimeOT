#!/usr/bin/env python3
"""Accuracy of the smoothed Waddington-OT field on EB (interpolation setup).

exp6_eb_bootstrap_v2.py shows that a Nadaraya-Watson smoothing of the
Waddington-OT displacements with the median-distance bandwidth is far more
stable under resampling than either the nearest-neighbour lookup or the
all-time network.  This script measures what that smoothing does to the
one accuracy that EB can check, held-out marginal tracking.

Setup (identical to exp6_eb_baselines_interp.py): Sinkhorn(day 9 -> day 21)
with eps = 0.05 * median cost, barycentric displacement D(x) = T(x) - x for
each day-9 cell, prediction of day 15 as x + 0.5 * D_h(x), where D_h is the
Nadaraya-Watson smoothing of D with Gaussian bandwidth h = c * h_med and
h_med the median pairwise distance among day-9 cells.  c = 0 is the raw
McCann prediction of the paper, c = 1 is the rule of the stability table,
and c = infinity is a pure translation by the mean displacement.

For every c the script reports
  - SW2 and MMD of the prediction against held-out day 15 (same projections,
    seeds and MMD bandwidth as exp6_eb_evaluate_interp.py),
  - the fraction of the field's energy carried by its spatial variation,
    sum_x |D_h(x) - mean D_h|^2 / sum_x |D_h(x)|^2,
  - bootstrap stability over K = 20 resamples of 80 % of the day-9 and
    day-21 cells, evaluated at a fixed set of 1000 day-9 query cells, as
    median s_q and median rsd_q in the units of the stability table
    (velocity = displacement / 0.5).

Output
------
- output/exp6/smoothed_accuracy.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alltime_ot.simulate import sliced_w2
from experiments.exp6_eb_baselines_interp import (
    load_pools, _sinkhorn_bary, EPS_FACTOR, N_PRED, SEED,
)
from experiments.exp6_eb_evaluate_interp import mmd_gauss, median_pair, N_PROJ

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp6")
os.makedirs(OUT, exist_ok=True)

C_LIST = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, np.inf]
K_BOOT = 20
BOOT_FRAC = 0.80
N_QUERY = 1000
DT = 0.5                      # day 9 -> day 21 in the units of the paper
TAU = 0.5                     # held-out day 15 sits halfway


def median_distance(X, n=400, seed=0):
    """Same rule as eval_wot_smooth in exp6_eb_bootstrap_v2.py."""
    sub = X[np.random.default_rng(seed).choice(len(X), min(len(X), n),
                                               replace=False)]
    return float(np.median(np.linalg.norm(sub[:, None, :] - sub[None, :, :],
                                          axis=-1)))


def nw_smooth(X_src, D, x_eval, h):
    """Nadaraya-Watson regression of D on X_src, evaluated at x_eval.
    h = 0 is the nearest-neighbour rule, h = inf the global mean."""
    if h == 0.0:
        d2 = (np.sum(x_eval ** 2, 1)[:, None] + np.sum(X_src ** 2, 1)[None, :]
              - 2.0 * x_eval @ X_src.T)
        return D[np.argmin(d2, axis=1)]
    if not np.isfinite(h):
        return np.tile(D.mean(0), (len(x_eval), 1))
    d2 = (np.sum(x_eval ** 2, 1)[:, None] + np.sum(X_src ** 2, 1)[None, :]
          - 2.0 * x_eval @ X_src.T)
    logw = -np.maximum(d2, 0.0) / (2.0 * h * h)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return (w @ D) / w.sum(axis=1, keepdims=True)


def spatial_fraction(F):
    """Fraction of the field energy carried by spatial variation."""
    return float(((F - F.mean(0)) ** 2).sum() / np.maximum((F ** 2).sum(), 1e-12))


def label(c):
    return "nn" if c == 0.0 else ("translation" if not np.isfinite(c) else f"c{c:g}")


def main():
    pools = load_pools()
    X_true = pools[15.0]
    h_mmd = median_pair(X_true)
    print(f"MMD bandwidth {h_mmd:.3f}, N_PROJ {N_PROJ}", flush=True)

    # ----- full-data coupling, identical to the paper's WOT row -------
    rng = np.random.default_rng(SEED)
    p9, p21 = pools[9.0], pools[21.0]
    X9 = p9[rng.choice(len(p9), N_PRED, replace=False)]
    X21 = p21[rng.choice(len(p21), N_PRED, replace=False)]
    tic = time.time()
    T9, eps, _ = _sinkhorn_bary(X9, X21, eps_factor=EPS_FACTOR)
    print(f"Sinkhorn(9->21) eps={eps:.2f} [{time.time()-tic:.0f}s]", flush=True)
    D = T9 - X9
    h_med = median_distance(X9)
    print(f"h_med (day 9) = {h_med:.3f}, mean |D| = "
          f"{np.linalg.norm(D, axis=1).mean():.3f}", flush=True)

    acc = {}
    for c in C_LIST:
        Dh = nw_smooth(X9, D, X9, c * h_med if np.isfinite(c) else np.inf)
        pred = X9 + TAU * Dh
        acc[label(c)] = {
            "c": None if not np.isfinite(c) else float(c),
            "h": None if not np.isfinite(c) else float(c * h_med),
            "sw2": float(sliced_w2(pred, X_true, n_proj=N_PROJ, seed=2027)),
            "mmd": float(mmd_gauss(pred, X_true, h=h_mmd)),
            "spatial_fraction": spatial_fraction(Dh),
            "mean_speed": float(np.linalg.norm(Dh / DT, axis=1).mean()),
        }
        a = acc[label(c)]
        print(f"  {label(c):<12} SW2 {a['sw2']:.4f}  MMD {a['mmd']:.4f}  "
              f"spatial frac {a['spatial_fraction']:.3f}  "
              f"speed {a['mean_speed']:.2f}", flush=True)
    # zero-drift reference, as in the paper
    acc["zero"] = {"sw2": float(sliced_w2(X9, X_true, n_proj=N_PROJ, seed=2027)),
                   "mmd": float(mmd_gauss(X9, X_true, h=h_mmd))}
    print(f"  {'zero':<12} SW2 {acc['zero']['sw2']:.4f}  "
          f"MMD {acc['zero']['mmd']:.4f}", flush=True)

    # ----- bootstrap stability at fixed day-9 queries -----------------
    q_rng = np.random.default_rng(SEED + 100)
    x_q = p9[q_rng.choice(len(p9), N_QUERY, replace=False)]
    runs = {label(c): np.zeros((K_BOOT, N_QUERY, X9.shape[1])) for c in C_LIST}
    for k in range(K_BOOT):
        tic = time.time()
        brng = np.random.default_rng(SEED + k)
        n9 = int(round(BOOT_FRAC * len(p9))); n21 = int(round(BOOT_FRAC * len(p21)))
        B9 = p9[brng.choice(len(p9), n9, replace=False)]
        B21 = p21[brng.choice(len(p21), n21, replace=False)]
        TB, _, _ = _sinkhorn_bary(B9, B21, eps_factor=EPS_FACTOR)
        DB = TB - B9
        hb = median_distance(B9)
        for c in C_LIST:
            runs[label(c)][k] = nw_smooth(
                B9, DB, x_q, c * hb if np.isfinite(c) else np.inf) / DT
        print(f"  boot {k+1}/{K_BOOT} [{time.time()-tic:.0f}s]", flush=True)

    stab = {}
    for name, R in runs.items():
        sd = R.std(axis=0, ddof=1); mu = R.mean(axis=0)
        s_q = np.sqrt((sd ** 2).sum(axis=1))
        rsd = np.linalg.norm(sd, axis=1) / np.maximum(np.linalg.norm(mu, axis=1), 1e-12)
        stab[name] = {"median_s_q": float(np.median(s_q)),
                      "iqr_s_q": [float(np.percentile(s_q, 25)),
                                  float(np.percentile(s_q, 75))],
                      "median_rsd": float(np.median(rsd)),
                      "mean_speed": float(np.linalg.norm(mu, axis=1).mean())}
        s = stab[name]
        print(f"  {name:<12} median s_q {s['median_s_q']:.3f}  "
              f"median rsd {s['median_rsd']:.3f}  speed {s['mean_speed']:.2f}",
              flush=True)

    print("\nc        h      SW2     MMD     spatial  rsd    s_q")
    for c in C_LIST:
        a, s = acc[label(c)], stab[label(c)]
        hs = "-" if a["h"] is None else f"{a['h']:.2f}"
        print(f"{label(c):<8} {hs:>6}  {a['sw2']:.3f}  {a['mmd']:.4f}  "
              f"{a['spatial_fraction']:.3f}   {s['median_rsd']:.3f}  "
              f"{s['median_s_q']:.2f}")

    path = os.path.join(OUT, "smoothed_accuracy.json")
    with open(path, "w") as f:
        json.dump({"h_med": h_med, "h_mmd": h_mmd, "eps": float(eps),
                   "K_BOOT": K_BOOT, "n_query": N_QUERY,
                   "accuracy": acc, "stability": stab}, f, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
