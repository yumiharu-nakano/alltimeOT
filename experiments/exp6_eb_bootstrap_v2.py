#!/usr/bin/env python3
"""Exp 6 bootstrap stability, revised.

experiments/exp6_eb_bootstrap_stability.py fits each method on five
80%-subsamples and compares the across-bootstrap variance of the inferred
velocity at a fixed query set.  Three things limit what that comparison can
say, and this script addresses them.

1.  Five bootstrap runs give a noisy variance estimate.  Here K = 20.

2.  A raw variance is not comparable across methods whose velocity fields
    have different scales.  Here we also report the relative standard
    deviation at each query,

        rsd(q) = || std_k u_k(q) || / || mean_k u_k(q) ||,

    and summarize it by its median over queries.

3.  The original comparison pits a spatially continuous network against a
    Waddington-OT field obtained by nearest-neighbour lookup, so it mixes
    statistical stability with the interpolation rule.  Here Waddington-OT is
    evaluated in two ways: the original nearest-neighbour rule, and a
    Nadaraya-Watson smoothing of the same barycentric displacements with the
    median-distance bandwidth, which yields a continuous field like the
    network.  The three-way comparison separates the two effects.

Output
------
- output/exp6/bootstrap_stability_v2.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.exp6_eb_bootstrap_stability as bs

K_BOOT = 20
OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp6")
os.makedirs(OUT, exist_ok=True)


def eval_wot_smooth(couplings, t_q, x_q):
    """Waddington-OT velocity as a Nadaraya-Watson regression of the
    barycentric displacement, rather than a nearest-neighbour lookup."""
    Q, D = len(t_q), x_q.shape[1]
    out = np.zeros((Q, D))
    cache = {}
    for q in range(Q):
        i = next((k for k in range(len(bs.TS) - 1)
                  if bs.TS[k] <= t_q[q] < bs.TS[k + 1]), len(bs.TS) - 2)
        if i not in cache:
            X_i, _, T_ij = couplings[i]
            dt = bs.TS[i + 1] - bs.TS[i]
            disp = (T_ij - X_i) / dt
            sub = X_i[np.random.default_rng(0).choice(
                len(X_i), min(len(X_i), 400), replace=False)]
            h = float(np.median(np.linalg.norm(
                sub[:, None, :] - sub[None, :, :], axis=-1)))
            cache[i] = (X_i, disp, max(h, 1e-8))
        X_i, disp, h = cache[i]
        w = np.exp(-np.sum((X_i - x_q[q]) ** 2, axis=1) / (2.0 * h * h))
        s = w.sum()
        out[q] = disp.mean(0) if s <= 1e-300 else (w[:, None] * disp).sum(0) / s
    return out


def summarize(runs):
    """runs: (K, Q, D).  Returns raw variance and relative-std summaries."""
    sd = runs.std(axis=0, ddof=1)                      # (Q, D)
    mu = runs.mean(axis=0)                             # (Q, D)
    var_q = (sd ** 2).sum(axis=1)                      # (Q,)
    rsd = np.linalg.norm(sd, axis=1) / np.maximum(
        np.linalg.norm(mu, axis=1), 1e-12)             # (Q,)
    f = lambda a: {"mean": float(np.mean(a)), "median": float(np.median(a)),
                   "q25": float(np.percentile(a, 25)),
                   "q75": float(np.percentile(a, 75))}
    return {"total_variance": f(var_q), "relative_sd": f(rsd),
            "mean_speed": float(np.mean(np.linalg.norm(mu, axis=1)))}


def main():
    print(f"K_BOOT = {K_BOOT}", flush=True)
    X_all, days = bs.load_data()
    X_dim = X_all.shape[1]
    t_q, x_q = bs.make_query_set(X_all, days, seed=bs.BASE_SEED + 100)
    Q = len(t_q)
    print(f"d = {X_dim}, queries = {Q}", flush=True)

    ours = np.zeros((K_BOOT, Q, X_dim))
    wot_nn = np.zeros((K_BOOT, Q, X_dim))
    wot_sm = np.zeros((K_BOOT, Q, X_dim))

    for k in range(K_BOOT):
        tic = time.time()
        pools = bs.bootstrap_pools(X_all, days, frac=bs.BOOT_FRAC,
                                   seed=bs.BASE_SEED + k)
        model = bs.train_alltime(pools, X_dim, seed=bs.BASE_SEED + k)
        ours[k] = bs.eval_alltime(model, t_q, x_q)
        cpl = bs.train_wot(pools)
        wot_nn[k] = bs.eval_wot(cpl, t_q, x_q)
        wot_sm[k] = eval_wot_smooth(cpl, t_q, x_q)
        print(f"  boot {k+1}/{K_BOOT} [{time.time()-tic:.0f}s]", flush=True)

    res = {"ours": summarize(ours),
           "wot_nearest_neighbour": summarize(wot_nn),
           "wot_smoothed": summarize(wot_sm)}
    for name, r in res.items():
        print(f"{name:<24} median rel.sd = {r['relative_sd']['median']:.3f}"
              f"   median total var = {r['total_variance']['median']:.4g}"
              f"   mean speed = {r['mean_speed']:.3f}", flush=True)

    path = os.path.join(OUT, "bootstrap_stability_v2.json")
    with open(path, "w") as f:
        json.dump({"K_BOOT": K_BOOT, "n_query": Q, "results": res}, f, indent=2)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
