#!/usr/bin/env python3
"""Exp 6 baselines (interpolation setup): Waddington-OT + zero drift.

Interpolation evaluation: predict held-out day 15 from day 9 (the
nearest training snapshot) using:

WOT
---
Direct entropic OT (Sinkhorn) coupling between day 9 and day 21,
followed by McCann (displacement) interpolation at fraction
tau = (15-9)/(21-9) = 0.5.  This is the natural WOT use case
(coupling adjacent training snapshots that bracket the held-out
time point), as opposed to the multi-step forward-propagation setup.

Zero drift
----------
Cells stay at their day-9 position; the day-15 prediction equals
the day-9 sample.

Output
------
``output/exp6/wot_predictions_interp.npz`` and
``output/exp6/zero_predictions_interp.npz``.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.distance import cdist


def sinkhorn(a: np.ndarray, b: np.ndarray, M: np.ndarray, *,
             reg: float, num_iter: int = 2000, tol: float = 1e-8) -> np.ndarray:
    log_a = np.log(np.maximum(a, 1e-300))
    log_b = np.log(np.maximum(b, 1e-300))
    log_K = -M / reg
    log_u = np.zeros_like(log_a)
    log_v = np.zeros_like(log_b)
    for it in range(num_iter):
        log_u_new = log_a - _logsumexp(log_K + log_v[None, :], axis=1)
        log_v_new = log_b - _logsumexp(log_K + log_u_new[:, None], axis=0)
        if (np.max(np.abs(log_u_new - log_u)) < tol
                and np.max(np.abs(log_v_new - log_v)) < tol):
            log_u, log_v = log_u_new, log_v_new
            break
        log_u, log_v = log_u_new, log_v_new
    return np.exp(log_K + log_u[:, None] + log_v[None, :])


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp6")
os.makedirs(OUT, exist_ok=True)

EPS_FACTOR = 0.05
N_PRED = 1500
SEED = 2026


def load_pools():
    payload = np.load(os.path.join(OUT, "eb_pca.npz"))
    X, days = payload["X"], payload["days"]
    pools = {}
    for day in [3.0, 9.0, 15.0, 21.0, 27.0]:
        pools[day] = X[days == day].copy()
    return pools


def _sinkhorn_bary(X_src, X_dst, *, eps_factor):
    M = cdist(X_src, X_dst, "sqeuclidean")
    median_sq = float(np.median(M))
    eps = eps_factor * median_sq
    a = np.ones(X_src.shape[0]) / X_src.shape[0]
    b = np.ones(X_dst.shape[0]) / X_dst.shape[0]
    pi = sinkhorn(a, b, M, reg=eps, num_iter=2000, tol=1e-8)
    row_sums = pi.sum(1, keepdims=True)
    T = (pi @ X_dst) / np.maximum(row_sums, 1e-12)
    return T, eps, median_sq


def run_wot_interp(pools):
    """Interpolation setup: Sinkhorn(day 9 -> day 21) + McCann at fraction 0.5.

    This is the canonical WOT use: couple the two training snapshots
    that bracket the held-out time point, then displacement-interpolate.
    """
    rng = np.random.default_rng(SEED)
    p9, p21 = pools[9.0], pools[21.0]
    idx9 = rng.choice(len(p9), N_PRED, replace=False)
    idx21 = rng.choice(len(p21), N_PRED, replace=False)
    X9 = p9[idx9]
    X21 = p21[idx21]

    print("  Sinkhorn(day 9 -> day 21) ...")
    T_9to21, eps, med = _sinkhorn_bary(X9, X21, eps_factor=EPS_FACTOR)
    print(f"    median^2={med:.2f}, eps={eps:.2f}")

    # McCann interpolation at tau = (15-9)/(21-9) = 0.5
    tau = 0.5
    X15_pred = (1.0 - tau) * X9 + tau * T_9to21

    return {
        "pred_at_held_out": X15_pred,
        "X9_used": X9,
        "X21_used": X21,
        "barycentric_9to21": T_9to21,
        "eps_9to21": eps,
        "tau": tau,
    }


def run_zero_interp(pools):
    """Zero drift starting from day 9: cells stay at day 9."""
    rng = np.random.default_rng(SEED + 1)
    p9 = pools[9.0]
    idx9 = rng.choice(len(p9), N_PRED, replace=False)
    X9 = p9[idx9]
    return {"pred_at_held_out": X9.copy(), "x_start": X9}


def main() -> None:
    print("=" * 72)
    print("Exp 6 baselines (interpolation): WOT + zero drift")
    print("  predict day 15 from day 9 (1-step, neighbor-interpolation)")
    print("=" * 72)
    pools = load_pools()
    for day, pool in pools.items():
        print(f"  pool day {day}: {pool.shape[0]} cells")

    print("\n-- WOT (interpolation: Sinkhorn(day 9 -> day 21) + McCann@0.5) --")
    wot = run_wot_interp(pools)
    out_path = os.path.join(OUT, "wot_predictions_interp.npz")
    np.savez_compressed(
        out_path,
        held_out_day=15.0,
        held_out_t=0.5,
        eps_9to21=wot["eps_9to21"],
        tau=wot["tau"],
        **{k: v for k, v in wot.items()
           if k.startswith("pred_") or k.startswith("X")
              or k.startswith("barycentric_")},
    )
    print(f"  saved: {out_path}")

    print("\n-- Zero drift (day 9 samples held fixed) --")
    zd = run_zero_interp(pools)
    out_path = os.path.join(OUT, "zero_predictions_interp.npz")
    np.savez_compressed(out_path, held_out_day=15.0, held_out_t=0.5, **zd)
    print(f"  saved: {out_path}")

    with open(os.path.join(OUT, "baselines_meta_interp.json"), "w") as f:
        json.dump({
            "wot": {"eps_9to21": float(wot["eps_9to21"]),
                    "tau": wot["tau"],
                    "n_pred": N_PRED},
            "zero": {"n_pred": N_PRED, "start_day": 9.0},
        }, f, indent=2)


if __name__ == "__main__":
    main()
