#!/usr/bin/env python3
"""Exp 6 baselines: Waddington-OT (Sinkhorn) and zero drift on EB scRNA-seq.

Both baselines predict the held-out day 15 distribution.

WOT
---
Entropic OT (Sinkhorn) is computed between every pair of consecutive
*training* time points (day 3, 9, 21, 27).  To predict day 15, we use
the (day 9, day 21) coupling and apply McCann (displacement)
interpolation at fraction $\\tau = (15-9)/(21-9) = 0.5$.

Zero drift
----------
``u\\equiv 0`` means cells do not move; the day-15 prediction equals
the day-3 sample.

Output
------
``output/exp6/wot_predictions.npz`` and
``output/exp6/zero_predictions.npz`` containing predicted day-15
samples.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.distance import cdist


def sinkhorn(a: np.ndarray, b: np.ndarray, M: np.ndarray, *,
             reg: float, num_iter: int = 2000, tol: float = 1e-8) -> np.ndarray:
    """Sinkhorn-Knopp entropic OT in log-domain.

    Solves min <pi, M> + reg * KL(pi | a (x) b) over couplings pi with
    marginals (a, b).  Returns the optimal coupling pi.
    """
    log_a = np.log(np.maximum(a, 1e-300))
    log_b = np.log(np.maximum(b, 1e-300))
    log_K = -M / reg                          # (n, m)
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

EPS_FACTOR = 0.05         # entropic-regularization scale (fraction of median^2)
N_PRED = 1500             # cells to use for prediction (matches all-time N_SIM)
SEED = 2026


def load_pools():
    payload = np.load(os.path.join(OUT, "eb_pca.npz"))
    X, days = payload["X"], payload["days"]
    pools = {}
    for day in [3.0, 9.0, 15.0, 21.0, 27.0]:
        pools[day] = X[days == day].copy()
    return pools


def _sinkhorn_bary(X_src, X_dst, *, eps_factor):
    """Sinkhorn between two empirical clouds; return barycentric projection."""
    M = cdist(X_src, X_dst, "sqeuclidean")
    median_sq = float(np.median(M))
    eps = eps_factor * median_sq
    a = np.ones(X_src.shape[0]) / X_src.shape[0]
    b = np.ones(X_dst.shape[0]) / X_dst.shape[0]
    pi = sinkhorn(a, b, M, reg=eps, num_iter=2000, tol=1e-8)
    row_sums = pi.sum(1, keepdims=True)
    T = (pi @ X_dst) / np.maximum(row_sums, 1e-12)
    return T, eps, median_sq


def run_wot(pools):
    """Forward-simulate from day 3 to day 15 by composing
    Sinkhorn(3 -> 9) with McCann interpolation along Sinkhorn(9 -> 21).

    This matches the all-time method's evaluation, which also starts
    from day 3 samples and propagates forward to day 15.
    """
    rng = np.random.default_rng(SEED)
    p3, p9, p21 = pools[3.0], pools[9.0], pools[21.0]
    idx3 = rng.choice(len(p3), N_PRED, replace=False)
    idx9 = rng.choice(len(p9), N_PRED, replace=False)
    idx21 = rng.choice(len(p21), N_PRED, replace=False)
    X3 = p3[idx3]
    X9 = p9[idx9]
    X21 = p21[idx21]

    # Step 1: Sinkhorn(3 -> 9), barycentric projection of X3.
    print("  step 1: Sinkhorn (day 3 -> day 9) ...")
    T_3to9, eps1, med1 = _sinkhorn_bary(X3, X9, eps_factor=EPS_FACTOR)
    print(f"    median^2={med1:.2f}, eps={eps1:.2f}")
    X9_predicted = T_3to9   # cells at day 9 according to WOT, starting from day 3

    # Step 2: Sinkhorn(9 -> 21), barycentric projection of X9_predicted.
    print("  step 2: Sinkhorn (day 9_predicted -> day 21) ...")
    T_9to21, eps2, med2 = _sinkhorn_bary(X9_predicted, X21, eps_factor=EPS_FACTOR)
    print(f"    median^2={med2:.2f}, eps={eps2:.2f}")
    # McCann (displacement) interpolation at tau = (15-9)/(21-9) = 0.5
    tau = 0.5
    X15_pred = (1.0 - tau) * X9_predicted + tau * T_9to21

    return {
        "pred_t_0.50": X15_pred,
        "X3_used": X3,
        "X9_predicted": X9_predicted,
        "X21_used": X21,
        "barycentric_3to9": T_3to9,
        "barycentric_9to21": T_9to21,
        "eps_3to9": eps1,
        "eps_9to21": eps2,
    }


def run_zero(pools):
    """Zero drift: cells stay at their initial position (day 3)."""
    rng = np.random.default_rng(SEED + 1)
    p3 = pools[3.0]
    idx3 = rng.choice(len(p3), N_PRED, replace=False)
    X0 = p3[idx3]
    return {"pred_t_0.50": X0.copy(), "x0": X0}


def main() -> None:
    print("=" * 72)
    print("Exp 6 baselines: WOT (Sinkhorn) + zero drift")
    print("=" * 72)
    pools = load_pools()
    for day, pool in pools.items():
        print(f"  pool day {day}: {pool.shape[0]} cells")

    print("\n-- WOT (forward-simulate 3 -> 9 -> 15 via Sinkhorn + McCann) --")
    wot = run_wot(pools)
    out_path = os.path.join(OUT, "wot_predictions.npz")
    np.savez_compressed(
        out_path,
        held_out_day=15.0,
        held_out_t=0.5,
        eps_3to9=wot["eps_3to9"],
        eps_9to21=wot["eps_9to21"],
        **{k: v for k, v in wot.items()
           if k.startswith("pred_") or k.startswith("X")
              or k.startswith("barycentric_")},
    )
    print(f"  saved: {out_path}")

    print("\n-- Zero drift (day 3 samples held fixed) --")
    zd = run_zero(pools)
    out_path = os.path.join(OUT, "zero_predictions.npz")
    np.savez_compressed(out_path, held_out_day=15.0, held_out_t=0.5, **zd)
    print(f"  saved: {out_path}")

    with open(os.path.join(OUT, "baselines_meta.json"), "w") as f:
        json.dump({
            "wot": {"eps_3to9": float(wot["eps_3to9"]),
                    "eps_9to21": float(wot["eps_9to21"]),
                    "n_pred": N_PRED},
            "zero": {"n_pred": N_PRED},
        }, f, indent=2)


if __name__ == "__main__":
    main()
