#!/usr/bin/env python3
"""Evaluate Exp 6 predictions against the held-out day-15 distribution.

Loads ``alltime_predictions.npz`` (our method),
``wot_predictions.npz`` (Sinkhorn baseline) and
``zero_predictions.npz`` (zero drift) and the held-out day-15
ground truth from the preprocessed PCA archive, then computes

- mean sliced Wasserstein-2 (over 200 random projections),
- Gaussian-kernel MMD with bandwidth equal to the median heuristic,

for each method.  Outputs a LaTeX table (stdout), a JSON summary,
and a 2D PCA visualisation comparing predicted and held-out clouds.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alltime_ot.simulate import sliced_w2

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp6")
N_PROJ = 200
RNG_PROJ = np.random.default_rng(2026)


def mmd_gauss(X: np.ndarray, Y: np.ndarray, *, h: float) -> float:
    """Gaussian-kernel MMD^2 (unbiased estimator)."""
    a = 1.0 / (h * h)
    nX, nY = X.shape[0], Y.shape[0]
    # XX
    dXX = np.sum(X * X, axis=1)[:, None] + np.sum(X * X, axis=1)[None, :] - 2 * X @ X.T
    KXX = np.exp(-0.5 * a * dXX)
    KXX[np.diag_indices(nX)] = 0.0
    # YY
    dYY = np.sum(Y * Y, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - 2 * Y @ Y.T
    KYY = np.exp(-0.5 * a * dYY)
    KYY[np.diag_indices(nY)] = 0.0
    # XY
    dXY = np.sum(X * X, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - 2 * X @ Y.T
    KXY = np.exp(-0.5 * a * dXY)
    return float(KXX.sum() / (nX * (nX - 1)) + KYY.sum() / (nY * (nY - 1))
                 - 2.0 * KXY.mean())


def median_pair(X: np.ndarray, n: int = 500) -> float:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs = X[idx]
    d = np.sqrt(((Xs[:, None] - Xs[None, :]) ** 2).sum(-1))
    return float(np.median(d[d > 0]))


def main() -> None:
    print("=" * 72)
    print("Exp 6 evaluation: marginal-tracking metrics at all 5 time points")
    print("(All-time OT propagates forward via ODE; WOT chains Sinkhorn maps)")
    print("=" * 72)

    payload = np.load(os.path.join(OUT, "eb_pca.npz"))
    X_all, days = payload["X"], payload["days"]
    X_true = X_all[days == 15.0]
    print(f"  held-out day 15: {X_true.shape}")

    h = median_pair(X_true)
    print(f"  MMD bandwidth (median heuristic on held-out): {h:.3f}")

    methods = {}
    # All-time
    a = np.load(os.path.join(OUT, "alltime_predictions.npz"))
    methods["All-time (ours)"] = a["sim_t_0.50"]
    # WOT
    w = np.load(os.path.join(OUT, "wot_predictions.npz"))
    methods["WOT (Sinkhorn)"] = w["pred_t_0.50"]
    # Zero
    z = np.load(os.path.join(OUT, "zero_predictions.npz"))
    methods["Zero drift"] = z["pred_t_0.50"]

    # Reference: bootstrap floor — sliced W2 between two halves of true day 15.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_true))
    half = len(X_true) // 2
    A = X_true[perm[:half]]
    B = X_true[perm[half:2 * half]]
    sw2_floor = sliced_w2(A, B, n_proj=N_PROJ, seed=2026)
    mmd_floor = mmd_gauss(A, B, h=h)
    print(f"\nMonte-Carlo floor (split true day 15 in half):"
          f"  SW2 = {sw2_floor:.4f},  MMD = {mmd_floor:.4f}")

    results = {"floor": {"sw2": sw2_floor, "mmd": mmd_floor, "h": h}}
    print(f"\n  {'method':24s} {'SW2':>10s} {'MMD':>10s}")
    print("  " + "-" * 46)
    for name, pred in methods.items():
        sw = sliced_w2(pred, X_true, n_proj=N_PROJ, seed=2027)
        mm = mmd_gauss(pred, X_true, h=h)
        results[name] = {"sw2": float(sw), "mmd": float(mm)}
        print(f"  {name:24s} {sw:10.4f} {mm:10.4f}")

    # LaTeX table.
    print("\nLaTeX table rows:")
    for name, r in results.items():
        if name == "floor":
            label = "MC floor (split true)"
        else:
            label = name
        print(f"  {label:24s} & ${r['sw2']:.4f}$ & ${r['mmd']:.4f}$ \\\\")

    with open(os.path.join(OUT, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 2D PCA visualisation: project all clouds to first 2 PCA directions.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True, sharey=True)
    cmap = {
        "All-time (ours)": "C1",
        "WOT (Sinkhorn)": "C2",
        "Zero drift": "C7",
    }
    # Use first 2 PCA dims of pre-computed PCA.
    for i, (name, pred) in enumerate(methods.items()):
        ax = axes[i]
        ax.scatter(X_true[:, 0], X_true[:, 1], s=4, alpha=0.25, color="C0",
                   label="held-out day 15")
        ax.scatter(pred[:, 0], pred[:, 1], s=4, alpha=0.4, color=cmap[name],
                   label=f"predicted ({name})")
        ax.set(title=f"{name}\nSW$_2$ = {results[name]['sw2']:.3f},  MMD = {results[name]['mmd']:.3f}",
               xlabel="PC1", ylabel="PC2" if i == 0 else "")
        ax.legend(fontsize=8, loc="best")
    # Reference panel: true vs true split.
    ax = axes[3]
    ax.scatter(A[:, 0], A[:, 1], s=4, alpha=0.3, color="C0", label="true day 15 (half A)")
    ax.scatter(B[:, 0], B[:, 1], s=4, alpha=0.3, color="C3", label="true day 15 (half B)")
    ax.set(title=f"MC floor: split true\nSW$_2$ = {sw2_floor:.3f},  MMD = {mmd_floor:.3f}",
           xlabel="PC1")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Experiment 6 (EB scRNA-seq, $d=30$): held-out day 15 prediction",
                 y=1.01, fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(OUT, "eb_evaluation.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")


if __name__ == "__main__":
    main()
