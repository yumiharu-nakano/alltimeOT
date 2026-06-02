#!/usr/bin/env python3
"""Evaluate Exp 6 (interpolation) predictions against held-out day 15.

Compares: All-time (day 9 -> 15, 1-step ODE), WOT (Sinkhorn(9,21) +
McCann@0.5), zero drift (cells stay at day 9).  Reports SW2 and MMD
against the held-out day-15 distribution.
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


def mmd_gauss(X: np.ndarray, Y: np.ndarray, *, h: float) -> float:
    a = 1.0 / (h * h)
    nX, nY = X.shape[0], Y.shape[0]
    dXX = np.sum(X * X, axis=1)[:, None] + np.sum(X * X, axis=1)[None, :] - 2 * X @ X.T
    KXX = np.exp(-0.5 * a * dXX)
    KXX[np.diag_indices(nX)] = 0.0
    dYY = np.sum(Y * Y, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - 2 * Y @ Y.T
    KYY = np.exp(-0.5 * a * dYY)
    KYY[np.diag_indices(nY)] = 0.0
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
    print("Exp 6 (interpolation) evaluation: held-out day 15")
    print("  All-time: day 9 -> day 15 (1-step ODE)")
    print("  WOT:      Sinkhorn(day 9, day 21) + McCann@0.5")
    print("  Zero:     stay at day 9")
    print("=" * 72)

    payload = np.load(os.path.join(OUT, "eb_pca.npz"))
    X_all, days = payload["X"], payload["days"]
    X_true = X_all[days == 15.0]
    print(f"  held-out day 15: {X_true.shape}")

    h = median_pair(X_true)
    print(f"  MMD bandwidth (median heuristic on held-out): {h:.3f}")

    methods = {}
    # All-time (interpolation setup)
    a = np.load(os.path.join(OUT, "alltime_predictions_interp.npz"))
    methods["All-time (ours)"] = a["pred_at_held_out"]
    # WOT (interpolation setup)
    w = np.load(os.path.join(OUT, "wot_predictions_interp.npz"))
    methods["WOT (Sinkhorn)"] = w["pred_at_held_out"]
    # Zero (interpolation setup)
    z = np.load(os.path.join(OUT, "zero_predictions_interp.npz"))
    methods["Zero drift"] = z["pred_at_held_out"]

    # MC floor.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_true))
    half = len(X_true) // 2
    A = X_true[perm[:half]]
    B = X_true[perm[half:2 * half]]
    sw2_floor = sliced_w2(A, B, n_proj=N_PROJ, seed=2026)
    mmd_floor = mmd_gauss(A, B, h=h)
    print(f"\nMonte-Carlo floor (split true day 15):"
          f"  SW2 = {sw2_floor:.4f},  MMD = {mmd_floor:.4f}")

    results = {"floor": {"sw2": sw2_floor, "mmd": mmd_floor, "h": h}}
    print(f"\n  {'method':24s} {'SW2':>10s} {'MMD':>10s}")
    print("  " + "-" * 46)
    for name, pred in methods.items():
        sw = sliced_w2(pred, X_true, n_proj=N_PROJ, seed=2027)
        mm = mmd_gauss(pred, X_true, h=h)
        results[name] = {"sw2": float(sw), "mmd": float(mm)}
        print(f"  {name:24s} {sw:10.4f} {mm:10.4f}")

    print("\nLaTeX table rows:")
    for name, r in results.items():
        if name == "floor":
            label = "Monte-Carlo floor (true split)"
        else:
            label = name
        print(f"  {label:30s} & ${r['sw2']:.4f}$ & ${r['mmd']:.4f}$ \\\\")

    with open(os.path.join(OUT, "evaluation_interp.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Visualisation
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True, sharey=True)
    cmap = {
        "All-time (ours)": "C1",
        "WOT (Sinkhorn)": "C2",
        "Zero drift": "C7",
    }
    for i, (name, pred) in enumerate(methods.items()):
        ax = axes[i]
        ax.scatter(X_true[:, 0], X_true[:, 1], s=4, alpha=0.25, color="C0",
                   label="held-out day 15")
        ax.scatter(pred[:, 0], pred[:, 1], s=4, alpha=0.4, color=cmap[name],
                   label=f"predicted ({name})")
        ax.set(title=f"{name}\nSW$_2$ = {results[name]['sw2']:.3f},  MMD = {results[name]['mmd']:.3f}",
               xlabel="PC1", ylabel="PC2" if i == 0 else "")
        ax.legend(fontsize=8, loc="best")
    ax = axes[3]
    ax.scatter(A[:, 0], A[:, 1], s=4, alpha=0.3, color="C0", label="true day 15 (half A)")
    ax.scatter(B[:, 0], B[:, 1], s=4, alpha=0.3, color="C3", label="true day 15 (half B)")
    ax.set(title=f"MC floor: split true\nSW$_2$ = {sw2_floor:.3f},  MMD = {mmd_floor:.3f}",
           xlabel="PC1")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle("Experiment 6 (interpolation, $d=30$): predict day 15 from day 9",
                 y=1.01, fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(OUT, "eb_evaluation_interp.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")


if __name__ == "__main__":
    main()
