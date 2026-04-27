#!/usr/bin/env python3
"""Appendix A: sensitivity analysis on (M, N, lambda).

Base problem: Experiment 1 (1-d Gaussian translation), affine drift model
u(t, x) = w0 + w1 t + w2 x.  We sweep each of M (time slices), N
(particles per slice), and lambda (penalty weight) one at a time,
keeping the other two at their default values, and measure grid MSE
of the learned drift.  Each configuration runs K_seed independent
realisations, each ensemble-averaged over K_ens RNG draws.
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alltime_ot import EnsembleObjective, ensemble_lbfgs, feat_affine
from alltime_ot.problems import gaussian_translation

# ---- Problem ---------------------------------------------------------
T = 1.0

def mean_fn(t: float) -> np.ndarray:
    return np.array([-1.0 + 2.0 * t])

U_TRUE = 2.0
H = 1.0

OUT = os.environ.get("ALLTIME_OT_OUT", "output/appA")
os.makedirs(OUT, exist_ok=True)


def run_one_point(M: int, N: int, lam: float, K_ens: int, seed_offset: int) -> float:
    """Return grid MSE for one realisation (ensemble-averaged over K_ens)."""
    N0 = max(30, 2 * N)
    provider = gaussian_translation(mean_fn=mean_fn, d=1, M=M, N=N, N0=N0, T=T)
    objective = EnsembleObjective(
        batch_provider=provider,
        feat_fn=feat_affine,
        n_params=3, d=1,
        lam=lam, h=H, T=T,
        K_ens=K_ens, seed_offset=seed_offset,
    )
    w, _, _ = ensemble_lbfgs(
        objective, [np.zeros(3)],
        maxiter=300, verbose=False,
    )
    # Grid MSE on [0, 1] x [-3, 3]
    te = np.linspace(0, T, 30)
    xe = np.linspace(-3, 3, 50)
    TT, XX = np.meshgrid(te, xe, indexing="ij")
    uh = w[0] + w[1] * TT + w[2] * XX
    return float(((uh - U_TRUE) ** 2).mean())


def sweep(values, varying, defaults, K_seed, K_ens):
    print(f"\n-- Sweep: {varying} --")
    means, stds = [], []
    for v in values:
        args = dict(defaults)
        args[varying] = v
        tic = time.time()
        mse_list = [
            run_one_point(
                M=args["M"], N=args["N"], lam=args["lam"],
                K_ens=K_ens, seed_offset=10_000 * s + 111,
            )
            for s in range(K_seed)
        ]
        mu = float(np.mean(mse_list))
        sd = float(np.std(mse_list))
        means.append(mu)
        stds.append(sd)
        print(f"  {varying}={v:<8}  MSE = {mu:.5f} +/- {sd:.5f}  "
              f"(K_seed={K_seed}, t={time.time() - tic:.1f}s)")
    return np.array(means), np.array(stds)


def main() -> None:
    defaults = {"M": 30, "N": 20, "lam": 1000.0}
    K_seed = 4
    K_ens = 5

    M_values = [10, 15, 20, 30, 50]
    N_values = [5, 10, 15, 25, 40]
    lam_values = [1e1, 1e2, 1e3, 1e4, 1e5]

    print("Appendix A: Sensitivity analysis on (M, N, lambda)")
    print(f"  Defaults: {defaults}, K_seed={K_seed}, K_ens={K_ens}")

    M_mean, M_std = sweep(M_values, "M", defaults, K_seed, K_ens)
    N_mean, N_std = sweep(N_values, "N", defaults, K_seed, K_ens)
    la_mean, la_std = sweep(lam_values, "lam", defaults, K_seed, K_ens)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, values, means, stds, xlabel, marker, color, title in [
        (axes[0], M_values, M_mean, M_std, "$M$ (time slices)", "o", "C0",
         f"vs $M$  (N={defaults['N']}, $\\lambda$={int(defaults['lam'])})"),
        (axes[1], N_values, N_mean, N_std, "$N$ (particles per slice)", "s", "C2",
         f"vs $N$  (M={defaults['M']}, $\\lambda$={int(defaults['lam'])})"),
        (axes[2], lam_values, la_mean, la_std, r"$\lambda$", "^", "C3",
         f"vs $\\lambda$  (M={defaults['M']}, N={defaults['N']})"),
    ]:
        ax.errorbar(values, means, yerr=stds, fmt=f"{marker}-", color=color,
                    lw=2, ms=7, capsize=4)
        ax.set(xlabel=xlabel, xscale="log", yscale="log", title=title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"grid MSE$(\hat u, u^*)$")
    fig.suptitle("Appendix A: sensitivity analysis (Experiment 1 base problem)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "defaults": defaults,
        "K_seed": K_seed, "K_ens": K_ens,
        "M": {"values": M_values, "mean": M_mean.tolist(), "std": M_std.tolist()},
        "N": {"values": N_values, "mean": N_mean.tolist(), "std": N_std.tolist()},
        "lam": {"values": lam_values, "mean": la_mean.tolist(), "std": la_std.tolist()},
    }
    with open(f"{OUT}/sensitivity.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved figure and JSON to {OUT}/")


if __name__ == "__main__":
    main()
