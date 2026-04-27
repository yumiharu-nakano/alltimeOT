#!/usr/bin/env python3
"""Appendix B: dimension scaling of the RKHS all-time OT estimator.

Base problem: Gaussian translation in spatial dimension d.
    mu_t = N(m_t, I_d),  m_t = t * 1_d / sqrt(d),  T = 1
    true drift u*(t, x) = 1_d / sqrt(d)  (constant, |u*| = 1)

We vary d in {1, 2, 3, 5, 8, 10} and record grid MSE of the learned
drift (mean +/- std over K_seed seeds), per-optimisation wall-clock
time, and the number of free parameters n_params = d (d + 2).
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

def mu_mean_d(t: float, d: int) -> np.ndarray:
    return np.full(d, t / np.sqrt(d))


def u_star_d(d: int) -> np.ndarray:
    return np.full(d, 1.0 / np.sqrt(d))


# ---- Hyper-parameters ------------------------------------------------
LAM = 1000.0
M, N, N0 = 25, 20, 50
K_ENS = 5
K_SEED = 3
H = 1.0

OUT = os.environ.get("ALLTIME_OT_OUT", "output/appB")
os.makedirs(OUT, exist_ok=True)


def run_one_dim(d: int, seed_offset: int) -> tuple[float, float, int, float, int]:
    p_feat = d + 2
    n_params = p_feat * d

    provider = gaussian_translation(
        mean_fn=lambda t: mu_mean_d(t, d), d=d,
        M=M, N=N, N0=N0, T=T,
    )
    objective = EnsembleObjective(
        batch_provider=provider,
        feat_fn=feat_affine,
        n_params=n_params,
        d=d,
        lam=LAM, h=H, T=T,
        K_ens=K_ENS, seed_offset=seed_offset,
    )

    tic = time.time()
    w, _, logs = ensemble_lbfgs(
        objective, [np.zeros(n_params)],
        maxiter=300, verbose=False,
    )
    elapsed = time.time() - tic
    nit = logs[0]["nit"]
    W_hat = w.reshape(p_feat, d)

    # Grid MSE on [0, 1] x [-3, 3]^d (Monte Carlo in x for large d)
    n_grid_t = 15
    n_grid_x = 2000
    te = np.linspace(0, T, n_grid_t)
    rng_g = np.random.default_rng(2024)
    xg = rng_g.uniform(-3, 3, size=(n_grid_x, d))
    tg = np.repeat(te, n_grid_x)
    xg_full = np.tile(xg, (n_grid_t, 1))
    Phi_g = np.column_stack([np.ones(tg.size), tg, xg_full])
    uh = Phi_g @ W_hat
    us = np.broadcast_to(u_star_d(d), uh.shape)
    mse = float(((uh - us) ** 2).sum(-1).mean())

    W_star = np.zeros_like(W_hat)
    W_star[0, :] = 1.0 / np.sqrt(d)
    w_err = float(np.linalg.norm(W_hat - W_star))
    return mse, elapsed, nit, w_err, n_params


def main() -> None:
    print("Appendix B: dimension scaling")
    print(f"  h={H}, lam={LAM}, M={M}, N={N}, N0={N0}, "
          f"K_ens={K_ENS}, K_seed={K_SEED}")
    print("  mu_t = N(t * 1_d / sqrt(d), I_d),  u* = 1_d / sqrt(d)")

    dims = [1, 2, 3, 5, 8, 10]
    results: dict[int, dict] = {}
    for d in dims:
        print(f"\n-- d = {d} (n_params = {d * (d + 2)}) --")
        mses, times, werrs = [], [], []
        for s in range(K_SEED):
            mse, tt, nit, w_err, _ = run_one_dim(d, 8000 + 1000 * s)
            mses.append(mse)
            times.append(tt)
            werrs.append(w_err)
            print(f"  seed {s}: MSE={mse:.5f}, time={tt:.1f}s, "
                  f"nit={nit}, |W-W*|={w_err:.4f}")
        results[d] = {
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "time_mean": float(np.mean(times)),
            "werr_mean": float(np.mean(werrs)),
            "n_params": d * (d + 2),
        }
        print(f"  -- mean MSE = {results[d]['mse_mean']:.5f} "
              f"+/- {results[d]['mse_std']:.5f}, "
              f"mean time = {results[d]['time_mean']:.1f}s")

    with open(f"{OUT}/scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_scaling(dims, results)
    print(f"\nSaved figure and JSON to {OUT}/")


def plot_scaling(dims, results: dict[int, dict]) -> None:
    ds = np.array(dims)
    mses = np.array([results[d]["mse_mean"] for d in dims])
    stds = np.array([results[d]["mse_std"] for d in dims])
    times = np.array([results[d]["time_mean"] for d in dims])
    nps = np.array([results[d]["n_params"] for d in dims])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].errorbar(ds, mses, yerr=stds, fmt="o-", color="C0", lw=2, ms=7, capsize=4)
    axes[0].set(xlabel="spatial dimension $d$",
                ylabel=r"grid MSE$(\hat u, u^*)$",
                title="(a) Total grid MSE vs $d$")
    axes[0].grid(True, alpha=0.3)

    per_dim = mses / ds
    per_dim_std = stds / ds
    axes[1].errorbar(ds, per_dim, yerr=per_dim_std, fmt="s-", color="C2",
                     lw=2, ms=7, capsize=4)
    axes[1].set(xlabel="spatial dimension $d$", ylabel="per-dim MSE",
                title="(b) MSE per output channel vs $d$")
    axes[1].axhline(per_dim[0], color="k", ls=":", lw=1,
                    label=f"$d=1$ baseline $= {per_dim[0]:.3f}$")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(ds, times, "s-", color="C3", lw=2, ms=7, label="wall-clock (s)")
    ax.set(xlabel="spatial dimension $d$", ylabel="optimisation time (s)",
           title="(c) Runtime and model size vs $d$")
    ax.tick_params(axis="y", labelcolor="C3")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ds, nps, "D--", color="C2", lw=1.5, ms=6,
             label=r"$n_{\mathrm{params}}$")
    ax2.set_ylabel(r"$n_{\mathrm{params}} = d(d+2)$", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")

    fig.suptitle("Appendix B: dimension scaling of the RKHS estimator",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
