#!/usr/bin/env python3
"""Experiment 1: Gaussian translation (d=1, sigma=0).

    mu_t = N(-1 + 2t, 1),    T = 1
    true drift: u*(t, x) = 2  (constant)
    model:      u(t, x) = w0 + w1 * t + w2 * x  (3 parameters)
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from alltime_ot import (
    EnsembleObjective,
    LinearDriftModel,
    ensemble_lbfgs,
    euler_simulate,
    feat_affine,
    sorted_w2,
)
from alltime_ot.problems import gaussian_translation


# ---- Problem ---------------------------------------------------------
T = 1.0
def mean_fn(t: float) -> np.ndarray:
    return np.array([-1.0 + 2.0 * t])

U_STAR_CONST = 2.0  # true drift is a constant

# ---- Hyper-parameters ------------------------------------------------
LAM = 1000.0
M, N, N0 = 50, 25, 50
K_ENS = 30
H = 1.0

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp1")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    print("Exp 1: Gaussian translation (d=1, sigma=0)")
    print(f"  h={H}, lam={LAM}, M={M}, N={N}, N0={N0}, K_ens={K_ENS}")

    provider = gaussian_translation(
        mean_fn=mean_fn, d=1, M=M, N=N, N0=N0, T=T
    )
    objective = EnsembleObjective(
        batch_provider=provider,
        feat_fn=feat_affine,
        n_params=3,
        d=1,
        lam=LAM,
        h=H,
        T=T,
        K_ens=K_ENS,
        seed_offset=1000,
    )

    # Multi-start L-BFGS-B
    inits = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([3.0, -1.0, 0.0]),
    ]
    tic = time.time()
    w, best_loss, _ = ensemble_lbfgs(objective, inits)
    print(f"\nBest w  = {np.round(w, 6).tolist()}")
    print("True w* = [2.000000, 0.000000, 0.000000]")
    print(f"Optimisation time: {time.time() - tic:.1f}s")

    # Drift grid MSE on [0, T] x [-4, 4]
    te = np.linspace(0, T, 50)
    xe = np.linspace(-4, 4, 100)
    TT, XX = np.meshgrid(te, xe, indexing="ij")
    u_hat = w[0] + w[1] * TT + w[2] * XX
    mse = float(((u_hat - U_STAR_CONST) ** 2).mean())
    print(f"drift grid MSE(u_hat, u*) on [0,1]x[-4,4] = {mse:.6f}")

    model = LinearDriftModel(w, feat_affine, d=1)
    plot_drift(model, w)
    verify_marginals(model)


def plot_drift(model: LinearDriftModel, w: np.ndarray) -> None:
    """Drift comparison plot and parameter bar chart."""
    x_pl = np.linspace(-4, 4, 200)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
    for i, tv in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        ax = axes[i]
        u_hat = model(tv, x_pl)
        pdf = norm.pdf(x_pl, loc=mean_fn(tv)[0])
        ax.fill_between(x_pl, 0, pdf * 3, alpha=0.15, color="C0")
        ax.plot(x_pl, np.full_like(x_pl, U_STAR_CONST), "k--", lw=1.5, label="$u^*=2$")
        ax.plot(x_pl, u_hat, "C1-", lw=2, label="learned")
        ax.set(title=f"$t={tv}$", xlabel="$x$", xlim=(-4, 4), ylim=(-1, 5))
        if i == 0:
            ax.set_ylabel("$u(t,x)$")
        if i == 4:
            ax.legend(fontsize=8)
    fig.suptitle(
        r"Exp 1: Learned vs true drift ($\sigma=0$, Gaussian translation)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/drift_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    idx = np.arange(3)
    ax.bar(idx - 0.15, [2.0, 0.0, 0.0], 0.3, label=r"true $w^*$", color="C0", alpha=0.7)
    ax.bar(idx + 0.15, w, 0.3, label=r"learned $\hat w$", color="C1", alpha=0.7)
    ax.set_xticks(idx)
    ax.set_xticklabels(["$w_0$", "$w_1$", "$w_2$"])
    ax.set(ylabel="value", title="Parameter comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/param_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def verify_marginals(model: LinearDriftModel) -> None:
    """Euler-integrate the learned ODE and measure W_2 to the true marginals."""
    print("\nMarginal verification (ODE, Euler) ...")
    N_sim = 5000
    rng = np.random.default_rng(123)
    x0 = mean_fn(0.0)[0] + rng.standard_normal(N_sim)

    def u_ode(t: float, x: np.ndarray) -> np.ndarray:
        return model(t, x)

    eval_t = [0.0, 0.25, 0.5, 0.75, 1.0]
    snaps = euler_simulate(u_ode, x0, T=T, n_step=1000, eval_t=eval_t)

    print(f"  {'t':>5s}  {'W2':>8s}  {'mean_sim':>10s}  {'mean_true':>10s}")
    w2s = []
    for tv in eval_t:
        xs = np.sort(snaps[tv])
        mt = mean_fn(tv)[0]
        xt = norm.ppf((np.arange(N_sim) + 0.5) / N_sim, loc=mt, scale=1.0)
        w2 = sorted_w2(xs, xt)
        w2s.append(w2)
        print(f"  {tv:5.2f}  {w2:8.4f}  {xs.mean():10.4f}  {mt:10.4f}")
    print(f"  mean W2 = {float(np.mean(w2s)):.4f},  max W2 = {float(np.max(w2s)):.4f}")

    # Marginal density comparison
    fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
    for i, tv in enumerate(eval_t):
        ax = axes[i]
        mt = mean_fn(tv)[0]
        ax.hist(snaps[tv], bins=60, density=True, alpha=0.5, color="C0", label="ODE sim")
        xr = np.linspace(mt - 4, mt + 4, 200)
        ax.plot(xr, norm.pdf(xr, mt, 1), "k-", lw=1.5, label=r"$\mu_t$")
        ax.set(title=f"$t={tv}$", xlabel="$x$")
        if i == 0:
            ax.set_ylabel("density")
        if i == 4:
            ax.legend(fontsize=8)
    fig.suptitle("Marginal verification (ODE simulation)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/marginal_verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(eval_t, w2s, "C0o-", lw=1.5, ms=6)
    ax.set(xlabel="$t$", ylabel="$W_2$", title=r"$W_2(\hat\mu_t, \mu_t)$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/w2_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll figures saved to {OUT}/")


if __name__ == "__main__":
    main()
