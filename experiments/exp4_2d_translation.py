#!/usr/bin/env python3
"""Experiment 4: 2-d Gaussian translation (d=2, sigma=0).

    mu_t = N((-1+2t, 0.5 t)^T, I_2),   T = 1
    true drift: u*(t, x) = (2, 0.5)^T  (constant, same in time and space)

Validates that the RKHS all-time OT estimator works in d > 1.
Parametrisation: affine per output channel
    u_i(t, x) = w_{i,0} + w_{i,1} t + w_{i,2} x_1 + w_{i,3} x_2,  i=1,2
(8 parameters in total).
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from alltime_ot import (
    EnsembleObjective,
    LinearDriftModel,
    ensemble_lbfgs,
    euler_simulate,
    feat_affine,
    mmd_gauss,
    sliced_w2,
)
from alltime_ot.problems import gaussian_translation

# ---- Problem ---------------------------------------------------------
T = 1.0
D = 2

def mean_fn(t: float) -> np.ndarray:
    return np.array([-1.0 + 2.0 * t, 0.5 * t])

U_TRUE = np.array([2.0, 0.5])

# ---- Hyper-parameters ------------------------------------------------
LAM = 1000.0
M, N, N0 = 25, 20, 50
K_ENS = 15
H = 1.0
P_FEAT = 4          # [1, t, x_1, x_2]
N_PARAMS = P_FEAT * D

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp4")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    print("Exp 4: 2D Gaussian translation (d=2, sigma=0)")
    print("  mu_t = N((-1+2t, 0.5t), I_2),  u* = (2, 0.5)")
    print(f"  h={H}, lam={LAM}, M={M}, N={N}, N0={N0}, K_ens={K_ENS}")
    print(f"  Features: [1, t, x_1, x_2]  =>  {N_PARAMS} parameters")

    provider = gaussian_translation(mean_fn=mean_fn, d=D, M=M, N=N, N0=N0, T=T)
    objective = EnsembleObjective(
        batch_provider=provider,
        feat_fn=feat_affine,
        n_params=N_PARAMS,
        d=D,
        lam=LAM,
        h=H,
        T=T,
        K_ens=K_ENS,
        seed_offset=5000,
    )

    inits = [
        np.zeros(N_PARAMS),
        np.array([2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    tic = time.time()
    w, _, _ = ensemble_lbfgs(objective, inits)
    print(f"\nOptimisation time: {time.time() - tic:.1f}s")

    W_hat = w.reshape(P_FEAT, D)
    print("Learned W (rows = features, cols = output dim):")
    for row, name, true in zip(
        W_hat, ("intercept", "t        ", "x_1      ", "x_2      "),
        (U_TRUE, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
    ):
        print(f"  {name}: [{row[0]:+.5f}, {row[1]:+.5f}]    "
              f"true [{true[0]:+.5f}, {true[1]:+.5f}]")

    # Drift grid MSE on [0, T] x [-4, 4]^2
    te = np.linspace(0, T, 20)
    g = np.linspace(-4, 4, 30)
    GT, GX, GY = np.meshgrid(te, g, g, indexing="ij")
    Phi_g = np.column_stack([
        np.ones(GT.size), GT.ravel(), GX.ravel(), GY.ravel()
    ])
    u_hat_grid = (Phi_g @ W_hat).reshape(*GT.shape, 2)
    mse = float(((u_hat_grid - U_TRUE) ** 2).sum(-1).mean())
    print(f"\ndrift grid MSE(u_hat, u*) on [0,1]x[-4,4]^2 = {mse:.5e}")

    model = LinearDriftModel(w, feat_affine, d=D)
    plot_quiver(model)
    plot_params(W_hat)
    verify_marginals(model)


def plot_quiver(model: LinearDriftModel) -> None:
    xg = np.linspace(-3, 3, 10)
    yg = np.linspace(-2, 2, 8)
    XG, YG = np.meshgrid(xg, yg)
    XY = np.column_stack([XG.ravel(), YG.ravel()])

    fig, axes = plt.subplots(1, 5, figsize=(17, 4), sharex=True, sharey=True)
    for i, tv in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        ax = axes[i]
        u_hat_xy = model(tv, XY)
        ax.quiver(XY[:, 0], XY[:, 1], u_hat_xy[:, 0], u_hat_xy[:, 1],
                  color="C1", scale=20, width=0.006, label="learned")
        ax.quiver(XY[:, 0], XY[:, 1],
                  np.full(len(XY), U_TRUE[0]), np.full(len(XY), U_TRUE[1]),
                  color="k", scale=20, width=0.003, alpha=0.4, label="true $u^*$")
        m = mean_fn(tv)
        ax.plot(m[0], m[1], "r*", ms=14,
                label=r"$\mu_t$ mean" if i == 0 else None)
        ax.set(title=f"$t={tv}$", xlim=(-3.3, 3.3), ylim=(-2.2, 2.2))
        ax.set_aspect("equal")
        if i == 0:
            ax.set(xlabel="$x_1$", ylabel="$x_2$")
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Exp 4 (d=2): learned drift vs true $u^*=(2, 0.5)$",
                 y=1.05, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/drift_quiver.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_params(W_hat: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    labels = ["$w_0$", "$w_1^{(t)}$", "$w_2^{(x_1)}$", "$w_3^{(x_2)}$"]
    for mu in range(D):
        ax = axes[mu]
        true_vec = [U_TRUE[mu], 0.0, 0.0, 0.0]
        idx = np.arange(4)
        ax.bar(idx - 0.15, true_vec, 0.3, label=r"true $w^*$", color="C0", alpha=0.7)
        ax.bar(idx + 0.15, W_hat[:, mu], 0.3, label=r"learned $\hat w$",
               color="C1", alpha=0.7)
        ax.set_xticks(idx)
        ax.set_xticklabels(labels)
        ax.set_title(f"output dimension {mu + 1}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        if mu == 0:
            ax.legend(fontsize=9)
    fig.suptitle("Exp 4: parameter recovery", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/param_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def verify_marginals(model: LinearDriftModel) -> None:
    print("\nMarginal verification (ODE Euler) ...")
    N_sim, n_step = 3000, 600
    rng = np.random.default_rng(777)
    x0 = mean_fn(0.0)[None, :] + rng.standard_normal((N_sim, D))

    def u_ode(t: float, x: np.ndarray) -> np.ndarray:
        return model(t, x)

    eval_t = [0.0, 0.25, 0.5, 0.75, 1.0]
    snaps = euler_simulate(u_ode, x0, T=T, n_step=n_step, eval_t=eval_t)

    rng_ref = np.random.default_rng(999)
    sw2_vals, mmd_vals = [], []
    print(f"  {'t':>5s}  {'SW_2':>8s}  {'MMD':>8s}  "
          f"{'mean_sim':>22s}  {'mean_true':>22s}")
    for tv in eval_t:
        ref = mean_fn(tv)[None, :] + rng_ref.standard_normal((N_sim, D))
        sw = sliced_w2(snaps[tv], ref)
        m = mmd_gauss(snaps[tv], ref, h=1.0)
        sw2_vals.append(sw)
        mmd_vals.append(m)
        mt = mean_fn(tv)
        ms = snaps[tv].mean(0)
        print(f"  {tv:5.2f}  {sw:8.4f}  {m:8.4f}  "
              f"({ms[0]:+.3f},{ms[1]:+.3f})    "
              f"({mt[0]:+.3f},{mt[1]:+.3f})")

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    rng_ref2 = np.random.default_rng(999)
    for i, tv in enumerate(eval_t):
        ax = axes[0, i]
        xs = snaps[tv]
        ax.scatter(xs[:, 0], xs[:, 1], s=3, alpha=0.3, color="C1")
        gg = np.linspace(-4, 4, 50)
        XXc, YYc = np.meshgrid(gg, gg)
        Zc = multivariate_normal(mean_fn(tv), np.eye(2)).pdf(np.dstack([XXc, YYc]))
        ax.contour(XXc, YYc, Zc, levels=5, colors="k", linewidths=1, alpha=0.6)
        ax.set(title=f"$t={tv}$", xlim=(-4, 4), ylim=(-4, 4))
        ax.set_aspect("equal")
        if i == 0:
            ax.set_ylabel("learned ODE\n$x_2$", fontsize=9)

        ax = axes[1, i]
        ref = mean_fn(tv)[None, :] + rng_ref2.standard_normal((N_sim, D))
        ax.scatter(ref[:, 0], ref[:, 1], s=3, alpha=0.3, color="C0")
        ax.contour(XXc, YYc, Zc, levels=5, colors="k", linewidths=1, alpha=0.6)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="$x_1$")
        ax.set_aspect("equal")
        if i == 0:
            ax.set_ylabel(r"true $\mu_t$" + "\n$x_2$", fontsize=9)

    fig.suptitle("Exp 4 (d=2): marginal verification via ODE simulation",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/marginal_verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, (ax_w, ax_m) = plt.subplots(1, 2, figsize=(11, 3.5))
    ax_w.plot(eval_t, sw2_vals, "C0o-", lw=2, ms=7)
    ax_w.set(xlabel="$t$", ylabel="Sliced $W_2$",
             title=r"Sliced $W_2(\hat\mu_t, \mu_t)$")
    ax_w.grid(True, alpha=0.3)
    ax_m.plot(eval_t, mmd_vals, "C1s-", lw=2, ms=7)
    ax_m.set(xlabel="$t$", ylabel=r"$\mathrm{MMD}$",
             title=r"$\mathrm{MMD}(\hat\mu_t, \mu_t)$ "
                   r"(Gaussian kernel, $h=1$)")
    ax_m.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/sw2_mmd_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nMean sliced W2 = {np.mean(sw2_vals):.4f},  "
          f"max sliced W2 = {np.max(sw2_vals):.4f}")
    print(f"Mean MMD       = {np.mean(mmd_vals):.4f},  "
          f"max MMD       = {np.max(mmd_vals):.4f}")
    print(f"\nAll figures saved to {OUT}/")


if __name__ == "__main__":
    main()
