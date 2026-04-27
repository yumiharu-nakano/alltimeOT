#!/usr/bin/env python3
"""Train the affine drift on the 1d Nelson problem (sigma=1).

Problem
-------
mu_t = N(-1 + 2t, 1) on t in [0, 1] with diffusion sigma = 1
(stochastic Nelson extension).
For this setting the unique optimal drift in the affine class
u(t,x) = w0 + w1 t + w2 x is u^*(t,x) = -x/2 + 3/2 + t,
i.e. w^* = (1.5, 1.0, -0.5).

Training
--------
We use the RKHS all-time OT loss with sigma=1 (extended Stein
operator) and optimise the three affine weights with Adam
(15 000 iterations, cosine LR 5e-4 -> 5e-5, lambda=1000, M=N=15,
h=1).  Mini-batches are drawn from a pre-cached ensemble of
K_ens=20 batches re-used at each iteration.

Outputs
-------
- output/exp_stochastic/weights.json :  learned w_hat
- output/exp_stochastic/drift_final.png : five-time-slice drift figure
- output/exp_stochastic/training_curve.png : loss vs iteration
"""

from __future__ import annotations

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from alltime_ot.problems import gaussian_translation
from alltime_ot.rkhs import rkhs_all_time_loss_from_drift

# ----- Problem ---------------------------------------------------------
T = 1.0
SIGMA = 1.0
D = 1


def mean_fn(t: float) -> np.ndarray:
    return np.array([-1.0 + 2.0 * t])


def u_true(t: float, x: np.ndarray) -> np.ndarray:
    return -0.5 * x + 1.5 + t


# ----- Hyperparameters -------------------------------------------------
LAM = 1000.0
H = 1.0
M = 30
N = 30
N0 = 60
K_ENS = 30
N_ITER = 15_000
LR_HIGH = 5e-4
LR_LOW = 5e-5
SEED = 2026

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp_stochastic")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    print("Exp Stochastic 1d: training affine drift on Nelson problem")
    print(f"  mu_t = N(-1+2t, 1), sigma={SIGMA}, T={T}")
    print(f"  Hyperparams: M={M}, N={N}, N0={N0}, lam={LAM}, h={H}")
    print(f"               iters={N_ITER}, K_ens={K_ENS}, lr={LR_HIGH}->{LR_LOW}")
    sys.stdout.flush()

    provider = gaussian_translation(mean_fn=mean_fn, d=D, M=M, N=N, N0=N0, T=T)

    # Pre-cache K_ENS independent batches (CPU, float64).
    print("Pre-caching batches...")
    sys.stdout.flush()
    cached = []
    for k in range(K_ENS):
        t_s, X, X0 = provider(SEED + k)
        cached.append((t_s.double(), X.double(), X0.double()))

    # Affine drift parameters: w[0] + w[1]*t + w[2]*x  (3 params).
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    optim = torch.optim.Adam([w], lr=LR_HIGH)

    def cosine_lr(it: int) -> float:
        # iter 0 -> LR_HIGH, iter N_ITER-1 -> LR_LOW
        frac = it / max(N_ITER - 1, 1)
        return LR_LOW + 0.5 * (LR_HIGH - LR_LOW) * (1.0 + np.cos(np.pi * frac))

    losses: list[float] = []
    print(f"Training for {N_ITER} iterations...")
    sys.stdout.flush()
    tic = time.time()
    for it in range(N_ITER):
        for g in optim.param_groups:
            g["lr"] = cosine_lr(it)
        t_s, X, X0 = cached[it % K_ENS]
        tg = t_s.repeat_interleave(N)
        xg = X.reshape(M * N, D)
        # Drift evaluation: uf = w0 + w1*t + w2*x.
        uf = (w[0] + w[1] * tg + w[2] * xg[:, 0])[:, None]
        loss = rkhs_all_time_loss_from_drift(
            uf, tg, xg, X0,
            M=M, N=N, t_s=t_s,
            lam=LAM, h=H, T=T, sigma=SIGMA,
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if (it + 1) % 1000 == 0 or it == 0:
            print(
                f"  iter {it+1:5d}/{N_ITER}  loss={loss.item():.5f}"
                f"  w=({w[0].item():.4f}, {w[1].item():.4f}, {w[2].item():.4f})"
                f"  lr={cosine_lr(it):.2e}"
            )
            sys.stdout.flush()
    elapsed = time.time() - tic

    w_hat = tuple(float(x) for x in w.detach().numpy())
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  learned w_hat = {w_hat}")
    print(f"  true     w*   = (1.5, 1.0, -0.5)")

    # Drift grid MSE on x in [-3, 3].
    grid_t = np.linspace(0, T, 50)
    grid_x = np.linspace(-3.0, 3.0, 200)
    GT, GX = np.meshgrid(grid_t, grid_x, indexing="ij")
    u_true_grid = u_true(GT, GX)
    u_hat_grid = w_hat[0] + w_hat[1] * GT + w_hat[2] * GX
    drift_mse = float(((u_hat_grid - u_true_grid) ** 2).mean())
    print(f"  drift grid MSE = {drift_mse:.5f} on [0,1]x[-3,3]")

    # Save weights JSON.
    weights_path = os.path.join(OUT, "weights.json")
    with open(weights_path, "w") as f:
        json.dump(
            {
                "w_hat": list(w_hat),
                "w_star": [1.5, 1.0, -0.5],
                "drift_grid_mse": drift_mse,
                "sigma": SIGMA,
                "lam": LAM,
                "h": H,
                "M": M,
                "N": N,
                "N0": N0,
                "K_ens": K_ENS,
                "n_iter": N_ITER,
                "training_time_s": elapsed,
            },
            f,
            indent=2,
        )
    print(f"Saved: {weights_path}")

    # Drift figure (five time slices).
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
    for i, tv in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        ax = axes[i]
        ust = u_true(tv, grid_x)
        uhat = w_hat[0] + w_hat[1] * tv + w_hat[2] * grid_x
        ax.plot(grid_x, ust, "k--", lw=1.5, label="$u^*$")
        ax.plot(grid_x, uhat, "C1-", lw=2, label=r"$\hat u$")
        ax.set(title=f"$t={tv}$", xlim=(-3, 3))
        ax.set_xlabel("$x$")
        if i == 0:
            ax.set_ylabel(r"$u(t,x)$")
        if i == 4:
            ax.legend(fontsize=9, loc="upper right")
    fig.suptitle(
        r"Stochastic 1d Gaussian: learned drift $\hat u$ vs true $u^*=-x/2+3/2+t$",
        y=1.05,
    )
    fig.tight_layout()
    drift_path = os.path.join(OUT, "drift_final.png")
    fig.savefig(drift_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {drift_path}")

    # Training curve.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(losses, lw=0.6, alpha=0.7)
    ax.set(
        xlabel="iteration", ylabel="loss",
        title="Stochastic 1d: training curve",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = os.path.join(OUT, "training_curve.png")
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {curve_path}")


if __name__ == "__main__":
    main()
