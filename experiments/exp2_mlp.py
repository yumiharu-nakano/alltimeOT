#!/usr/bin/env python3
"""Experiment 2 (model comparison): Roundtrip motion, linear vs MLP drift.

Same problem as exp2_roundtrip.py:
    mu_t = N(2 sin(pi t), 1),  T = 1,  sigma = 0
    true drift: u*(t, x) = 2 pi cos(pi t)

Two drift parametrisations are compared:

  (A) Quadratic (4 params): u = w0 + w1 t + w2 t^2 + w3 x
      -- t and x are separated additively, matching the fact that the
         true drift is constant in x.  Optimised by ensemble L-BFGS-B.

  (B) MLP (~4.4 k params): [2 -> 64 -> 64 -> 1] with tanh activations,
      trained by Adam with minibatch SGD over K_ens pre-cached batches.
      Completely model-agnostic; no t-x structural prior.

This demonstrates that the RKHS all-time loss is model-agnostic (the
same objective drives both fits) and that the NN recovers marginal
consistency competitive with the structurally-optimal linear baseline.
"""

from __future__ import annotations

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from alltime_ot import (
    EnsembleObjective,
    LinearDriftModel,
    ensemble_lbfgs,
    euler_simulate,
    feat_quadratic_t,
    mmd_gauss,
    rkhs_all_time_loss_from_drift,
    sorted_w2,
)
from alltime_ot.problems import gaussian_translation

# ---- Problem ---------------------------------------------------------
T_HORIZON = 1.0
A_AMP = 2.0


def mean_fn(t: float) -> np.ndarray:
    return np.array([A_AMP * np.sin(np.pi * t)])


def u_true(t: float) -> float:
    return A_AMP * np.pi * np.cos(np.pi * t)


# ---- Shared hyper-parameters -----------------------------------------
LAM = 1000.0
H = 1.0
T_HOR = T_HORIZON

# Data budget (shared between L-BFGS and SGD paths)
M, N, N0 = 25, 15, 30
K_ENS = 60

# MLP training
K_BATCH = 4
LR_INIT = 3e-3
LR_MIN = 1e-5
N_ITER = 6000
PRINT_EVERY = 1000
GRAD_CLIP = 10.0

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp2")
os.makedirs(OUT, exist_ok=True)


# ---- MLP model -------------------------------------------------------
class DriftMLP(nn.Module):
    """Two-hidden-layer tanh MLP: (t, x) -> u(t, x) in R^1."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, tg: torch.Tensor, xg: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([tg[:, None], xg], dim=1)
        return self.net(inp)


# ---- Training helpers ------------------------------------------------
def train_linear(feat_fn, n_params: int, inits, label: str) -> np.ndarray:
    """Train a linear-in-parameters model via ensemble L-BFGS-B."""
    provider = gaussian_translation(
        mean_fn=mean_fn, d=1, M=M, N=N, N0=N0, T=T_HORIZON,
    )
    objective = EnsembleObjective(
        batch_provider=provider,
        feat_fn=feat_fn,
        n_params=n_params,
        d=1,
        lam=LAM,
        h=H,
        T=T_HORIZON,
        K_ens=K_ENS,
        seed_offset=1000,
    )
    print(f"\n[{label}] training ({n_params} params, {K_ENS} batches) ...")
    tic = time.time()
    w, loss, _ = ensemble_lbfgs(objective, inits, verbose=False)
    print(f"  loss={loss:.4f}  time={time.time() - tic:.1f}s")
    return w


def train_mlp() -> tuple[DriftMLP, list[float]]:
    """Train the MLP via Adam + minibatch SGD over cached batches."""
    torch.manual_seed(42)
    dtype = torch.float64

    provider = gaussian_translation(
        mean_fn=mean_fn, d=1, M=M, N=N, N0=N0, T=T_HORIZON,
    )
    batches = [provider(1000 + k) for k in range(K_ENS)]

    model = DriftMLP(hidden=64).to(dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[MLP] training ({n_params} params, Adam {LR_INIT}->{LR_MIN}, "
          f"{N_ITER} iters, {K_BATCH}/{K_ENS} batches per step) ...")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_ITER, eta_min=LR_MIN,
    )
    rng_train = np.random.default_rng(99)

    tic = time.time()
    loss_history: list[float] = []
    for it in range(1, N_ITER + 1):
        optimizer.zero_grad()
        idxs = rng_train.choice(K_ENS, size=K_BATCH, replace=False)
        loss_sum = torch.zeros((), dtype=dtype)
        for idx in idxs:
            t_s, X, X0 = batches[idx]
            Mb, Nb, d = X.shape
            tg = t_s.repeat_interleave(Nb)
            xg = X.reshape(Mb * Nb, d)
            uf = model(tg, xg)
            loss_sum = loss_sum + rkhs_all_time_loss_from_drift(
                uf, tg, xg, X0,
                M=Mb, N=Nb, t_s=t_s, lam=LAM, h=H, T=T_HORIZON,
            )
        loss = loss_sum / K_BATCH
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        loss_history.append(float(loss.detach()))
        if it % PRINT_EVERY == 0 or it == 1:
            print(f"  iter {it:5d}  loss={loss_history[-1]:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
    print(f"  time={time.time() - tic:.1f}s")
    model.eval()
    return model, loss_history


# ---- Drift callables (numpy) -----------------------------------------
def make_mlp_callable(model: DriftMLP):
    dtype = torch.float64

    def u(t: float, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tg = torch.full((len(x),), t, dtype=dtype)
            xg = torch.tensor(x, dtype=dtype)[:, None]
            return model(tg, xg).numpy()[:, 0]

    return u


# ---- Main ------------------------------------------------------------
def main() -> None:
    print("Exp 2 (model comparison): Roundtrip (d=1, sigma=0)")
    print(f"  lam={LAM}, M={M}, N={N}, N0={N0}, K_ens={K_ENS}")

    # ---- Train two models --------------------------------------------
    # (A) Quadratic (linear-in-parameters, L-BFGS-B)
    w_quad = train_linear(
        feat_quadratic_t,
        n_params=4,
        inits=[
            np.zeros(4),
            np.array([6.0, -12.0, 0.0, 0.0]),
            np.array([3.0, 0.0, -6.0, 0.0]),
            np.array([0.0, 10.0, -10.0, 0.0]),
        ],
        label="Quadratic",
    )
    model_quad = LinearDriftModel(w_quad, feat_quadratic_t, d=1)

    # (B) MLP (Adam + minibatch SGD)
    mlp, loss_history = train_mlp()
    u_mlp_np = make_mlp_callable(mlp)

    # ---- Grid MSE ----------------------------------------------------
    te = np.linspace(0, T_HORIZON, 50)
    xe = np.linspace(-4, 4, 100)
    TT, XX = np.meshgrid(te, xe, indexing="ij")
    u_star_grid = 2 * np.pi * np.cos(np.pi * TT)

    def grid_eval(u_func) -> np.ndarray:
        return np.array([u_func(tv, xe) for tv in te])

    u_quad_grid = grid_eval(lambda t, x: model_quad(t, x))
    u_mlp_grid = grid_eval(u_mlp_np)

    mse = {
        "Quadratic": float(((u_quad_grid - u_star_grid) ** 2).mean()),
        "MLP": float(((u_mlp_grid - u_star_grid) ** 2).mean()),
        "Zero": float((u_star_grid ** 2).mean()),
    }

    # ---- ODE marginal verification -----------------------------------
    print("\nMarginal verification (ODE, Euler) ...")
    N_sim = 5000
    rng = np.random.default_rng(123)
    x0 = mean_fn(0.0)[0] + rng.standard_normal(N_sim)
    eval_t = [0.0, 0.25, 0.5, 0.75, 1.0]

    def simulate(u_func):
        return euler_simulate(u_func, x0, T=T_HORIZON, n_step=1000, eval_t=eval_t)

    snaps = {
        "Quadratic": simulate(lambda t, x: model_quad(t, x)),
        "MLP": simulate(u_mlp_np),
        "Zero": simulate(lambda t, x: np.zeros_like(x)),
        "True": simulate(lambda t, x: np.full_like(x, u_true(t))),
    }
    w2: dict[str, list[float]] = {}
    mmd: dict[str, list[float]] = {}
    rng_ref = np.random.default_rng(321)
    ref_mmd_per_t = {
        tv: mean_fn(tv)[0] + rng_ref.standard_normal(N_sim) for tv in eval_t
    }
    for name, snap in snaps.items():
        ws, ms = [], []
        for tv in eval_t:
            ref = norm.ppf((np.arange(N_sim) + 0.5) / N_sim,
                           loc=mean_fn(tv)[0], scale=1.0)
            ws.append(sorted_w2(snap[tv], ref))
            # Same Gaussian kernel (h=1) as the RKHS loss; subsample for speed.
            ms.append(mmd_gauss(snap[tv], ref_mmd_per_t[tv], h=1.0))
        w2[name] = ws
        mmd[name] = ms

    # ---- Summary table -----------------------------------------------
    print("\n" + "=" * 84)
    print(f"{'Model':12s} {'#params':>10s} {'drift MSE':>12s} "
          f"{'mean W2':>10s} {'mean MMD':>10s} {'max W2':>10s}")
    print("-" * 84)
    counts = {
        "Quadratic": 4,
        "MLP": sum(p.numel() for p in mlp.parameters()),
        "Zero": 0,
        "True": "-",
    }
    for name in ("True", "Quadratic", "MLP", "Zero"):
        mean_w2 = float(np.mean(w2[name]))
        max_w2 = float(np.max(w2[name]))
        mean_mmd = float(np.mean(mmd[name]))
        mse_val = 0.0 if name == "True" else mse[name]
        cnt = counts[name]
        cnt_s = f"{cnt}" if not isinstance(cnt, str) else cnt
        print(f"  {name:10s} {cnt_s:>10s} {mse_val:12.4f} "
              f"{mean_w2:10.4f} {mean_mmd:10.4f} {max_w2:10.4f}")
    print("=" * 84)

    # ---- Plots -------------------------------------------------------
    plot_time_profile(model_quad, mlp)
    plot_drift_slices(model_quad, mlp)
    plot_w2_comparison(eval_t, w2, mmd)
    plot_loss_curve(loss_history)
    plot_marginals(snaps, eval_t)

    print(f"\nAll figures saved to {OUT}/")


# ---- Plotting --------------------------------------------------------
def plot_time_profile(model_quad, mlp: DriftMLP) -> None:
    """Drift at x = 0 as a function of t."""
    tt = np.linspace(0, 1, 200)
    u_star_t = 2 * np.pi * np.cos(np.pi * tt)
    u_quad_t = model_quad(tt, np.zeros(len(tt)))
    with torch.no_grad():
        tg = torch.tensor(tt, dtype=torch.float64)
        xg = torch.zeros(len(tt), 1, dtype=torch.float64)
        u_mlp_t = mlp(tg, xg).numpy()[:, 0]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(tt, u_star_t, "k-", lw=2, label=r"$u^*(t) = 2\pi\cos(\pi t)$")
    ax.plot(tt, u_quad_t, "C0--", lw=1.8, label="Quadratic (4 params)")
    ax.plot(tt, u_mlp_t, "C1-", lw=2, label="MLP (~4.4k params)")
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set(xlabel="$t$", ylabel="$u(t, x=0)$",
           title="Roundtrip drift at $x=0$: Quadratic vs MLP")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison_time_profile.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drift_slices(model_quad, mlp: DriftMLP) -> None:
    x_pl = np.linspace(-4, 4, 200)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
    for i, tv in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        ax = axes[i]
        u_tr = u_true(tv)
        u_q = model_quad(tv, x_pl)
        with torch.no_grad():
            tg = torch.full((len(x_pl),), tv, dtype=torch.float64)
            xg = torch.tensor(x_pl, dtype=torch.float64)[:, None]
            u_m = mlp(tg, xg).numpy()[:, 0]
        pdf = norm.pdf(x_pl, loc=mean_fn(tv)[0])
        ax.fill_between(x_pl, -8, pdf * 4 - 8, alpha=0.15, color="C0")
        ax.axhline(u_tr, color="k", ls="--", lw=1.5, label=f"$u^*={u_tr:.2f}$")
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.plot(x_pl, u_q, "C0-", lw=1.5, label="Quad")
        ax.plot(x_pl, u_m, "C1-", lw=1.5, label="MLP")
        ax.set(title=f"$t={tv}$", xlabel="$x$", xlim=(-4, 4), ylim=(-8, 10))
        if i == 0:
            ax.set_ylabel("$u(t,x)$")
        if i == 4:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(r"Exp 2: learned drift vs true $u^*$ -- Quadratic / MLP", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_w2_comparison(
    eval_t: list[float],
    w2: dict[str, list[float]],
    mmd: dict[str, list[float]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    panels = [
        (axes[0], w2, r"$W_2(\hat\mu_t, \mu_t)$", "Wasserstein-2"),
        (axes[1], mmd, r"$\mathrm{MMD}(\hat\mu_t, \mu_t)$",
         "MMD (Gaussian kernel, $h=1$)"),
    ]
    for ax, metric, ylabel, title in panels:
        ax.plot(eval_t, metric["True"], "ko--", lw=1.2, ms=5, label="True $u^*$")
        ax.plot(eval_t, metric["Quadratic"], "C0s-", lw=2, ms=6, label="Quadratic")
        ax.plot(eval_t, metric["MLP"], "C1o-", lw=2, ms=6, label="MLP")
        ax.plot(eval_t, metric["Zero"], "gray", ls=":", lw=2, marker="^", ms=6,
                label=r"Zero ($u\equiv 0$)")
        ax.set(xlabel="$t$", ylabel=ylabel, title=title, yscale="log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("Marginal tracking: Quadratic vs MLP", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison_w2_mmd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(loss_history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(loss_history, "C0-", lw=0.8)
    ax.set(xlabel="iteration", ylabel="loss", title="MLP training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mlp_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marginals(snaps: dict, eval_t: list[float]) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(16, 7.5), sharey="row")
    rows = [
        ("True $u^*$", snaps["True"], "k"),
        ("Quadratic", snaps["Quadratic"], "C0"),
        ("MLP", snaps["MLP"], "C1"),
    ]
    for r, (lab, snap, c) in enumerate(rows):
        for cc, tv in enumerate(eval_t):
            ax = axes[r, cc]
            ax.hist(snap[tv], bins=60, density=True, alpha=0.5, color=c,
                    range=(-5, 5))
            xr = np.linspace(-5, 5, 200)
            ax.plot(xr, norm.pdf(xr, mean_fn(tv)[0], 1), "k-", lw=1.5)
            if r == 0:
                ax.set_title(f"$t={tv}$")
            if cc == 0:
                ax.set_ylabel(lab, fontsize=9)
            ax.set_xlim(-5, 5)
    fig.suptitle("Exp 2: marginal verification -- Quadratic vs MLP", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/model_comparison_marginals.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
