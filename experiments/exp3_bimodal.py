#!/usr/bin/env python3
"""Experiment 3: bimodal merging flow (d=1, sigma=0).

    mu_t = 1/2 N(-2+2t, 1) + 1/2 N(2-2t, 1),  T = 1
    true drift: u*(t, x) = -2 * tanh(2 * (1 - t) * x)

Three parametric drift classes are compared under the *same* all-time
RKHS loss, the *same* sample budget, and the *same* random batches:

    (A) Bilinear : Phi = [1, t, x, t*x]                     (4 params)
                   -- linear in x, unbounded in tails; illustrates that
                   a misspecified linear model can track marginals
                   reasonably while producing a grossly wrong drift MSE
                   (decoupling between pointwise and distributional error).

    (B) Tanh dict: Phi = [1, t, x, t*x, tanh(x), t*tanh(x),
                          tanh(2x), t*tanh(2x)]             (8 params)
                   -- structured dictionary that cannot realize
                   -2*tanh(2(1-t)x) exactly (only amplitude scaling is
                   available, not argument scaling), but captures the
                   L^2-best projection.

    (C) MLP      : [2 -> 48 -> 48 -> 1] tanh MLP             (~2.5 k params)
                   -- universal approximator, trained by Adam on the
                   identical RKHS loss via rkhs_all_time_loss_from_drift.

Outputs unified figures placing all three model classes side by side.
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
    feat_bilinear,
    feat_tanh_merger,
    mmd_gauss,
    rkhs_all_time_loss_from_drift,
    sorted_w2,
)
from alltime_ot.features import feat_count
from alltime_ot.problems import gaussian_mixture_1d


# ---- Problem ---------------------------------------------------------
T = 1.0


def m1(t: float) -> float:
    return -2.0 + 2.0 * t


def m2(t: float) -> float:
    return 2.0 - 2.0 * t


def means_fn(t: float) -> tuple[float, float]:
    return m1(t), m2(t)


def rho_t(t: float, x: np.ndarray) -> np.ndarray:
    return 0.5 * norm.pdf(x, m1(t), 1.0) + 0.5 * norm.pdf(x, m2(t), 1.0)


def u_star(t, x):
    return -2.0 * np.tanh(2.0 * (1.0 - t) * x)


# ---- Hyper-parameters ------------------------------------------------
LAM = 5000.0
H = 1.0
M, N, N0 = 50, 30, 60
K_ENS = 20
SEED_OFFSET = 4000

# MLP training
MLP_HIDDEN = 48
K_BATCH = 3
LR_INIT = 3e-3
LR_MIN = 1e-5
N_ITER = 4000
PRINT_EVERY = 500
GRAD_CLIP = 10.0

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp3")
os.makedirs(OUT, exist_ok=True)

LINEAR_MODELS = {
    "bilin": feat_bilinear,
    "tanh": feat_tanh_merger,
}
INITS = {
    "bilin": [np.zeros(4), np.array([0.0, 0.0, -1.0, 0.5])],
    "tanh": [
        np.zeros(8),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 2.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.5, -1.0, 1.0]),
    ],
}
LABELS = {
    "bilin": "bilinear",
    "tanh": "tanh dict",
    "mlp": "MLP",
    "zero": "zero",
}
COLORS = {"bilin": "C2", "tanh": "C1", "mlp": "C3", "zero": "C7"}


# ---- MLP model -------------------------------------------------------
class DriftMLP(nn.Module):
    """Two-hidden-layer tanh MLP: (t, x) -> u(t, x) in R^1."""

    def __init__(self, hidden: int = MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight, gain=0.5)
                nn.init.zeros_(mod.bias)

    def forward(self, tg: torch.Tensor, xg: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([tg[:, None], xg], dim=1)
        return self.net(inp)


# ---- Training --------------------------------------------------------
def train_linear(provider, feat_fn, inits, label: str) -> np.ndarray:
    p_feat = feat_count(feat_fn, d=1)
    objective = EnsembleObjective(
        batch_provider=provider,
        feat_fn=feat_fn,
        n_params=p_feat,
        d=1,
        lam=LAM,
        h=H,
        T=T,
        K_ens=K_ENS,
        seed_offset=SEED_OFFSET,
    )
    print(f"\n[{label}] training ({p_feat} params, {K_ENS} batches) ...")
    tic = time.time()
    w, loss, _ = ensemble_lbfgs(objective, inits, verbose=False)
    print(f"  loss={loss:.4f}  time={time.time() - tic:.1f}s")
    return w


def train_mlp(provider) -> tuple[DriftMLP, list[float]]:
    torch.manual_seed(42)
    dtype = torch.float64
    batches = [provider(SEED_OFFSET + k) for k in range(K_ENS)]

    model = DriftMLP().to(dtype)
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
                M=Mb, N=Nb, t_s=t_s, lam=LAM, h=H, T=T,
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


def make_mlp_callable(model: DriftMLP):
    dtype = torch.float64

    def u(t, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        with torch.no_grad():
            tg = torch.full((len(x),), float(t), dtype=dtype)
            xg = torch.tensor(x, dtype=dtype)[:, None]
            return model(tg, xg).numpy()[:, 0]

    return u


# ---- Main ------------------------------------------------------------
def main() -> None:
    print("Exp 3: Bimodal merging (d=1, sigma=0) -- bilinear / tanh / MLP")
    print(r"  mu_t = 1/2 N(-2+2t,1) + 1/2 N(2-2t,1), u* = -2 tanh(2(1-t)x)")
    print(f"  h={H}, lam={LAM}, M={M}, N={N}, N0={N0}, K_ens={K_ENS}")

    provider = gaussian_mixture_1d(
        means_fn=means_fn, M=M, N=N, N0=N0, T=T,
    )

    # ---- Grid -------------------------------------------------------
    te = np.linspace(0, T, 50)
    xe = np.linspace(-4, 4, 100)
    TT, XX = np.meshgrid(te, xe, indexing="ij")
    u_star_grid = u_star(TT, XX)
    mse_zero = float((u_star_grid ** 2).mean())

    # ---- Train three models ----------------------------------------
    results: dict[str, dict] = {}
    for key, feat in LINEAR_MODELS.items():
        w = train_linear(provider, feat, INITS[key], LABELS[key])
        model = LinearDriftModel(w, feat, d=1)
        u_hat = model(TT.ravel(), XX.reshape(-1, 1)).reshape(TT.shape)
        mse = float(((u_hat - u_star_grid) ** 2).mean())
        print(f"  drift grid MSE({LABELS[key]}) = {mse:.5f}")
        results[key] = {
            "w": w,
            "mse": mse,
            "u_fn": lambda t, x, m=model: m(t, x),
            "n_params": feat_count(feat, d=1),
        }

    mlp, loss_history = train_mlp(provider)
    u_mlp_np = make_mlp_callable(mlp)
    u_mlp_grid = np.stack([u_mlp_np(tv, xe) for tv in te], axis=0)
    mse_mlp = float(((u_mlp_grid - u_star_grid) ** 2).mean())
    print(f"  drift grid MSE(MLP) = {mse_mlp:.5f}")
    results["mlp"] = {
        "mse": mse_mlp,
        "u_fn": u_mlp_np,
        "n_params": sum(p.numel() for p in mlp.parameters()),
        "loss_history": loss_history,
    }

    print(f"\nzero-drift grid MSE: {mse_zero:.4f}")

    # ---- Marginal verification -------------------------------------
    w2, mmd, snaps = verify_marginals(results)

    # ---- Summary ---------------------------------------------------
    print("\n" + "=" * 84)
    print(f"{'Model':12s} {'#params':>10s} {'drift MSE':>12s} "
          f"{'mean W2':>10s} {'mean MMD':>10s} {'max W2':>10s}")
    print("-" * 84)
    for key in ("bilin", "tanh", "mlp"):
        r = results[key]
        print(f"  {LABELS[key]:10s} {r['n_params']:>10d} {r['mse']:12.4f} "
              f"{np.mean(w2[key]):10.4f} {np.mean(mmd[key]):10.4f} "
              f"{np.max(w2[key]):10.4f}")
    print(f"  {'zero':10s} {'0':>10s} {mse_zero:12.4f} "
          f"{np.mean(w2['zero']):10.4f} {np.mean(mmd['zero']):10.4f} "
          f"{np.max(w2['zero']):10.4f}")
    print("=" * 84)

    # ---- Plots -----------------------------------------------------
    plot_drift_comparison(results, mlp)
    plot_mse_bar(results, mse_zero)
    plot_w2_mmd(w2, mmd)
    plot_marginals(snaps)
    plot_loss_curve(loss_history)

    print(f"\nFigures saved to {OUT}/")


# ---- Marginal verification ------------------------------------------
EVAL_T = [0.0, 0.25, 0.5, 0.75, 1.0]


def verify_marginals(results: dict) -> tuple[dict, dict, dict]:
    print("\nMarginal verification (ODE, Euler) ...")
    N_sim = 5000
    rng = np.random.default_rng(123)
    pick = rng.random(N_sim) < 0.5
    x0 = np.where(
        pick,
        m1(0.0) + rng.standard_normal(N_sim),
        m2(0.0) + rng.standard_normal(N_sim),
    )

    def simulate(u_func):
        return euler_simulate(u_func, x0, T=T, n_step=1000, eval_t=EVAL_T)

    snaps_true = simulate(u_star)
    snaps_zero = simulate(lambda t, x: np.zeros_like(x))
    snaps_model = {
        key: simulate(results[key]["u_fn"]) for key in ("bilin", "tanh", "mlp")
    }

    w2 = {k: [] for k in ("bilin", "tanh", "mlp", "zero")}
    mmd = {k: [] for k in ("bilin", "tanh", "mlp", "zero")}
    for tv in EVAL_T:
        ref = snaps_true[tv]
        for key in ("bilin", "tanh", "mlp"):
            w2[key].append(sorted_w2(snaps_model[key][tv], ref))
            mmd[key].append(mmd_gauss(snaps_model[key][tv], ref, h=1.0))
        w2["zero"].append(sorted_w2(snaps_zero[tv], ref))
        mmd["zero"].append(mmd_gauss(snaps_zero[tv], ref, h=1.0))

    snaps_all = {"true": snaps_true, "zero": snaps_zero, **snaps_model}
    return w2, mmd, snaps_all


# ---- Plots ----------------------------------------------------------
def plot_drift_comparison(results: dict, mlp: DriftMLP) -> None:
    x_pl = np.linspace(-4, 4, 300)
    fig, axes = plt.subplots(3, 5, figsize=(16, 8), sharey="row", sharex=True)
    rows = [
        ("bilinear", results["bilin"]["u_fn"]),
        ("tanh dict", results["tanh"]["u_fn"]),
        ("MLP",     results["mlp"]["u_fn"]),
    ]
    for r, (lab, u_fn) in enumerate(rows):
        for c, tv in enumerate(EVAL_T):
            ax = axes[r, c]
            ust = u_star(tv, x_pl)
            u_hat = u_fn(tv, x_pl)
            pdf = rho_t(tv, x_pl)
            ax.fill_between(x_pl, -3.5, pdf * 4 - 3.5, alpha=0.15, color="C0")
            ax.plot(x_pl, ust, "k--", lw=1.5, label="$u^*$")
            ax.plot(x_pl, u_hat, "-", color=COLORS[["bilin","tanh","mlp"][r]],
                    lw=2.0, label="learned")
            if r == 0:
                ax.set_title(f"$t={tv}$")
            if c == 0:
                ax.set_ylabel(f"{lab}\n$u(t,x)$", fontsize=9)
            if r == 2:
                ax.set_xlabel("$x$")
            ax.set(xlim=(-4, 4), ylim=(-3.5, 3.5))
            if (r, c) == (0, 4):
                ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        r"Exp 3: Learned vs true drift $u^*(t,x)=-2\tanh(2(1-t)x)$",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/drift_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mse_bar(results: dict, mse_zero: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["zero", "bilinear", "tanh dict", "MLP"]
    mse_vals = [mse_zero, results["bilin"]["mse"],
                results["tanh"]["mse"], results["mlp"]["mse"]]
    colors = [COLORS["zero"], COLORS["bilin"], COLORS["tanh"], COLORS["mlp"]]
    ax.bar(names, mse_vals, color=colors, alpha=0.85)
    for i, v in enumerate(mse_vals):
        ax.text(i, v * 1.1, f"{v:.3f}", ha="center", fontsize=10)
    ax.set(ylabel=r"grid MSE$(\hat u, u^*)$",
           title="Exp 3: drift-recovery MSE by model class",
           yscale="log")
    fig.tight_layout()
    fig.savefig(f"{OUT}/mse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_w2_mmd(w2: dict, mmd: dict) -> None:
    fig, (ax_w, ax_m) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylab, title in [
        (ax_w, w2,  r"$W_2(\hat\mu_t,\mu_t)$", "Wasserstein-2"),
        (ax_m, mmd, r"$\mathrm{MMD}(\hat\mu_t,\mu_t)$",
         "MMD (Gaussian kernel, $h=1$)"),
    ]:
        ax.plot(EVAL_T, metric["zero"], "C7^--", lw=1.5, ms=6, label="zero")
        ax.plot(EVAL_T, metric["bilin"], "C2D-", lw=2, ms=7, label="bilinear")
        ax.plot(EVAL_T, metric["tanh"], "C1o-", lw=2, ms=7, label="tanh dict")
        ax.plot(EVAL_T, metric["mlp"],  "C3s-", lw=2, ms=7, label="MLP")
        ax.set(xlabel="$t$", ylabel=ylab, title=title, yscale="log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("Exp 3: Marginal tracking by model class", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/w2_mmd_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marginals(snaps: dict) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(16, 10), sharey="row")
    rows = [
        ("True $u^*$",  snaps["true"], "k"),
        ("Bilinear",    snaps["bilin"], COLORS["bilin"]),
        ("Tanh dict",   snaps["tanh"], COLORS["tanh"]),
        ("MLP",         snaps["mlp"],  COLORS["mlp"]),
    ]
    for r, (lab, snap, c) in enumerate(rows):
        for cc, tv in enumerate(EVAL_T):
            ax = axes[r, cc]
            ax.hist(snap[tv], bins=60, density=True, alpha=0.55,
                    color=c, range=(-5, 5))
            xr = np.linspace(-5, 5, 300)
            ax.plot(xr, rho_t(tv, xr), "k-", lw=1.5)
            if r == 0:
                ax.set_title(f"$t={tv}$")
            if cc == 0:
                ax.set_ylabel(lab, fontsize=9)
            ax.set_xlim(-5, 5)
    fig.suptitle("Exp 3: Marginal verification (ODE simulation)", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/marginal_verification.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(loss_history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(loss_history, "C3-", lw=0.8)
    ax.set(xlabel="iteration", ylabel="loss",
           title="Exp 3 MLP training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mlp_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
