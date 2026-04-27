#!/usr/bin/env python3
"""Experiment 5: 2-d bifurcation / bimodal merging (d=2, sigma=0).

    mu_t(x_1, x_2) = [1/2 N(-2+2t, 1) + 1/2 N(2-2t, 1)](x_1) * N(0, 1)(x_2)

The x_1 marginal is the Exp 4 bimodal merge; x_2 is N(0, 1) for all t,
so u*_1(t, x) = -2 tanh(2 (1-t) x_1) and u*_2(t, x) = 0.

Three feature dictionaries with the same features in both output
channels are compared; the estimator has to learn that channel 2 is
identically zero.
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
    feat_bilinear,
    feat_tanh_merger_2d,
    mmd_gauss,
    sliced_w2,
)
from alltime_ot.features import feat_count
from alltime_ot.problems import gaussian_mixture_x1_times_normal_x2

# ---- Problem ---------------------------------------------------------
T = 1.0
D = 2

def m1(t: float) -> float:
    return -2.0 + 2.0 * t

def m2(t: float) -> float:
    return 2.0 - 2.0 * t

def means_fn(t: float) -> tuple[float, float]:
    return m1(t), m2(t)

# ---- Hyper-parameters ------------------------------------------------
LAM = 3000.0
M, N, N0 = 25, 25, 60
K_ENS = 10
H = 1.0

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp5")
os.makedirs(OUT, exist_ok=True)

MODELS = {
    "affine": feat_affine,        # 4 feats x 2 channels = 8 params
    "bilin": feat_bilinear,       # 6 feats x 2 channels = 12 params
    "tanh": feat_tanh_merger_2d,  # 10 feats x 2 channels = 20 params
}


def main() -> None:
    print("Exp 5: 2D bifurcation (d=2, sigma=0)")
    print("  mu_t = [1/2 N(-2+2t,1)+1/2 N(2-2t,1)] tensor N(0,1)")
    print("  u* = (-2 tanh(2(1-t) x_1), 0)")
    print(f"  h={H}, lam={LAM}, M={M}, N={N}, N0={N0}, K_ens={K_ENS}")

    provider = gaussian_mixture_x1_times_normal_x2(
        means_fn=means_fn, M=M, N=N, N0=N0, T=T,
    )

    # Grid for MSE
    te = np.linspace(0, T, 20)
    g1 = np.linspace(-4, 4, 25)
    g2 = np.linspace(-3, 3, 15)
    GT, GX1, GX2 = np.meshgrid(te, g1, g2, indexing="ij")
    grid_x = np.column_stack([GX1.ravel(), GX2.ravel()])
    grid_t = GT.ravel()
    u_star_grid = np.zeros((grid_t.size, 2))
    u_star_grid[:, 0] = -2.0 * np.tanh(2.0 * (1.0 - grid_t) * grid_x[:, 0])

    inits_of = {
        "affine": [np.zeros(4 * 2),
                   np.array([0., 0., -0.5, 0., 0., 0., 0., 0.])],
        "bilin": [np.zeros(6 * 2),
                  np.array([0., 0., -1.0, 0., 0.5, 0., 0., 0., 0., 0., 0., 0.])],
        "tanh": [
            np.zeros(10 * 2),
            np.array([0., 0., 0., 0., 0., 0., 0., 0., -2.0, 2.0,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            np.array([0., 0., 0., 0., 0., 0., -0.5, 0.5, -1.0, 1.0,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        ],
    }

    results: dict[str, dict] = {}
    for name, feat in MODELS.items():
        p_feat = feat_count(feat, d=D)
        n_params = p_feat * D
        print(f"\n-- Model: {name} ({n_params} params) --")

        objective = EnsembleObjective(
            batch_provider=provider,
            feat_fn=feat,
            n_params=n_params,
            d=D,
            lam=LAM,
            h=H,
            T=T,
            K_ens=K_ENS,
            seed_offset=6000,
        )
        tic = time.time()
        w, _, _ = ensemble_lbfgs(objective, inits_of[name])
        elapsed = time.time() - tic

        W_hat = w.reshape(p_feat, D)
        model = LinearDriftModel(w, feat, d=D)
        u_hat_grid = model(grid_t, grid_x)
        mse_total = float(((u_hat_grid - u_star_grid) ** 2).sum(-1).mean())
        mse_c1 = float(((u_hat_grid[:, 0] - u_star_grid[:, 0]) ** 2).mean())
        mse_c2 = float((u_hat_grid[:, 1] ** 2).mean())
        print(f"  drift grid MSE total = {mse_total:.5f}  "
              f"(ch1={mse_c1:.5f}, ch2={mse_c2:.5f})  time={elapsed:.1f}s")

        results[name] = {
            "W": W_hat, "model": model, "feat": feat,
            "mse": mse_total, "mse_c1": mse_c1, "mse_c2": mse_c2,
        }

    mse_zero = float((u_star_grid ** 2).sum(-1).mean())
    print(f"\nZero-drift grid MSE = {mse_zero:.5f}")

    plot_slices(results)
    plot_mse_bars(results, mse_zero)
    verify_marginals(results)
    save_table(results, mse_zero)
    print(f"\nAll figures saved to {OUT}/")


def plot_slices(results: dict[str, dict]) -> None:
    x1_pl = np.linspace(-4, 4, 200)
    zero_col = np.zeros_like(x1_pl)
    t_eval = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(3, 5, figsize=(16, 8), sharey="row", sharex=True)
    for row, name in enumerate(("affine", "bilin", "tanh")):
        model = results[name]["model"]
        for col, tv in enumerate(t_eval):
            ax = axes[row, col]
            xpts = np.column_stack([x1_pl, zero_col])
            u_hat = model(tv, xpts)
            ust1 = -2.0 * np.tanh(2.0 * (1.0 - tv) * x1_pl)
            pdf = 0.5 * norm.pdf(x1_pl, m1(tv), 1.0) + 0.5 * norm.pdf(x1_pl, m2(tv), 1.0)
            ax.fill_between(x1_pl, -3.5, pdf * 4 - 3.5, alpha=0.15, color="C0")
            ax.plot(x1_pl, ust1, "k--", lw=1.5, label="$u_1^*$")
            ax.plot(x1_pl, u_hat[:, 0], "C1-", lw=2, label=r"$\hat u_1$")
            ax.plot(x1_pl, u_hat[:, 1], "C3-", lw=1.5, label=r"$\hat u_2$")
            ax.axhline(0, color="gray", lw=0.5)
            if row == 0:
                ax.set_title(f"$t={tv}$")
            if col == 0:
                ax.set_ylabel(f"{name}\n$u$", fontsize=9)
            if row == 2:
                ax.set_xlabel("$x_1$")
            ax.set(xlim=(-4, 4), ylim=(-3.5, 3.5))
            if (row, col) == (0, 4):
                ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(r"Exp 5: drift along the $x_2=0$ slice ($u^*_2\equiv 0$)", y=1.00)
    fig.tight_layout()
    fig.savefig(f"{OUT}/drift_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mse_bars(results: dict[str, dict], mse_zero: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["zero", "affine", "bilin", "tanh"]
    mse_vals = [mse_zero] + [results[k]["mse"] for k in names[1:]]
    colors = ["C7", "C0", "C2", "C1"]
    ax.bar(names, mse_vals, color=colors, alpha=0.8)
    for i, v in enumerate(mse_vals):
        ax.text(i, v * 1.08, f"{v:.3f}", ha="center", fontsize=10)
    ax.set(ylabel=r"grid MSE$(\hat u, u^*)$",
           title="Exp 5: drift-recovery MSE by model class", yscale="log")
    fig.tight_layout()
    fig.savefig(f"{OUT}/mse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def verify_marginals(results: dict[str, dict]) -> None:
    N_sim, n_step = 4000, 600
    eval_t = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Shared initial samples and reference samples (same noise for fair comparison).
    rng = np.random.default_rng(888)
    pick = rng.random(N_sim) < 0.5
    x1 = np.where(pick, m1(0.0) + rng.standard_normal(N_sim),
                  m2(0.0) + rng.standard_normal(N_sim))
    x2 = rng.standard_normal(N_sim)
    x0 = np.column_stack([x1, x2])

    refs: dict[float, np.ndarray] = {}
    rng_ref = np.random.default_rng(999)
    for tv in eval_t:
        pref = rng_ref.random(N_sim) < 0.5
        x1r = np.where(pref, m1(tv) + rng_ref.standard_normal(N_sim),
                       m2(tv) + rng_ref.standard_normal(N_sim))
        x2r = rng_ref.standard_normal(N_sim)
        refs[tv] = np.column_stack([x1r, x2r])

    for name in ("affine", "bilin", "tanh"):
        model = results[name]["model"]
        print(f"\nMarginal verification (ODE Euler) -- {name} model ...")

        def u_ode(t: float, x: np.ndarray, _m=model) -> np.ndarray:
            return _m(t, x)

        snaps = euler_simulate(u_ode, x0, T=T, n_step=n_step, eval_t=eval_t)

        sw2_vals, mmd_vals = [], []
        print(f"  {'t':>5s}  {'SW_2':>8s}  {'MMD':>8s}")
        for tv in eval_t:
            sw = sliced_w2(snaps[tv], refs[tv])
            m = mmd_gauss(snaps[tv], refs[tv], h=1.0)
            sw2_vals.append(sw)
            mmd_vals.append(m)
            print(f"  {tv:5.2f}  {sw:8.4f}  {m:8.4f}")
        mean_sw2 = float(np.mean(sw2_vals))
        max_sw2 = float(np.max(sw2_vals))
        mean_mmd = float(np.mean(mmd_vals))
        max_mmd = float(np.max(mmd_vals))
        print(f"  Mean sliced W2 = {mean_sw2:.4f},  max sliced W2 = {max_sw2:.4f}")
        print(f"  Mean MMD       = {mean_mmd:.4f},  max MMD       = {max_mmd:.4f}")

        results[name].update({
            "sw2": sw2_vals, "mean_sw2": mean_sw2, "max_sw2": max_sw2,
            "mmd": mmd_vals, "mean_mmd": mean_mmd, "max_mmd": max_mmd,
            "snaps": snaps,
        })

    # Marginal verification figure uses the tanh model.
    snaps = results["tanh"]["snaps"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    gg = np.linspace(-4, 4, 80)
    XXc, YYc = np.meshgrid(gg, gg)

    for i, tv in enumerate(eval_t):
        den = (0.5 * norm.pdf(XXc, m1(tv), 1.0) + 0.5 * norm.pdf(XXc, m2(tv), 1.0)) \
            * norm.pdf(YYc, 0.0, 1.0)

        ax = axes[0, i]
        s = snaps[tv]
        ax.scatter(s[:, 0], s[:, 1], s=3, alpha=0.3, color="C1")
        ax.contour(XXc, YYc, den, levels=5, colors="k", linewidths=1, alpha=0.6)
        ax.set(title=f"$t={tv}$", xlim=(-4, 4), ylim=(-4, 4))
        ax.set_aspect("equal")
        if i == 0:
            ax.set_ylabel("learned ODE\n$x_2$", fontsize=9)

        ax = axes[1, i]
        local = np.random.default_rng(2000 + i)
        pref = local.random(N_sim) < 0.5
        x1r = np.where(pref, m1(tv) + local.standard_normal(N_sim),
                       m2(tv) + local.standard_normal(N_sim))
        x2r = local.standard_normal(N_sim)
        ax.scatter(x1r, x2r, s=3, alpha=0.3, color="C0")
        ax.contour(XXc, YYc, den, levels=5, colors="k", linewidths=1, alpha=0.6)
        ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="$x_1$")
        ax.set_aspect("equal")
        if i == 0:
            ax.set_ylabel(r"true $\mu_t$" + "\n$x_2$", fontsize=9)

    fig.suptitle("Exp 5: marginal verification (tanh model)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/marginal_verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_table(results: dict[str, dict], mse_zero: float) -> None:
    with open(f"{OUT}/mse_table.txt", "w") as f:
        f.write(f"zero   {mse_zero:.5f}\n")
        for name in ("affine", "bilin", "tanh"):
            r = results[name]
            f.write(
                f"{name:6s}  total={r['mse']:.5f}  "
                f"u1={r['mse_c1']:.5f}  u2={r['mse_c2']:.5f}  "
                f"mean_SW2={r.get('mean_sw2', float('nan')):.5f}  "
                f"mean_MMD={r.get('mean_mmd', float('nan')):.5f}\n"
            )


if __name__ == "__main__":
    main()
