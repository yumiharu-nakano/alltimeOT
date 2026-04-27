#!/usr/bin/env python3
"""Experiment 2: Roundtrip motion (d=1, sigma=0) -- the paper's key experiment.

    mu_t = N(A sin(pi t), 1),  A = 2,  T = 1
    true drift: u*(t, x) = A pi cos(pi t) = 2 pi cos(pi t)

mu_0 = mu_1 = N(0, 1), so two-marginal OT gives u == 0.  The all-time
marginal constraint instead recovers the non-trivial u*.
The model is quadratic in t to capture the cos(pi t) shape:
    u(t, x) = w0 + w1 t + w2 t^2 + w3 x  (4 parameters).
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
    feat_quadratic_t,
    mmd_gauss,
    sorted_w2,
)
from alltime_ot.problems import gaussian_translation

# ---- Problem ---------------------------------------------------------
T = 1.0
A_AMP = 2.0


def mean_fn(t: float) -> np.ndarray:
    return np.array([A_AMP * np.sin(np.pi * t)])


def u_true_const(t: float) -> float:
    return A_AMP * np.pi * np.cos(np.pi * t)


# ---- Hyper-parameters ------------------------------------------------
LAM = 1000.0
M, N, N0 = 50, 25, 50
K_ENS = 30
H = 1.0

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp2")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    print("Exp 2: Roundtrip (d=1, sigma=0)")
    print(f"  h={H}, lam={LAM}, M={M}, N={N}, N0={N0}, K_ens={K_ENS}")
    print("  Model: u = w0 + w1 t + w2 t^2 + w3 x  (4 params)")

    provider = gaussian_translation(mean_fn=mean_fn, d=1, M=M, N=N, N0=N0, T=T)
    objective = EnsembleObjective(
        batch_provider=provider,
        feat_fn=feat_quadratic_t,
        n_params=4,
        d=1,
        lam=LAM,
        h=H,
        T=T,
        K_ens=K_ENS,
        seed_offset=1000,
    )

    inits = [
        np.zeros(4),
        np.array([6.0, -12.0, 0.0, 0.0]),     # rough fit to 2 pi cos(pi t)
        np.array([3.0, 0.0, -6.0, 0.0]),
        np.array([0.0, 10.0, -10.0, 0.0]),
    ]
    tic = time.time()
    w, _, _ = ensemble_lbfgs(objective, inits)
    print(f"\nBest w = {np.round(w, 4).tolist()}")
    print(f"Optimisation time: {time.time() - tic:.1f}s")

    # Least-squares quadratic fit to 2*pi*cos(pi*t) for reference
    tt_fit = np.linspace(0, 1, 1000)
    A_fit = np.column_stack([np.ones_like(tt_fit), tt_fit, tt_fit ** 2])
    w_ls = np.linalg.lstsq(A_fit, 2 * np.pi * np.cos(np.pi * tt_fit), rcond=None)[0]
    print(f"L2-best quadratic fit of 2 pi cos(pi t): {np.round(w_ls, 4).tolist()}")

    # Drift grid MSE on [0, T] x [-4, 4]
    te = np.linspace(0, T, 50)
    xe = np.linspace(-4, 4, 100)
    TT, XX = np.meshgrid(te, xe, indexing="ij")
    u_learned = w[0] + w[1] * TT + w[2] * TT ** 2 + w[3] * XX
    u_star = 2 * np.pi * np.cos(np.pi * TT)
    mse = float(((u_learned - u_star) ** 2).mean())
    mse_zero = float((u_star ** 2).mean())
    print(f"drift grid MSE(u_hat, u*) on [0,1]x[-4,4] = {mse:.6f}")
    print(f"drift grid MSE(0, u*)    on [0,1]x[-4,4] = {mse_zero:.4f}  (2-marginal baseline)")

    model = LinearDriftModel(w, feat_quadratic_t, d=1)
    plot_all(model, w, w_ls, mse, mse_zero)


def plot_all(
    model: LinearDriftModel,
    w: np.ndarray,
    w_ls: np.ndarray,
    mse: float,
    mse_zero: float,
) -> None:
    x_pl = np.linspace(-4, 4, 200)

    # Drift comparison at five time slices
    fig, axes = plt.subplots(1, 5, figsize=(16, 3), sharey=True)
    for i, tv in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        ax = axes[i]
        u_hat = model(tv, x_pl)
        u_tr = u_true_const(tv)
        pdf = norm.pdf(x_pl, loc=mean_fn(tv)[0])
        ax.fill_between(x_pl, -8, pdf * 4 - 8, alpha=0.15, color="C0")
        ax.axhline(u_tr, color="k", ls="--", lw=1.5, label=f"$u^*={u_tr:.2f}$")
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.plot(x_pl, u_hat, "C1-", lw=2, label="learned")
        ax.set(title=f"$t={tv}$", xlabel="$x$", xlim=(-4, 4), ylim=(-8, 10))
        if i == 0:
            ax.set_ylabel("$u(t,x)$")
        if i == 4:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(r"Exp 2: Roundtrip -- learned vs true drift ($\sigma=0$)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/drift_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Time profile at x = 0
    tt = np.linspace(0, 1, 200)
    u_hat_t = w[0] + w[1] * tt + w[2] * tt ** 2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, 2 * np.pi * np.cos(np.pi * tt), "k-", lw=2,
            label=r"$u^*(t) = 2\pi\cos(\pi t)$")
    ax.plot(tt, u_hat_t, "C1-", lw=2, label="All-time learned")
    ax.axhline(0, color="C2", ls="--", lw=2, label=r"2-marginal OT ($u\equiv 0$)")
    ax.set(xlabel="$t$", ylabel="$u(t, x=0)$",
           title="Drift at $x=0$: all-time vs 2-marginal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/time_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Parameter bar chart
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    idx = np.arange(4)
    ax.bar(idx - 0.15, [*w_ls, 0.0], 0.3, label=r"$L^2$-best quadratic",
           color="C0", alpha=0.7)
    ax.bar(idx + 0.15, w, 0.3, label=r"learned $\hat w$", color="C1", alpha=0.7)
    ax.set_xticks(idx)
    ax.set_xticklabels(["$w_0$", "$w_1$", "$w_2$", "$w_3$"])
    ax.set(ylabel="value", title="Parameter comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/param_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ODE verification: learned / zero / true drifts
    print("\nMarginal verification (ODE, Euler) ...")
    N_sim = 5000
    rng = np.random.default_rng(123)
    x0 = mean_fn(0.0)[0] + rng.standard_normal(N_sim)
    eval_t = [0.0, 0.25, 0.5, 0.75, 1.0]

    def learned(t: float, x: np.ndarray) -> np.ndarray:
        return model(t, x)

    def zero(t: float, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def true_drift(t: float, x: np.ndarray) -> np.ndarray:
        return np.full_like(x, u_true_const(t))

    snaps_l = euler_simulate(learned, x0, T=T, n_step=1000, eval_t=eval_t)
    snaps_z = euler_simulate(zero, x0, T=T, n_step=1000, eval_t=eval_t)
    snaps_t = euler_simulate(true_drift, x0, T=T, n_step=1000, eval_t=eval_t)

    print(f"  {'t':>5s}  {'W2_learn':>10s}  {'W2_zero':>10s}  {'W2_true':>10s}  "
          f"{'MMD_learn':>10s}  {'MMD_zero':>10s}  {'MMD_true':>10s}")
    w2_l, w2_z, w2_t = [], [], []
    mmd_l, mmd_z, mmd_t = [], [], []
    rng_ref = np.random.default_rng(321)
    for tv in eval_t:
        ref = norm.ppf((np.arange(N_sim) + 0.5) / N_sim,
                       loc=mean_fn(tv)[0], scale=1.0)
        w2_l.append(sorted_w2(snaps_l[tv], ref))
        w2_z.append(sorted_w2(snaps_z[tv], ref))
        w2_t.append(sorted_w2(snaps_t[tv], ref))
        # MMD uses the same Gaussian kernel (h=1.0) as the RKHS loss.
        # Fresh iid reference sample to avoid the artificial quantile-grid ordering.
        ref_mmd = mean_fn(tv)[0] + rng_ref.standard_normal(N_sim)
        mmd_l.append(mmd_gauss(snaps_l[tv], ref_mmd, h=1.0))
        mmd_z.append(mmd_gauss(snaps_z[tv], ref_mmd, h=1.0))
        mmd_t.append(mmd_gauss(snaps_t[tv], ref_mmd, h=1.0))
        print(f"  {tv:5.2f}  {w2_l[-1]:10.4f}  {w2_z[-1]:10.4f}  {w2_t[-1]:10.4f}  "
              f"{mmd_l[-1]:10.4f}  {mmd_z[-1]:10.4f}  {mmd_t[-1]:10.4f}")

    # Three-row marginal panel
    fig, axes = plt.subplots(3, 5, figsize=(16, 8), sharey="row")
    labels = ["True $u^*$", "All-time learned", r"2-marginal ($u\equiv 0$)"]
    snap_list = [snaps_t, snaps_l, snaps_z]
    for row, (snaps, lab) in enumerate(zip(snap_list, labels)):
        for col, tv in enumerate(eval_t):
            ax = axes[row, col]
            ax.hist(snaps[tv], bins=60, density=True, alpha=0.5,
                    color=f"C{row}", range=(-5, 5))
            xr = np.linspace(-5, 5, 200)
            ax.plot(xr, norm.pdf(xr, mean_fn(tv)[0], 1), "k-", lw=1.5)
            if row == 0:
                ax.set_title(f"$t={tv}$")
            if col == 0:
                ax.set_ylabel(lab, fontsize=9)
            ax.set_xlim(-5, 5)
    fig.suptitle("Exp 2: Marginal verification -- all-time vs 2-marginal", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT}/marginal_verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # W_2 and MMD comparison panels
    fig, (ax_w, ax_m) = plt.subplots(1, 2, figsize=(12, 4))
    ax_w.plot(eval_t, w2_t, "ko--", lw=1.5, ms=6, label="True $u^*$")
    ax_w.plot(eval_t, w2_l, "C1o-", lw=2, ms=7, label="All-time learned")
    ax_w.plot(eval_t, w2_z, "C2s-", lw=2, ms=7, label=r"2-marginal ($u\equiv 0$)")
    ax_w.set(xlabel="$t$", ylabel=r"$W_2(\hat\mu_t, \mu_t)$",
             title="Wasserstein-2")
    ax_w.legend(fontsize=9)
    ax_w.grid(True, alpha=0.3)

    ax_m.plot(eval_t, mmd_t, "ko--", lw=1.5, ms=6, label="True $u^*$")
    ax_m.plot(eval_t, mmd_l, "C1o-", lw=2, ms=7, label="All-time learned")
    ax_m.plot(eval_t, mmd_z, "C2s-", lw=2, ms=7, label=r"2-marginal ($u\equiv 0$)")
    ax_m.set(xlabel="$t$",
             ylabel=r"$\mathrm{MMD}(\hat\mu_t, \mu_t)$",
             title=f"MMD (Gaussian kernel, $h=1$)")
    ax_m.legend(fontsize=9)
    ax_m.grid(True, alpha=0.3)

    fig.suptitle("Marginal tracking: all-time vs 2-marginal", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/w2_mmd_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Summary ===")
    print(f"  All-time learned:  mean W2 = {np.mean(w2_l):.4f},  "
          f"mean MMD = {np.mean(mmd_l):.4f}")
    print(f"  2-marginal (u=0):  mean W2 = {np.mean(w2_z):.4f},  "
          f"mean MMD = {np.mean(mmd_z):.4f}")
    print(f"  True drift:        mean W2 = {np.mean(w2_t):.4f},  "
          f"mean MMD = {np.mean(mmd_t):.4f}")
    print(f"  drift grid MSE (learned) = {mse:.4f}")
    print(f"  drift grid MSE (zero)    = {mse_zero:.4f}")
    print(f"\nAll figures saved to {OUT}/")


if __name__ == "__main__":
    main()
