#!/usr/bin/env python3
"""Exp 2 -- Comparison: all-time OT vs Waddington-OT (Schiebinger et al.).

WOT solves entropic OT (Sinkhorn) between consecutive marginal snapshots
and reads the implied drift off the barycentric projection.  For the
roundtrip problem WOT can track marginals well given enough snapshots,
but its drift recovery degrades as M grows: the O(1/sqrt(N)) sample
noise is divided by dt = T/M, amplifying error as O(M/sqrt(N)).
"""

from __future__ import annotations

import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from alltime_ot import euler_simulate, mmd_gauss, sorted_w2
from alltime_ot.baselines import sinkhorn_wot_drift

warnings.filterwarnings("ignore")

# ---- Problem ---------------------------------------------------------
T = 1.0
A_AMP = 2.0

def mu_mean(t: float) -> float:
    return A_AMP * np.sin(np.pi * t)

def u_true(t: float) -> float:
    return A_AMP * np.pi * np.cos(np.pi * t)

def sample_fn(t: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return mu_mean(t) + rng.standard_normal(n)

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp2")
os.makedirs(OUT, exist_ok=True)

# ---- ODE simulation + W2 ---------------------------------------------
N_SIM = 5000
N_STEP = 1000
EVAL_T = [0.0, 0.25, 0.5, 0.75, 1.0]


def simulate_and_w2(u_func, seed: int = 123) -> tuple[list[float], list[float]]:
    """Return (W_2, MMD) lists at each time in ``EVAL_T``."""
    rng = np.random.default_rng(seed)
    x0 = mu_mean(0.0) + rng.standard_normal(N_SIM)
    snaps = euler_simulate(u_func, x0, T=T, n_step=N_STEP, eval_t=EVAL_T)
    rng_ref = np.random.default_rng(seed + 1)
    w2s, mmds = [], []
    for tv in EVAL_T:
        ref = norm.ppf((np.arange(N_SIM) + 0.5) / N_SIM, loc=mu_mean(tv), scale=1.0)
        w2s.append(sorted_w2(snaps[tv], ref))
        ref_mmd = mu_mean(tv) + rng_ref.standard_normal(N_SIM)
        mmds.append(mmd_gauss(snaps[tv], ref_mmd, h=1.0))
    return w2s, mmds


# ---- All-time baseline (from exp2_roundtrip.py) ----------------------
W_ALLTIME = np.array([7.2044, -14.9183, 1.4829, -0.0001])


def u_alltime(t: float, x: np.ndarray) -> np.ndarray:
    return W_ALLTIME[0] + W_ALLTIME[1] * t + W_ALLTIME[2] * t ** 2 + W_ALLTIME[3] * x


def main() -> None:
    print("=" * 60)
    print("Exp 2 -- WOT comparison")
    print("=" * 60)

    M_list = (5, 10, 20, 50)
    N_wot = 200

    tt_plot = np.linspace(0.02, 0.98, 200)
    u_true_plot = u_true(tt_plot)
    u_alltime_plot = np.array([u_alltime(t, np.array([0.0]))[0] for t in tt_plot])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_wot = ["C2", "C3", "C4", "C5"]

    ax = axes[0]
    ax.plot(tt_plot, u_true_plot, "k-", lw=2.5, label="True $u^*$", zorder=10)
    ax.plot(tt_plot, u_alltime_plot, "C1-", lw=2, label="All-time (ours)")
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)

    wot_cache: dict[int, tuple[np.ndarray, np.ndarray, callable, float]] = {}
    for i, M_wot in enumerate(M_list):
        tic = time.time()
        out = sinkhorn_wot_drift(
            sample_fn, M_wot=M_wot, N_sample=N_wot,
            eps_factor=1.0, T=T, seed=42,
        )
        elapsed = time.time() - tic
        u_true_mid = np.array([u_true(t) for t in out["t_mid"]])
        mse = float(((out["u_mid"] - u_true_mid) ** 2).mean())
        wot_cache[M_wot] = (out["t_mid"], out["u_mid"], out["u_func"], mse)
        ax.plot(out["t_mid"], out["u_mid"], "o-", color=colors_wot[i], ms=4, lw=1.5,
                label=f"WOT ($M{{{M_wot}}}$)", alpha=0.8)
        print(f"  WOT M={M_wot:3d}, N={N_wot}: drift MSE={mse:.4f}  ({elapsed:.1f}s)")

    ax.set(xlabel="$t$", ylabel="$u(t, x{=}0)$",
           title="Drift time profile at $x=0$", ylim=(-10, 10))
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Right panel: W2 marginal tracking
    ax = axes[1]
    w2_true, mmd_true = simulate_and_w2(lambda t, x: u_true(t) * np.ones_like(x))
    w2_alltime, mmd_alltime = simulate_and_w2(u_alltime)
    w2_zero, mmd_zero = simulate_and_w2(lambda t, x: np.zeros_like(x))

    ax.plot(EVAL_T, w2_true, "ko--", lw=1.5, ms=6, label="True $u^*$")
    ax.plot(EVAL_T, w2_alltime, "C1o-", lw=2, ms=7, label="All-time (ours)")
    ax.plot(EVAL_T, w2_zero, "gray", ls="--", lw=2, marker="s", ms=6,
            label=r"2-marg ($u\equiv 0$)")

    results_table = [
        ("True u*", float(np.mean(w2_true)), float(np.max(w2_true)),
         float(np.mean(mmd_true)), 0.0),
        ("All-time (ours)", float(np.mean(w2_alltime)),
         float(np.max(w2_alltime)), float(np.mean(mmd_alltime)), 0.64),
    ]
    for i, M_wot in enumerate(M_list):
        _, _, u_wot_f, mse = wot_cache[M_wot]
        w2_wot, mmd_wot = simulate_and_w2(u_wot_f)
        ax.plot(EVAL_T, w2_wot, "o-", color=colors_wot[i], ms=5, lw=1.5,
                label=f"WOT ($M{{{M_wot}}}$)", alpha=0.8)
        results_table.append((f"WOT M={M_wot}",
                              float(np.mean(w2_wot)), float(np.max(w2_wot)),
                              float(np.mean(mmd_wot)), mse))
    results_table.append(("2-marginal (u=0)",
                          float(np.mean(w2_zero)), float(np.max(w2_zero)),
                          float(np.mean(mmd_zero)), 20.13))

    ax.set(xlabel="$t$", ylabel=r"$W_2(\hat\mu_t, \mu_t)$",
           title="Marginal tracking comparison", yscale="log", ylim=(0.01, 3))
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 2: All-time OT vs Waddington-OT", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/wot_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 82)
    print(f"{'Method':30s} {'mean W2':>10s} {'max W2':>10s} "
          f"{'mean MMD':>10s} {'drift MSE':>12s}")
    print("-" * 82)
    for name, mw2, maxw2, mmd, mse in results_table:
        print(f"  {name:28s} {mw2:10.4f} {maxw2:10.4f} {mmd:10.4f} {mse:12.4f}")
    print("=" * 82)

    # Convergence sweep
    print("\n=== WOT drift MSE, mean W2, mean MMD vs M ===")
    M_sweep = (3, 5, 10, 20, 50)
    mse_list, w2_list, mmd_list = [], [], []
    for M_s in M_sweep:
        out = sinkhorn_wot_drift(
            sample_fn, M_wot=M_s, N_sample=N_wot, eps_factor=1.0, T=T, seed=42,
        )
        u_true_mid = np.array([u_true(t) for t in out["t_mid"]])
        mse_s = float(((out["u_mid"] - u_true_mid) ** 2).mean())
        w2_vals, mmd_vals = simulate_and_w2(out["u_func"])
        w2_s = float(np.mean(w2_vals))
        mmd_s = float(np.mean(mmd_vals))
        mse_list.append(mse_s)
        w2_list.append(w2_s)
        mmd_list.append(mmd_s)
        print(f"  M={M_s:3d}: drift MSE={mse_s:.4f}, mean W2={w2_s:.4f}, "
              f"mean MMD={mmd_s:.4f}")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.semilogy(M_sweep, mse_list, "C2o-", lw=2, ms=8, label="WOT drift MSE")
    ax1.axhline(0.64, color="C1", ls="--", lw=1.5, label="All-time drift MSE")
    ax1.set_xlabel("Number of WOT snapshots $M$")
    ax1.set_ylabel("Drift MSE", color="C2")
    ax1.tick_params(axis="y", labelcolor="C2")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(M_sweep, w2_list, "C4s--", lw=2, ms=7, label="WOT mean $W_2$")
    ax2.axhline(float(np.mean(w2_alltime)), color="C1", ls=":", lw=1.5,
                label="All-time mean $W_2$")
    ax2.set_ylabel("Mean $W_2$", color="C4")
    ax2.tick_params(axis="y", labelcolor="C4")
    ax2.legend(fontsize=8, loc="right")

    ax1.set_title("WOT trade-off: drift recovery vs marginal tracking")
    fig.tight_layout()
    fig.savefig(f"{OUT}/wot_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll figures saved to {OUT}/")


if __name__ == "__main__":
    main()
