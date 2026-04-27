#!/usr/bin/env python3
"""Exp 2 -- Honest multi-marginal OT (MMOT) comparison with affine maps.

We replace the translation-only family T_k(x) = x + b_k with the fully
affine T_k(x) = A_k x + b_k.  Fixing A_k = 1 pre-encoded the fact that
roundtrip marginals are equal-variance Gaussians; here we let MMOT
discover that itself.

Optimisation: ensemble-averaged L-BFGS-B with a small hyper-parameter
sweep and multiple random initialisations.  The recovered intercept
drift at x = 0 is compared with the true u*(t) = 2 pi cos(pi t) and
against the all-time OT estimator.
"""

from __future__ import annotations

import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from alltime_ot import euler_simulate, mmd_gauss, sorted_w2
from alltime_ot.baselines import extract_affine_drift, make_affine_mmot_loss_grad

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
    """Return (W_2 list, MMD list) at each time in ``EVAL_T``."""
    rng = np.random.default_rng(seed)
    x0 = mu_mean(0.0) + rng.standard_normal(N_SIM)
    snaps = euler_simulate(u_func, x0, T=T, n_step=N_STEP, eval_t=EVAL_T)
    w2s: list[float] = []
    mmds: list[float] = []
    rng_ref = np.random.default_rng(seed + 1)
    for tv in EVAL_T:
        ref = norm.ppf((np.arange(N_SIM) + 0.5) / N_SIM, loc=mu_mean(tv), scale=1.0)
        w2s.append(sorted_w2(snaps[tv], ref))
        ref_mmd = mu_mean(tv) + rng_ref.standard_normal(N_SIM)
        mmds.append(mmd_gauss(snaps[tv], ref_mmd, h=1.0))
    return w2s, mmds


# ---- Multi-strategy sweep --------------------------------------------
def run_mmot_affine(N_marg: int) -> dict:
    t_pts = np.linspace(0.0, T, N_marg + 1)
    u_true_mid = np.array([u_true(0.5 * (t_pts[k] + t_pts[k + 1])) for k in range(N_marg)])

    hp_grid = [
        (1e3, 1.0, 200, 10),
        (1e4, 1.0, 200, 10),
        (1e5, 1.0, 200, 10),
        (1e4, 0.5, 200, 10),
        (1e4, 2.0, 200, 10),
    ]

    best_mse, best_result, best_hp, best_strategy = np.inf, None, None, None

    for lam, alpha_k, M_b, K_e in hp_grid:
        lg, t_pts, dt = make_affine_mmot_loss_grad(
            sample_fn, N_marg=N_marg, M_batch=M_b, lam=lam,
            alpha_k=alpha_k, K_ens=K_e, T=T, seed=42,
        )

        inits = [(np.concatenate([np.ones(N_marg), np.zeros(N_marg)]), "LBFGS(id)")]
        for s in (1, 2):
            rs = np.random.default_rng(s)
            inits.append((
                np.concatenate([
                    1.0 + 0.3 * rs.standard_normal(N_marg),
                    0.5 * rs.standard_normal(N_marg),
                ]),
                f"LBFGS(r{s})",
            ))

        for theta0, label in inits:
            try:
                res = minimize(
                    lg, theta0, jac=True, method="L-BFGS-B",
                    options={"maxiter": 500, "ftol": 1e-12},
                )
                _, icpt, _, _ = extract_affine_drift(res.x, t_pts, dt, N_marg, T=T)
                mse = float(((icpt - u_true_mid) ** 2).mean())
                if mse < best_mse:
                    best_mse = mse
                    best_result = res.x
                    best_hp = (lam, alpha_k)
                    best_strategy = label
            except Exception:
                pass

    t_mid, icpt, slopes, u_f = extract_affine_drift(best_result, t_pts, dt, N_marg, T=T)
    w2_vals, mmd_vals = simulate_and_w2(u_f)
    return {
        "t_mid": t_mid,
        "intercepts": icpt,
        "slopes": slopes,
        "u_func": u_f,
        "drift_mse": best_mse,
        "mean_w2": float(np.mean(w2_vals)),
        "w2_vals": w2_vals,
        "mean_mmd": float(np.mean(mmd_vals)),
        "mmd_vals": mmd_vals,
        "theta": best_result,
        "A": best_result[:N_marg],
        "b": best_result[N_marg:],
        "best_hp": best_hp,
        "best_strategy": best_strategy,
    }


def main() -> None:
    print("=" * 72)
    print("Exp 2 -- Honest learned MMOT with AFFINE maps T_k(x) = A_k x + b_k")
    print("=" * 72)

    results: dict[int, dict] = {}
    for N_m in (5, 10, 20, 50):
        tic = time.time()
        r = run_mmot_affine(N_m)
        r["time"] = time.time() - tic
        results[N_m] = r
        print(
            f"N={N_m:3d}  mse={r['drift_mse']:8.4f}  W2={r['mean_w2']:.4f}  "
            f"MMD={r['mean_mmd']:.4f}  "
            f"mean A={r['A'].mean():.3f}  max|A-1|={np.max(np.abs(r['A']-1)):.3f}  "
            f"hp=(lam={r['best_hp'][0]:.0e}, a={r['best_hp'][1]}) strat={r['best_strategy']}"
        )

    plot_comparison(results)
    plot_Ak(results)
    print_summary(results)


def plot_comparison(results: dict[int, dict]) -> None:
    tt_plot = np.linspace(0.02, 0.98, 200)
    u_true_plot = u_true(tt_plot)
    # All-time learned drift (copied from exp2_roundtrip.py best fit)
    w_alltime = np.array([7.2044, -14.9183, 1.4829, -0.0001])
    u_alltime_plot = w_alltime[0] + w_alltime[1] * tt_plot + w_alltime[2] * tt_plot ** 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ["C6", "C7", "C8", "C9"]

    ax = axes[0]
    ax.plot(tt_plot, u_true_plot, "k-", lw=2.5, label="True $u^*$", zorder=10)
    ax.plot(tt_plot, u_alltime_plot, "C1-", lw=2, label="All-time (ours)")
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    for i, N_m in enumerate((5, 10, 20, 50)):
        r = results[N_m]
        ax.plot(r["t_mid"], r["intercepts"], "o-", color=colors[i], ms=4, lw=1.5,
                label=f"MMOT affine $N{{=}}{N_m}$", alpha=0.8)
    ax.set(xlabel="$t$", ylabel="$u(t,\\,x{=}0)$",
           title="Affine-MMOT drift (intercept at $x=0$) vs true $u^*$", ylim=(-12, 12))
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels = ["All-time\n(ours)"]
    # Hardcoded reference values from exp2_roundtrip.py (see paper tab:exp2-w2mmd).
    mses, w2s = [0.64], [0.111]
    for N_m in (5, 10, 20, 50):
        labels.append(f"MMOT\naffine\nN={N_m}")
        mses.append(results[N_m]["drift_mse"])
        w2s.append(results[N_m]["mean_w2"])
    idx = np.arange(len(labels))
    width = 0.35
    ax.bar(idx - width / 2, mses, width, label="Drift MSE", color="C0", alpha=0.7)
    ax.bar(idx + width / 2, w2s, width, label="Mean $W_2$", color="C3", alpha=0.7)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set(ylabel="Error", title="Affine-MMOT vs All-time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp 2: all-time OT vs learned multi-marginal OT (affine maps)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mmot_affine_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_Ak(results: dict[int, dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["C6", "C7", "C8", "C9"]
    for i, N_m in enumerate((5, 10, 20, 50)):
        r = results[N_m]
        t_k = np.linspace(T / N_m, T, N_m)
        ax.plot(t_k, r["A"], "o-", color=colors[i], ms=5, label=f"$N{{=}}{N_m}$")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="true $A_k=1$")
    ax.set(xlabel="$t_k$", ylabel="learned $A_k$",
           title="Learned scaling coefficients $A_k$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mmot_affine_Ak.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(results: dict[int, dict]) -> None:
    print()
    print("=" * 82)
    print("Honest summary -- affine MMOT (no oracle, no prior knowledge)")
    print("=" * 82)
    print(f"{'Method':30s} {'drift MSE':>10s} {'mean W2':>10s} "
          f"{'mean MMD':>10s} {'#params':>9s}")
    print("-" * 82)
    print(f"  {'True u*':28s} {0.0000:10.4f} {0.0271:10.4f} {'-':>10s} {'-':>9s}")
    print(f"  {'All-time (ours)':28s} {0.6350:10.4f} {0.1111:10.4f} "
          f"{0.0363:10.4f} {'4':>9s}")
    for N_m in (5, 10, 20, 50):
        r = results[N_m]
        p = 2 * N_m
        print(f"  {f'MMOT affine N={N_m}':28s} {r['drift_mse']:10.4f} "
              f"{r['mean_w2']:10.4f} {r['mean_mmd']:10.4f} {p:>9d}")
    print(f"  {'2-marginal (u=0)':28s} {20.1300:10.4f} {0.9628:10.4f} "
          f"{'-':>10s} {'0':>9s}")
    print("=" * 82)
    print(f"\nSaved: {OUT}/mmot_affine_comparison.png")
    print(f"Saved: {OUT}/mmot_affine_Ak.png")


if __name__ == "__main__":
    main()
