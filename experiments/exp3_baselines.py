#!/usr/bin/env python3
"""Exp 3 -- Baseline comparisons on the bimodal merging flow.

Two competitor methods run on the same problem as :mod:`exp4_bimodal`:

  (1) Affine MMOT -- multi-marginal OT with learned maps T_k(x) = A_k x + b_k,
      optimised by ensemble L-BFGS-B against an MMD U-statistic marginal
      loss plus a kinetic cost.  Affine maps cannot push a bimodal onto
      another bimodal except via scaling, so we expect a large
      irreducible error.

  (2) Waddington-OT -- entropic OT (Sinkhorn) between consecutive
      bimodal snapshots, barycentric-projection transport map, drift
      (T(x) - x) / dt per source particle.  The drift is position
      dependent, so we interpolate it to a regular grid for MSE.
"""

from __future__ import annotations

import json
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
from alltime_ot.baselines import (
    extract_affine_drift,
    make_affine_mmot_loss_grad,
    sinkhorn_wot_drift,
)

warnings.filterwarnings("ignore")

# ---- Problem ---------------------------------------------------------
T_END = 1.0

def m1(t: float) -> float:
    return -2.0 + 2.0 * t

def m2(t: float) -> float:
    return 2.0 - 2.0 * t

def u_true(t: float, x: np.ndarray) -> np.ndarray:
    return -2.0 * np.tanh(2.0 * (1.0 - t) * x)

def rho_t(t: float, x: np.ndarray) -> np.ndarray:
    return 0.5 * norm.pdf(x, m1(t), 1.0) + 0.5 * norm.pdf(x, m2(t), 1.0)

def sample_bimodal(t: float, n: int, rng: np.random.Generator) -> np.ndarray:
    pick = rng.random(n) < 0.5
    return np.where(pick, m1(t) + rng.standard_normal(n),
                    m2(t) + rng.standard_normal(n))

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp3")
os.makedirs(OUT, exist_ok=True)

# Evaluation grid (same as exp4_bimodal)
t_grid = np.linspace(0, T_END, 50)
x_grid = np.linspace(-4, 4, 100)
TT, XX = np.meshgrid(t_grid, x_grid, indexing="ij")
U_STAR = -2.0 * np.tanh(2.0 * (1.0 - TT) * XX)

# ---- ODE simulation + W1 ---------------------------------------------
N_SIM = 5000
EVAL_T = [0.0, 0.25, 0.5, 0.75, 1.0]


def simulate(u_func, seed: int = 123) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = sample_bimodal(0.0, N_SIM, rng)
    return euler_simulate(u_func, x0, T=T_END, n_step=1000, eval_t=EVAL_T)


# ---- Affine MMOT ------------------------------------------------------
def run_affine_mmot(N_marg: int) -> dict:
    dt = T_END / N_marg
    hp_grid = [
        (1e3, 1.0, 200, 8),
        (1e4, 1.0, 200, 8),
        (1e5, 1.0, 200, 8),
        (1e4, 0.5, 200, 8),
        (1e4, 2.0, 200, 8),
    ]

    best_mse, best_theta, best_hp = np.inf, None, None

    for lam, alpha_k, Mb, Ke in hp_grid:
        lg, _, _ = make_affine_mmot_loss_grad(
            sample_bimodal, N_marg=N_marg, M_batch=Mb,
            lam=lam, alpha_k=alpha_k, K_ens=Ke, T=T_END, seed=42,
        )
        inits = [np.concatenate([np.ones(N_marg), np.zeros(N_marg)])]
        for s in (1, 2):
            rs = np.random.default_rng(s)
            inits.append(np.concatenate([
                1.0 + 0.3 * rs.standard_normal(N_marg),
                0.5 * rs.standard_normal(N_marg),
            ]))

        for th0 in inits:
            try:
                res = minimize(
                    lg, th0, jac=True, method="L-BFGS-B",
                    options={"maxiter": 500, "ftol": 1e-12},
                )
                A = np.concatenate([[1.0], res.x[:N_marg]])
                b = np.concatenate([[0.0], res.x[N_marg:]])
                slopes = (A[1:] / A[:-1] - 1.0) / dt
                intercepts = (b[1:] * A[:-1] - b[:-1] * A[1:]) / (A[:-1] * dt)
                u_grid = np.zeros_like(TT)
                for i_t, t in enumerate(t_grid):
                    k = min(max(int(t / dt), 0), N_marg - 1)
                    u_grid[i_t] = slopes[k] * x_grid + intercepts[k]
                mse = float(((u_grid - U_STAR) ** 2).mean())
                if mse < best_mse:
                    best_mse = mse
                    best_theta = res.x
                    best_hp = (lam, alpha_k)
            except Exception:
                pass

    A = np.concatenate([[1.0], best_theta[:N_marg]])
    b = np.concatenate([[0.0], best_theta[N_marg:]])
    slopes = (A[1:] / A[:-1] - 1.0) / dt
    intercepts = (b[1:] * A[:-1] - b[:-1] * A[1:]) / (A[:-1] * dt)

    def u_func(t: float, x: np.ndarray) -> np.ndarray:
        k = min(max(int(t / dt), 0), N_marg - 1)
        return slopes[k] * x + intercepts[k]

    return {
        "mse": best_mse,
        "theta": best_theta,
        "A": A[1:], "b": b[1:],
        "u_func": u_func,
        "hp": best_hp,
    }


# ---- WOT -------------------------------------------------------------
def run_wot(N_marg: int, N_sample: int = 200, eps_factor: float = 1.0, seed: int = 42) -> dict:
    out = sinkhorn_wot_drift(
        sample_bimodal, M_wot=N_marg, N_sample=N_sample,
        eps_factor=eps_factor, T=T_END, seed=seed,
        position_dependent=True,
    )
    dt = out["dt"]
    interpolators = out["interpolators"]
    u_grid = np.zeros_like(TT)
    for i_t, t in enumerate(t_grid):
        k = min(max(int(t / dt), 0), N_marg - 1)
        u_grid[i_t] = interpolators[k][1](x_grid)
    mse = float(((u_grid - U_STAR) ** 2).mean())
    return {
        "mse": mse,
        "u_func": out["u_func"],
        "drift_interp": interpolators,
        "hp": (N_sample, eps_factor),
    }


# ---- Reference numbers from exp3_bimodal.py (all-time method) --------
# Placeholders; overwritten automatically when exp3_bimodal.py is rerun
# and the values are copy-pasted here.  Fresh 2026 run values below.
ALLTIME = {
    "mse_tanh": 0.187,
    "mean_w2_tanh": 0.074,
    "mean_mmd_tanh": 0.019,
    "mse_bilin": 5.264,
    "mean_w2_bilin": 0.242,
    "mean_mmd_bilin": 0.092,
    "mse_affine": 2.0047,
    "mean_w2_affine": 0.2250,
    "mean_mmd_affine": 0.0688,
    "mse_zero": 2.5443,
    "mean_w2_zero": 0.7559,
    "mean_mmd_zero": 0.3212,
}


def evaluate(u_func, snaps_true: dict[float, np.ndarray]):
    """Return (snaps, w2_list, mmd_list) against the true bimodal marginals."""
    snaps = simulate(u_func)
    w2 = [sorted_w2(snaps[tv], snaps_true[tv]) for tv in EVAL_T]
    mmd = [mmd_gauss(snaps[tv], snaps_true[tv], h=1.0) for tv in EVAL_T]
    return snaps, w2, mmd


def main() -> None:
    print("=" * 72)
    print("Exp 3 -- Baseline comparisons on bimodal merging")
    print("=" * 72)

    snaps_true = simulate(u_true)

    mmot_results: dict[int, dict] = {}
    for N_m in (3, 5, 10, 20):
        tic = time.time()
        r = run_affine_mmot(N_m)
        snaps, w2, mmd = evaluate(r["u_func"], snaps_true)
        r.update({
            "w2": w2, "mean_w2": float(np.mean(w2)),
            "mmd": mmd, "mean_mmd": float(np.mean(mmd)),
            "snaps": snaps, "time": time.time() - tic,
        })
        mmot_results[N_m] = r
        print(f"MMOT N={N_m:3d}  MSE={r['mse']:8.4f}  "
              f"W2={r['mean_w2']:.4f}  MMD={r['mean_mmd']:.4f}  "
              f"A_mean={r['A'].mean():.3f}  "
              f"hp=(lam={r['hp'][0]:.0e},a={r['hp'][1]})  ({r['time']:.1f}s)")

    wot_results: dict[int, dict] = {}
    for N_m in (5, 10, 20, 50):
        tic = time.time()
        r = run_wot(N_m, N_sample=200)
        snaps, w2, mmd = evaluate(r["u_func"], snaps_true)
        r.update({
            "w2": w2, "mean_w2": float(np.mean(w2)),
            "mmd": mmd, "mean_mmd": float(np.mean(mmd)),
            "snaps": snaps, "time": time.time() - tic,
        })
        wot_results[N_m] = r
        print(f"WOT N={N_m:3d}  MSE={r['mse']:8.4f}  "
              f"W2={r['mean_w2']:.4f}  MMD={r['mean_mmd']:.4f}  "
              f"({r['time']:.1f}s)")

    print_summary(mmot_results, wot_results)
    plot_drifts(mmot_results, wot_results)
    plot_bars(mmot_results, wot_results)
    save_json(mmot_results, wot_results)


def print_summary(mmot: dict, wot: dict) -> None:
    print()
    print("=" * 86)
    print("Summary (bimodal merging, drift MSE on [0,1]x[-4,4], "
          "W_2 / MMD over 5 slices)")
    print("=" * 86)
    print(f"{'Method':32s} {'#params':>10s} {'drift MSE':>12s} "
          f"{'mean W2':>10s} {'mean MMD':>10s}")
    print("-" * 86)
    print(f"  {'True u*':30s} {'-':>10s} {0.0:12.4f} "
          f"{0.0:10.4f} {0.0:10.4f}")
    print(f"  {'All-time tanh (ours, 8p)':30s} {'8':>10s} "
          f"{ALLTIME['mse_tanh']:12.4f} {ALLTIME['mean_w2_tanh']:10.4f} "
          f"{ALLTIME['mean_mmd_tanh']:10.4f}")
    print(f"  {'All-time bilin (ours, 4p)':30s} {'4':>10s} "
          f"{ALLTIME['mse_bilin']:12.4f} {ALLTIME['mean_w2_bilin']:10.4f} "
          f"{ALLTIME['mean_mmd_bilin']:10.4f}")
    for N_m in (3, 5, 10, 20):
        r = mmot[N_m]
        p = 2 * N_m
        print(f"  {f'MMOT affine N={N_m}':30s} {p:>10d} "
              f"{r['mse']:12.4f} {r['mean_w2']:10.4f} {r['mean_mmd']:10.4f}")
    for N_m in (5, 10, 20, 50):
        r = wot[N_m]
        print(f"  {f'WOT N={N_m}, Nsample=200':30s} "
              f"{'non-param':>10s} {r['mse']:12.4f} "
              f"{r['mean_w2']:10.4f} {r['mean_mmd']:10.4f}")
    print(f"  {'Zero drift':30s} {'0':>10s} "
          f"{ALLTIME['mse_zero']:12.4f} {ALLTIME['mean_w2_zero']:10.4f} "
          f"{ALLTIME['mean_mmd_zero']:10.4f}")
    print("=" * 86)


def plot_drifts(mmot: dict, wot: dict) -> None:
    best_mmot_N = min(mmot, key=lambda k: mmot[k]["mse"])
    best_wot_N = min(wot, key=lambda k: wot[k]["mse"])
    print(f"Best MMOT N={best_mmot_N}, best WOT N={best_wot_N}")

    x_pl = np.linspace(-4, 4, 300)
    t_eval_fig = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharey="row", sharex=True)
    r_mmot = mmot[best_mmot_N]
    dt_mmot = T_END / best_mmot_N
    for cc, tv in enumerate(t_eval_fig):
        ax = axes[0, cc]
        ust = -2.0 * np.tanh(2.0 * (1.0 - tv) * x_pl)
        k_idx = min(max(int(tv / dt_mmot), 0), best_mmot_N - 1)
        A_full = np.concatenate([[1.0], r_mmot["A"]])
        b_full = np.concatenate([[0.0], r_mmot["b"]])
        slope = (A_full[k_idx + 1] / A_full[k_idx] - 1.0) / dt_mmot
        icpt = (b_full[k_idx + 1] * A_full[k_idx] - b_full[k_idx] * A_full[k_idx + 1]) \
            / (A_full[k_idx] * dt_mmot)
        u_mmot = slope * x_pl + icpt
        pdf = rho_t(tv, x_pl)
        ax.fill_between(x_pl, -3.5, pdf * 4 - 3.5, alpha=0.15, color="C0")
        ax.plot(x_pl, ust, "k--", lw=1.5, label="$u^*$")
        ax.plot(x_pl, u_mmot, "C3-", lw=2, label=f"MMOT $N{{=}}{best_mmot_N}$")
        ax.set(title=f"$t={tv}$", xlim=(-4, 4), ylim=(-3.5, 3.5))
        if cc == 0:
            ax.set_ylabel("Affine MMOT\n$u(t,x)$", fontsize=9)
        if cc == 4:
            ax.legend(fontsize=8, loc="lower left")

    r_wot = wot[best_wot_N]
    dt_w = T_END / best_wot_N
    for cc, tv in enumerate(t_eval_fig):
        ax = axes[1, cc]
        ust = -2.0 * np.tanh(2.0 * (1.0 - tv) * x_pl)
        k_idx = min(max(int(tv / dt_w), 0), best_wot_N - 1)
        f = r_wot["drift_interp"][k_idx][1]
        u_w = f(x_pl)
        pdf = rho_t(tv, x_pl)
        ax.fill_between(x_pl, -3.5, pdf * 4 - 3.5, alpha=0.15, color="C0")
        ax.plot(x_pl, ust, "k--", lw=1.5, label="$u^*$")
        ax.plot(x_pl, u_w, "C2-", lw=2, label=f"WOT $M{{=}}{best_wot_N}$")
        ax.set(xlim=(-4, 4), ylim=(-3.5, 3.5), xlabel="$x$")
        if cc == 0:
            ax.set_ylabel("WOT\n$u(t,x)$", fontsize=9)
        if cc == 4:
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        r"Exp 3 baselines: affine MMOT (top) and WOT (bottom) vs true "
        r"$u^*=-2\tanh(2(1-t)x)$",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/baselines_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bars(mmot: dict, wot: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    labels = ["All-time\ntanh\n(8p)", "All-time\nbilin\n(4p)"]
    mses = [ALLTIME["mse_tanh"], ALLTIME["mse_bilin"]]
    w2s = [ALLTIME["mean_w2_tanh"], ALLTIME["mean_w2_bilin"]]
    mmds = [ALLTIME["mean_mmd_tanh"], ALLTIME["mean_mmd_bilin"]]
    colors = ["C1", "C0"]
    for N_m in (3, 5, 10, 20):
        labels.append(f"MMOT\naffine\nN={N_m}")
        mses.append(mmot[N_m]["mse"])
        w2s.append(mmot[N_m]["mean_w2"])
        mmds.append(mmot[N_m]["mean_mmd"])
        colors.append("C3")
    for N_m in (5, 10, 20, 50):
        labels.append(f"WOT\nM={N_m}")
        mses.append(wot[N_m]["mse"])
        w2s.append(wot[N_m]["mean_w2"])
        mmds.append(wot[N_m]["mean_mmd"])
        colors.append("C2")
    labels.append("Zero")
    mses.append(ALLTIME["mse_zero"])
    w2s.append(ALLTIME["mean_w2_zero"])
    mmds.append(ALLTIME["mean_mmd_zero"])
    colors.append("C7")

    x_bar = np.arange(len(labels))
    for ax, values, title, ylab in [
        (axes[0], mses, "Drift recovery error", "drift grid MSE"),
        (axes[1], w2s, "Marginal tracking ($W_2$)", "mean $W_2$"),
        (axes[2], mmds, "Marginal tracking (MMD)", "mean $\\mathrm{MMD}$"),
    ]:
        ax.bar(x_bar, values, color=colors, alpha=0.8)
        ax.set_xticks(x_bar)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set(ylabel=ylab, title=title, yscale="log")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp 3 baseline summary: bimodal merging", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/baselines_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT}/baselines_drift.png")
    print(f"Saved: {OUT}/baselines_summary.png")


def save_json(mmot: dict, wot: dict) -> None:
    payload = {
        "mmot": {
            str(k): {
                "mse": float(v["mse"]),
                "mean_w2": float(v["mean_w2"]),
                "mean_mmd": float(v["mean_mmd"]),
                "A": v["A"].tolist(),
                "b": v["b"].tolist(),
            }
            for k, v in mmot.items()
        },
        "wot": {
            str(k): {
                "mse": float(v["mse"]),
                "mean_w2": float(v["mean_w2"]),
                "mean_mmd": float(v["mean_mmd"]),
            }
            for k, v in wot.items()
        },
        "alltime": ALLTIME,
    }
    path = f"{OUT}/baselines_numbers.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
