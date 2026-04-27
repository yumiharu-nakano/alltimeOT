#!/usr/bin/env python3
"""Flow matching baseline on the roundtrip flow.

Vanilla conditional flow matching (Lipman et al., 2023) with a
straight-line interpolant between samples of mu_0 = mu_1 = N(0, 1).
Since the endpoint distributions are identical, the conditional FM
target E[x_1 - x_0 | x_t] is the zero field up to finite-sample noise,
so the method reproduces the two-marginal OT solution u ~= 0.

This script is used as a contrast to the all-time OT estimator in the
paper -- FM only sees mu_0 and mu_1, not the interior marginals mu_t,
and therefore cannot recover the non-trivial u* = 2 pi cos(pi t).
"""

from __future__ import annotations

import numpy as np

from alltime_ot import mmd_gauss

RNG_SEED_BASE = 0
N_PAIRS = 200
N_ENS = 30
N_PARTICLES = 5000
N_STEPS = 1000


# ---- Ground truth (same roundtrip flow as exp2_roundtrip.py) ---------
def u_star(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.cos(np.pi * t)


def mu_t_mean(t: np.ndarray | float) -> np.ndarray:
    return 2.0 * np.sin(np.pi * t)


# ---- Linear model ---------------------------------------------------
# u_theta(t, x) = w0 + w1 t + w2 t^2 + w3 x   (same class as the main exp)
def features(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(t), t, t ** 2, x], axis=-1)


def run_fm(n_pairs: int = N_PAIRS, n_ens: int = N_ENS, seed: int = 0) -> np.ndarray:
    """Conditional FM with straight-line interpolant; averaged over n_ens trials."""
    rng = np.random.default_rng(seed)
    W = np.zeros(4)
    for _ in range(n_ens):
        x0 = rng.standard_normal(n_pairs)
        x1 = rng.standard_normal(n_pairs)
        ts = rng.uniform(0.0, 1.0, size=n_pairs)
        xt = (1.0 - ts) * x0 + ts * x1
        target = x1 - x0            # conditional FM velocity
        Phi = features(ts, xt)
        w, *_ = np.linalg.lstsq(Phi, target, rcond=None)
        W += w
    return W / n_ens


def drift_mse(W: np.ndarray, n_t: int = 101, n_x: int = 201) -> float:
    t_grid = np.linspace(0.0, 1.0, n_t)
    x_grid = np.linspace(-4.0, 4.0, n_x)
    T, X = np.meshgrid(t_grid, x_grid, indexing="ij")
    Phi = features(T.ravel(), X.ravel())
    u_hat = (Phi @ W).reshape(T.shape)
    return float(np.mean((u_hat - u_star(T, X)) ** 2))


def simulate_marginals(
    W: np.ndarray,
    n_particles: int = N_PARTICLES,
    n_steps: int = N_STEPS,
    seed: int = 1,
) -> tuple[dict[float, float], dict[float, float]]:
    """Euler-simulate the learned ODE; return (W_2, MMD) dicts per time."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_particles)     # x0 ~ mu_0 = N(0, 1)
    dt = 1.0 / n_steps
    eval_ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    snaps: dict[float, np.ndarray] = {0.0: x.copy()}
    next_idx = 1
    for step in range(1, n_steps + 1):
        t = step * dt
        Phi = features(np.full_like(x, t - 0.5 * dt), x)
        x = x + dt * (Phi @ W)
        if next_idx < len(eval_ts) and abs(t - eval_ts[next_idx]) < 0.5 * dt:
            snaps[float(eval_ts[next_idx])] = x.copy()
            next_idx += 1
    w2s: dict[float, float] = {}
    mmds: dict[float, float] = {}
    rng_ref = np.random.default_rng(seed + 1)
    for tv in eval_ts:
        xs = np.sort(snaps[float(tv)])
        ref = np.sort(rng.normal(loc=mu_t_mean(tv), scale=1.0, size=n_particles))
        w2s[float(tv)] = float(np.sqrt(np.mean((xs - ref) ** 2)))
        # MMD with the same Gaussian kernel (h=1) as the paper's RKHS loss.
        ref_mmd = mu_t_mean(tv) + rng_ref.standard_normal(n_particles)
        mmds[float(tv)] = mmd_gauss(snaps[float(tv)], ref_mmd, h=1.0)
    return w2s, mmds


def main() -> None:
    W = run_fm(seed=RNG_SEED_BASE)
    print(f"Learned weights (w0, w1, w2, w3) = {np.round(W, 4).tolist()}")
    mse = drift_mse(W)
    print(f"Drift grid MSE = {mse:.4f}")
    w2s, mmds = simulate_marginals(W)
    print("t       W2        MMD")
    for t_val in sorted(w2s):
        print(f"  t={t_val:.2f}: W2 = {w2s[t_val]:.4f},  MMD = {mmds[t_val]:.4f}")
    mean_w2 = float(np.mean(list(w2s.values())))
    mean_mmd = float(np.mean(list(mmds.values())))
    print(f"Mean W2 = {mean_w2:.4f},  mean MMD = {mean_mmd:.4f}")


if __name__ == "__main__":
    main()
