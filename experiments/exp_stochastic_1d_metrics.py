#!/usr/bin/env python3
"""Marginal-consistency metrics for the 1d stochastic Gaussian.

Loads the learned affine drift weights produced by
:mod:`exp_stochastic_1d_train` (saved as ``weights.json``) and
evaluates the Wasserstein-2 and Gaussian MMD between the
Euler--Maruyama-simulated marginals and the prescribed
``mu_t = N(-1 + 2 t, 1)``.  Run the training script first.

Outputs a LaTeX-ready table to stdout.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from alltime_ot.simulate import sorted_w2, mmd_gauss

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp_stochastic")
os.makedirs(OUT, exist_ok=True)


# ----- Problem ---------------------------------------------------------
T = 1.0
SIGMA = 1.0
M_OF_T = lambda t: -1.0 + 2.0 * t          # true mean of mu_t
U_TRUE = lambda t, x: -x / 2.0 + 1.5 + t   # true optimal drift
U_ZERO = lambda t, x: np.zeros_like(x)


def load_weights() -> tuple[float, float, float]:
    """Load (w0, w1, w2) from the training script's JSON output."""
    weights_path = os.path.join(OUT, "weights.json")
    if not os.path.exists(weights_path):
        sys.exit(
            f"Missing {weights_path}. Run experiments/exp_stochastic_1d_train.py first."
        )
    with open(weights_path) as f:
        payload = json.load(f)
    w = payload["w_hat"]
    return float(w[0]), float(w[1]), float(w[2])


W_HAT = load_weights()
U_LEARN = lambda t, x: W_HAT[0] + W_HAT[1] * t + W_HAT[2] * x


# ----- Forward simulation ---------------------------------------------
def euler_maruyama(u_func, x0, *, sigma=SIGMA, T=T, n_step=2000, eval_t):
    dt = T / n_step
    sdt = np.sqrt(dt)
    x = np.array(x0, dtype=np.float64, copy=True)
    eval_t = list(eval_t)
    snaps = {}
    if abs(eval_t[0]) < dt / 2:
        snaps[eval_t[0]] = x.copy()
    tc = 0.0
    rng = np.random.default_rng(0)
    for _ in range(n_step):
        eps = rng.standard_normal(x.shape)
        x = x + u_func(tc, x) * dt + sigma * sdt * eps
        tc += dt
        for tv in eval_t:
            if tv not in snaps and abs(tc - tv) < dt / 2:
                snaps[tv] = x.copy()
    return snaps


# ----- Main -----------------------------------------------------------
def main():
    eval_t = [0.0, 0.25, 0.5, 0.75, 1.0]
    N_PART = 20_000

    rng0 = np.random.default_rng(42)
    x0 = M_OF_T(0.0) + rng0.standard_normal(N_PART)

    results = {}
    for name, u in [("true", U_TRUE), ("learned", U_LEARN), ("zero", U_ZERO)]:
        snaps = euler_maruyama(u, x0, eval_t=eval_t)
        w2s, mmds = [], []
        rng_ref = np.random.default_rng(7)
        for tv in eval_t:
            ref_samples = M_OF_T(tv) + rng_ref.standard_normal(N_PART)
            w2s.append(sorted_w2(snaps[tv], ref_samples))
            mmds.append(mmd_gauss(snaps[tv], ref_samples, h=1.0))
        results[name] = {"w2": w2s, "mmd": mmds}
        print(f"{name:>8s}  W2 @ t: "
              + "  ".join(f"{v:.4f}" for v in w2s)
              + f"   mean={np.mean(w2s):.4f}  max={np.max(w2s):.4f}")
        print(f"{name:>8s}  MMD @ t: "
              + "  ".join(f"{v:.4f}" for v in mmds)
              + f"   mean={np.mean(mmds):.4f}  max={np.max(mmds):.4f}")

    print()
    print("LaTeX table rows:")
    for name, label in [("true", "True $u^*$"),
                        ("learned", "Learned $\\hat u$"),
                        ("zero", "Zero drift")]:
        r = results[name]
        print(f"  {label:18s} & {np.mean(r['w2']):.4f} & {np.max(r['w2']):.4f} & "
              f"{np.mean(r['mmd']):.4f} & {np.max(r['mmd']):.4f} \\\\")

    out_path = os.path.join(OUT, "marginal_metrics.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "eval_t": eval_t,
                "N_particles": N_PART,
                "sigma": SIGMA,
                "w_hat": list(W_HAT),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
