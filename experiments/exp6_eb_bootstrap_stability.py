#!/usr/bin/env python3
"""Exp 6 diagnostic: bootstrap stability of the inferred velocity field
on the EB scRNA-seq benchmark.

Mirrors the Sec 4.3 ``drift-recovery stability'' diagnostic to real
data, where no ground-truth drift is available.  We train both methods
on K independent 80%-subsamples of the EB data and evaluate the
inferred velocity at a fixed set of query points (t_q, x_q).  The
across-bootstrap variance of these velocities at each query is the
stability metric: lower is more stable.

WOT velocity at (t_q, x_q):
    - find the bracketing pair (t_i, t_{i+1}) with t_i <= t_q < t_{i+1};
    - find x_nn = nearest neighbour of x_q in the snapshot-i sample;
    - barycentric target T(x_nn) under the Sinkhorn(i, i+1) coupling;
    - return  (T(x_nn) - x_nn) / (t_{i+1} - t_i).

All-time velocity at (t_q, x_q):
    - direct MLP evaluation.

Both methods use the same query set and the same bootstrap seeds, so
the variance comparison is paired.  We report the mean, median, and
IQR of the per-query squared variance ``sum_d Var_k u^d(t_q, x_q)''.

Outputs ``output/exp6/bootstrap_stability.json`` and a summary figure.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from alltime_ot.rkhs import rkhs_all_time_loss_from_drift


OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp6")
os.makedirs(OUT, exist_ok=True)

# ----- Problem ---------------------------------------------------------
DAYS = [3.0, 9.0, 15.0, 21.0, 27.0]
TS = [0.0, 0.25, 0.5, 0.75, 1.0]
T_END = 1.0

# Hyperparameters
H_KERNEL = 8.0
LAM = 10000.0
N_PER_TIME = 80
N0_BATCH = 160
K_ENS = 25
MLP_HIDDEN = 96
N_ITER = 8_000
LR_HIGH = 5e-4
LR_LOW = 5e-6
EPS_FACTOR = 0.05  # WOT entropic regularization

# Bootstrap & query
K_BOOT = 5
BOOT_FRAC = 0.80
N_QUERY = 1000
QUERY_T_LOW = 0.05
QUERY_T_HIGH = 0.95
BASE_SEED = 2026


def load_data():
    payload = np.load(os.path.join(OUT, "eb_pca.npz"))
    return payload["X"], payload["days"]


def make_query_set(X_all, days, *, seed=2027):
    """Fixed query set: random (t_q, x_q) with x_q sampled from the full data."""
    rng = np.random.default_rng(seed)
    t_q = rng.uniform(QUERY_T_LOW, QUERY_T_HIGH, size=N_QUERY)
    idx = rng.choice(len(X_all), N_QUERY, replace=False)
    x_q = X_all[idx]
    return t_q, x_q


def bootstrap_pools(X_all, days, *, frac, seed):
    """Subsample `frac` cells per snapshot."""
    rng = np.random.default_rng(seed)
    pools = {}
    for day in DAYS:
        full = X_all[days == day]
        n = max(2, int(round(frac * len(full))))
        idx = rng.choice(len(full), n, replace=False)
        pools[day] = full[idx]
    return pools


def make_batch_provider(pools, rng_seed, X_dim):
    rng = np.random.default_rng(rng_seed)
    M = len(DAYS)
    N = N_PER_TIME
    batches = []
    for k in range(K_ENS):
        Xb = np.zeros((M, N, X_dim))
        for i, day in enumerate(DAYS):
            pool = pools[day]
            idx = rng.choice(len(pool), N, replace=(len(pool) < N))
            Xb[i] = pool[idx]
        pool0 = pools[DAYS[0]]
        idx0 = rng.choice(len(pool0), N0_BATCH, replace=(len(pool0) < N0_BATCH))
        X0b = pool0[idx0]
        batches.append((
            torch.tensor(TS, dtype=torch.float64),
            torch.from_numpy(Xb).double(),
            torch.from_numpy(X0b).double(),
        ))
    return batches


class MLP(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, d_out),
        )
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.zeros_(layer.bias)
                torch.nn.init.normal_(layer.weight, std=0.05)
        self.double()

    def forward(self, t, x):
        z = torch.cat([t.unsqueeze(-1), x], dim=-1)
        return self.net(z)


def train_alltime(pools, X_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(d_in=1 + X_dim, d_out=X_dim, hidden=MLP_HIDDEN)
    optim = torch.optim.Adam(model.parameters(), lr=LR_HIGH)
    batches = make_batch_provider(pools, seed, X_dim)

    def cosine_lr(it: int) -> float:
        frac = it / max(N_ITER - 1, 1)
        return LR_LOW + 0.5 * (LR_HIGH - LR_LOW) * (1.0 + np.cos(np.pi * frac))

    M_b = len(DAYS)
    N_b = N_PER_TIME
    for it in range(N_ITER):
        for g in optim.param_groups:
            g["lr"] = cosine_lr(it)
        t_s, X_b, X0_b = batches[it % K_ENS]
        tg = t_s.repeat_interleave(N_b)
        xg = X_b.reshape(M_b * N_b, X_dim)
        uf = model(tg, xg)
        loss = rkhs_all_time_loss_from_drift(
            uf, tg, xg, X0_b,
            M=M_b, N=N_b, t_s=t_s,
            lam=LAM, h=H_KERNEL, T=T_END,
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
    return model


def eval_alltime(model, t_q, x_q):
    model.eval()
    tq_t = torch.tensor(t_q, dtype=torch.float64)
    xq_t = torch.tensor(x_q, dtype=torch.float64)
    with torch.no_grad():
        u = model(tq_t, xq_t).numpy()
    return u


def _logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn(a, b, M, reg, num_iter=2000, tol=1e-8):
    log_a = np.log(np.maximum(a, 1e-300))
    log_b = np.log(np.maximum(b, 1e-300))
    log_K = -M / reg
    log_u = np.zeros_like(log_a)
    log_v = np.zeros_like(log_b)
    for it in range(num_iter):
        log_u_new = log_a - _logsumexp(log_K + log_v[None, :], axis=1)
        log_v_new = log_b - _logsumexp(log_K + log_u_new[:, None], axis=0)
        if (np.max(np.abs(log_u_new - log_u)) < tol
                and np.max(np.abs(log_v_new - log_v)) < tol):
            log_u, log_v = log_u_new, log_v_new
            break
        log_u, log_v = log_u_new, log_v_new
    return np.exp(log_K + log_u[:, None] + log_v[None, :])


def train_wot(pools):
    """Compute Sinkhorn couplings + barycentric maps between adjacent snapshots.

    Returns a dict: (t_i, t_{i+1}) -> (X_i, X_{i+1}, T_i_to_i+1)
    where T(x) for x in X_i is the barycentric target in X_{i+1}.
    """
    couplings = {}
    for i in range(len(DAYS) - 1):
        X_i = pools[DAYS[i]]
        X_j = pools[DAYS[i + 1]]
        M_cost = cdist(X_i, X_j, "sqeuclidean")
        eps = EPS_FACTOR * float(np.median(M_cost))
        a = np.ones(len(X_i)) / len(X_i)
        b = np.ones(len(X_j)) / len(X_j)
        pi = sinkhorn(a, b, M_cost, reg=eps, num_iter=2000, tol=1e-8)
        row_sums = pi.sum(1, keepdims=True)
        T_ij = (pi @ X_j) / np.maximum(row_sums, 1e-12)
        couplings[i] = (X_i, X_j, T_ij)
    return couplings


def eval_wot(couplings, t_q, x_q):
    """Evaluate WOT velocity at query (t_q, x_q) via nearest-neighbour."""
    Q = len(t_q)
    D = x_q.shape[1]
    out = np.zeros((Q, D))
    for q in range(Q):
        # Find bracketing interval (i, i+1)
        i = None
        for k in range(len(TS) - 1):
            if TS[k] <= t_q[q] < TS[k + 1]:
                i = k
                break
        if i is None:
            i = len(TS) - 2  # tail
        X_i, X_j, T_ij = couplings[i]
        dt = TS[i + 1] - TS[i]
        # Nearest neighbour in X_i
        dists = np.sum((X_i - x_q[q]) ** 2, axis=1)
        nn = int(np.argmin(dists))
        out[q] = (T_ij[nn] - X_i[nn]) / dt
    return out


def main() -> None:
    print("=" * 72)
    print("Exp 6 bootstrap stability diagnostic")
    print(f"  K_BOOT = {K_BOOT}, BOOT_FRAC = {BOOT_FRAC}, N_QUERY = {N_QUERY}")
    print("=" * 72)
    sys.stdout.flush()

    X_all, days = load_data()
    X_dim = X_all.shape[1]
    print(f"  d = {X_dim}, total cells = {X_all.shape[0]}")

    t_q, x_q = make_query_set(X_all, days, seed=BASE_SEED + 100)
    print(f"  query set: {t_q.shape}, t_q in [{t_q.min():.3f}, {t_q.max():.3f}]")
    sys.stdout.flush()

    u_alltime_runs = np.zeros((K_BOOT, N_QUERY, X_dim))
    u_wot_runs = np.zeros((K_BOOT, N_QUERY, X_dim))

    for k in range(K_BOOT):
        seed = BASE_SEED + k
        print(f"\n[bootstrap {k+1}/{K_BOOT}, seed={seed}]")
        sys.stdout.flush()
        pools = bootstrap_pools(X_all, days, frac=BOOT_FRAC, seed=seed)
        print(f"  pool sizes: " + ", ".join(f"d={d}: {len(p)}" for d, p in pools.items()))
        sys.stdout.flush()

        print("  training all-time ...")
        sys.stdout.flush()
        t0 = time.time()
        model = train_alltime(pools, X_dim, seed)
        print(f"    done in {time.time()-t0:.1f}s")
        u_alltime_runs[k] = eval_alltime(model, t_q, x_q)
        print(f"    eval norm: mean |u| = {np.linalg.norm(u_alltime_runs[k], axis=1).mean():.3f}")

        print("  training WOT ...")
        sys.stdout.flush()
        t0 = time.time()
        couplings = train_wot(pools)
        print(f"    done in {time.time()-t0:.1f}s")
        u_wot_runs[k] = eval_wot(couplings, t_q, x_q)
        print(f"    eval norm: mean |u| = {np.linalg.norm(u_wot_runs[k], axis=1).mean():.3f}")
        sys.stdout.flush()

    # Per-query variance: sum_d Var_k u_k^d, then sqrt to get scale
    var_alltime = np.var(u_alltime_runs, axis=0, ddof=1)  # (Q, D)
    var_wot = np.var(u_wot_runs, axis=0, ddof=1)
    per_query_alltime = var_alltime.sum(axis=1)  # (Q,)
    per_query_wot = var_wot.sum(axis=1)
    # Take sqrt to get a stddev-like scale
    std_alltime = np.sqrt(per_query_alltime)
    std_wot = np.sqrt(per_query_wot)

    summary = {
        "K_BOOT": K_BOOT,
        "BOOT_FRAC": BOOT_FRAC,
        "N_QUERY": N_QUERY,
        "alltime": {
            "per_query_std_mean": float(np.mean(std_alltime)),
            "per_query_std_median": float(np.median(std_alltime)),
            "per_query_std_q25": float(np.quantile(std_alltime, 0.25)),
            "per_query_std_q75": float(np.quantile(std_alltime, 0.75)),
            "mean_norm_u": float(np.linalg.norm(u_alltime_runs, axis=2).mean()),
        },
        "wot": {
            "per_query_std_mean": float(np.mean(std_wot)),
            "per_query_std_median": float(np.median(std_wot)),
            "per_query_std_q25": float(np.quantile(std_wot, 0.25)),
            "per_query_std_q75": float(np.quantile(std_wot, 0.75)),
            "mean_norm_u": float(np.linalg.norm(u_wot_runs, axis=2).mean()),
        },
    }

    print("\n" + "=" * 72)
    print("Summary:")
    for method, stats in [("All-time", summary["alltime"]), ("WOT", summary["wot"])]:
        print(f"  {method}:")
        print(f"    per-query std: mean={stats['per_query_std_mean']:.4f}, "
              f"median={stats['per_query_std_median']:.4f}, "
              f"IQR=[{stats['per_query_std_q25']:.4f}, {stats['per_query_std_q75']:.4f}]")
        print(f"    mean |u| across queries and bootstraps: {stats['mean_norm_u']:.4f}")
    ratio_mean = summary["wot"]["per_query_std_mean"] / summary["alltime"]["per_query_std_mean"]
    ratio_median = summary["wot"]["per_query_std_median"] / summary["alltime"]["per_query_std_median"]
    print(f"  WOT/All-time mean ratio: {ratio_mean:.2f}x  (median ratio: {ratio_median:.2f}x)")
    summary["wot_over_alltime_ratio_mean"] = float(ratio_mean)
    summary["wot_over_alltime_ratio_median"] = float(ratio_median)

    with open(os.path.join(OUT, "bootstrap_stability.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-query arrays for downstream analysis
    np.savez_compressed(
        os.path.join(OUT, "bootstrap_stability_raw.npz"),
        t_q=t_q, x_q=x_q,
        u_alltime_runs=u_alltime_runs,
        u_wot_runs=u_wot_runs,
        std_alltime=std_alltime,
        std_wot=std_wot,
    )

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    bins = np.linspace(0, max(std_alltime.max(), std_wot.max()), 40)
    ax.hist(std_alltime, bins=bins, alpha=0.6, color="C1", label="All-time (ours)")
    ax.hist(std_wot, bins=bins, alpha=0.6, color="C0", label="WOT (Sinkhorn)")
    ax.set(xlabel=r"per-query velocity std (across bootstraps)",
           ylabel="count",
           title=r"(a) Distribution of $\sqrt{\sum_d \mathrm{Var}_k\,\hat u_k^d(t_q, x_q)}$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.boxplot([std_alltime, std_wot], labels=["All-time", "WOT"], showfliers=False)
    ax.set(ylabel="per-query velocity std",
           title=f"(b) Box-plot of stability (lower is more stable)\n"
                 f"WOT/All-time ratio: mean={ratio_mean:.2f}x, median={ratio_median:.2f}x")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Exp 6 bootstrap stability: K={K_BOOT}, frac={BOOT_FRAC}, "
                 f"N_query={N_QUERY}", y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(OUT, "bootstrap_stability.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {fig_path}")
    print(f"Saved JSON:   {os.path.join(OUT, 'bootstrap_stability.json')}")


if __name__ == "__main__":
    main()
