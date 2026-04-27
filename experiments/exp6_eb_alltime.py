#!/usr/bin/env python3
"""Experiment 6 (real data): all-time OT on EB scRNA-seq.

Trains an MLP drift on the embryoid body PCA representation
(PCA-30 of HVG-restricted log-normalised counts).  Held-out day 15
is used for evaluation; training uses days {3, 9, 21, 27}.

Outputs ``output/exp6/alltime_predictions.npz`` containing forward-
simulated samples at the held-out time, plus a training-curve PNG.
The baseline comparisons (Waddington-OT, zero drift) live in
:mod:`exp6_eb_baselines`; the synthesis figure / table is produced
by :mod:`exp6_eb_evaluate`.
"""

from __future__ import annotations

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from alltime_ot.rkhs import rkhs_all_time_loss_from_drift

OUT = os.environ.get("ALLTIME_OT_OUT", "output/exp6")
os.makedirs(OUT, exist_ok=True)

# ----- Problem ---------------------------------------------------------
# Time points: days 3, 9, 15, 21, 27 -> t = (day-3)/24 \in {0, 0.25, 0.5, 0.75, 1}
HELD_OUT_DAY = 15.0
HELD_OUT_T = 0.5
TRAIN_DAYS = [3.0, 9.0, 21.0, 27.0]
TRAIN_TS = [0.0, 0.25, 0.75, 1.0]
T_END = 1.0

# Hyperparameters
H_KERNEL = 8.0            # half of median pairwise distance
LAM = 10000.0             # penalty weight (high to enforce marginal constraint)
N_PER_TIME = 80           # mini-batch size per time point
N0_BATCH = 160            # mu_0 batch size
K_ENS = 25                # number of pre-cached batches
MLP_HIDDEN = 96
N_ITER = 8_000
LR_HIGH = 5e-4
LR_LOW = 5e-6
SEED = 2026
N_SIM = 1500              # forward simulation particles (<= subsample size)


def load_data():
    payload = np.load(os.path.join(OUT, "eb_pca.npz"))
    X = payload["X"]
    days = payload["days"]
    return X, days


def build_pools(X, days):
    """Return dict day -> (n_day, d) array."""
    pools = {}
    for day in [3.0, 9.0, 15.0, 21.0, 27.0]:
        pools[day] = X[days == day].copy()
    return pools


def make_batch_provider(pools, rng_seed):
    rng = np.random.default_rng(rng_seed)
    M = len(TRAIN_DAYS)
    N = N_PER_TIME
    # Pre-cache K_ENS batches
    batches = []
    for k in range(K_ENS):
        Xb = np.zeros((M, N, X_dim))
        for i, day in enumerate(TRAIN_DAYS):
            pool = pools[day]
            idx = rng.choice(len(pool), N, replace=False)
            Xb[i] = pool[idx]
        # X0 from day 3 (= initial marginal)
        idx0 = rng.choice(len(pools[3.0]), N0_BATCH, replace=False)
        X0b = pools[3.0][idx0]
        batches.append((
            torch.tensor(TRAIN_TS, dtype=torch.float64),
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

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([t.unsqueeze(-1), x], dim=-1)
        return self.net(z)


def simulate(model, x0, n_step=200):
    """Euler ODE simulation forward from t=0, returning trajectory dict."""
    dt = T_END / n_step
    x = torch.from_numpy(x0).double()
    eval_t = [0.0, 0.25, 0.5, 0.75, 1.0]
    snaps = {0.0: x.clone()}
    tc = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(n_step):
            t_in = torch.full((x.shape[0],), tc, dtype=torch.float64)
            u = model(t_in, x)
            x = x + u * dt
            tc += dt
            for tv in eval_t:
                if tv not in snaps and abs(tc - tv) < dt / 2:
                    snaps[tv] = x.clone()
    if T_END not in snaps:
        snaps[T_END] = x.clone()
    return {tv: v.numpy() for tv, v in snaps.items()}


def main() -> None:
    print("=" * 72)
    print("Exp 6: all-time OT on EB scRNA-seq (held-out day 15)")
    print("=" * 72)
    X, days = load_data()
    global X_dim
    X_dim = X.shape[1]
    print(f"  d={X_dim}, total cells={X.shape[0]}")
    print(f"  train days: {TRAIN_DAYS}, held-out: {HELD_OUT_DAY}")
    print(f"  hyperparams: h={H_KERNEL}, lam={LAM}, N_per_t={N_PER_TIME},"
          f" hidden={MLP_HIDDEN}, iters={N_ITER}")
    sys.stdout.flush()

    pools = build_pools(X, days)
    for day, pool in pools.items():
        print(f"  pool day {day}: {pool.shape[0]} cells")

    print("\nPre-caching batches ...")
    sys.stdout.flush()
    batches = make_batch_provider(pools, SEED)

    torch.manual_seed(SEED)
    model = MLP(d_in=1 + X_dim, d_out=X_dim, hidden=MLP_HIDDEN)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MLP params: {n_params}")

    optim = torch.optim.Adam(model.parameters(), lr=LR_HIGH)

    def cosine_lr(it: int) -> float:
        frac = it / max(N_ITER - 1, 1)
        return LR_LOW + 0.5 * (LR_HIGH - LR_LOW) * (1.0 + np.cos(np.pi * frac))

    M_b = len(TRAIN_DAYS)
    N_b = N_PER_TIME
    losses = []
    print(f"\nTraining for {N_ITER} iterations ...")
    sys.stdout.flush()
    tic = time.time()
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
        losses.append(loss.item())
        if (it + 1) % 500 == 0 or it == 0:
            print(f"  iter {it+1:5d}/{N_ITER}  loss={loss.item():.4f}"
                  f"  lr={cosine_lr(it):.2e}")
            sys.stdout.flush()
    elapsed = time.time() - tic
    print(f"\nDone in {elapsed:.1f}s")

    print("\nSimulating forward from day 3 ...")
    rng_sim = np.random.default_rng(SEED + 1)
    pool0 = pools[3.0]
    init_idx = rng_sim.choice(len(pool0), N_SIM, replace=False)
    x0 = pool0[init_idx]
    snaps = simulate(model, x0, n_step=200)
    for tv, arr in snaps.items():
        print(f"  t={tv:.2f}: {arr.shape}")

    # Save predictions and training curve
    np.savez_compressed(
        os.path.join(OUT, "alltime_predictions.npz"),
        held_out_day=HELD_OUT_DAY,
        held_out_t=HELD_OUT_T,
        sim_t=np.array(list(snaps.keys())),
        **{f"sim_t_{tv:.2f}": v for tv, v in snaps.items()},
        x0=x0,
        n_params=n_params,
        training_time_s=elapsed,
    )
    with open(os.path.join(OUT, "alltime_meta.json"), "w") as f:
        json.dump({
            "method": "all-time MLP",
            "n_params": int(n_params),
            "iters": N_ITER,
            "h": H_KERNEL, "lam": LAM,
            "M": M_b, "N": N_b, "K_ens": K_ENS,
            "training_time_s": elapsed,
            "final_loss": float(losses[-1]),
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(losses, lw=0.6, alpha=0.7)
    ax.set(xlabel="iteration", ylabel="loss",
           title="Exp 6: all-time training loss (EB scRNA-seq, d=30)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = os.path.join(OUT, "alltime_training_curve.png")
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {curve_path}")
    print(f"Saved: {os.path.join(OUT, 'alltime_predictions.npz')}")


if __name__ == "__main__":
    main()
