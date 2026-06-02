#!/usr/bin/env python3
"""Experiment 6 (re-design): all-time OT on EB scRNA-seq — interpolation setup.

Same training data and hyperparameters as ``exp6_eb_alltime.py``
(MLP drift trained on days {3, 9, 21, 27}, i.e. t in {0, 0.25, 0.75, 1.0}).
Held-out day 15 is predicted by **forward-simulating from day 9
(t=0.25) to day 15 (t=0.5)**, a one-step prediction from the
nearest training snapshot.  This is the standard trajectory-inference
interpolation setup and matches the natural use case of pairwise OT
baselines (which couple adjacent snapshots).

Outputs ``output/exp6/alltime_predictions_interp.npz``.
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
# Time points: days 3, 9, 15, 21, 27 -> t = (day-3)/24 in {0, 0.25, 0.5, 0.75, 1}
HELD_OUT_DAY = 15.0
HELD_OUT_T = 0.5
TRAIN_DAYS = [3.0, 9.0, 21.0, 27.0]
TRAIN_TS = [0.0, 0.25, 0.75, 1.0]
T_END = 1.0

# Interpolation evaluation: start from day 9 (t=0.25), integrate to day 15 (t=0.5)
PRED_T_START = 0.25
PRED_T_END = 0.5
PRED_DAY_START = 9.0

# Hyperparameters (identical to exp6_eb_alltime.py)
H_KERNEL = 8.0
LAM = 10000.0
N_PER_TIME = 80
N0_BATCH = 160
K_ENS = 25
MLP_HIDDEN = 96
N_ITER = 8_000
LR_HIGH = 5e-4
LR_LOW = 5e-6
SEED = 2026
N_SIM = 1500


def load_data():
    payload = np.load(os.path.join(OUT, "eb_pca.npz"))
    X = payload["X"]
    days = payload["days"]
    return X, days


def build_pools(X, days):
    pools = {}
    for day in [3.0, 9.0, 15.0, 21.0, 27.0]:
        pools[day] = X[days == day].copy()
    return pools


def make_batch_provider(pools, rng_seed):
    rng = np.random.default_rng(rng_seed)
    M = len(TRAIN_DAYS)
    N = N_PER_TIME
    batches = []
    for k in range(K_ENS):
        Xb = np.zeros((M, N, X_dim))
        for i, day in enumerate(TRAIN_DAYS):
            pool = pools[day]
            idx = rng.choice(len(pool), N, replace=False)
            Xb[i] = pool[idx]
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


def simulate_from(model, x0, t_start: float, t_end: float, n_step: int = 100):
    """Euler ODE simulation from t_start to t_end, returning final state."""
    dt = (t_end - t_start) / n_step
    x = torch.from_numpy(x0).double()
    tc = t_start
    model.eval()
    with torch.no_grad():
        for _ in range(n_step):
            t_in = torch.full((x.shape[0],), tc, dtype=torch.float64)
            u = model(t_in, x)
            x = x + u * dt
            tc += dt
    return x.numpy()


def main() -> None:
    print("=" * 72)
    print("Exp 6 (interpolation): all-time OT on EB scRNA-seq")
    print(f"  predict day {HELD_OUT_DAY} (t={HELD_OUT_T}) from day {PRED_DAY_START} (t={PRED_T_START})")
    print("=" * 72)
    X, days = load_data()
    global X_dim
    X_dim = X.shape[1]
    print(f"  d={X_dim}, total cells={X.shape[0]}")
    print(f"  train days: {TRAIN_DAYS}, held-out: {HELD_OUT_DAY}")
    print(f"  prediction start: day {PRED_DAY_START} (t={PRED_T_START})")
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

    # --- Forward simulation from day 9 (t=0.25) to day 15 (t=0.5) ---
    print(f"\nSimulating forward from day {PRED_DAY_START} to day {HELD_OUT_DAY} ...")
    rng_sim = np.random.default_rng(SEED + 1)
    pool_start = pools[PRED_DAY_START]
    init_idx = rng_sim.choice(len(pool_start), N_SIM, replace=False)
    x_start = pool_start[init_idx]
    x_pred = simulate_from(model, x_start, PRED_T_START, PRED_T_END, n_step=100)
    print(f"  start day {PRED_DAY_START}: {x_start.shape}")
    print(f"  predicted day {HELD_OUT_DAY}: {x_pred.shape}")

    # Save predictions
    np.savez_compressed(
        os.path.join(OUT, "alltime_predictions_interp.npz"),
        held_out_day=HELD_OUT_DAY,
        held_out_t=HELD_OUT_T,
        pred_day_start=PRED_DAY_START,
        pred_t_start=PRED_T_START,
        pred_t_end=PRED_T_END,
        pred_at_held_out=x_pred,
        x_start=x_start,
        n_params=n_params,
        training_time_s=elapsed,
    )
    with open(os.path.join(OUT, "alltime_meta_interp.json"), "w") as f:
        json.dump({
            "method": "all-time MLP (interpolation setup)",
            "n_params": int(n_params),
            "iters": N_ITER,
            "h": H_KERNEL, "lam": LAM,
            "M": M_b, "N": N_b, "K_ens": K_ENS,
            "pred_day_start": PRED_DAY_START,
            "pred_t_start": PRED_T_START,
            "pred_t_end": PRED_T_END,
            "training_time_s": elapsed,
            "final_loss": float(losses[-1]),
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(losses, lw=0.6, alpha=0.7)
    ax.set(xlabel="iteration", ylabel="loss",
           title="Exp 6 (interp): all-time training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    curve_path = os.path.join(OUT, "alltime_training_curve_interp.png")
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {curve_path}")
    print(f"Saved: {os.path.join(OUT, 'alltime_predictions_interp.npz')}")


if __name__ == "__main__":
    main()
