"""Baseline methods used for comparison in the paper.

These helpers are shared between the experiment-2 and experiment-4
comparison scripts.

* :func:`affine_mmot_loss_grad` -- Multi-Marginal OT Affine flow with T_k(x)=A_k x+b_k
  optimised under an MMD U-statistic marginal loss plus a kinetic cost.
* :func:`sinkhorn_wot_drift`    -- Waddington-OT drift estimator built from
  consecutive entropic OT couplings (via POT's Sinkhorn solver).
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import logsumexp

SampleFn = Callable[[float, int, np.random.Generator], np.ndarray]


# ---------------------------------------------------------------------
# Pure-NumPy log-domain Sinkhorn (for WOT baseline)
# ---------------------------------------------------------------------
def _sinkhorn_log(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    reg: float,
    *,
    num_iter: int = 2000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Log-domain Sinkhorn iteration for regularised OT.

    Solves  min_{P in Pi(a,b)}  <P, C> + reg * H(P)  where Pi(a, b) is
    the set of couplings with marginals (a, b).  Uses log-space updates
    for numerical stability; avoids the POT dependency which has
    platform-specific binary issues.

    Parameters
    ----------
    a, b : (n,), (m,) ndarray, non-negative, sum-to-one
    C    : (n, m) ndarray, cost matrix
    reg  : float, entropic regularisation weight
    num_iter, tol : standard Sinkhorn stopping criteria

    Returns
    -------
    gamma : (n, m) ndarray, the optimal coupling
    """
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)
    log_K = -C / reg                        # (n, m)
    f = np.zeros(a.shape[0])                # log-scaling for rows
    g = np.zeros(b.shape[0])                # log-scaling for cols
    for _ in range(num_iter):
        f_new = log_a - logsumexp(log_K + g[None, :], axis=1)
        g_new = log_b - logsumexp(log_K + f_new[:, None], axis=0)
        if (
            np.max(np.abs(f_new - f)) < tol
            and np.max(np.abs(g_new - g)) < tol
        ):
            f, g = f_new, g_new
            break
        f, g = f_new, g_new
    # Symmetric finalization: one extra f-update so both marginals are
    # satisfied up to the same tolerance.
    f = log_a - logsumexp(log_K + g[None, :], axis=1)
    return np.exp(f[:, None] + log_K + g[None, :])


# ---------------------------------------------------------------------
# Affine MMOT (multi-marginal OT with learned affine maps)
# ---------------------------------------------------------------------
def make_affine_mmot_loss_grad(
    sample_fn: SampleFn,
    *,
    N_marg: int,
    M_batch: int,
    lam: float,
    alpha_k: float,
    K_ens: int = 10,
    T: float = 1.0,
    seed: int = 42,
) -> Tuple[Callable[[np.ndarray], Tuple[float, np.ndarray]], np.ndarray, float]:
    """Build a deterministic (loss, grad) callable for an affine MMOT problem.

    The parameters ``theta`` are packed as ``[A_1,...,A_N, b_1,...,b_N]``
    with fixed ``A_0 = 1`` and ``b_0 = 0`` (identity map).  The loss is
    the kinetic cost of the chained affine maps plus an MMD U-statistic
    penalty matching each T_k # mu_0 to samples from mu_{t_k}.

    Parameters
    ----------
    sample_fn : callable ``(t, n, rng) -> (n,) ndarray``
        Draws ``n`` samples from the marginal mu_t.
    """
    rng = np.random.default_rng(seed)
    t_pts = np.linspace(0.0, T, N_marg + 1)
    dt = T / N_marg

    X0_ens = [sample_fn(0.0, M_batch, rng) for _ in range(K_ens)]
    Y_ens = [
        [sample_fn(t_pts[k + 1], M_batch, rng) for k in range(N_marg)]
        for _ in range(K_ens)
    ]

    def loss_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        A_full = np.concatenate([[1.0], theta[:N_marg]])
        b_full = np.concatenate([[0.0], theta[N_marg:]])

        total = 0.0
        gA = np.zeros(N_marg)
        gb = np.zeros(N_marg)

        for ens in range(K_ens):
            X0 = X0_ens[ens]
            Mb = M_batch

            # -- Kinetic cost for the chained affine maps --
            cost = 0.0
            gAc = np.zeros(N_marg + 1)
            gbc = np.zeros(N_marg + 1)
            for k in range(1, N_marg + 1):
                diff = (A_full[k] - A_full[k - 1]) * X0 + (b_full[k] - b_full[k - 1])
                cost += (diff * diff).sum() / (Mb * dt)
                ga = (2.0 / (Mb * dt)) * (diff * X0).sum()
                gbv = (2.0 / (Mb * dt)) * diff.sum()
                gAc[k] += ga
                gAc[k - 1] -= ga
                gbc[k] += gbv
                gbc[k - 1] -= gbv
            grad_cost_A = gAc[1:]
            grad_cost_b = gbc[1:]

            # -- MMD U-statistic penalties on T_k # mu_0 vs mu_{t_k} --
            penalty = 0.0
            grad_pen_A = np.zeros(N_marg)
            grad_pen_b = np.zeros(N_marg)
            for k in range(N_marg):
                Ak = A_full[k + 1]
                bk = b_full[k + 1]
                Tk_X = Ak * X0 + bk
                Yk = Y_ens[ens][k]

                dxx = Tk_X[:, None] - Tk_X[None, :]
                Kxx = np.exp(-alpha_k * dxx ** 2)
                dxy = Tk_X[:, None] - Yk[None, :]
                Kxy = np.exp(-alpha_k * dxy ** 2)
                mask = 1.0 - np.eye(Mb)
                Kyy = np.exp(-alpha_k * (Yk[:, None] - Yk[None, :]) ** 2)
                mmd2 = (
                    (Kxx * mask).sum() / (Mb * (Mb - 1))
                    - 2.0 * Kxy.mean()
                    + (Kyy * mask).sum() / (Mb * (Mb - 1))
                )
                penalty += mmd2

                dKxx = -2.0 * alpha_k * dxx * Kxx
                dKxy = -2.0 * alpha_k * dxy * Kxy
                dz = (
                    (2.0 / (Mb * (Mb - 1))) * (dKxx * mask).sum(axis=1)
                    - (2.0 / (Mb * Mb)) * dKxy.sum(axis=1)
                )
                grad_pen_A[k] += (dz * X0).sum()
                grad_pen_b[k] += dz.sum()

            total += (cost + lam * penalty) / K_ens
            gA += (grad_cost_A + lam * grad_pen_A) / K_ens
            gb += (grad_cost_b + lam * grad_pen_b) / K_ens

        return total, np.concatenate([gA, gb])

    return loss_grad, t_pts, dt


def extract_affine_drift(
    theta: np.ndarray, t_pts: np.ndarray, dt: float, N_marg: int, T: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[float, np.ndarray], np.ndarray]]:
    """Recover piecewise-linear (in t), position-dependent drift from theta.

    In the interval (t_{k-1}, t_k] with T_{k-1}(x) = A_{k-1} x + b_{k-1},
    T_k(x) = A_k x + b_k, a particle at position y has velocity
        u(y) = [(A_k/A_{k-1} - 1) y + (b_k A_{k-1} - b_{k-1} A_k) / A_{k-1}] / dt.
    """
    A = np.concatenate([[1.0], theta[:N_marg]])
    b = np.concatenate([[0.0], theta[N_marg:]])
    t_mid = 0.5 * (t_pts[:-1] + t_pts[1:])

    slopes = (A[1:] / A[:-1] - 1.0) / dt
    intercepts = (b[1:] * A[:-1] - b[:-1] * A[1:]) / (A[:-1] * dt)

    slope_fn = interp1d(
        np.concatenate([[0.0], t_mid, [T]]),
        np.concatenate([[slopes[0]], slopes, [slopes[-1]]]),
        kind="linear",
        fill_value="extrapolate",
    )
    icpt_fn = interp1d(
        np.concatenate([[0.0], t_mid, [T]]),
        np.concatenate([[intercepts[0]], intercepts, [intercepts[-1]]]),
        kind="linear",
        fill_value="extrapolate",
    )

    def u_func(t: float, x: np.ndarray) -> np.ndarray:
        return slope_fn(t) * x + icpt_fn(t)

    return t_mid, intercepts, slopes, u_func


# ---------------------------------------------------------------------
# Waddington-OT via pairwise Sinkhorn
# ---------------------------------------------------------------------
def sinkhorn_wot_drift(
    sample_fn: SampleFn,
    *,
    M_wot: int,
    N_sample: int,
    eps_factor: float = 1.0,
    T: float = 1.0,
    seed: int = 42,
    position_dependent: bool = False,
) -> dict:
    """Chain consecutive entropic OT couplings to recover a drift.

    Schiebinger et al. (2019) solve entropic OT between consecutive
    marginals, use the barycentric projection as a transport map, and
    read the drift off the displacement divided by dt.

    When ``position_dependent`` is False we return the per-interval mean
    drift (as used in :mod:`exp2_wot_comparison`); when True we keep the
    source particles and return a 1d interpolator for each interval
    (as needed by :mod:`exp4_baselines` where the drift is non-constant
    in x).
    """
    rng = np.random.default_rng(seed)
    t_pts = np.linspace(0.0, T, M_wot + 1)
    dt = T / M_wot
    eps = eps_factor * dt

    samples = [sample_fn(t_pts[k], N_sample, rng) for k in range(M_wot + 1)]

    t_mid = np.zeros(M_wot)
    u_mid = np.zeros(M_wot)
    interpolators: List[Tuple[float, Callable[[np.ndarray], np.ndarray]]] = []

    for k in range(M_wot):
        X_src = samples[k]
        X_tgt = samples[k + 1]
        n = len(X_src)
        C = (X_src[:, None] - X_tgt[None, :]) ** 2
        a_hist = np.full(n, 1.0 / n)
        b_hist = np.full(n, 1.0 / n)
        gamma = _sinkhorn_log(a_hist, b_hist, C, reg=eps)
        T_map = (gamma @ X_tgt) / (gamma.sum(axis=1) + 1e-30)
        v = (T_map - X_src) / dt

        t_mid[k] = t_pts[k] + dt / 2.0
        u_mid[k] = float(v.mean())

        if position_dependent:
            order = np.argsort(X_src)
            xs = X_src[order]
            vs = v[order]
            _, idx = np.unique(xs, return_index=True)
            xs_u = xs[np.sort(idx)]
            vs_u = vs[np.sort(idx)]
            f = interp1d(
                xs_u, vs_u,
                kind="linear",
                fill_value=(vs_u[0], vs_u[-1]),
                bounds_error=False,
            )
            interpolators.append((t_mid[k], f))

    if position_dependent:
        def u_func(t: float, x: np.ndarray) -> np.ndarray:
            k = min(max(int(t / dt), 0), M_wot - 1)
            return interpolators[k][1](x)
    else:
        t_ext = np.concatenate([[0.0], t_mid, [T]])
        u_ext = np.concatenate([[u_mid[0]], u_mid, [u_mid[-1]]])
        u_interp_t = interp1d(t_ext, u_ext, kind="linear", fill_value="extrapolate")

        def u_func(t: float, x: np.ndarray) -> np.ndarray:
            return u_interp_t(t) * np.ones_like(x)

    return {
        "t_mid": t_mid,
        "u_mid": u_mid,
        "u_func": u_func,
        "interpolators": interpolators,
        "dt": dt,
    }
