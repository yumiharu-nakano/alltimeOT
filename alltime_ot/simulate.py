"""Forward ODE simulation and Wasserstein-2 / MMD helpers.

The drift u(t, x) learned by the RKHS estimator defines the ODE
    dX/dt = u(t, X),  X(0) ~ mu_0.
We integrate it with explicit Euler (sufficient for the smooth, linear
drifts used in the paper) and evaluate the marginal distributions at a
handful of time slices using the sorted 2-Wasserstein distance (or the
sliced variant in d>1) and a Gaussian-kernel MMD.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

DriftFn = Callable[[float, np.ndarray], np.ndarray]


def euler_simulate(
    u_func: DriftFn,
    x0: np.ndarray,
    *,
    T: float = 1.0,
    n_step: int = 1000,
    eval_t: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Dict[float, np.ndarray]:
    """Integrate dX/dt = u(t, X) with forward Euler.

    Parameters
    ----------
    u_func : callable (t, x) -> drift
        Accepts a scalar time and an array x of shape (N,) or (N, d);
        returns an array of the same shape.
    x0 : (N,) or (N, d) ndarray
    eval_t : iterable of float
        Times at which to record snapshots.  ``eval_t[0]`` is typically
        0 and always stores a copy of ``x0``.

    Returns
    -------
    snaps : dict mapping each requested t to the particle cloud at that time.
    """
    dt = T / n_step
    x = np.array(x0, dtype=np.float64, copy=True)
    eval_t = list(eval_t)
    snaps: Dict[float, np.ndarray] = {}
    if eval_t and abs(eval_t[0]) < dt / 2:
        snaps[eval_t[0]] = x.copy()
    tc = 0.0
    for _ in range(n_step):
        x = x + u_func(tc, x) * dt
        tc += dt
        for tv in eval_t:
            if tv not in snaps and abs(tc - tv) < dt / 2:
                snaps[tv] = x.copy()
    return snaps


def sorted_w2(a: np.ndarray, b: np.ndarray) -> float:
    """Exact 1d 2-Wasserstein distance between equal-sized samples.

    Computed as the L^2 norm of the difference of sorted order
    statistics: W_2(a, b) = sqrt( mean_i (sort(a)_i - sort(b)_i)^2 ).
    This matches the quadratic-cost Benamou-Brenier objective used in
    the paper's loss functional.
    """
    sa = np.sort(np.asarray(a).ravel())
    sb = np.sort(np.asarray(b).ravel())
    return float(np.sqrt(np.mean((sa - sb) ** 2)))


def sliced_w2(
    A: np.ndarray,
    B: np.ndarray,
    *,
    n_proj: int = 50,
    seed: int = 0,
) -> float:
    """Monte Carlo sliced 2-Wasserstein distance for (n, d) samples."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    rng = np.random.default_rng(seed)
    d = A.shape[1]
    dirs = rng.standard_normal((n_proj, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sw_sq = 0.0
    for direction in dirs:
        pa = np.sort(A @ direction)
        pb = np.sort(B @ direction)
        sw_sq += float(np.mean((pa - pb) ** 2))
    return float(np.sqrt(sw_sq / n_proj))


def mmd2_gauss(
    x: np.ndarray,
    y: np.ndarray,
    *,
    h: float = 1.0,
    max_samples: int | None = 2000,
    seed: int = 0,
) -> float:
    """Unbiased U-statistic estimator of the squared MMD with a Gaussian kernel.

    Uses the same Gaussian kernel family as the paper's RKHS loss:
        K(x, y) = exp( -||x - y||^2 / (2 h^2) ).
    The MMD is computed between two sample sets ``x`` and ``y``:
        MMD^2(x, y) = E[K(X, X')] - 2 E[K(X, Y)] + E[K(Y, Y')].
    The self-terms use an unbiased U-statistic (diagonal removed).

    Parameters
    ----------
    x, y : (n, d) or (n,) ndarray
        Sample sets.  1-d arrays are reshaped to (n, 1).
    h : float
        Kernel bandwidth; matches the paper's ``h`` convention.
    max_samples : int or None
        Cap on the sample count per set to control the O(n^2) cost
        (default 2000).  ``None`` uses all samples.
    seed : int
        RNG seed for the random subsample, if subsampling is triggered.

    Returns
    -------
    mmd2 : float
        U-statistic MMD^2.  Can be slightly negative due to finite-sample
        variance; clip at zero before taking the square root if desired.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    if max_samples is not None:
        rng = np.random.default_rng(seed)
        if x.shape[0] > max_samples:
            x = x[rng.choice(x.shape[0], max_samples, replace=False)]
        if y.shape[0] > max_samples:
            y = y[rng.choice(y.shape[0], max_samples, replace=False)]

    n, m = x.shape[0], y.shape[0]
    a = 1.0 / (h * h)

    # Pairwise squared distances
    dxx = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    dyy = ((y[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    dxy = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)

    Kxx = np.exp(-0.5 * a * dxx)
    Kyy = np.exp(-0.5 * a * dyy)
    Kxy = np.exp(-0.5 * a * dxy)

    # Unbiased U-statistic: remove the diagonal.
    term_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    term_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    term_xy = Kxy.mean()
    return float(term_xx - 2.0 * term_xy + term_yy)


def mmd_gauss(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    """MMD (square root of the unbiased MMD^2, clipped at zero)."""
    return float(np.sqrt(max(0.0, mmd2_gauss(x, y, **kwargs))))


