"""Reusable sampling utilities for the paper's benchmark problems.

Each function returns a ``batch_provider`` closure suitable for
:class:`alltime_ot.ensemble.EnsembleObjective`.  The provider takes an
integer seed and returns ``(t_s, X, X0)`` as torch tensors.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch

Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
BatchProvider = Callable[[int], Batch]


def _grid_times(M: int, T: float) -> np.ndarray:
    return np.linspace(0.01 * T, 0.99 * T, M)


def gaussian_translation(
    *,
    mean_fn: Callable[[float], np.ndarray],
    d: int,
    M: int,
    N: int,
    N0: int,
    T: float = 1.0,
) -> BatchProvider:
    """mu_t = N(mean_fn(t), I_d).  ``mean_fn`` returns a (d,) array."""

    def provider(seed: int) -> Batch:
        rng = np.random.default_rng(seed)
        t_s = _grid_times(M, T)
        X = np.zeros((M, N, d))
        for m in range(M):
            X[m] = mean_fn(t_s[m])[None, :] + rng.standard_normal((N, d))
        X0 = mean_fn(0.0)[None, :] + rng.standard_normal((N0, d))
        return (
            torch.from_numpy(t_s),
            torch.from_numpy(X),
            torch.from_numpy(X0),
        )

    return provider


def gaussian_mixture_1d(
    *,
    means_fn: Callable[[float], Tuple[float, float]],
    weights: Tuple[float, float] = (0.5, 0.5),
    M: int,
    N: int,
    N0: int,
    T: float = 1.0,
) -> BatchProvider:
    """Bimodal mu_t = w1*N(m1(t),1) + w2*N(m2(t),1) in 1-d."""

    w1 = weights[0]

    def sample_mixture(t: float, n: int, rng: np.random.Generator) -> np.ndarray:
        pick = rng.random(n) < w1
        m1, m2 = means_fn(t)
        return np.where(pick, m1 + rng.standard_normal(n), m2 + rng.standard_normal(n))

    def provider(seed: int) -> Batch:
        rng = np.random.default_rng(seed)
        t_s = _grid_times(M, T)
        X = np.zeros((M, N, 1))
        for m in range(M):
            X[m, :, 0] = sample_mixture(t_s[m], N, rng)
        X0 = sample_mixture(0.0, N0, rng)[:, None]
        return (
            torch.from_numpy(t_s),
            torch.from_numpy(X),
            torch.from_numpy(X0),
        )

    return provider


def gaussian_mixture_x1_times_normal_x2(
    *,
    means_fn: Callable[[float], Tuple[float, float]],
    M: int,
    N: int,
    N0: int,
    T: float = 1.0,
) -> BatchProvider:
    """2-d product: bimodal on x_1, standard normal on x_2."""

    def sample(t: float, n: int, rng: np.random.Generator) -> np.ndarray:
        pick = rng.random(n) < 0.5
        m1, m2 = means_fn(t)
        x1 = np.where(pick, m1 + rng.standard_normal(n), m2 + rng.standard_normal(n))
        x2 = rng.standard_normal(n)
        return np.column_stack([x1, x2])

    def provider(seed: int) -> Batch:
        rng = np.random.default_rng(seed)
        t_s = _grid_times(M, T)
        X = np.zeros((M, N, 2))
        for m in range(M):
            X[m] = sample(t_s[m], N, rng)
        X0 = sample(0.0, N0, rng)
        return (
            torch.from_numpy(t_s),
            torch.from_numpy(X),
            torch.from_numpy(X0),
        )

    return provider
