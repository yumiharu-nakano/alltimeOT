"""Linear-in-parameters feature dictionaries for the drift model.

Each feature function takes two tensors
    tg : (P,)            time values
    xg : (P, d)          space values
and returns a design matrix Phi of shape (P, p_feat).  The drift is
then u(t, x) = Phi(t, x) @ W with W of shape (p_feat, d).

All functions are pure PyTorch so that the composed loss is fully
differentiable by autograd.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import Tensor

FeatureFn = Callable[[Tensor, Tensor], Tensor]


def _ones_like_t(tg: Tensor) -> Tensor:
    return torch.ones_like(tg)


def feat_affine(tg: Tensor, xg: Tensor) -> Tensor:
    """[1, t, x_1, ..., x_d] -> (P, d+2)."""
    return torch.cat([_ones_like_t(tg)[:, None], tg[:, None], xg], dim=1)


def feat_quadratic_t(tg: Tensor, xg: Tensor) -> Tensor:
    """[1, t, t^2, x_1, ..., x_d] -> (P, d+3).

    Useful for the roundtrip problem where the drift has the shape
    2*pi*cos(pi*t), which is well approximated by a quadratic in t.
    """
    return torch.cat(
        [_ones_like_t(tg)[:, None], tg[:, None], (tg * tg)[:, None], xg],
        dim=1,
    )


def feat_bilinear(tg: Tensor, xg: Tensor) -> Tensor:
    """[1, t, x_1, ..., x_d, t*x_1, ..., t*x_d] -> (P, 2d+2)."""
    return torch.cat(
        [_ones_like_t(tg)[:, None], tg[:, None], xg, tg[:, None] * xg],
        dim=1,
    )


def feat_tanh_merger(tg: Tensor, xg: Tensor) -> Tensor:
    """Feature dictionary tuned to the 1-d bimodal merging flow.

    8 features: [1, t, x, t*x, tanh(x), t*tanh(x), tanh(2x), t*tanh(2x)].
    Targets drift of the form -2*tanh(2*(1-t)*x).
    """
    assert xg.shape[1] == 1, "feat_tanh_merger expects d=1"
    x = xg[:, 0]
    th1 = torch.tanh(x)
    th2 = torch.tanh(2.0 * x)
    return torch.stack(
        [
            _ones_like_t(tg),
            tg,
            x,
            tg * x,
            th1,
            tg * th1,
            th2,
            tg * th2,
        ],
        dim=1,
    )


def feat_tanh_merger_2d(tg: Tensor, xg: Tensor) -> Tensor:
    """2-d analogue of :func:`feat_tanh_merger`.

    10 features: [1, t, x1, x2, t*x1, t*x2, tanh(x1), t*tanh(x1),
                   tanh(2*x1), t*tanh(2*x1)].
    """
    assert xg.shape[1] == 2, "feat_tanh_merger_2d expects d=2"
    x1 = xg[:, 0]
    x2 = xg[:, 1]
    th1 = torch.tanh(x1)
    th2 = torch.tanh(2.0 * x1)
    return torch.stack(
        [
            _ones_like_t(tg),
            tg,
            x1,
            x2,
            tg * x1,
            tg * x2,
            th1,
            tg * th1,
            th2,
            tg * th2,
        ],
        dim=1,
    )


def make_feat_rbf_grid(
    t_centers: Tensor,
    x_centers: Tensor,
    sigma: float,
    include_bias: bool = True,
) -> FeatureFn:
    """Build a Gaussian-RBF feature map on a tensor-product grid of centres.

    Centres are placed at every ``(t_i, x_j, 0, ..., 0)`` with x only active
    in the first spatial coordinate; for d > 1 additional spatial
    directions are handled by also placing centres on the relevant axes
    via a separate call and :func:`concat_features`.

    The feature for centre c = (t_c, x_c) at a sample (t, x) is
        phi_c(t, x) = exp( -((t - t_c)^2 + (x_1 - x_c)^2) / (2 * sigma^2) ).

    Parameters
    ----------
    t_centers : (T_c,) tensor
        Centre locations along the time axis.
    x_centers : (X_c,) tensor
        Centre locations along the first spatial axis.
    sigma : float
        Common isotropic RBF bandwidth.
    include_bias : bool
        If True, prepend a constant-1 feature so the model can represent a
        non-zero asymptote outside the RBF support.

    Returns
    -------
    feat_fn : callable (tg, xg) -> Phi of shape (P, T_c * X_c [+ 1])
    """
    t_centers = t_centers.reshape(-1)
    x_centers = x_centers.reshape(-1)
    # Tensor-product grid of (T_c * X_c, 2) centres.
    tc, xc = torch.meshgrid(t_centers, x_centers, indexing="ij")
    centres = torch.stack([tc.reshape(-1), xc.reshape(-1)], dim=1)  # (C, 2)
    two_sigma_sq = 2.0 * sigma * sigma

    def feat(tg: Tensor, xg: Tensor) -> Tensor:
        # Project onto (t, x_1); other spatial dims are ignored by this map.
        tx = torch.stack([tg, xg[:, 0]], dim=1)   # (P, 2)
        d2 = ((tx[:, None, :] - centres[None, :, :]) ** 2).sum(-1)  # (P, C)
        phi = torch.exp(-d2 / two_sigma_sq)
        if include_bias:
            phi = torch.cat([torch.ones_like(tg)[:, None], phi], dim=1)
        return phi

    return feat


def concat_features(*fns: FeatureFn) -> FeatureFn:
    """Horizontally concatenate multiple feature maps."""

    def combined(tg: Tensor, xg: Tensor) -> Tensor:
        return torch.cat([f(tg, xg) for f in fns], dim=1)

    return combined


def feat_count(fn: FeatureFn, d: int) -> int:
    """Return the number of output features for a given dim by a dry run."""
    tg = torch.zeros(1)
    xg = torch.zeros(1, d)
    return int(fn(tg, xg).shape[1])
