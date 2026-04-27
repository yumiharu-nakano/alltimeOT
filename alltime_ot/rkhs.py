"""Core RKHS all-time OT loss (sigma >= 0; sigma=0 is the deterministic case).

Notation
--------
mu_t           : target marginal at time t (density is not required;
                 only samples X[m, i] ~ mu_{t_s[m]} and X0[i] ~ mu_0)
u(t, x)        : drift field, parametrised linearly as u = Phi(t, x) @ W
                 where W has shape (p_feat, d).
Gaussian kernel: K((t,x),(t',x')) = exp(-a/2 * ((t-t')^2 + ||x-x'||^2)),
                 bandwidth a = 1 / h^2.

Loss
----
L(W) = kinetic(u)  +  lam * (J1 + J46_6 + J46_4)

    kinetic   : T / (MN) * sum_p ||u_p||^2
    J1        : block-diagonal-excluded sample-based estimator of
                (A^u_y K, A^u_y' K) weighted by T - max(t, t')
    J46_6     : cross term between interior samples and the t=0 marginal
    J46_4     : chronological boundary term summed over t_l

Block-diagonal exclusion drops only self pairs (m=l, i=j), keeping the
same-time cross-particle pairs.  This is quadrature-consistent and
eliminates the finite-sample bias of full diagonal exclusion.

See the companion paper for the derivation of each term from the
first-order variation of the Benamou-Brenier energy under an
all-time marginal constraint.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor

FeatureFn = Callable[[Tensor, Tensor], Tensor]


def rkhs_all_time_loss_from_drift(
    uf: Tensor,
    tg: Tensor,
    xg: Tensor,
    X0: Tensor,
    *,
    M: int,
    N: int,
    t_s: Tensor,
    lam: float,
    h: float = 1.0,
    T: float = 1.0,
    sigma: float = 0.0,
) -> Tensor:
    """Model-agnostic RKHS loss computed from pre-evaluated drift values.

    This is the core loss function.  It does not depend on how ``uf``
    was computed (linear model, MLP, etc.), only on the drift values
    themselves.  Autograd propagates gradients back through ``uf`` to
    whatever produced it.

    Parameters
    ----------
    uf : (P, d) tensor
        Drift values at each sample point; differentiable.
    tg : (P,) tensor
        Time coordinate of each sample.
    xg : (P, d) tensor
        Spatial coordinate of each sample.
    X0 : (N0, d) tensor
        Samples from the initial marginal mu_0.
    M, N : int
        Number of time slices and particles per slice (P = M * N).
    t_s : (M,) tensor
        The M distinct time points (needed for the block-diagonal mask
        and the chronological weight).
    lam, h, T : float
        Penalty weight, kernel bandwidth, and time horizon.
    sigma : float, default 0.0
        Diffusion coefficient.  ``sigma = 0`` recovers the deterministic
        case; ``sigma > 0`` augments the Stein operator with the
        ``(sigma**2 / 2) * Laplacian`` term required for the Nelson
        problem (Section "Stochastic extension" of the paper).
    """
    a = 1.0 / (h * h)
    c = 0.5 * sigma * sigma
    d = uf.shape[1]
    device = uf.device
    dtype = uf.dtype

    # Pairwise differences and the Gaussian kernel matrix.
    dt = tg[:, None] - tg[None, :]               # (P, P)
    dx = xg[:, None, :] - xg[None, :, :]         # (P, P, d)
    r2 = (dx * dx).sum(-1)                       # (P, P)
    Km = torch.exp(-0.5 * a * (dt * dt + r2))

    # Stein-operator quantities.
    tau1 = dt + (uf[:, None, :] * dx).sum(-1)    # (P, P)
    tau2 = dt + (uf[None, :, :] * dx).sum(-1)    # (P, P)
    uu = (uf[:, None, :] * uf[None, :, :]).sum(-1)
    beta = -d + a * r2                           # (P, P)

    # Block-diagonal mask: exclude only self-pairs (m=l, i=j).
    eye_N = torch.eye(N, device=device, dtype=dtype)
    off_block = 1.0 - torch.block_diag(*([eye_N] * M))

    # ---- J1: bulk term weighted by T - max(t, t') ----
    # Deterministic part.
    AA = -a * a * tau1 * tau2 + a * (1.0 + uu)
    if c != 0.0:
        # Sigma-correction for the double Stein operator
        # A^u_y A^u_{y'} K, derived in the paper.
        AA = AA + (
            a * a * c * beta * (tau2 - tau1)
            + a * a * c * c * beta * beta
            + 2.0 * a * a * c * (tau1 - tau2)
            + 2.0 * a * a * c * c * d
            - 4.0 * a * a * a * c * c * r2
        )
    AA = Km * AA
    tmax = torch.maximum(t_s[:, None], t_s[None, :])
    w_tt = (T - tmax).repeat_interleave(N, 0).repeat_interleave(N, 1)
    Wmsk = w_tt * off_block
    c1 = T * T / (M * M * N * N)
    J1 = c1 * (AA * Wmsk).sum()

    # ---- J46_6: initial-marginal boundary term ----
    dt6 = tg[:, None]
    dx6 = xg[:, None, :] - X0[None, :, :]
    r26 = (dx6 * dx6).sum(-1)
    K6 = torch.exp(-0.5 * a * (dt6 * dt6 + r26))
    tau6 = dt6 + (uf[:, None, :] * dx6).sum(-1)
    if c != 0.0:
        beta6 = -d + a * r26
        Au6 = a * K6 * (-tau6 + c * beta6)
    else:
        Au6 = -a * K6 * tau6
    tw = (T - t_s).repeat_interleave(N)
    J46_6 = (2.0 * T / (M * N * X0.shape[0])) * (Au6.sum(1) * tw).sum()

    # ---- J46_4: chronological boundary term over t_l ----
    if c != 0.0:
        Au_m = a * Km * (-tau1 + c * beta)
    else:
        Au_m = -a * Km * tau1
    chron = (t_s[:, None] <= t_s[None, :]).to(dtype)
    chron = chron.repeat_interleave(N, 0).repeat_interleave(N, 1)
    M4 = chron * off_block
    c4 = -2.0 * T * T / (M * M * N * N)
    J46_4 = c4 * (Au_m * M4).sum()

    kin = (T / (M * N)) * (uf * uf).sum()
    return kin + lam * (J1 + J46_6 + J46_4)


def rkhs_all_time_loss(
    W: Tensor,
    t_s: Tensor,
    X: Tensor,
    X0: Tensor,
    feat_fn: FeatureFn,
    *,
    lam: float,
    h: float = 1.0,
    T: float = 1.0,
    sigma: float = 0.0,
) -> Tensor:
    """RKHS loss for a linear-in-parameters drift u = feat_fn(t, x) @ W.

    Thin wrapper around :func:`rkhs_all_time_loss_from_drift`.
    See that function for the full mathematical description.

    Parameters
    ----------
    W : (p_feat, d) tensor
    t_s : (M,) tensor
    X : (M, N, d) tensor
    X0 : (N0, d) tensor
    feat_fn : callable (tg, xg) -> Phi of shape (P, p_feat)
    lam, h, T : float
    sigma : float, default 0.0
        Diffusion coefficient (sigma > 0 enables the Nelson extension).
    """
    M, N, d = X.shape
    tg = t_s.repeat_interleave(N)
    xg = X.reshape(M * N, d)
    Phi = feat_fn(tg, xg)
    uf = Phi @ W
    return rkhs_all_time_loss_from_drift(
        uf, tg, xg, X0, M=M, N=N, t_s=t_s, lam=lam, h=h, T=T, sigma=sigma,
    )


class LinearDriftModel:
    """Linear-in-parameters drift u(t, x) = feat_fn(t, x) @ W.

    Convenient wrapper used by experiment scripts to evaluate the
    learned drift after optimisation without re-implementing the
    feature map.  Supports both NumPy and PyTorch inputs.
    """

    def __init__(self, W: np.ndarray | Tensor, feat_fn: FeatureFn, d: int):
        W = np.asarray(W, dtype=np.float64)
        if W.ndim == 1:
            # Flat parameter vector: infer (p_feat, d).
            assert W.size % d == 0, "parameter length must be divisible by d"
            W = W.reshape(-1, d)
        self.W = W
        self.feat_fn = feat_fn
        self.d = d

    @property
    def n_params(self) -> int:
        return self.W.size

    def __call__(self, t: float | np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate u(t, x).

        ``x`` is (P, d) or (P,) for d=1; ``t`` is a scalar or (P,).
        Returns an array of shape (P, d), or (P,) when d == 1 to match
        the convention used by downstream 1d plotting helpers.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        P = x_arr.shape[0]
        if np.isscalar(t):
            t_arr = np.full(P, float(t))
        else:
            t_arr = np.asarray(t, dtype=np.float64)
        tg = torch.from_numpy(t_arr)
        xg = torch.from_numpy(x_arr)
        Phi = self.feat_fn(tg, xg).numpy()
        u = Phi @ self.W
        return u[:, 0] if self.d == 1 else u
