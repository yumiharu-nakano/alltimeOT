"""Ensemble-averaged scipy L-BFGS-B driver.

The experiment scripts repeatedly draw K_ens independent sample batches,
evaluate the RKHS loss on each, and average.  The resulting objective is
differentiable w.r.t. the model parameters, so scipy's L-BFGS-B on the
averaged loss + averaged gradient gives a stable estimator with far less
variance than a single-batch fit.

We compute gradients with torch autograd and expose the result to scipy
as a plain (loss, grad) numpy callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from .rkhs import FeatureFn, rkhs_all_time_loss

# A "batch provider" is any callable that returns
#     (t_s, X, X0) as torch tensors.
BatchProvider = Callable[[int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


@dataclass
class EnsembleObjective:
    """Averaged (loss, grad) callable for scipy L-BFGS-B.

    Parameters
    ----------
    batch_provider : callable(int) -> (t_s, X, X0)
        Returns a batch given an integer seed.  Implementations usually
        cache the batches once at construction time.
    feat_fn : callable
        Feature map for the linear-in-parameters drift.
    n_params : int
        Total length of the flat parameter vector.
    d : int
        Spatial dimension.
    lam, h, T : float
        RKHS loss hyper-parameters.
    K_ens : int
        Number of batches to average.
    seed_offset : int
        Base seed; batch k uses seed_offset + k.
    """

    batch_provider: BatchProvider
    feat_fn: FeatureFn
    n_params: int
    d: int
    lam: float
    h: float = 1.0
    T: float = 1.0
    K_ens: int = 10
    seed_offset: int = 0

    # Cached torch tensors (populated lazily so multiple instances don't
    # re-materialise the same batches).
    _batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=list, init=False, repr=False
    )

    def _ensure_batches(self) -> None:
        if self._batches:
            return
        for k in range(self.K_ens):
            self._batches.append(self.batch_provider(self.seed_offset + k))

    def __call__(self, w_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        self._ensure_batches()
        p_feat = self.n_params // self.d
        W = torch.tensor(
            w_flat.reshape(p_feat, self.d), dtype=torch.float64, requires_grad=True
        )
        loss_sum = torch.zeros((), dtype=torch.float64)
        for t_s, X, X0 in self._batches:
            loss_sum = loss_sum + rkhs_all_time_loss(
                W, t_s, X, X0, self.feat_fn, lam=self.lam, h=self.h, T=self.T
            )
        loss = loss_sum / self.K_ens
        (grad,) = torch.autograd.grad(loss, W)
        return float(loss.detach()), grad.detach().numpy().ravel()


def ensemble_lbfgs(
    objective: EnsembleObjective,
    inits: Sequence[np.ndarray],
    *,
    maxiter: int = 500,
    ftol: float = 1e-12,
    gtol: float = 1e-8,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, List[dict]]:
    """Run L-BFGS-B from several starting points; return the best minimum.

    Returns
    -------
    best_w : np.ndarray
    best_loss : float
    logs : list of dict with keys ``init_idx``, ``w``, ``loss``, ``nit``
    """
    best_w, best_loss = None, np.inf
    logs: List[dict] = []
    for i, w0 in enumerate(inits):
        res = minimize(
            objective,
            np.asarray(w0, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
        )
        logs.append(
            {
                "init_idx": i,
                "w": res.x.copy(),
                "loss": float(res.fun),
                "nit": int(res.nit),
                "grad_norm": float(np.linalg.norm(res.jac)),
            }
        )
        if verbose:
            print(
                f"  init {i}: loss={res.fun:.5f}, nit={res.nit}, "
                f"|grad|={np.linalg.norm(res.jac):.2e}"
            )
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_w = res.x.copy()
    assert best_w is not None
    return best_w, best_loss, logs
