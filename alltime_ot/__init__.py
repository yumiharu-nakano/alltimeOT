"""All-time optimal transport: RKHS drift estimator from marginal snapshots.

Public API:
    rkhs_all_time_loss      - core loss (PyTorch, autograd-friendly)
    LinearDriftModel        - linear-in-parameters drift u(t,x) = Phi(t,x) @ W
    feat_affine, feat_bilinear, feat_tanh_merger  - feature dictionaries
    ensemble_lbfgs          - ensemble-averaged scipy L-BFGS-B driver
    euler_simulate          - Euler ODE forward simulation
    sorted_w2, sliced_w2    - Wasserstein-2 evaluation
    mmd_gauss, mmd2_gauss   - MMD with Gaussian kernel
"""

from .rkhs import rkhs_all_time_loss, rkhs_all_time_loss_from_drift, LinearDriftModel
from .features import (
    feat_affine,
    feat_quadratic_t,
    feat_bilinear,
    feat_tanh_merger,
    feat_tanh_merger_2d,
    make_feat_rbf_grid,
    concat_features,
)
from .ensemble import ensemble_lbfgs, EnsembleObjective
from .simulate import (
    euler_simulate,
    mmd2_gauss,
    mmd_gauss,
    sliced_w2,
    sorted_w2,
)

__all__ = [
    "rkhs_all_time_loss",
    "rkhs_all_time_loss_from_drift",
    "LinearDriftModel",
    "feat_affine",
    "feat_quadratic_t",
    "feat_bilinear",
    "feat_tanh_merger",
    "feat_tanh_merger_2d",
    "make_feat_rbf_grid",
    "concat_features",
    "ensemble_lbfgs",
    "EnsembleObjective",
    "euler_simulate",
    "sorted_w2",
    "sliced_w2",
    "mmd2_gauss",
    "mmd_gauss",
]
