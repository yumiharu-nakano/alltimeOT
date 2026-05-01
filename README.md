# All-time Optimal Transport

> Reference code for the paper *Continuum-marginal optimal transport: a mesh-free
> kernel method* (Y. Nakano, 2026).
> [arXiv:2604.24226](https://arxiv.org/abs/2604.24226)

A reference implementation of the **All-time OT** estimator: given samples
from every marginal of a time-indexed family of probability measures
`{μ_t}_{t ∈ [0, T]}`, recover the velocity field `u*(t, x)` that
transports `μ_0` through the entire family with minimal kinetic energy.

The estimator uses a Gaussian RKHS U-statistic of the Benamou–Brenier
continuity residual with quadrature-consistent block-diagonal exclusion,
and fits a linear-in-parameters drift
`u(t, x) = Φ(t, x) W` by ensemble-averaged L-BFGS-B.

Unlike two-marginal OT (which returns `u ≡ 0` whenever `μ_0 = μ_T`) and
Waddington-OT (which suffers `O(M/√N)` drift error with `M` snapshots),
the all-time estimator produces accurate non-trivial drifts from
**interior** marginals, as demonstrated in Experiments 2, 3, and 5.

## Repository layout

```
alltime_ot/              core Python package
    rkhs.py              PyTorch all-time OT loss (autograd-friendly)
    features.py          Linear-in-parameters feature dictionaries
    ensemble.py          Ensemble-averaged scipy L-BFGS-B driver
    problems.py          Sampling helpers for the benchmark problems
    simulate.py          Euler ODE integrator and W_2 / sliced-W_2 / MMD metrics
    baselines.py         Affine MMOT and Sinkhorn-WOT comparators

experiments/             Reproduction scripts for every paper figure
    exp1_gaussian.py         Exp 1: 1-d Gaussian translation
    exp2_roundtrip.py        Exp 2: 1-d roundtrip (key experiment, linear model)
    exp2_mlp.py              Exp 2: 1-d roundtrip (MLP neural network)
    exp2_flow_matching.py    Exp 2: Vanilla conditional flow matching baseline
    exp2_mmot_learned_affine.py  Exp 2: Multi-marginal OT with affine maps
    exp2_wot_comparison.py   Exp 2: Waddington-OT comparison
    exp3_bimodal.py          Exp 3: 1-d bimodal merging flow (bilinear, tanh dict, MLP)
    exp3_baselines.py        Exp 3: Bimodal MMOT and WOT baselines
    exp4_2d_translation.py   Exp 4: 2-d Gaussian translation
    exp5_2d_bifurcation.py   Exp 5: 2-d bifurcation / bimodal merging
    exp_stochastic_1d_train.py    §4.6: 1-d Nelson stochastic case (affine training)
    exp_stochastic_1d_metrics.py  §4.6: 1-d Nelson W2/MMD metrics (loads weights.json)
    exp6_eb_preprocess.py         §4.7: Embryoid body scRNA-seq preprocessing
    exp6_eb_alltime.py            §4.7: All-time MLP drift on EB data
    exp6_eb_baselines.py          §4.7: Waddington-OT and zero-drift baselines
    exp6_eb_evaluate.py           §4.7: Evaluation on held-out day 15
    appA_sensitivity.py      App A: Sensitivity sweep over (M, N, λ)
    appB_dim_scaling.py      App B: Dimension scaling (d = 1 … 10)
```

## Installation

```bash
git clone https://github.com/<you>/alltime_ot.git
cd alltime_ot
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the `alltime_ot` package plus its dependencies
(`numpy`, `scipy`, `torch`, `matplotlib`, `POT`).

## Reproducing an experiment

Each script in `experiments/` is a standalone entry point and writes
figures / tables to `output/expN/`:

```bash
python experiments/exp1_gaussian.py
python experiments/exp2_roundtrip.py
# ...
```

Set `ALLTIME_OT_OUT=/custom/path` to redirect the outputs.

## The core loss in one function

All experiments (1-d and 2-d, affine / bilinear / tanh-basis drift
classes) share the same PyTorch loss:

```python
from alltime_ot import rkhs_all_time_loss, feat_affine

# W : (p_feat, d) tensor, requires_grad=True
# t_s : (M,) interior time points
# X   : (M, N, d) samples from mu_{t_s[m]}
# X0  : (N0, d) samples from mu_0
loss = rkhs_all_time_loss(W, t_s, X, X0, feat_affine, lam=1000.0, h=1.0, T=1.0)
loss.backward()
```

The previous NumPy implementation duplicated ~150 lines of manual
gradient per experiment; autograd replaces all of it.  Adding a new
drift parametrisation only requires writing the forward feature map.

## Method summary

The Benamou–Brenier dual for a prescribed marginal flow
`∂_t μ_t + ∇·(u μ_t) = 0` gives

```
L(u) = ∫₀ᵀ E_{X ~ μ_t}[‖u(t, X)‖²] dt  +  λ · R(u),
```

where the penalty `R(u)` is the squared RKHS norm of the continuity
residual, expanded into three quadrature terms `J1`, `J46_6`, `J46_4`
(bulk, `t = 0` boundary, chronological boundary).  Each term is a
sample-based U-statistic of products of the Stein operator applied to a
Gaussian kernel; see `alltime_ot/rkhs.py` and the companion paper for
the derivation.

## Baselines

- **Flow Matching** (`exp2_flow_matching.py`): conditional FM with a
  straight-line interpolant — sees only `μ_0` and `μ_1`.
- **Affine MMOT** (`alltime_ot/baselines.py::make_affine_mmot_loss_grad`):
  multi-marginal OT with learned maps `T_k(x) = A_k x + b_k`, optimised
  under an MMD U-statistic.
- **Waddington-OT** (`alltime_ot/baselines.py::sinkhorn_wot_drift`):
  entropic OT between consecutive snapshots via `POT`.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@article{Nakano2026alltimeot,
  author  = {Nakano, Yumiharu},
  title   = {All-time optimal transport: a mesh-free kernel method},
  journal = {arXiv preprint arXiv:2604.24226},
  year    = {2026},
}
```


## License

MIT.
