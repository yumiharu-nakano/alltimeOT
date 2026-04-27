#!/usr/bin/env python3
"""Preprocess the embryoid body (EB) scRNA-seq dataset for Experiment 6.

The Moon et al. 2019 (PHATE) dataset has 5 time points
(days 0-3, 6-9, 12-15, 18-21, 24-27) of human embryoid body
development.  Raw counts are in 10x mtx format under
``/tmp/eb_data/scRNAseq/``.

This script loads, QC-filters, normalises, log-transforms, picks
highly variable genes, runs PCA (default 30 components), subsamples
to a manageable cell count per time point, and saves a clean
``.npz`` archive that the downstream experiment scripts use.
"""

from __future__ import annotations

import os

import numpy as np
import scanpy as sc

DATA_ROOT = "/tmp/eb_data/scRNAseq"
OUT_DIR = os.environ.get("ALLTIME_OT_OUT", "output/exp6")
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES = ["T0_1A", "T2_3B", "T4_5C", "T6_7D", "T8_9E"]
DAY_VALUES = [3.0, 9.0, 15.0, 21.0, 27.0]  # midpoint of each window
LABELS = ["Day 0-3", "Day 6-9", "Day 12-15", "Day 18-21", "Day 24-27"]

# Hyperparameters
MIN_GENES_PER_CELL = 500
MAX_PCT_MT = 20.0
N_TOP_GENES = 2000
N_PCA = 30
N_SUBSAMPLE = 1500   # per time point; total 5*1500 = 7500 cells
SEED = 2026


def main() -> None:
    sc.settings.verbosity = 2

    print("Loading 10x mtx data ...")
    adatas = []
    for sample, day, label in zip(SAMPLES, DAY_VALUES, LABELS):
        path = os.path.join(DATA_ROOT, sample)
        adata = sc.read_mtx(os.path.join(path, "matrix.mtx")).T
        # genes.tsv is two columns (id, symbol)
        genes = np.loadtxt(os.path.join(path, "genes.tsv"), dtype=str, delimiter="\t")
        barcodes = np.loadtxt(os.path.join(path, "barcodes.tsv"), dtype=str)
        adata.var_names = genes[:, 1] if genes.ndim == 2 else genes
        adata.var_names_make_unique()
        adata.obs_names = barcodes
        adata.obs["day"] = day
        adata.obs["timepoint"] = label
        print(f"  {sample}: {adata.shape}")
        adatas.append(adata)

    print("\nConcatenating ...")
    adata = sc.concat(adatas, label="sample", keys=SAMPLES, index_unique="-")
    print(f"  total: {adata.shape}")

    print("\nComputing QC metrics ...")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True,
    )

    print(f"  before QC: {adata.shape}")
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_PCT_MT].copy()
    print(f"  after  QC: {adata.shape}")

    print("\nNormalising and log-transforming ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print(f"\nSelecting top {N_TOP_GENES} HVGs ...")
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"  HVG-restricted shape: {adata.shape}")

    print("\nScaling and PCA ...")
    sc.pp.scale(adata, max_value=10.0)
    sc.tl.pca(adata, n_comps=N_PCA, random_state=SEED)
    X_pca = adata.obsm["X_pca"]
    print(f"  PCA shape: {X_pca.shape}")
    print(f"  explained variance ratio sum: {adata.uns['pca']['variance_ratio'].sum():.3f}")

    print(f"\nSubsampling to {N_SUBSAMPLE} cells per time point ...")
    rng = np.random.default_rng(SEED)
    selected = []
    for day in DAY_VALUES:
        idx = np.where(adata.obs["day"].values == day)[0]
        n_pick = min(N_SUBSAMPLE, len(idx))
        pick = rng.choice(idx, n_pick, replace=False)
        selected.append(pick)
        print(f"  day {day}: {n_pick} cells")
    selected = np.concatenate(selected)
    X = X_pca[selected].astype(np.float64)
    days = adata.obs["day"].values[selected].astype(np.float64)

    out_path = os.path.join(OUT_DIR, "eb_pca.npz")
    np.savez_compressed(
        out_path,
        X=X,
        days=days,
        day_values=np.array(DAY_VALUES),
        labels=np.array(LABELS),
        n_pca=N_PCA,
        explained_variance_ratio=adata.uns["pca"]["variance_ratio"],
    )
    print(f"\nSaved: {out_path}  (X shape={X.shape}, days shape={days.shape})")


if __name__ == "__main__":
    main()
