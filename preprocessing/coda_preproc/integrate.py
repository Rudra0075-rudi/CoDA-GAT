"""Stage 5: batch integration (notebook Cells 25 and 26).

Logic is unchanged from the notebook:
- ComBat (inmoose.pycombat_norm, parametric, no covariates) is fitted
  separately within the RNA-seq group and within the microarray group.
- Every loaded cohort enters its platform's fit, including the eight
  held-out LODO test cohorts and the 148 GEO control samples. ComBat sees
  no labels, but it does see held-out expression: the evaluation is
  transductive. Fixing that needs per-fold correction and retraining.
- There is no cross-platform correction. The two platform blocks keep
  different overall levels; within-cohort z-scoring in features.py removes
  that offset for the per-sample features.
"""
from collections import Counter

import numpy as np
import pandas as pd

from .config import RES_HEALTHY, RES_UNKNOWN
from .guards import check_expression

MAX_COHORT_COST = 3000


def intersection_cost(expr, cohorts):
    """Genes each cohort removes from the shared intersection."""
    cnt = Counter()
    for g in cohorts:
        cnt.update(set(expr[g].index))
    base = sum(1 for v in cnt.values() if v == len(cohorts))
    costs = {}
    for g in cohorts:
        others = [x for x in cohorts if x != g]
        c2 = Counter()
        for x in others:
            c2.update(set(expr[x].index))
        costs[g] = sum(1 for v in c2.values() if v == len(others)) - base
    return base, costs


def integrate_platform(expr, cohorts, run_combat=True, max_cost=MAX_COHORT_COST):
    """Intersect genes, drop zero-variance genes, ComBat across cohorts."""
    from inmoose.pycombat import pycombat_norm

    if len(cohorts) != len(set(cohorts)):
        raise ValueError("repeated cohort entries")
    for g in cohorts:
        check_expression(expr[g], g)

    _, costs = intersection_cost(expr, cohorts)
    blockers = [g for g, c in costs.items() if c > max_cost]
    if blockers:
        raise RuntimeError(f"{blockers} each cost >{max_cost} genes; check identifier mapping")

    genes = sorted(set.intersection(*(set(expr[g].index) for g in cohorts)))
    X = pd.concat([expr[g].loc[genes] for g in cohorts], axis=1)
    batch = np.concatenate([[g] * expr[g].shape[1] for g in cohorts])
    check_expression(X, "concatenated")

    var_ok = np.ones(X.shape[0], dtype=bool)
    for b in np.unique(batch):
        block = X.loc[:, batch == b]
        if block.shape[1] < 2:
            raise ValueError(f"batch {b} has fewer than two samples")
        var_ok &= block.var(axis=1).to_numpy() > 1e-8
    X = X.loc[var_ok].copy()

    if run_combat and len(np.unique(batch)) > 1:
        Xc = pycombat_norm(X, batch)
        Xc = Xc if isinstance(Xc, pd.DataFrame) else pd.DataFrame(np.asarray(Xc), index=X.index, columns=X.columns)
        if not (Xc.index.equals(X.index) and Xc.columns.equals(X.columns)):
            raise ValueError("ComBat changed gene/sample labels or order")
        X = check_expression(Xc.copy(), "ComBat output")
    return X, batch


def integrate_all(expr, platform_of, labels_raw):
    """Return (bulk_all, meta_all): ComBat-corrected, common genes, usable samples."""
    groups = {}
    for g in expr:
        groups.setdefault("microarray" if platform_of[g] == "microarray" else "rnaseq", []).append(g)
    rna, _ = integrate_platform(expr, groups["rnaseq"])
    mic, _ = integrate_platform(expr, groups["microarray"])
    common = sorted(set(rna.index) & set(mic.index))
    bulk = pd.concat([rna.loc[common], mic.loc[common]], axis=1)

    lab = labels_raw.set_index("sample_id")
    usable = set(lab.index[~lab.resolution.isin([RES_UNKNOWN, "duplicate_biopsy"])])
    bulk = bulk[[c for c in bulk.columns if c in usable]]
    meta = lab.loc[bulk.columns, ["cohort", "stage", "resolution", "tier", "etiology",
                                  "platform", "preservation", "y_T1_significant",
                                  "y_T2_ordinal", "y_T3_early_adv"]].copy()
    meta["is_healthy"] = meta.resolution == RES_HEALTHY
    meta["is_supervised"] = meta[["y_T1_significant", "y_T3_early_adv"]].notna().any(axis=1)
    return check_expression(bulk, "BULK"), meta


def split_supervised(bulk, meta):
    sup = meta[meta.is_supervised]
    return bulk[sup.index], sup, bulk[meta.index[meta.is_healthy]]


def uncorrected_like(expr, like, drop=()):
    """No-ComBat matrix on exactly the corrected matrix's genes and samples."""
    raw = pd.concat([expr[g] for g in expr if g not in drop], axis=1)
    if not raw.columns.is_unique:
        raise ValueError("duplicate sample IDs across cohorts")
    out = raw.loc[like.index, like.columns].copy()
    if float(np.abs((like - out).to_numpy()).mean()) <= 1e-6:
        raise ValueError("ComBat and no-ComBat matrices are effectively identical")
    return check_expression(out, "BULK_RAW")
