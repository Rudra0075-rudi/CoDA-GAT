"""Stage 9: LODO folds and Group C (notebook Cell 32).

Two things differ from the supplied notebook, both to reproduce the
archived tensors that trained the models:

1. Cohen's d. The archive's pooled SD uses population (ddof=0) per-class
   variances with (n-1) weights: sqrt(((n1-1)v1 + (n0-1)v0)/(n1+n0-2)).
   The notebook revision switched to ddof=1 variances, which changes
   features 21 (cohens_d) and 17 (disease_specificity) by up to ~0.1%.
   COHENS_D_MODE="archive" (default) reproduces the archive exactly;
   "notebook_v2" gives the revised formula.
2. Nothing else: fold construction, calibration rotation and the other
   eight Group C features are the notebook code.

Disclosure for the paper: Group C is fitted on the whole proper-train
split. The training scripts later carve a 15% early-stopping validation
set out of that same split, so validation labels shaped Group C.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from sklearn.metrics import roc_auc_score

from .config import PATHWAYS, TASK
from .guards import (assert_cohort_disjoint, assert_no_synthetic, assert_tensor_integrity,
                     assert_tier_usage)

COHENS_D_MODE = "archive"


def make_folds(sup, t1_cohorts, task=TASK, min_cal=30):
    folds = []
    for k, test_c in enumerate(t1_cohorts):
        has = sup[task].notna()
        test_idx = sup.index[(sup.cohort == test_c) & has].tolist()
        pool = sup.index[(sup.cohort != test_c) & has].tolist()
        cands = [c for c in t1_cohorts if c != test_c
                 and (sup.loc[pool].cohort == c).sum() >= min_cal
                 and sup.loc[[i for i in pool if sup.loc[i, "cohort"] == c], task].nunique() > 1]
        if not cands:
            raise ValueError(f"fold {k}: no eligible T1 calibration cohort")
        cal_c = cands[k % len(cands)]          # deterministic rotation
        cal_idx = [i for i in pool if sup.loc[i, "cohort"] == cal_c]
        tr_idx = [i for i in pool if sup.loc[i, "cohort"] != cal_c]

        assert_cohort_disjoint(sup.loc[tr_idx], sup.loc[cal_idx], sup.loc[test_idx], k)
        assert_tier_usage(sup.loc[test_idx], "test", k)
        assert_tier_usage(sup.loc[cal_idx], "calibration", k)
        assert_no_synthetic(cal_idx, "calibration", k)
        assert_no_synthetic(test_idx, "test", k)
        folds.append(dict(fold=k, test_cohort=test_c, calibration=f"cohort:{cal_c}",
                          train_idx=tr_idx, cal_idx=cal_idx, test_idx=test_idx))
    return folds


def _pooled_sd(pos, neg):
    n1, n0 = pos.shape[1], neg.shape[1]
    ddof = 0 if COHENS_D_MODE == "archive" else 1
    v1, v0 = pos.var(axis=1, ddof=ddof), neg.var(axis=1, ddof=ddof)
    return np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))


def group_C(train_ids, z, sup, panel, task=TASK):
    X = z.loc[panel, train_ids].to_numpy(dtype=float)
    y = sup.loc[train_ids, task].to_numpy(dtype=float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Group C requires binary training labels")
    pos, neg = X[:, y == 1], X[:, y == 0]
    pooled = _pooled_sd(pos, neg)
    if (~np.isfinite(pooled) | (pooled <= 0)).any():
        raise ValueError("Cohen's d undefined for some genes")
    d = (pos.mean(axis=1) - neg.mean(axis=1)) / pooled
    t = ttest_ind(pos, neg, axis=1, equal_var=False).statistic
    lfc = pos.mean(axis=1) - neg.mean(axis=1)
    auc = np.array([roc_auc_score(y, X[i]) for i in range(len(panel))])

    gi = {g: i for i, g in enumerate(panel)}
    coh = np.zeros(len(panel))
    for gs in PATHWAYS.values():
        mem = [gi[g] for g in gs if g in gi]
        if len(mem) > 1:
            C = np.corrcoef(X[mem])
            np.fill_diagonal(C, np.nan)
            for j, m in enumerate(mem):
                coh[m] = max(coh[m], np.nanmean(C[j]))

    et = sup.loc[train_ids, "etiology"].values
    xe = np.zeros(len(panel))
    groups = [e for e in np.unique(et) if (et == e).sum() >= 20]
    if len(groups) > 1:
        signs = [np.sign(X[:, (et == e) & (y == 1)].mean(1) - X[:, (et == e) & (y == 0)].mean(1))
                 for e in groups if len(np.unique(y[et == e])) > 1]
        if signs:
            xe = np.abs(np.mean(signs, axis=0))

    stg = sup.loc[train_ids, "y_T2_ordinal"].values.astype(float)   # exact stages only
    ok = np.isfinite(stg)
    prog = (np.array([spearmanr(stg[ok], X[i, ok]).statistic for i in range(len(panel))])
            if ok.sum() >= 30 else np.zeros(len(panel)))

    spec = np.abs(d) / (np.abs(d).max() + 1e-9)
    comp = np.mean([np.argsort(np.argsort(np.abs(v))) / len(v) for v in (d, t, lfc, auc - 0.5)], axis=0)
    C = np.column_stack([spec, coh, xe, np.nan_to_num(prog), d, np.nan_to_num(t), lfc, auc, comp])
    return np.nan_to_num(C.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fold_tensors(fold, tensor_AB, order, z, sup, panel):
    C = group_C(fold["train_idx"], z, sup, panel)
    pos = {s: i for i, s in enumerate(order)}
    out = {}
    for split, key in (("train", "train_idx"), ("calibration", "cal_idx"), ("test", "test_idx")):
        rows = [pos[i] for i in fold[key]]
        ab = tensor_AB[rows]
        out[split] = np.concatenate([ab, np.broadcast_to(C[None], (len(rows),) + C.shape)],
                                    axis=2).astype(np.float32)
        assert_tensor_integrity(out[split], f"fold{fold['fold']}_{split}")
    return out
