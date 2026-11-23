"""Stage 10: write the handoff package the training scripts read
(notebook Cells 32, 33 and 34).

Writes every file of the archived package: shared/ (14 files)
and, per fold, train/ calibration/ test/ (X.npy, y.csv, sample_ids.csv)
plus fold_info.json. Archive-compatible by default: the same file names,
columns and JSON fields. Not written: the flat top-level copy of fold 0
and the README/notes, which Cell 34 added for inspection only.
"""
import json
import time

import numpy as np
import pandas as pd

from .config import FEATURES_A, FEATURES_B, FEATURES_C, META_COLS_ARCHIVE, TASK
from .folds import fold_table


def add_v2_label_columns(df):
    out = df.copy()
    out["exact_stage"] = out["stage"].where(out["resolution"].eq("exact"))
    out["eligible_T1"] = out["y_T1_significant"].notna()
    out["eligible_T2"] = out["y_T2_ordinal"].notna()
    out["eligible_T3"] = out["y_T3_early_adv"].notna()
    return out


def write_package(out, *, bulk_sup, bulk_raw, healthy, sup, dev_targets, panel_tab,
                  edges, topo, order, folds, tensors, meta_all, head3, archive_compatible=True):
    sh = out / "shared"
    sh.mkdir(parents=True, exist_ok=True)
    bulk_sup.to_parquet(sh / "bulk_expression.parquet")
    bulk_raw.to_parquet(sh / "bulk_expression_nocombat.parquet")
    healthy.to_parquet(sh / "healthy_reference.parquet")
    labels = sup if archive_compatible else add_v2_label_columns(sup)
    labels.to_parquet(sh / "labels.parquet")
    dev_targets.to_parquet(sh / "deviation_targets.parquet")
    panel_tab.to_csv(sh / "panel_genes.csv", index=False)
    edges.to_csv(sh / "graph_edges.csv", index=False)
    topo.to_csv(sh / "graph_topology.csv")
    pd.Series(order, name="sample_id").to_csv(sh / "tensor_sample_order.csv", index=False)
    pd.DataFrame({"idx": range(17), "feature": FEATURES_A + FEATURES_B,
                  "group": ["A"] * 8 + ["B"] * 9,
                  "leakage_class": ["dynamic_global"] * 8 + ["static_global"] * 9}
                 ).to_csv(sh / "feature_schema_AB.csv", index=False)
    pd.DataFrame({"idx": range(17, 26), "feature": FEATURES_C, "group": "C",
                  "leakage_class": "fold_fitted"}).to_csv(sh / "feature_schema_C.csv", index=False)

    cols = META_COLS_ARCHIVE if archive_compatible else META_COLS_ARCHIVE + [
        "exact_stage", "eligible_T1", "eligible_T2", "eligible_T3"]
    rows = []
    for f in folds:
        base = out / "folds" / f"fold{f['fold']}_{f['test_cohort']}"
        for split, key in (("train", "train_idx"), ("calibration", "cal_idx"), ("test", "test_idx")):
            d = base / split
            d.mkdir(parents=True, exist_ok=True)
            ids = f[key]
            np.save(d / "X.npy", tensors[f["fold"]][split])
            m = labels.loc[ids, cols].copy()
            m.index.name = "sample_id"
            m.to_csv(d / "y.csv")
            pd.Series(ids, name="sample_id").to_csv(d / "sample_ids.csv", index=False)
            rows += [dict(fold=f["fold"], sample_id=i,
                          split="proper_train" if split == "train" else split,
                          cohort=sup.loc[i, "cohort"]) for i in ids]
        (base / "fold_info.json").write_text(json.dumps(dict(
            fold=f["fold"], test_cohort=f["test_cohort"], calibration_source=f["calibration"],
            shapes={s: list(tensors[f["fold"]][s].shape) for s in ("train", "calibration", "test")},
            positive_rate=dict(train=float(sup.loc[f["train_idx"], TASK].mean()),
                               calibration=float(sup.loc[f["cal_idx"], TASK].mean()),
                               test=float(sup.loc[f["test_idx"], TASK].mean())),
            train_cohorts=sorted(sup.loc[f["train_idx"], "cohort"].unique().tolist()),
            rules=["augmentation applies to train/ ONLY",
                   "calibration/ and test/ are raw and must remain raw",
                   "features 17-25 (Group C) are already fitted on THIS fold's train"],
        ), indent=2))
    pd.DataFrame(rows).to_csv(sh / "fold_sample_splits.csv", index=False)

    fd = fold_table(folds, sup)
    fd.drop(columns="JS_test_vs_cal").to_csv(sh / "fold_assignment_lodo.csv", index=False)
    fd["JS_test_vs_train"] = fd["JS"]
    handoff = dict(
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_supervised=int(len(sup)), n_healthy=int(meta_all.is_healthy.sum()),
        n_genes=int(bulk_sup.shape[0]), n_panel=int(len(panel_tab)),
        n_edges_undirected=int(len(edges) // 2), n_folds=len(folds),
        t1_cohorts=sorted(sup.loc[sup.tier == "T1", "cohort"].unique()),
        t2_cohorts=sorted(sup.loc[sup.tier == "T2", "cohort"].unique()),
        external=["GSE163211"], task_primary=TASK, calibration_mode="V1",
        features=dict(A=8, B=9, C=9, total=26), head3=head3,
        fold_label_shift=fd[["fold", "test_cohort", "JS_test_vs_cal", "JS_test_vs_train"]].to_dict("records"),
        notes=[
            "Group C must be recomputed inside each fold; never precompute.",
            "Augmentation: proper_train only. Never calibration, never test.",
            "GSE163211 (NanoString) stays outside the pipeline entirely.",
            "Head 3 is a 3-arm ablation, not a fixed setting.",
            "GSE213621 lacks ALB/APOA1/APOB/FGA/FGB \u2014 annotation gap, footnote it.",
            "Two cohorts (GSE31803, GSE213621 partial) are grouped-label: "
            "usable for T1/T3, excluded from the ordinal task.",
        ])
    (sh / "handoff.json").write_text(json.dumps(handoff, indent=2))
