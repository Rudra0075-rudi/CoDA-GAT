"""Stage 10: write the handoff package the training scripts read
(notebook Cells 33, 33b, 34, reduced to the files those scripts use).

Archive-compatible by default: the same file names and y.csv columns as
the August 2026 package that trained the models. Note that
scripts/baselines_b3.py (edited 2026-10-05, after the baseline run)
reads shared/geo_control_reference.parquet, a name that exists only in
the later export; the archive holds the same 148 controls as
shared/healthy_reference.parquet. write_package writes both names.
"""
import json

import numpy as np
import pandas as pd

from .config import FEATURES_A, FEATURES_B, FEATURES_C, META_COLS_ARCHIVE


def add_v2_label_columns(df):
    out = df.copy()
    out["exact_stage"] = out["stage"].where(out["resolution"].eq("exact"))
    out["eligible_T1"] = out["y_T1_significant"].notna()
    out["eligible_T2"] = out["y_T2_ordinal"].notna()
    out["eligible_T3"] = out["y_T3_early_adv"].notna()
    return out


def write_package(out, *, bulk_sup, bulk_raw, healthy, sup, dev_targets, panel_tab,
                  edges, topo, order, folds, tensors, archive_compatible=True):
    sh = out / "shared"
    sh.mkdir(parents=True, exist_ok=True)
    bulk_sup.to_parquet(sh / "bulk_expression.parquet")
    bulk_raw.to_parquet(sh / "bulk_expression_nocombat.parquet")
    healthy.to_parquet(sh / "healthy_reference.parquet")
    healthy.to_parquet(sh / "geo_control_reference.parquet")
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
        ), indent=2))
    pd.DataFrame(rows).to_csv(sh / "fold_sample_splits.csv", index=False)
