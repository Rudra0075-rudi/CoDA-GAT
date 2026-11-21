"""
Issue 4: AUROC aggregation, primary-metric invariance, ablation summaries.

Two aggregations, never to be interchanged in the text:
  per_seed_auc_mean        AUROC of each seed's predictions, averaged over the 5
                           seeds within a cohort, then averaged over the 8 cohorts.
                           Describes the expected single trained model.
  seed_ensemble_auc_mean   the 5 seeds' P(F2+) averaged per sample first, AUROC of
                           that 5-seed ensemble per cohort, then averaged over cohorts.
                           Describes a 5-model ensemble; it is a different model.
Between-cohort SD is over the 8 cohort-level values, ddof=1 (pandas default),
with ddof=0 given alongside. Baselines were run at a single seed, so for them
the two aggregations coincide.

Also re-derives every B7 run's AUROC / macro-F1 / balanced accuracy from the
saved probs_test.npy / predictions_test.npy and checks master_results.csv, and
recomputes the B9 ablation deltas and Wilcoxon tests from the saved tables.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import guard

REPO, OUT = guard.REPO, guard.OUT / "aggregation"
SEEDS = [42, 43, 44, 45, 46]


def fold_dirs():
    return sorted(p for p in (REPO / "folds").iterdir() if p.is_dir())


def test_labels(fold_dir, lab):
    ids = pd.read_csv(fold_dir / "test" / "sample_ids.csv")["sample_id"]
    return ids, lab.loc[ids, "y_T1_significant"].to_numpy().astype(int)


def arm_runs(arm):
    """(root, run subdir pattern) for each saved GAT arm."""
    if arm.startswith("b7_"):
        _, m, mode = arm.split("_", 2)
        root = {"m0": "results_b7", "m1": "results_b7_m1", "m3": "results_b7_m3"}[m]
        return REPO / root, f"mode_{mode}"
    return REPO / "ablations_b9", arm.removeprefix("b9_")


ARMS = ["b7_m0_off", "b7_m0_marginal", "b7_m0_conditional", "b7_m1_off", "b7_m3_off",
        "b9_shuffled_graph", "b9_random_graph", "b9_empty_graph", "b9_no_h3_deviation",
        "b9_no_pathway_features", "b9_no_fold_fitted_features"]


def gat_arm(arm, lab):
    root, sub = arm_runs(arm)
    per_run, per_cohort = [], []
    for fd in fold_dirs():
        ids, y = test_labels(fd, lab)
        cohort = json.loads((fd / "fold_info.json").read_text())["test_cohort"]
        fold = int(fd.name[4:fd.name.index("_")])
        ps = []
        for seed in SEEDS:
            rd = root / fd.name / sub / f"seed_{seed}"
            probs = np.load(rd / "probs_test.npy")
            assert probs.shape == (len(ids), 2)
            p = probs[:, 1]
            pred_path = rd / "predictions_test.npy"
            pred = np.load(pred_path) if pred_path.exists() else probs.argmax(axis=1)
            ps.append(p)
            per_run.append({"arm": arm, "fold": fold, "test_cohort": cohort, "seed": seed,
                            "auc": roc_auc_score(y, p),
                            "macro_f1": f1_score(y, pred, average="macro"),
                            "balanced_acc": balanced_accuracy_score(y, pred),
                            "pred_equals_argmax": bool((pred == probs.argmax(axis=1)).all())})
        pm = np.mean(ps, axis=0)
        per_cohort.append({"arm": arm, "fold": fold, "test_cohort": cohort, "n": len(y),
                           "seed_ensemble_auc": roc_auc_score(y, pm),
                           "seed_ensemble_macro_f1": f1_score(y, (pm >= 0.5).astype(int), average="macro")})
    runs, coh = pd.DataFrame(per_run), pd.DataFrame(per_cohort)
    coh = coh.merge(runs.groupby("fold")[["auc", "macro_f1", "balanced_acc"]].mean()
                    .rename(columns=lambda c: f"per_seed_{c}_mean").reset_index(), on="fold")
    return runs, coh


def baseline_arms(lab):
    rows = []
    root = REPO / "baselines" / "per_sample_predictions"
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for fd in fold_dirs():
            cohort = json.loads((fd / "fold_info.json").read_text())["test_cohort"]
            f = d / f"{fd.name}.csv"
            df = pd.read_csv(f, float_precision="round_trip")
            ids, y = test_labels(fd, lab)
            assert df["sample_id"].tolist() == ids.tolist() and (df["y_true"].to_numpy() == y).all()
            rows.append({"arm": d.name, "fold": int(fd.name[4:fd.name.index("_")]), "test_cohort": cohort,
                         "auc": roc_auc_score(y, df["p_test"]),
                         "macro_f1": f1_score(y, df["pred_test"], average="macro"),
                         "balanced_acc": balanced_accuracy_score(y, df["pred_test"])})
    return pd.DataFrame(rows)


def summarise(coh, col, label):
    v = coh[col]
    return {"aggregation": label, "mean": float(v.mean()), "sd_between_cohort_ddof1": float(v.std(ddof=1)),
            "sd_between_cohort_ddof0": float(v.std(ddof=0)), "n_cohorts": int(v.notna().sum())}


def master_check(runs_all):
    """Saved master tables vs metrics re-derived from saved per-sample arrays."""
    out = []
    for root, arms in (("results_b7", ["b7_m0_off", "b7_m0_marginal", "b7_m0_conditional"]),
                       ("results_b7_m1", ["b7_m1_off"]), ("results_b7_m3", ["b7_m3_off"])):
        mr = pd.read_csv(REPO / root / "master_results.csv")
        for arm in arms:
            mode = arm.split("_", 2)[2]
            r = runs_all[runs_all["arm"] == arm].merge(mr[mr["mode"] == mode], on=["fold", "seed"], suffixes=("", "_saved"))
            out.append({"table": f"{root}/master_results.csv", "arm": arm, "n_runs": len(r),
                        **{f"max_abs_diff_{m}": float((r[m] - r[f"{m}_saved"]).abs().max())
                           for m in ("auc", "macro_f1", "balanced_acc")},
                        "predictions_equal_argmax": bool(r["pred_equals_argmax"].all())})
    ab = pd.read_csv(REPO / "ablations_b9" / "master_ablation_results.csv")
    for arm in [a for a in ARMS if a.startswith("b9_")]:
        name = arm.removeprefix("b9_")
        r = runs_all[runs_all["arm"] == arm].merge(ab[ab["ablation"] == name], on=["fold", "seed"], suffixes=("", "_saved"))
        out.append({"table": "ablations_b9/master_ablation_results.csv", "arm": arm, "n_runs": len(r),
                    **{f"max_abs_diff_{m}": float((r[m] - r[f"{m}_saved"]).abs().max())
                       for m in ("auc", "macro_f1", "balanced_acc")},
                    "predictions_equal_argmax": bool(r["pred_equals_argmax"].all())})
    return pd.DataFrame(out)


def ablation_table():
    """Recomputes ablations_b9/ablation_report.txt from the saved tables (per-fold
    seed means; paired over the 8 folds; two-sided Wilcoxon, as in ablations_b9.py)."""
    ab = pd.read_csv(REPO / "ablations_b9" / "master_ablation_results.csv")
    b7 = pd.read_csv(REPO / "results_b7" / "master_results.csv")
    base = b7[(b7["mode"] == "off") & (b7["m"] == 0)].groupby("fold")[["auc", "macro_f1", "balanced_acc"]].mean()
    rows = []
    for abl, g in ab.groupby("ablation"):
        per = g.groupby("fold")[["auc", "macro_f1", "balanced_acc"]].mean()
        for metric in ("auc", "macro_f1", "balanced_acc"):
            j = per[[metric]].join(base[[metric]], lsuffix="_abl", rsuffix="_base").dropna()
            d = j[f"{metric}_abl"] - j[f"{metric}_base"]
            rows.append({"ablation": abl, "metric": metric, "n_folds": len(j),
                         "baseline_mean": float(j[f"{metric}_base"].mean()),
                         "ablation_mean": float(j[f"{metric}_abl"].mean()),
                         "mean_diff_ablation_minus_full": float(d.mean()), "sd_diff": float(d.std()),
                         "wilcoxon_p_two_sided": float(wilcoxon(j[f"{metric}_abl"], j[f"{metric}_base"]).pvalue)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(REPO / "shared" / "labels.parquet")
    runs_all, coh_all, summ = [], [], []
    for arm in ARMS:
        runs, coh = gat_arm(arm, lab)
        runs_all.append(runs)
        coh_all.append(coh)
        for col, label in (("per_seed_auc_mean", "per_seed_auc_mean"), ("seed_ensemble_auc", "seed_ensemble_auc_mean"),
                           ("per_seed_macro_f1_mean", "per_seed_macro_f1_mean"),
                           ("seed_ensemble_macro_f1", "seed_ensemble_macro_f1_mean")):
            summ.append({"model": arm, "kind": "gat_5seed", **summarise(coh, col, label)})
    runs_all, coh_all = pd.concat(runs_all), pd.concat(coh_all)
    base = baseline_arms(lab)
    for arm, g in base.groupby("arm"):
        summ.append({"model": arm, "kind": "baseline_single_seed", **summarise(g, "auc", "single_seed_auc_mean")})
        summ.append({"model": arm, "kind": "baseline_single_seed", **summarise(g, "macro_f1", "single_seed_macro_f1_mean")})
    summ = pd.DataFrame(summ)

    runs_all.to_csv(OUT / "per_run_metrics_recomputed.csv", index=False)
    coh_all.to_csv(OUT / "per_cohort_gat.csv", index=False)
    base.to_csv(OUT / "per_cohort_baselines.csv", index=False)
    summ.to_csv(OUT / "auroc_aggregation_summary.csv", index=False)
    chk = master_check(runs_all)
    chk.to_csv(OUT / "primary_metric_invariance.csv", index=False)
    abl = ablation_table()
    abl.to_csv(OUT / "ablation_summary_recomputed.csv", index=False)

    auc_rows = summ[summ["aggregation"].isin(["per_seed_auc_mean", "seed_ensemble_auc_mean", "single_seed_auc_mean"])]
    lines = ["AUROC aggregation (8 held-out cohorts; SD = between-cohort, ddof=1 [ddof=0])", "=" * 78]
    for _, r in auc_rows.sort_values("mean", ascending=False).iterrows():
        lines.append(f"  {r['model']:34s} {r['aggregation']:24s} {r['mean']:.5f}  SD {r['sd_between_cohort_ddof1']:.5f} "
                     f"[{r['sd_between_cohort_ddof0']:.5f}]")
    lines += ["", "Primary-metric invariance (saved tables vs saved per-sample arrays):", chk.to_string(index=False),
              "", "Ablations (ablation minus full model; per-fold seed means; Wilcoxon two-sided, n=8):",
              abl.to_string(index=False)]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]), flush=True)
    tol = 1e-12
    inv_ok = bool((chk[[c for c in chk.columns if c.startswith("max_abs")]] <= tol).all().all()
                  and chk["predictions_equal_argmax"].all())
    return {"summary": summ, "invariance_ok": inv_ok, "invariance": chk, "ablation": abl}
