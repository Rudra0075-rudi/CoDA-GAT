"""Summarise the five-seed refit of the tabular models (manuscript table of the five-seed refit).

Usage (from the repository root):
    python paper_analyses/baseline_5seed/summarise_seeds.py paper_analyses/baseline_5seed

Reads per_sample_seed{42..46}/recomputed_metrics_tabular.csv (written by
scripts/baselines_b3.py --only tabular --seed S), the saved single-seed metrics in
baselines/per_sample_predictions/recomputed_metrics_tabular.csv and the CoDA-GAT
per-cohort AUROC (primary configuration, mean per seed) in
corrections_2026-11/outputs/aggregation/per_cohort_gat.csv. Writes summary.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

D = Path(sys.argv[1])
SEEDS = [42, 43, 44, 45, 46]

per_seed = []
for s in SEEDS:
    m = pd.read_csv(D / f"per_sample_seed{s}" / "recomputed_metrics_tabular.csv")
    per_seed.append(m.assign(seed=s)[["seed", "fold", "model", "auc"]])
per_seed = pd.concat(per_seed)
saved = pd.read_csv("baselines/per_sample_predictions/recomputed_metrics_tabular.csv")
gat = pd.read_csv("corrections_2026-11/outputs/aggregation/per_cohort_gat.csv")
gat = gat[gat.arm == "b7_m0_off"].set_index("fold")["per_seed_auc_mean"].sort_index()

rows = []
for model, g in per_seed.groupby("model"):
    by_fold = g.groupby("fold")["auc"].mean().sort_index()          # five-seed mean per cohort
    seed_means = g.groupby("seed")["auc"].mean()                      # eight-cohort mean per seed
    diff = by_fold - gat.reindex(by_fold.index)
    rows.append(dict(
        model=model,
        saved_single_seed=saved[saved.model == model]["auc"].mean(),
        rerun_5seed_mean=by_fold.mean(),
        sd_between_cohort=by_fold.std(ddof=1),
        seed_min=seed_means.min(), seed_max=seed_means.max(), seed_sd=seed_means.std(ddof=1),
        minus_gat=diff.mean(),
        cohorts_better=int((diff > 0).sum()),
        wilcoxon_p=wilcoxon(by_fold.values, gat.reindex(by_fold.index).values).pvalue,
    ))
out = pd.DataFrame(rows).sort_values("rerun_5seed_mean", ascending=False)
out.to_csv(D / "summary.csv", index=False)
print(out.round(4).to_string(index=False))
