"""Reference values that the manuscript states without a separate output file.

1. Chance overlap of a random edge set with STRING (learned-graph paragraph of T2):
   expected shared edges = m_a * m_b / C(n, 2) and the matching Jaccard index, for two
   undirected edge sets of the STRING size on the 72-gene panel.
2. Random-shortlist baselines for the planted-error recall (review-flag paragraph of T5):
   a 192-flag shortlist drawn from all test samples, and one drawn only from the exact
   F1 and F2 records, the only records whose labels the planted-error test flips.
3. Benjamini-Hochberg q values over the twelve AUROC and macro-F1 ablation comparisons
   (ablation paragraph of T2), from the Wilcoxon p values of the recomputed ablation summary.

Usage (from the repository root, Zenodo data restored):
    python paper_analyses/additional_analyses/derived_numbers.py
Output: derived_numbers.txt
"""
from math import comb

import pandas as pd

edges = pd.read_csv("shared/graph_edges.csv")
genes = pd.read_csv("shared/panel_genes.csv")
m = int((edges.src_idx < edges.dst_idx).sum())          # undirected STRING edges
n_pairs = comb(len(genes), 2)
shared = m * m / n_pairs
print(f"panel genes {len(genes)}, gene pairs {n_pairs}, STRING undirected edges {m}")
print(f"expected shared edges for a random edge set of size {m}: {shared:.1f}")
print(f"expected Jaccard: {shared / (2 * m - shared):.3f}")

elig = pd.read_csv("corrections_2026-11/outputs/stage_enrichment/planted_error_eligibility.csv")
row = elig[elig.resolution == "ALL"].iloc[0]
n_flags = 192
print(f"\nshortlist of {n_flags} flags")
print(f"random from all {row.n_test} test samples: expected recall {n_flags / row.n_test:.3f}")
print(f"random from the {row.eligible_total} exact F1/F2 records: expected recall {n_flags / row.eligible_total:.3f}")

abl = pd.read_csv("corrections_2026-11/outputs/aggregation/ablation_summary_recomputed.csv")
abl = abl[abl.metric.isin(["auc", "macro_f1"])].sort_values("wilcoxon_p_two_sided").reset_index(drop=True)
k = len(abl)
q = (abl.wilcoxon_p_two_sided * k / (abl.index + 1))[::-1].cummin()[::-1].clip(upper=1)
abl["bh_q"] = q
print(f"\nBenjamini-Hochberg over {k} ablation comparisons (AUROC and macro-F1)")
print(abl[["ablation", "metric", "mean_diff_ablation_minus_full", "wilcoxon_p_two_sided", "bh_q"]]
      .round(4).to_string(index=False))
print(f"smallest q: {abl.bh_q.min():.3f}")
