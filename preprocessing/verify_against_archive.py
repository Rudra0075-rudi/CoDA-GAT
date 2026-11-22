"""Check the cleaned preprocessing against the archived handoff package.

Usage:
  python verify_against_archive.py --archive <dir with shared/ and folds/>
         --gtex-tpm <GTEx v8 gene_tpm.gct.gz> --gtex-attr <SampleAttributesDS.txt>
         [--out report.csv]

Starts from the archived ComBat matrix, GEO controls and labels. It
rebuilds everything downstream (deviation targets, panel order, topology
from the archived edge list, Group A/B/C features, fold splits) and
compares each with the archived files. It trains nothing and writes only
the report.

Not checked here, because the inputs are not in the archive: per-cohort
expression loading, identifier mapping, ComBat itself, and STRING
filtering (the edge list is checked by hash against the notebook run).
"""
import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coda_preproc import config, features, folds, graph, reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--gtex-tpm", required=True)
    ap.add_argument("--gtex-attr", required=True)
    ap.add_argument("--gtex-cache", default=None, help="optional parquet cache of the GTEx liver matrix")
    ap.add_argument("--out", default="verification_report.csv")
    a = ap.parse_args()
    S = a.archive / "shared"
    rows = []

    def record(check, value, tol, note=""):
        rows.append(dict(check=check, max_abs_diff=float(value), tolerance=tol,
                         passed=bool(value <= tol), note=note))
        print(f"{'PASS' if value <= tol else 'FAIL'}  {check:48s} {value:.3e}  {note}")

    bulk_sup = pd.read_parquet(S / "bulk_expression.parquet")
    healthy = pd.read_parquet(S / "healthy_reference.parquet")
    sup = pd.read_parquet(S / "labels.parquet")
    order = pd.read_csv(S / "tensor_sample_order.csv").sample_id.tolist()
    sup = sup.loc[order]
    bulk_sup = bulk_sup[order]

    if a.gtex_cache and Path(a.gtex_cache).exists():
        gtex = pd.read_parquet(a.gtex_cache)
    else:
        gtex = reference.load_gtex_liver(a.gtex_tpm, a.gtex_attr)
        if a.gtex_cache:
            gtex.to_parquet(a.gtex_cache)
    record("GTEx liver samples == 226", abs(gtex.shape[1] - 226), 0)

    # deviation targets (needs supervised + control samples, 1,514 columns)
    bulk_all = pd.concat([bulk_sup, healthy], axis=1)
    disp = reference.rank_displacement(bulk_all, gtex)
    dev_pg = reference.z_in_sample(disp)
    arch_dev = pd.read_parquet(S / "deviation_targets.parquet")
    record("deviation genes == 10,734", abs(disp.shape[0] - 10734), 0)
    dirv = disp.loc[[g for g in config.ECM_UP if g in disp.index]].mean(0) - \
        disp.loc[[g for g in config.HEP_DOWN if g in disp.index]].mean(0)
    record("deviation_score (directional) vs archive",
           np.abs(dirv.loc[arch_dev.index] - arch_dev.deviation_score).max(), 1e-12)
    record("dev_raw vs archive", np.abs(disp.abs().mean(0).loc[arch_dev.index] - arch_dev.raw).max(), 1e-12)

    # panel and topology
    panel, dropped = graph.select_panel(bulk_sup)
    arch_panel = pd.read_csv(S / "panel_genes.csv")
    record("panel genes and order vs archive", float(panel != arch_panel.gene.tolist()), 0,
           f"excluded by alphabetical truncation: {dropped}")
    record("panel pathway labels vs archive",
           float(not graph.panel_table(panel).equals(arch_panel)), 0)
    E = pd.read_csv(S / "graph_edges.csv")
    G = nx.Graph()
    G.add_nodes_from(panel)
    G.add_edges_from(zip(E.src, E.dst))
    topo_arch = pd.read_csv(S / "graph_topology.csv", index_col=0)
    topo = pd.DataFrame({
        "degree_centrality": nx.degree_centrality(G), "betweenness_centrality": nx.betweenness_centrality(G),
        "closeness_centrality": nx.closeness_centrality(G), "clustering_coefficient": nx.clustering(G)}).loc[panel]
    record("graph: undirected pairs == 321", abs(G.number_of_edges() - 321), 0)
    record("topology recomputed from edge list",
           np.abs(topo.values - topo_arch.loc[panel, topo.columns].values).max(), 1e-12)

    # Group A + B
    T, order2, z = features.build_tensor_AB(bulk_sup, sup, panel, topo_arch, gtex, dev_pg)
    assert order2 == order

    # folds + Group C
    F = folds.make_folds(sup, config.T1_COHORTS)
    splits = pd.read_csv(S / "fold_sample_splits.csv")
    worst = np.zeros(26)
    split_mismatch = 0
    for f in F:
        fd = a.archive / "folds" / f"fold{f['fold']}_{f['test_cohort']}"
        ts = folds.fold_tensors(f, T, order, z, sup, panel)
        for split, key in (("train", "train_idx"), ("calibration", "cal_idx"), ("test", "test_idx")):
            ids = pd.read_csv(fd / split / "sample_ids.csv").sample_id.tolist()
            split_mismatch += int(ids != f[key])
            X = np.load(fd / split / "X.npy")
            worst = np.maximum(worst, np.abs(ts[split] - X).max(axis=(0, 1)))
        s = splits[splits.fold == f["fold"]]
        split_mismatch += int(set(s[s.split == "proper_train"].sample_id) != set(f["train_idx"]))
    record("fold membership and order vs archive", split_mismatch, 0)
    names = config.FEATURES_A + config.FEATURES_B + config.FEATURES_C
    for i, n in enumerate(names):
        record(f"feature {i:2d} {n}", worst[i], 1e-6,
               "")

    folds.COHENS_D_MODE = "notebook_v2"
    C2 = folds.group_C(F[0]["train_idx"], z, sup, panel)
    X0 = np.load(a.archive / "folds" / f"fold0_{F[0]['test_cohort']}" / "train" / "X.npy")[0, :, 17:]
    record("[info] notebook-v2 Cohen's d vs archive (expected to differ)",
           np.abs(C2[:, 4] - X0[:, 4]).max(), np.inf, "shows why COHENS_D_MODE='archive' is the default")
    folds.COHENS_D_MODE = "archive"

    rep = pd.DataFrame(rows)
    rep.to_csv(a.out, index=False)
    bad = rep[~rep.passed]
    print(f"\n{len(rep) - len(bad)}/{len(rep)} checks passed -> {a.out}")
    sys.exit(1 if len(bad) else 0)


if __name__ == "__main__":
    main()
