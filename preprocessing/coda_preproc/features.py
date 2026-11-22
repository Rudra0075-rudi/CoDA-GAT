"""Stage 8: Group A (8 per-sample) and Group B (9 per-gene) node features
(notebook Cell 30, parts B and C). Verified bit-identical to the archive.

Disclosures for the paper:
- expression_z is a within-cohort z-score, so each held-out cohort is
  standardised with its own (unlabelled) distribution: transductive.
- expression_variability and cross_platform_consistency are computed over
  all 1,366 supervised samples, including held-out cohorts (no labels).
- pw_* for gene g is the leave-one-out mean of the other pathway members'
  expression_z. Masking gene g (graph_utility.apply_mask) replaces g's own
  features 0-7 but leaves g's contribution inside its neighbours' pw_*
  features, so masking is feature perturbation, not gene removal.
"""
import numpy as np
import pandas as pd

from .config import FEATURES_A, FEATURES_B, PATHWAYS
from .guards import assert_tensor_integrity


def within_cohort_z(bulk_sup, sup, panel):
    z = bulk_sup.loc[panel].copy()
    for c in sup.cohort.unique():
        cols = [x for x in sup.index[sup.cohort == c] if x in z.columns]
        blk = z[cols]
        z[cols] = blk.sub(blk.mean(axis=1), axis=0).div(blk.std(axis=1).replace(0, np.nan), axis=0)
    return z.fillna(0.0)


def pathway_loo(z, panel):
    N, P = z.shape[1], len(panel)
    gidx = {g: i for i, g in enumerate(panel)}
    pw = np.zeros((N, P, len(PATHWAYS)), dtype=np.float32)
    for k, gs in enumerate(PATHWAYS.values()):
        mem = [g for g in gs if g in gidx]
        if not mem:
            continue
        tot = z.loc[mem].values.sum(axis=0)
        for g in mem:
            pw[:, gidx[g], k] = (tot - z.loc[g].values) / max(len(mem) - 1, 1)
    return pw


def _minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def static_features(bulk_sup, sup, panel, topo, gtex):
    pe = bulk_sup.loc[panel]
    gl = pd.Series({g: float(gtex.loc[g].mean()) if g in gtex.index else np.nan for g in panel})
    gs = pd.Series({g: float(gtex.loc[g].std()) if g in gtex.index else np.nan for g in panel})
    rna = [c for c in pe.columns if sup.loc[c, "platform"] == "rnaseq"]
    mic = [c for c in pe.columns if sup.loc[c, "platform"] == "microarray"]
    xplat = 1.0 - (pe[rna].mean(axis=1).rank() - pe[mic].mean(axis=1).rank()).abs() / len(panel)
    topo = topo.loc[panel]
    return pd.DataFrame({
        "pathway_membership_count": topo.pathway_membership_count.values,
        "degree_centrality": topo.degree_centrality.values,
        "betweenness_centrality": topo.betweenness_centrality.values,
        "closeness_centrality": topo.closeness_centrality.values,
        "clustering_coefficient": topo.clustering_coefficient.values,
        "gtex_expression_level": _minmax(gl.fillna(gl.median())).loc[panel].values,
        "gtex_expression_std": _minmax(gs.fillna(gs.median())).loc[panel].values,
        "expression_variability": _minmax(pe.std(axis=1) / (pe.mean(axis=1).abs() + 1e-9)).loc[panel].values,
        "cross_platform_consistency": xplat.loc[panel].values,
    }, index=panel)[FEATURES_B]


def build_tensor_AB(bulk_sup, sup, panel, topo, gtex, dev_per_gene):
    """Return (tensor (N, 72, 17), sample order, expression_z frame)."""
    z = within_cohort_z(bulk_sup, sup, panel)
    N, P = z.shape[1], len(panel)
    A = np.zeros((N, P, len(FEATURES_A)), dtype=np.float32)
    A[:, :, 0] = z.values.T
    A[:, :, 1] = dev_per_gene.loc[panel, z.columns].values.T
    A[:, :, 2:] = pathway_loo(z, panel)
    B = static_features(bulk_sup, sup, panel, topo, gtex).values.astype(np.float32)
    T = np.concatenate([A, np.broadcast_to(B[None], (N, P, B.shape[1]))], axis=2)
    assert_tensor_integrity(T, "TENSOR_AB")
    return T, list(z.columns), z
