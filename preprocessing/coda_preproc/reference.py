"""Stage 6: GTEx liver reference and deviation targets (notebook Cells 27, 27b, 28, 31).

The six deviation variants are computed exactly as in the notebook. The
target the training scripts read (deviation_score) is the 'directional'
variant, fixed by notebook Cell 31 ("defer, don't decide").

Disclosure for the paper: the notebook's variant gate (evaluate/passes)
and its stage correlations use stage labels from the whole collection,
including the held-out cohorts. The gate output was not used to pick the
default target ('directional' failed the gate), but the variants were
explored with collection-wide labels.
"""
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

from .config import ECM_UP, HEP_DOWN, MIN_ECM, MIN_HEP, RES_EXACT


def load_gtex_liver(tpm_gct_gz, sample_attributes):
    """226 GTEx v8 liver samples -> log2(TPM+1), duplicate symbols by median."""
    att = pd.read_csv(sample_attributes, sep="\t", low_memory=False)
    liver = set(att.loc[att.SMTSD == "Liver", "SAMPID"].astype(str))
    chunks, keep = [], None
    for ch in pd.read_csv(tpm_gct_gz, sep="\t", skiprows=2, chunksize=4000, low_memory=False):
        if keep is None:
            keep = [c for c in ch.columns if c in liver]
        chunks.append(ch[["Name", "Description"] + keep])
    g = pd.concat(chunks, ignore_index=True)
    g["symbol"] = g["Description"].astype(str).str.strip()
    g = g[g.symbol.notna() & ~g.symbol.isin(["", "nan"])]
    vals = g.drop(columns=["Name", "Description"]).groupby("symbol").median()
    return np.log2(vals.clip(lower=0) + 1)


def rank_displacement(bulk_all, gtex):
    """Per-sample rank percentile minus the GTEx median-rank percentile."""
    genes = sorted(set(gtex.index) & set(bulk_all.index))
    gtex_rank = gtex.loc[genes].median(axis=1).rank(pct=True)
    return bulk_all.loc[genes].rank(axis=0, pct=True).sub(gtex_rank, axis=0)


def z_in_sample(d):
    return d.sub(d.mean(axis=0), axis=1).div(d.std(axis=0).replace(0, np.nan), axis=1).fillna(0.0)


def _std_within(s, groups):
    return s.groupby(groups).transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))


def deviation_variants(disp, meta):
    ecm = [g for g in ECM_UP if g in disp.index]
    hep = [g for g in HEP_DOWN if g in disp.index]
    if len(ecm) < MIN_ECM or len(hep) < MIN_HEP:
        raise ValueError(f"gene sets too sparse: ECM {len(ecm)}, hepatocyte {len(hep)}")
    info = meta.loc[disp.columns]
    raw = disp.abs().mean(axis=0)
    dirv = disp.loc[ecm].mean(axis=0) - disp.loc[hep].mean(axis=0)
    out = pd.DataFrame({
        "raw": raw,
        "platform_std": _std_within(raw, info.platform),
        "cohort_std": _std_within(raw, info.cohort),
        "directional": dirv,
        "directional_ps": _std_within(dirv, info.platform),
        "directional_cs": _std_within(dirv, info.cohort),
    })
    out["deviation_score"] = out["directional"]       # default Head-3 target
    return out, {"n_ecm": len(ecm), "n_hep": len(hep)}


def describe_variant(score, meta):
    """Exploratory stage association. Uses labels from every cohort: report as such."""
    d = meta.loc[score.index].assign(dev=score)
    ex = d[d.resolution.isin([RES_EXACT, "exact_ishak_mapped"]) & d.stage.notna()]
    rho, p = spearmanr(ex.stage.str[1].astype(float), ex.dev)
    grp = [ex.loc[ex.stage == s, "dev"].values for s in ["F0", "F1", "F2", "F3", "F4"]
           if (ex.stage == s).sum() > 3]
    return dict(rho=rho, p_rho=p, kruskal_p=kruskal(*grp).pvalue,
                **{f"mean_{s}": float(ex.loc[ex.stage == s, "dev"].mean())
                   for s in ["F0", "F1", "F2", "F3", "F4"]})
