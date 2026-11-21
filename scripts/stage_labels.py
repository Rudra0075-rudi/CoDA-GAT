"""
CoDA-GAT — stage-label helpers shared by the adjacency / consensus analyses.

Torch-free on purpose: the analyses that only read saved predictions import
this module instead of test_a_adjacency.py, which pulls in the model and
training code.

WHY THIS EXISTS (2026-11 audit). labels.parquet stores grouped histology
under a single stage code:
  resolution == "grouped_F0_F1"  (137 records) -> stage "F0"
  resolution == "grouped_F3_F4"  (127 records) -> stage "F3"
Their binary F2+ label is valid, but their exact stage is not known, so they
cannot enter any test that asks WHICH stage a record has. In particular a
grouped F0-F1 record may be an F1 (adjacent to the F1/F2 boundary) and was
being counted as a distant F0. Stage-level enrichment is therefore computed
on the exact-stage subset only; grouped records are reported separately and
kept in every binary-label analysis.
"""

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, fisher_exact

ADJACENT_STAGES = {"F1", "F2"}
ALL_STAGES = ["F0", "F1", "F2", "F3", "F4"]
EXACT = "exact"
GROUPED_RESOLUTIONS = ["grouped_F0_F1", "grouped_F3_F4"]


def exact_stage_mask(df):
    """True where the recorded stage is an exact F0-F4 stage. Refuses to guess
    when the resolution column is missing: a stage column alone cannot tell an
    exact F0 from a grouped F0-F1."""
    if "resolution" not in df.columns:
        raise KeyError("stage-level analysis needs a 'resolution' column "
                       "(from shared/labels.parquet); refusing to treat grouped stages as exact")
    return (df["resolution"] == EXACT).to_numpy()


def enrichment_test(df, flagged, exact_only=True):
    """2x5 chi-square and one-sided 2x2 Fisher (flagged at F1/F2 vs F0/F3/F4).

    exact_only=True (the default) restricts both tests to exact-stage records.
    exact_only=False reproduces the historical behaviour, in which grouped
    records are counted under their stored stage code; use it only to show
    what was previously reported.
    """
    df = df.copy()
    df["flagged"] = np.asarray(flagged, dtype=bool)
    if exact_only:
        df = df[exact_stage_mask(df)]
    df["adjacency_class"] = np.where(df["stage"].isin(ADJACENT_STAGES), "adjacent", "distant")

    # descriptive: full 2x5 (flagged/unflagged x F0..F4)
    counts_5 = {stage: [int(((df["stage"] == stage) & df["flagged"]).sum()),
                         int(((df["stage"] == stage) & ~df["flagged"]).sum())]
                for stage in ALL_STAGES}
    table_5 = pd.DataFrame(counts_5, index=["flagged", "unflagged"])
    try:
        chi2, p_chi2, _, _ = chi2_contingency(table_5.loc[:, table_5.sum(axis=0) > 0].to_numpy())
    except ValueError:
        chi2, p_chi2 = float("nan"), float("nan")

    # primary: 2x2 Fisher's exact (adjacent F1/F2 vs distant F0/F3/F4), one-sided
    n_flag_adj = int(((df["adjacency_class"] == "adjacent") & df["flagged"]).sum())
    n_flag_dist = int(((df["adjacency_class"] == "distant") & df["flagged"]).sum())
    n_unflag_adj = int(((df["adjacency_class"] == "adjacent") & ~df["flagged"]).sum())
    n_unflag_dist = int(((df["adjacency_class"] == "distant") & ~df["flagged"]).sum())
    table_2 = np.array([[n_flag_adj, n_flag_dist], [n_unflag_adj, n_unflag_dist]])
    odds_ratio, p_fisher = fisher_exact(table_2, alternative="greater")

    return table_5, (chi2, p_chi2), table_2, (float(odds_ratio), float(p_fisher))


def grouped_table(df, flagged):
    """Flagged/unflagged counts for the grouped-label records, which the
    exact-stage tests leave out. Columns are resolutions, not stages."""
    flagged = np.asarray(flagged, dtype=bool)
    counts = {res: [int(((df["resolution"] == res).to_numpy() & flagged).sum()),
                    int(((df["resolution"] == res).to_numpy() & ~flagged).sum())]
              for res in GROUPED_RESOLUTIONS}
    return pd.DataFrame(counts, index=["flagged", "unflagged"])
