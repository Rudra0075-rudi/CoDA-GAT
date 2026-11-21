"""
Corrected figures 17-21 and the flagging_v2 stage-distribution panel, drawn
only from files in outputs/ and the historical flag files. Style constants are
copied from scripts/make_figures.py (that module cannot be imported under the
analysis-only guard) so the regenerated figures match the other 21.

Every bar carries a visible value label; the plotted numbers are also written
as CSV next to each figure.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import guard  # noqa: E402

REPO, OUT = guard.REPO, guard.OUT / "figures"
CONS = guard.OUT / "consensus_failures"

OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
CODAGAT = "#D55E00"
NEUTRAL = "#9AA0A6"
STAGES = ["F0", "F1", "F2", "F3", "F4"]
GROUPED = ["grouped_F0_F1", "grouped_F3_F4"]
GROUPED_LABEL = {"grouped_F0_F1": "F0–F1\n(grouped)", "grouped_F3_F4": "F3–F4\n(grouped)"}
FAMILY = {}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.prop_cycle": plt.cycler(color=OKABE),
})


def families():
    import consensus_failures as cf
    return {m: s["architecture_family"] for m, s in cf.MODEL_REGISTRY.items()}


def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()


def save(fig, tag, table=None):
    fig.savefig(OUT / f"{tag}.png")
    fig.savefig(OUT / f"{tag}.pdf")
    plt.close(fig)
    if table is not None:
        table.to_csv(OUT / f"{tag}.csv")


def label_bars(ax, bars, fmt="{:.0f}%", dy=0.6):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=7)


def stage_panel(df, flags, title, group_names):
    """Left: exact-stage shares (flagged vs rest). Right: grouped records, share flagged."""
    exact = (df["resolution"] == "exact").to_numpy()
    flags = np.asarray(flags, dtype=bool)
    g1, g2 = group_names
    tab = pd.DataFrame({g1: [int(((df["stage"] == s).to_numpy() & exact & flags).sum()) for s in STAGES],
                        g2: [int(((df["stage"] == s).to_numpy() & exact & ~flags).sum()) for s in STAGES]},
                       index=STAGES).T
    prop = tab.div(tab.sum(axis=1), axis=0) * 100
    gtab = pd.DataFrame({r: [int(((df["resolution"] == r).to_numpy() & flags).sum()),
                             int(((df["resolution"] == r).to_numpy() & ~flags).sum())] for r in GROUPED},
                        index=[g1, g2])

    fig, (ax, axg) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [5, 2]})
    x = np.arange(len(STAGES))
    for i, (grp, color) in enumerate([(g1, CODAGAT), (g2, NEUTRAL)]):
        bars = ax.bar(x + (i - 0.5) * 0.4, prop.loc[grp], 0.38, color=color,
                      label=f"{grp} (n={int(tab.loc[grp].sum())} exact-stage)")
        label_bars(ax, bars)
    ax.axvspan(0.5, 2.5, color=OKABE[2], alpha=0.10, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(STAGES)
    ax.set_xlabel("Recorded fibrosis stage (exact-stage records only)")
    ax.set_ylabel("Share of group (%)")
    ax.set_ylim(0, max(prop.max().max() * 1.18, 10))
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")

    share = gtab.loc[g1] / gtab.sum(axis=0) * 100
    bars = axg.bar(np.arange(len(GROUPED)), share, 0.55, color=CODAGAT)
    for b, r in zip(bars, GROUPED):
        axg.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                 f"{int(gtab.loc[g1, r])}/{int(gtab[r].sum())}", ha="center", va="bottom", fontsize=7)
    axg.set_xticks(np.arange(len(GROUPED))); axg.set_xticklabels([GROUPED_LABEL[r] for r in GROUPED])
    axg.set_ylabel(f"% of grouped records that are {g1.split(' (')[0]}")
    axg.set_ylim(0, max(share.max() * 1.25, 5))
    axg.set_title("Grouped labels\n(binary only; no exact stage)", fontsize=9)
    fig.tight_layout()
    out = pd.concat([tab.add_prefix("exact_"), gtab.add_prefix("n_")], axis=1)
    return fig, out


def fig17():
    d = pd.read_csv(CONS / "pairwise_agreement.csv")
    fams = families()
    models = sorted(set(d["model_a"]) | set(d["model_b"]), key=lambda m: (fams.get(m, "zz"), m))
    idx = {m: i for i, m in enumerate(models)}
    M = np.full((len(models), len(models)), np.nan)
    for _, r in d.iterrows():
        i, j = idx[r["model_a"]], idx[r["model_b"]]
        M[i, j] = M[j, i] = r["jaccard_observed"]
    np.fill_diagonal(M, 1.0)
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1)
    labs = [f"{m}  [{fams.get(m, '?')}]" for m in models]
    ax.set_xticks(range(len(models))); ax.set_xticklabels(labs, rotation=90, fontsize=6)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(labs, fontsize=6)
    prev = None
    for i, m in enumerate(models):
        if prev is not None and fams.get(m) != prev:
            ax.axhline(i - 0.5, color="white", lw=1.4); ax.axvline(i - 0.5, color="white", lw=1.4)
        prev = fams.get(m)
    ax.set_title(f"Error-set agreement across {len(models)} models (corrected baseline decisions)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, label="Jaccard of error sets")
    save(fig, "fig17_error_agreement_heatmap", pd.DataFrame(M, index=models, columns=models))


def fig18():
    d = pd.read_csv(CONS / "family_agreement_summary.csv")
    top = d[(d["level"] == "architecture_family") & (d["relation"] == "between")].copy()
    top["label"] = np.where(top["weighting"] == "pair", "all between-\nfamily pairs\n(model-pair\nweighted)",
                            "all between-\nfamily pairs\n(family-pair\nweighted)")
    pw = d[(d["level"] == "architecture_family_pairwise")].copy()
    pw["label"] = (pw["relation"].str.replace("tree_ensemble", "tree").str.replace(" vs ", "\nvs\n")
                   + "\n(" + pw["n_pairs"].astype(int).astype(str) + " pairs)")
    d = pd.concat([top, pw], ignore_index=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    series = [("mean_jaccard_observed", "observed", CODAGAT),
              ("mean_jaccard_expected_chance", "analytical independence expectation", NEUTRAL),
              ("mean_jaccard_perm_null_mean", "within-fold permutation null (mean)", OKABE[0])]
    for k, (col, lab, color) in enumerate(series):
        bars = ax.bar(x + (k - 1) * 0.27, d[col], 0.25, color=color, label=lab)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=6, rotation=90)
    ax.axvline(1.5, color="black", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(d["label"], fontsize=7.5, linespacing=1.1)
    ax.set_ylabel("Mean Jaccard of error sets")
    ax.set_ylim(0, 1.0)
    ax.set_title("Error agreement vs two different chance references (architecture families)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig18_family_agreement_vs_chance",
         d.set_index("label")[["weighting", "n_pairs"] + [s[0] for s in series]])


def fig19():
    em = pd.read_csv(CONS / "error_matrix.csv")
    models = [c for c in em.columns if c not in {"sample_id", "fold", "stage", "resolution", "y_true", "n_models_wrong"}]
    k = em[models].apply(as_bool).sum(axis=1).to_numpy()
    need = int(np.ceil(0.8 * len(models)))
    fig, tab = stage_panel(em, k >= need, f"Consensus failures (≥{need}/{len(models)} models wrong), "
                           f"n={int((k >= need).sum())} of {len(em)}",
                           ("consensus failure", "all other records"))
    save(fig, "fig19_consensus_failure_stages", tab)


def fig20():
    em = pd.read_csv(CONS / "error_matrix.csv")
    models = [c for c in em.columns if c not in {"sample_id", "fold", "stage", "resolution", "y_true", "n_models_wrong"}]
    rates = pd.Series({m: as_bool(em[m]).mean() for m in models}).sort_values()
    fams = families()
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.26 * len(rates))))
    colors = [CODAGAT if "codagat" in m else NEUTRAL for m in rates.index]
    bars = ax.barh(range(len(rates)), rates.values * 100, color=colors)
    for b in bars:
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=6.5)
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{m}  [{fams.get(m, '?')}]" for m in rates.index], fontsize=6.5)
    ax.set_xlabel(f"Error rate on the {len(em)} held-out records (%)")
    ax.set_title(f"Error rates: {rates.min() * 100:.1f}–{rates.max() * 100:.1f}% (corrected baseline decisions)")
    save(fig, "fig20_error_rate_all_models", rates.rename("error_rate").to_frame())


def fig21():
    lab = pd.read_parquet(REPO / "shared" / "labels.parquet")
    ta = pd.read_csv(REPO / "test_a_adjacency" / "per_sample_flags.csv")
    ta["resolution"] = lab.loc[ta["sample_id"], "resolution"].to_numpy()
    fig, tab = stage_panel(ta, as_bool(ta["flagged"]), f"Test A detector flags, n={int(as_bool(ta['flagged']).sum())} "
                           f"of {len(ta)}", ("flagged", "unflagged"))
    save(fig, "fig21_testA_flagged_stage_distribution", tab)


def flagging_v2_panel():
    lab = pd.read_parquet(REPO / "shared" / "labels.parquet")
    v2 = pd.read_csv(REPO / "flagging_v2" / "per_patient_flags.csv")
    v2["resolution"] = lab.loc[v2["sample_id"], "resolution"].to_numpy()
    meta = {"fold", "test_cohort", "sample_id", "stage", "resolution", "y_true", "p_f2plus", "uncertainty", "disagreement"}
    cols = [c for c in v2.columns if c not in meta]
    exact = v2[v2["resolution"] == "exact"]
    bg = exact["stage"].value_counts(normalize=True).reindex(STAGES, fill_value=0) * 100
    ncol, nrow = 3, int(np.ceil(len(cols) / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow), squeeze=False, sharey=True)
    rows, x = [], np.arange(len(STAGES))
    for ax, c in zip(axes.ravel(), cols):
        f = as_bool(exact[c])
        share = exact.loc[f, "stage"].value_counts(normalize=True).reindex(STAGES, fill_value=0) * 100
        ax.bar(x - 0.2, bg, 0.38, color=NEUTRAL, label="all exact-stage records")
        bars = ax.bar(x + 0.2, share, 0.38, color=CODAGAT, label="flagged")
        label_bars(ax, bars, dy=0.5)
        ax.axvspan(0.5, 2.5, color=OKABE[2], alpha=0.10, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(STAGES)
        ax.set_title(f"{c}\nn={int(f.sum())} exact of {int(as_bool(v2[c]).sum())} flagged", fontsize=8.5)
        rows.append({"variant": c, **{s: share[s] for s in STAGES}, "n_flagged_exact": int(f.sum())})
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("Share (%)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"flagging_v2: flagged stage distribution, exact-stage records only (n={len(exact)})", fontsize=11)
    fig.tight_layout()
    save(fig, "figS_flagging_v2_stage_distribution_exact", pd.DataFrame(rows).set_index("variant"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig17, fig18, fig19, fig20, fig21, flagging_v2_panel):
        fn()
        made.append(fn.__name__)
    print(f"[figures] made: {made}", flush=True)
    return made
