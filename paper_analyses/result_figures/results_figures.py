"""CoDA-GAT result figures (fig_p1 ... fig_p26), redrawn from the deposited outputs.

Every plotted number is read or recomputed from files in the code repository
(with the Zenodo data restored into shared/ and folds/) or from the analysis
files in paper_analyses/additional_analyses/. Nothing is typed in by
hand. The numbers behind each figure are written to figure_data/fig_pN_data.csv.

usage (from the repository root, with the Zenodo data restored):
    python paper_analyses/result_figures/results_figures.py . \
        paper_analyses/additional_analyses OUTDIR [N ...]
N limits the run to the listed figure numbers.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from style import *  # noqa: F401,F403
import style as S
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = Path(sys.argv[1])
ESMA = Path(sys.argv[2])
OUT = Path(sys.argv[3])
ONLY = [int(a) for a in sys.argv[4:]]
FIGDATA = OUT / "figure_data"
OUT.mkdir(parents=True, exist_ok=True)
FIGDATA.mkdir(parents=True, exist_ok=True)

SHARED, B7, B9 = REPO / "shared", REPO / "results_b7", REPO / "ablations_b9"
BASE, GUTIL, LEARNED = REPO / "baselines", REPO / "graph_utility", REPO / "learned_graph"
ATTN, VER = REPO / "attention_analysis_b9", REPO / "verification_checks"
B5, B8, TB, TC = REPO / "safety_checks_b5", REPO / "decomposition_b8", REPO / "test_b_rate", REPO / "test_c_cleaning"
CORR = REPO / "corrections_2026-11" / "outputs" / "figures"
FIGIN = Path(__file__).resolve().parent / "figure_data_in"   # reference values; the script stops if its recomputation differs

REG = {}


def fig(n):
    def deco(fn):
        REG[n] = fn
        return fn
    return deco


def fold_cohort():
    fa = pd.read_csv(SHARED / "fold_assignment_lodo.csv")
    return dict(zip(fa["fold"], fa["test_cohort"]))


def t_ci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return m, m - h, m + h


def labels_df():
    lab = pd.read_parquet(SHARED / "labels.parquet")
    return lab[lab["is_supervised"]].copy()


def cohort_order():
    lab = labels_df()
    return lab["cohort"].value_counts().sort_values(ascending=False).index.tolist()


T1 = None


def tier1():
    global T1
    if T1 is None:
        T1 = set(fold_cohort().values())
    return T1


def ylabels_with_tier(ax, names):
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for t, c in zip(ax.get_yticklabels(), names):
        t.set_color(S.INK if c in tier1() else S.GREY)


# ==========================================================================
# Evaluation set (p1-p3): one cohort order, tier 1 in colour, tier 2 in grey
# ==========================================================================
@fig(1)
def p1():
    lab = labels_df()
    order = cohort_order()
    n = lab["cohort"].value_counts().reindex(order)
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.55))
    y = np.arange(len(order))
    cols = [S.BLUE if c in tier1() else S.GREY_L for c in order]
    ax.barh(y, n.values, color=cols, height=0.68, zorder=2)
    for yi, v in zip(y, n.values):
        ax.text(v + 3, yi, f"{v}", va="center", fontsize=6.6, color=S.INK)
    ylabels_with_tier(ax, order)
    ax.invert_yaxis()
    ax.set_xlabel("Staged samples (n)")
    ax.set_xlim(0, n.max() * 1.1)
    only_bottom(ax)
    ax.legend(handles=[Patch(color=S.BLUE, label=f"Tier 1: held out in turn ({int(n[[c for c in order if c in tier1()]].sum()):,} samples)"),
                       Patch(color=S.GREY_L, label=f"Tier 2: training only ({int(n[[c for c in order if c not in tier1()]].sum()):,} samples)")],
              loc="lower right", title=f"Total {int(n.sum()):,}", title_fontsize=6.8, alignment="left")
    d = pd.DataFrame({"cohort": order, "n": n.values, "tier1": [c in tier1() for c in order]})
    return f, d


@fig(2)
def p2():
    lab = labels_df()
    order = cohort_order()
    elig = lab[lab["y_T1_significant"].notna()]
    share = elig.groupby("cohort")["y_T1_significant"].mean().reindex(order) * 100
    neli = elig.groupby("cohort").size().reindex(order)
    pooled = elig["y_T1_significant"].mean() * 100
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.55))
    y = np.arange(len(order))
    cols = [S.BLUE if c in tier1() else S.GREY_L for c in order]
    ax.barh(y, share.values, color=cols, height=0.68, zorder=2)
    for yi, v in zip(y, share.values):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=6.6)
    ax.axvline(pooled, color=S.INK, lw=0.7, ls=(0, (3, 2)), zorder=3)
    ax.text(pooled, -0.95, f"pooled {pooled:.1f}%", ha="center", va="bottom", fontsize=6.6)
    ylabels_with_tier(ax, order)
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.5, -1.2)
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("F2+ share of eligible samples (%)")
    only_bottom(ax)
    ax.legend(handles=[Patch(color=S.BLUE, label="Tier 1 (held out)"), Patch(color=S.GREY_L, label="Tier 2 (training only)")],
              loc="lower right")
    d = pd.DataFrame({"cohort": order, "f2plus_pct": share.values, "n_eligible": neli.values})
    return f, d


@fig(3)
def p3():
    lab = labels_df()
    order = cohort_order()
    cat = np.select([lab["resolution"] == "exact",
                     lab["resolution"].isin(["grouped_F0_F1", "grouped_F3_F4"]),
                     lab["resolution"] == "grouped_F0_F2"],
                    ["exact", "grouped", "grouped_F0_F2"], "other")
    assert (cat != "other").all()
    tab = pd.crosstab(lab["cohort"], cat).reindex(order).fillna(0).astype(int)
    for c in ("exact", "grouped", "grouped_F0_F2"):
        if c not in tab:
            tab[c] = 0
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.55))
    y = np.arange(len(order))
    left = np.zeros(len(order))
    spec = [("exact", S.BLUE, None, "Exact stage"),
            ("grouped", S.BLUE_L, None, "Grouped range (F0–F1 or F3–F4)"),
            ("grouped_F0_F2", "white", "////", "Grouped F0–F2 (not eligible)")]
    for col, c, h, lab_ in spec:
        v = tab[col].values
        keep = v > 0
        ax.barh(y[keep], v[keep], left=left[keep], color=c, height=0.68, hatch=h, edgecolor=S.GREY if h else c,
                lw=0.5 if h else 0, label=lab_, zorder=2)
        left += v
    for yi, (e, g, g2) in enumerate(tab[["exact", "grouped", "grouped_F0_F2"]].values):
        parts = [f"{e}"] + ([f"{g}"] if g else []) + ([f"{g2}"] if g2 else [])
        ax.text(e + g + g2 + 3, yi, " + ".join(parts), va="center", fontsize=6.4)
    ylabels_with_tier(ax, order)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.18)
    ax.set_xlabel("Staged samples (n)")
    only_bottom(ax)
    ax.legend(loc="lower right", title="Bar labels: exact + grouped", title_fontsize=6.4, alignment="left")
    test = tab.loc[[c for c in order if c in tier1()]]
    d = tab.reset_index().rename(columns={"index": "cohort", "row_0": "cohort"})
    d.attrs["test_exact"] = int(test["exact"].sum())
    print(f"   p3 check: test exact={int(test['exact'].sum())}, test grouped={int(test['grouped'].sum())}")
    return f, d


# ==========================================================================
# T1 performance (p4-p6)
# ==========================================================================
def per_sample_auc():
    rows = []
    for d in sorted((BASE / "per_sample_predictions").iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for fcsv in sorted(d.glob("fold*_*.csv")):
            t = pd.read_csv(fcsv)
            rows.append({"model": d.name, "fold": int(t["fold"].iloc[0]),
                         "auc": roc_auc_score(t["y_true"].astype(int), t["p_test"].astype(float))})
    return pd.DataFrame(rows)


def codagat_variant_folds():
    out = {}
    b7 = pd.read_csv(B7 / "master_results.csv")
    for mode in ("off", "marginal", "conditional"):
        out[f"b7_m0_{mode}"] = b7[(b7["mode"] == mode) & (b7["m"] == 0)].groupby("fold")["auc"].mean()
    for m in (1, 3):
        r = pd.read_csv(REPO / f"results_b7_m{m}" / "master_results.csv")
        r = r[r["mode"] == "off"] if "mode" in r else r
        out[f"b7_m{m}_off"] = r.groupby("fold")["auc"].mean()
    a = pd.read_csv(B9 / "master_ablation_results.csv")
    for ab, g in a.groupby("ablation"):
        out[f"b9_{ab}"] = g.groupby("fold")["auc"].mean()
    return out


@fig(4)
def p4():
    ref = pd.read_csv(FIGIN / "fig_p4_data.csv")
    ps = per_sample_auc()
    var = codagat_variant_folds()
    rec = {}
    for m, g in ps.groupby("model"):
        if g["fold"].nunique() == 8:
            rec[m] = (g["auc"].mean(), g["auc"].std(ddof=1))
    for k, s in var.items():
        rec[k] = (s.mean(), s.std(ddof=1))
    d = ref.copy()
    d["mean_recomputed"] = [rec.get(m, (np.nan, np.nan))[0] for m in d["model"]]
    d["sd_recomputed"] = [rec.get(m, (np.nan, np.nan))[1] for m in d["model"]]
    diff = (d["mean"] - d["mean_recomputed"]).abs().max()
    print(f"   p4 check: max |figure_data - recomputed| mean = {diff:.2e}, sd = {(d['sd_between_cohort_ddof1'] - d['sd_recomputed']).abs().max():.2e}")
    assert diff < 1e-9
    d = d.sort_values("mean").reset_index(drop=True)

    def group(m):
        if m == "b7_m0_off":
            return "primary"
        return "graph" if m.startswith("b7") or m.startswith("b9") else "tabular"

    d["group"] = [group(m) for m in d["model"]]
    col = {"primary": S.BLUE, "graph": S.BLUE_L, "tabular": S.GREEN}
    f, ax = plt.subplots(figsize=(S.TEXTW, 4.0))
    y = np.arange(len(d))
    prim = d.loc[d["group"] == "primary", "mean"].iloc[0]
    ax.axvline(prim, color=S.BLUE, lw=0.6, ls=(0, (3, 2)), zorder=0)
    for yi, r in d.iterrows():
        c = col[r["group"]]
        ax.plot([r["mean"] - r["sd_between_cohort_ddof1"], r["mean"] + r["sd_between_cohort_ddof1"]], [yi, yi],
                color=c if r["group"] != "graph" else S.BLUE2, lw=0.9, alpha=0.55, solid_capstyle="butt", zorder=1)
        ax.plot(r["mean"], yi, "o", ms=4.4 if r["group"] == "primary" else 3.6, color=c,
                mec=S.BLUE2 if r["group"] == "graph" else c, mew=0.6, zorder=3)
        ax.text(0.925, yi, f"{r['mean']:.3f}", va="center", ha="left", fontsize=6.4,
                fontweight="bold" if r["group"] == "primary" else "normal",
                color=S.BLUE if r["group"] == "primary" else S.INK)
    ax.set_yticks(y)
    short = (d["label"].str.replace("CoDA-GAT (primary, adversary off)", "CoDA-GAT (primary)", regex=False)
             .str.replace("72-gene panel + deviation + pathway", "Panel + dev. + pathway", regex=False)
             .str.replace("72-gene panel + deviation", "Panel + dev.", regex=False)
             .str.replace("top-1,000 genes", "Top-1,000 genes", regex=False)
             .str.replace("logistic regression", "logistic reg.", regex=False))
    d["short_label"] = short
    ax.set_yticklabels(short)
    for t, g in zip(ax.get_yticklabels(), d["group"]):
        if g == "primary":
            t.set_fontweight("bold"); t.set_color(S.BLUE)
    ax.set_xlim(0.70, 0.95)
    ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90])
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("Mean AUROC over eight held-out cohorts (line: ± between-cohort SD)")
    only_bottom(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.BLUE, label="CoDA-GAT (primary)"),
                       Line2D([], [], marker="o", ls="", color=S.BLUE_L, mec=S.BLUE2, label="Variants and ablations"),
                       Line2D([], [], marker="o", ls="", color=S.GREEN, label="Tabular models")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, columnspacing=1.0, handletextpad=0.2)
    return f, d


NICE = {
    "panel72_full__catboost": "Panel + dev. + pathway / CatBoost",
    "panel72_combined__catboost": "Panel + dev. / CatBoost",
    "panel72_full__logreg": "Panel + dev. + pathway / logistic reg.",
    "panel72_combined__logreg": "Panel + dev. / logistic reg.",
    "panel72_full__random_forest": "Panel + dev. + pathway / random forest",
    "panel72_combined__random_forest": "Panel + dev. / random forest",
    "panel72_full__xgboost": "Panel + dev. + pathway / XGBoost",
    "panel72_combined__xgboost": "Panel + dev. / XGBoost",
    "bulk_top1000__catboost": "Top-1,000 genes / CatBoost",
    "bulk_top1000__xgboost": "Top-1,000 genes / XGBoost",
    "bulk_top1000__random_forest": "Top-1,000 genes / random forest",
    "bulk_top1000__logreg": "Top-1,000 genes / logistic reg.",
    "boruta_lasso_random_forest": "Boruta–LASSO / random forest",
    "boruta_lasso_xgboost": "Boruta–LASSO / XGBoost",
    "CoDA-GAT": "CoDA-GAT (primary)",
    "abl: empty_graph": "Ablation: disconnected graph",
    "abl: random_graph": "Ablation: random directed graph",
    "abl: shuffled_graph": "Ablation: feature-permuted STRING",
    "abl: no_pathway_features": "Ablation: no pathway features",
    "abl: no_fold_fitted_features": "Ablation: no fold-fitted features",
    "abl: no_h3_deviation": "Ablation: no deviation head",
}


@fig(5)
def p5():
    cl = fold_cohort()
    ps = per_sample_auc()
    b7 = pd.read_csv(B7 / "master_results.csv")
    rows = {"CoDA-GAT": b7[(b7["mode"] == "off") & (b7["m"] == 0)].groupby("fold")["auc"].mean()}
    for m, g in ps.groupby("model"):
        rows[m] = g.groupby("fold")["auc"].mean()
    a9 = pd.read_csv(B9 / "master_ablation_results.csv")
    for m, g in a9.groupby("ablation"):
        rows[f"abl: {m}"] = g.groupby("fold")["auc"].mean()
    single = {}
    for stage, key in (("tabular", "model"), ("hybrids", "model"), ("graph_controls", "control")):
        p = BASE / stage / "all_folds.csv"
        if not p.exists():
            continue
        dd = pd.read_csv(p)
        if key not in dd.columns:
            continue
        for m, g in dd.groupby(key):
            if str(m) in rows:
                continue
            single[str(m)] = g.groupby("fold")["auc"].mean()
    full = pd.DataFrame(rows).T
    full = full.loc[full.mean(axis=1).sort_values(ascending=False).index]
    part = pd.DataFrame(single).T.reindex(columns=full.columns)
    part = part.loc[part.mean(axis=1).sort_values(ascending=False).index]
    print("   p5 single-cohort rows:", {k: (v.index.tolist(), round(float(v.iloc[0]), 3)) for k, v in single.items()})

    SINGLE_NICE = {"empty_graph": "Disconnected graph", "flattened_mlp": "MLP, flattened features (no graph)",
                   "fully_connected": "Fully connected graph", "graphsage_supcon": "GraphSAGE re-implementation",
                   "knowledgegcn_settransformer": "Graph-convolution re-implementation",
                   "random_graph": "Random graph", "real_graph": "STRING graph", "shuffled_graph": "Feature-permuted STRING"}
    names = [NICE.get(m, m) for m in full.index] + [SINGLE_NICE.get(m, m) + " †" for m in part.index]
    mat = pd.concat([full, part])
    mat.columns = [cl[c] for c in mat.columns]
    nfull = len(full)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#FFFFFF", S.BLUE_XL, S.BLUE_L, S.BLUE2, S.BLUE])
    cmap.set_bad("white")
    f = plt.figure(figsize=(S.TEXTW, 5.6))
    gap = 1.6
    ypos = np.r_[np.arange(nfull), np.arange(len(part)) + nfull + gap]
    ax = f.add_axes([0.37, 0.07, 0.56, 0.86])
    vmin, vmax = 0.70, 0.95
    for i, yy in enumerate(ypos):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                c = cmap((np.clip(v, vmin, vmax) - vmin) / (vmax - vmin))
                ax.add_patch(plt.Rectangle((j - 0.5, yy - 0.5), 1, 1, facecolor=c, edgecolor="white", lw=0.6))
                ax.text(j, yy, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v >= 0.865 else S.INK)
    for yy in np.r_[ypos]:
        pass
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(ypos[-1] + 0.5, -0.5)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=40, ha="left", fontsize=6.2)
    ax.xaxis.tick_top()
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=6.2)
    for t, m in zip(ax.get_yticklabels(), list(full.index) + list(part.index)):
        if m == "CoDA-GAT":
            t.set_fontweight("bold"); t.set_color(S.BLUE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(-0.45, nfull - 0.5 + (gap + 1) / 2, "† evaluated on one cohort (GSE130970) only; not ranked", fontsize=6.2,
            color=S.GREY, va="center", ha="left", style="italic")
    cax = f.add_axes([0.945, 0.40, 0.015, 0.30])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = f.colorbar(sm, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.set_label("AUROC", fontsize=6.5)
    d = mat.copy()
    d.index = names
    d["eight_cohort"] = [True] * nfull + [False] * len(part)
    return f, d.reset_index().rename(columns={"index": "model"})


@fig(6)
def p6():
    cl = fold_cohort()
    f, axes = plt.subplots(2, 4, figsize=(S.TEXTW, 3.05), sharex=True, sharey=True)
    rows = []
    for k, (fold, coh) in enumerate(sorted(cl.items())):
        fd = next((REPO / "folds").glob(f"fold{fold}_*"))
        y = pd.read_csv(fd / "test" / "y.csv", index_col=0)["y_T1_significant"].to_numpy().astype(int)
        probs = [np.load(s / "probs_test.npy")[:, 1] for s in sorted((B7 / f"fold{fold}_{coh}" / "mode_off").glob("seed_*"))]
        assert len(probs) == 5, (fold, len(probs))
        p = np.mean(probs, axis=0)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        rows.append({"fold": fold, "cohort": coh, "n": len(y), "n_pos": int(y.sum()), "auc_seed_avg_probs": auc})
        ax = axes.flat[k]
        ax.plot([0, 1], [0, 1], color=S.GREY_L, lw=0.6, ls=(0, (3, 2)))
        ax.fill_between(fpr, 0, tpr, step="post", color=S.BLUE_XL, lw=0)
        ax.step(fpr, tpr, where="post", color=S.BLUE, lw=1.1)
        ax.set_title(coh, fontsize=6.8, fontweight="normal", pad=2)
        ax.text(0.97, 0.05, f"AUROC {auc:.3f}\nn = {len(y)}", ha="right", va="bottom", fontsize=6.2, transform=ax.transAxes)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticklabels(["0", "0.5", "1"])
    f.supxlabel("False positive rate", fontsize=7.5, y=0.01)
    f.supylabel("True positive rate", fontsize=7.5, x=0.06)
    d = pd.DataFrame(rows)
    print(f"   p6 check: mean per-cohort AUROC of seed-averaged probabilities = {d['auc_seed_avg_probs'].mean():.4f}, sd = {d['auc_seed_avg_probs'].std(ddof=1):.4f}")
    f.subplots_adjust(left=0.11, right=0.99, bottom=0.11, top=0.94, wspace=0.12, hspace=0.32)
    return f, d


# ==========================================================================
# T3 (p7)
# ==========================================================================
@fig(7)
def p7():
    cl = fold_cohort()
    b7 = pd.read_csv(B7 / "master_results.csv")
    b7 = b7[b7["m"] == 0]
    per = b7.groupby(["mode", "fold"])["macro_f1"].mean().unstack(0)[["off", "marginal", "conditional"]]
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.6))
    x = np.arange(3)
    for fold, r in per.iterrows():
        ax.plot(x, r.values, color=S.GREY_L, lw=0.7, zorder=1)
        ax.plot(x, r.values, "o", ms=2.8, color=S.GREY, zorder=2)
    m = per.mean()
    ax.plot(x, m.values, color=S.BLUE, lw=1.6, zorder=3)
    ax.plot(x, m.values, "o", ms=5.5, color=S.BLUE, zorder=4)
    for xi, v in zip(x, m.values):
        ax.text(xi + 0.07, v - 0.004, f"{v:.3f}", ha="left", va="top", fontsize=6.6, color=S.BLUE, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels(["Off", "Marginal", "Class-\nconditional"])
    ax.set_xlim(-0.3, 2.45)
    ax.set_ylabel("Macro-F1 (mean of five seeds)")
    ax.set_xlabel("Adversary mode")
    ax.legend(handles=[Line2D([], [], color=S.GREY_L, marker="o", mfc=S.GREY, mec=S.GREY, ms=2.8, lw=0.7, label="One held-out cohort"),
                       Line2D([], [], color=S.BLUE, marker="o", ms=5, lw=1.6, label="Mean of eight cohorts")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    d = per.copy()
    d.index = [cl[i] for i in d.index]
    d.loc["mean"] = m
    print("   p7 means:", m.round(4).to_dict())
    return f, d.reset_index().rename(columns={"index": "cohort"})


# ==========================================================================
# T4 conformal (p8-p10)
# ==========================================================================
def aps_lac():
    d = pd.read_csv(VER / "aps_diagnostic_per_run.csv")
    recs = []
    for alpha, g in d.groupby("alpha"):
        for key, cov, size in (("LAC", "lac_coverage", "lac_set_size"),
                               ("APS (corrected)", "aps_coverage_corrected", "aps_set_size_corrected"),
                               ("APS (original)", "aps_coverage_original", "aps_set_size_original")):
            recs.append({"alpha": alpha, "method": key, "n_runs": len(g),
                         "coverage": g[cov].mean(), "coverage_sd": g[cov].std(),
                         "set_size": g[size].mean(), "set_size_sd": g[size].std()})
    return pd.DataFrame(recs)


CONF = {"LAC": (S.BLUE, "o", -1), "APS (corrected)": (S.GREEN, "s", 0), "APS (original)": (S.RED, "^", 1)}
CONF_LABEL = {"LAC": "LAC", "APS (corrected)": "APS, corrected", "APS (original)": "APS, original (no randomisation)"}


@fig(8)
def p8():
    t = aps_lac()
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.45))
    ax.plot([0.66, 1.02], [0.66, 1.02], color=S.GREY_L, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
    for meth, g in t.groupby("method", sort=False):
        c, mk, k = CONF[meth]
        g = g.sort_values("alpha")
        x = 1 - g["alpha"].values + k * 0.007
        ax.errorbar(x, g["coverage"], yerr=g["coverage_sd"], fmt=mk + "-", color=c, ms=3.8, lw=0.9,
                    elinewidth=0.8, mfc=c if meth != "APS (original)" else "white", mew=0.9, label=CONF_LABEL[meth], zorder=3)
    ax.set_xlim(0.76, 0.99); ax.set_ylim(0.68, 1.03)
    ax.set_xticks([0.80, 0.90, 0.95]); ax.set_xticklabels(["80%", "90%", "95%"])
    ax.set_yticks([0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["70%", "80%", "90%", "100%"])
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Achieved coverage")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    return f, t


@fig(9)
def p9():
    t = aps_lac()
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.45))
    ax.axhline(2.0, color=S.GREY_L, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
    ax.text(0.15, 2.01, "set contains both classes", fontsize=6.2, color=S.GREY, ha="center", va="bottom")
    for meth, g in t.groupby("method", sort=False):
        c, mk, k = CONF[meth]
        g = g.sort_values("alpha")
        ax.errorbar(g["alpha"].values + k * 0.004, g["set_size"], yerr=g["set_size_sd"], fmt=mk + "-", color=c, ms=3.8, lw=0.9,
                    elinewidth=0.8, mfc=c if meth != "APS (original)" else "white", mew=0.9, label=CONF_LABEL[meth], zorder=3)
    ax.set_xticks([0.05, 0.10, 0.20]); ax.set_xticklabels(["0.05", "0.10", "0.20"])
    ax.set_xlim(0.03, 0.215)
    ax.set_ylim(0.95, 2.1)
    ax.set_xlabel("Miscoverage level α")
    ax.set_ylabel("Mean prediction-set size")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
    return f, t


@fig(10)
def p10():
    cc = pd.read_csv(ESMA / "conformal_classcond.csv")
    mo = pd.read_csv(ESMA / "conformal_mondrian.csv")
    ver = pd.read_csv(VER / "aps_diagnostic_per_run.csv")
    cl = fold_cohort()
    cc["fold_i"] = cc["fold"].str.extract(r"fold(\d+)").astype(int)
    mo["fold_i"] = mo["fold"].str.extract(r"fold(\d+)").astype(int)
    # cross-check: LAC coverage in the class-conditional file equals the deposited verification run
    chk = cc.assign(seed_i=cc["seed"].str.extract(r"(\d+)").astype(int)).merge(
        ver, left_on=["fold_i", "seed_i", "alpha"], right_on=["fold", "seed", "alpha"])
    print(f"   p10 check: {len(chk)} runs matched, max |cov - lac_coverage| = {(chk['cov'] - chk['lac_coverage']).abs().max():.2e}")
    lac = cc.groupby(["fold_i", "alpha"])[["cov", "cov_neg", "cov_pos", "pi_test", "pi_cal"]].mean().reset_index()
    mon = mo[np.isclose(mo["alpha"], 0.1)].groupby("fold_i")[["cov", "cov_neg", "cov_pos"]].mean()
    mon["worst"] = mon[["cov_neg", "cov_pos"]].min(axis=1)
    folds = sorted(cl)
    names = [cl[k] for k in folds]
    y = np.arange(len(folds))

    f, axes = plt.subplots(1, 3, figsize=(S.TEXTW, 2.75), sharey=True, gridspec_kw={"width_ratios": [1.25, 1, 1]})
    # (a) LAC at three nominal levels
    ax = axes[0]
    lv = {0.05: ("95%", S.BLUE), 0.10: ("90%", S.BLUE2), 0.20: ("80%", S.BLUE_L)}
    for a, (lab_, c) in lv.items():
        ax.axvline(1 - a, color=c, lw=0.7, ls=(0, (3, 2)), zorder=0)
        v = lac[np.isclose(lac["alpha"], a)].set_index("fold_i").reindex(folds)["cov"].values
        ax.plot(v, y, "o", ms=4.2, color=c, mec="white", mew=0.3, label=lab_, zorder=3)
    for i, fo in enumerate(folds):
        vs = lac[lac["fold_i"] == fo]["cov"].values
        ax.plot([vs.min(), vs.max()], [i, i], color=S.GREY_XL, lw=2.2, zorder=1, solid_capstyle="round")
    ax.set_xlim(0.64, 1.01)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(["70%", "80%", "90%", "100%"])
    ax.set_title("LAC, three nominal levels", loc="left", pad=13)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.1)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.invert_yaxis()
    only_bottom(ax)
    # (b) LAC 90% by true class
    ax = axes[1]
    l90 = lac[np.isclose(lac["alpha"], 0.10)].set_index("fold_i").reindex(folds)
    ax.axvspan(0.5, 0.85, color=S.RED_XL, alpha=0.45, lw=0, zorder=0)
    ax.axvline(0.90, color=S.BLUE2, lw=0.7, ls=(0, (3, 2)), zorder=0)
    for i in range(len(folds)):
        ax.plot([l90["cov_neg"].values[i], l90["cov_pos"].values[i]], [i, i], color=S.GREY_L, lw=0.8, zorder=1)
    ax.plot(l90["cov_neg"], y, "o", ms=3.8, mfc="white", mec=S.BLUE, mew=1.0, label="F0–F1", zorder=3)
    ax.plot(l90["cov_pos"], y, "o", ms=3.8, color=S.BLUE, label="F2+", zorder=3)
    ax.set_xlim(0.64, 1.01)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(["70%", "80%", "90%", "100%"])
    ax.set_title("LAC at 90%, by true class", loc="left", pad=13)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.1)
    only_bottom(ax)
    # (c) Mondrian 90%
    ax = axes[2]
    ax.axvspan(0.5, 0.85, color=S.RED_XL, alpha=0.45, lw=0, zorder=0)
    ax.axvline(0.90, color=S.BLUE2, lw=0.7, ls=(0, (3, 2)), zorder=0)
    for i, fo in enumerate(folds):
        ax.plot([mon.loc[fo, "worst"], mon.loc[fo, "cov"]], [i, i], color=S.GREY_L, lw=0.8, zorder=1)
    ax.plot(mon.reindex(folds)["cov"], y, "D", ms=3.4, color=S.GREEN, label="Overall", zorder=3)
    ax.plot(mon.reindex(folds)["worst"], y, "D", ms=3.4, mfc="white", mec=S.GREEN, mew=1.0, label="Worst class", zorder=3)
    ax.set_xlim(0.64, 1.01)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0]); ax.set_xticklabels(["70%", "80%", "90%", "100%"])
    ax.set_title("Mondrian at 90%", loc="left", pad=13)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.1)
    only_bottom(ax)
    f.supxlabel("Achieved coverage (mean of five seeds)", fontsize=7.5, y=0.0)
    f.subplots_adjust(left=0.14, right=0.99, top=0.86, bottom=0.14, wspace=0.10)

    d = pd.DataFrame({"cohort": names,
                      "pi_test": l90["pi_test"].values, "pi_cal": l90["pi_cal"].values,
                      "lac95": lac[np.isclose(lac["alpha"], 0.05)].set_index("fold_i").reindex(folds)["cov"].values,
                      "lac90": l90["cov"].values,
                      "lac80": lac[np.isclose(lac["alpha"], 0.20)].set_index("fold_i").reindex(folds)["cov"].values,
                      "lac90_F0F1": l90["cov_neg"].values, "lac90_F2plus": l90["cov_pos"].values,
                      "mondrian90": mon.reindex(folds)["cov"].values, "mondrian90_worst": mon.reindex(folds)["worst"].values})
    d.loc[len(d)] = ["Mean", np.nan, np.nan] + d.iloc[:, 3:].mean().tolist()
    return f, d


def run():
    nums = ONLY or sorted(REG)
    for n in nums:
        print(f"[fig_p{n}]")
        target = 135 / 25.4
        S.TEXTW = target
        for _ in range(4):
            f, d = REG[n]()
            if getattr(f, "_fixed_canvas", False):
                w = target
                break
            f.canvas.draw()
            w = f.get_tightbbox(f.canvas.get_renderer()).width + 0.06
            if abs(w - target) < 0.02:
                break
            S.TEXTW *= target / w
            plt.close(f)
        print(f"   width {w*25.4:.1f} mm")
        kw = {} if getattr(f, "_fixed_canvas", False) else dict(bbox_inches="tight", pad_inches=0.03)
        f.savefig(OUT / f"fig_p{n}.png", dpi=600, **kw)
        f.savefig(OUT / f"fig_p{n}.pdf", **kw)
        plt.close(f)
        d.to_csv(FIGDATA / f"fig_p{n}_data.csv", index=False)


if __name__ == "__main__":
    _p2 = Path(__file__).resolve().parent / "results_figures_part2.py"
    if _p2.exists():
        exec(compile(_p2.read_text(), str(_p2), "exec"))   # registers p11-p26 in this namespace
    run()
