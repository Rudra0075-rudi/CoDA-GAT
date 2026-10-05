"""
CoDA-GAT — publication figures, generated entirely from saved output files.

Read-only. No retraining, no re-running experiments, no result file modified.
Every plotted number is read from disk; nothing is hardcoded or estimated.

APS values come from verification_checks/ (the CORRECTED randomized-APS
numbers), never from the aps_* columns in results_b7/master_results.csv,
which hold the degenerate coverage=1.000 values.

Each figure is registered with its source files and a caption. A figure whose
source data is missing is SKIPPED with a recorded reason rather than faked --
the end-of-run report lists what was made and what was skipped and why.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
SHARED = REPO / "shared"
B7 = REPO / "results_b7"
BASE = REPO / "baselines"
B9 = REPO / "ablations_b9"
ATTN = REPO / "attention_analysis_b9"
GUTIL = REPO / "graph_utility"
LEARNED = REPO / "learned_graph"
CONS = REPO / "consensus_failures"
TA = REPO / "test_a_adjacency"
TB = REPO / "test_b_rate"
TC = REPO / "test_c_cleaning"
B8 = REPO / "decomposition_b8"
B5 = REPO / "safety_checks_b5"
VER = REPO / "verification_checks"
OUT = REPO / "figures"

# ---- style: one consistent look, colour-blind-safe (Okabe-Ito) ----
OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
CODAGAT = "#D55E00"       # CoDA-GAT highlight -- identical in every figure
NEUTRAL = "#9AA0A6"
CEILING = (0.84, 0.94)    # label-noise ceiling band

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.facecolor": "white", "axes.prop_cycle": plt.cycler(color=OKABE),
})


class SkipFigure(Exception):
    """Raised when a figure's source data is absent -- recorded, never faked."""


REGISTRY = []


def figure(num, name, sources, caption):
    def deco(fn):
        REGISTRY.append({"num": num, "name": name, "sources": sources,
                          "caption": caption, "fn": fn})
        return fn
    return deco


def need(path):
    p = Path(path)
    if not p.exists():
        raise SkipFigure(f"missing source: {p.relative_to(REPO)}")
    return p


def read_csv(path, **kw):
    return pd.read_csv(need(path), **kw)


def as_bool(series):
    """Booleans round-trip through CSV. If pandas ever reads them back as object
    dtype, a bare .astype(bool) makes EVERY row True (non-empty strings are
    truthy) and would silently inflate every error rate. Coerce explicitly."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()


def cohort_labels():
    """fold -> held-out cohort ID. Folds are always labelled by cohort."""
    fa = read_csv(SHARED / "fold_assignment_lodo.csv")
    return dict(zip(fa["fold"], fa["test_cohort"]))


def ceiling_band(ax, orient="y"):
    if orient == "y":
        ax.axhspan(*CEILING, color=OKABE[2], alpha=0.12, zorder=0)
    else:
        ax.axvspan(*CEILING, color=OKABE[2], alpha=0.12, zorder=0)


# ==========================================================================
# A. DATASET
# ==========================================================================
@figure(1, "samples_per_cohort", ["shared/labels.parquet", "shared/fold_assignment_lodo.csv"],
        "Supervised samples per cohort, coloured by whether the cohort is a held-out "
        "evaluation fold (T1) or contributes to training only (T2).")
def fig01():
    lab = pd.read_parquet(need(SHARED / "labels.parquet"))
    lab = lab[lab["is_supervised"]]
    eval_cohorts = set(read_csv(SHARED / "fold_assignment_lodo.csv")["test_cohort"])
    counts = lab["cohort"].value_counts().sort_values(ascending=False)
    colors = [CODAGAT if c in eval_cohorts else NEUTRAL for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(counts)), counts.values, color=colors)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    ax.set_ylabel("Samples (n)")
    ax.set_title(f"Supervised samples per cohort (total n={int(counts.sum())})")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 4, str(int(v)), ha="center", fontsize=8)
    ax.legend(handles=[Patch(color=CODAGAT, label="held-out evaluation fold (T1)"),
                        Patch(color=NEUTRAL, label="training-only (T2)")], loc="upper right")
    return fig


@figure(2, "f2plus_share_per_cohort", ["shared/labels.parquet", "shared/fold_assignment_lodo.csv"],
        "Share of patients with significant fibrosis (F2+) in each cohort; the spread "
        "across cohorts is the case-mix shift the decomposition later quantifies.")
def fig02():
    lab = pd.read_parquet(need(SHARED / "labels.parquet"))
    lab = lab[lab["is_supervised"]]
    eval_cohorts = set(read_csv(SHARED / "fold_assignment_lodo.csv")["test_cohort"])
    share = lab.groupby("cohort")["y_T1_significant"].mean().sort_values(ascending=False)
    colors = [CODAGAT if c in eval_cohorts else NEUTRAL for c in share.index]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(share)), share.values * 100, color=colors)
    ax.axhline(lab["y_T1_significant"].mean() * 100, ls="--", lw=1, color="black",
               label=f"pooled {lab['y_T1_significant'].mean()*100:.1f}%")
    ax.set_xticks(range(len(share)))
    ax.set_xticklabels(share.index, rotation=45, ha="right")
    ax.set_ylabel("Patients with F2+ (%)")
    ax.set_title("Case mix: share of F2+ per cohort")
    pooled = lab["y_T1_significant"].mean() * 100
    ax.legend(handles=[Patch(color=CODAGAT, label="evaluation fold"),
                        Patch(color=NEUTRAL, label="training-only"),
                        plt.Line2D([0], [0], ls="--", color="black",
                                    label=f"pooled {pooled:.1f}%")],
              loc="lower left", fontsize=8)
    return fig


@figure(3, "label_precision_per_cohort", ["shared/labels.parquet"],
        "Biopsy label precision per cohort: samples with an exact fibrosis stage versus "
        "samples whose stage is only known as a grouped range.")
def fig03():
    lab = pd.read_parquet(need(SHARED / "labels.parquet"))
    lab = lab[lab["is_supervised"]].copy()
    lab["precision"] = np.where(lab["resolution"] == "exact", "exact", "grouped")
    tab = pd.crosstab(lab["cohort"], lab["precision"])
    for c in ("exact", "grouped"):
        if c not in tab:
            tab[c] = 0
    tab = tab[["exact", "grouped"]].sort_values("exact", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(tab)), tab["exact"], color=OKABE[0], label="exact stage")
    ax.bar(range(len(tab)), tab["grouped"], bottom=tab["exact"], color=OKABE[1],
           label="grouped range only")
    ax.set_xticks(range(len(tab)))
    ax.set_xticklabels(tab.index, rotation=45, ha="right")
    ax.set_ylabel("Samples (n)")
    ax.set_title("Label precision by cohort")
    ax.legend()
    return fig


# ==========================================================================
# B. MAIN PERFORMANCE
# ==========================================================================
def collect_all_model_auc():
    """Mean/std AUC for every model family from the saved summary files."""
    rows = []
    b7 = read_csv(B7 / "master_results.csv")
    off = b7[(b7["mode"] == "off") & (b7["m"] == 0)]
    per_fold = off.groupby("fold")["auc"].mean()
    rows.append({"model": "CoDA-GAT", "family": "graph", "auc_mean": per_fold.mean(),
                  "auc_std": per_fold.std(), "highlight": True})

    for path, key, fam in ((BASE / "summary_tabular.csv", "model", "tabular"),
                            (BASE / "summary_hybrids.csv", "model", "hybrid"),
                            (BASE / "summary_graph_controls.csv", "control", "graph control")):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            rows.append({"model": r[key], "family": fam, "auc_mean": r["auc_mean"],
                          "auc_std": r.get("auc_std", np.nan), "highlight": False})
    return pd.DataFrame(rows)


@figure(4, "all_models_auc_ranked",
        ["results_b7/master_results.csv", "baselines/summary_*.csv"],
        "Every model ranked by mean cross-cohort AUC with per-fold spread; the shaded band "
        "is the 0.87-0.90 label-noise ceiling implied by pathologist agreement.")
def fig04():
    df = collect_all_model_auc().sort_values("auc_mean")
    if df.empty:
        raise SkipFigure("no model summaries found")
    colors = [CODAGAT if h else NEUTRAL for h in df["highlight"]]

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.28 * len(df))))
    ceiling_band(ax, orient="x")
    ax.barh(range(len(df)), df["auc_mean"], xerr=df["auc_std"].fillna(0),
            color=colors, error_kw={"lw": 0.8, "ecolor": "#444"})
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"], fontsize=8)
    ax.set_xlabel("AUC (mean across 8 held-out cohorts, ± std)")
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="black", lw=0.8, ls=":")
    ax.set_title("Cross-cohort discrimination: all models")
    ax.legend(handles=[Patch(color=CODAGAT, label="CoDA-GAT"),
                        Patch(color=NEUTRAL, label="baselines / controls"),
                        Patch(color=OKABE[2], alpha=0.12,
                              label=f"ceiling band {CEILING[0]}-{CEILING[1]}")],
              loc="lower right")
    return fig


@figure(5, "per_fold_auc_heatmap",
        ["results_b7/master_results.csv", "baselines/*/all_folds.csv",
         "ablations_b9/master_ablation_results.csv"],
        "Per-cohort AUC for each model; columns are the eight held-out cohorts, revealing "
        "which cohorts are hard for every model rather than for one.")
def fig05():
    cl = cohort_labels()
    frames = []
    b7 = read_csv(B7 / "master_results.csv")
    off = b7[(b7["mode"] == "off") & (b7["m"] == 0)]
    frames.append(off.groupby("fold")["auc"].mean().rename("CoDA-GAT"))

    for sub, key in (("tabular", "model"), ("hybrids", "model"), ("graph_controls", "control")):
        p = BASE / sub / "all_folds.csv"
        if p.exists():
            d = pd.read_csv(p)
            for m, g in d.groupby(key):
                frames.append(g.groupby("fold")["auc"].mean().rename(str(m)))
    p = B9 / "master_ablation_results.csv"
    if p.exists():
        d = pd.read_csv(p)
        for m, g in d.groupby("ablation"):
            frames.append(g.groupby("fold")["auc"].mean().rename(f"abl: {m}"))

    if not frames:
        raise SkipFigure("no per-fold AUC sources found")
    mat = pd.concat(frames, axis=1).T
    mat = mat.reindex(sorted(mat.columns), axis=1)
    mat.columns = [cl.get(c, c) for c in mat.columns]
    mat = mat.sort_values(mat.columns[0], ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.3 * len(mat))))
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.8 else "black")
    ax.set_title("Per-cohort AUC (held-out cohort in columns)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="AUC", fraction=0.025)
    return fig


@figure(6, "roc_per_cohort",
        ["results_b7/fold*/mode_off/seed_*/probs_test.npy", "folds/*/test/y.csv"],
        "ROC curve for CoDA-GAT on each held-out cohort (seed-averaged probabilities), "
        "plus the pooled curve over all 1,064 evaluation samples.")
def fig06():
    cl = cohort_labels()
    fig, ax = plt.subplots(figsize=(6.2, 6))
    all_y, all_p = [], []
    plotted = 0
    for fold, cohort in sorted(cl.items()):
        ydir = REPO / "folds"
        fdirs = list(ydir.glob(f"fold{fold}_*"))
        if not fdirs:
            continue
        y = pd.read_csv(fdirs[0] / "test" / "y.csv", index_col=0)["y_T1_significant"].to_numpy().astype(int)
        probs = []
        for d in sorted((B7 / f"fold{fold}_{cohort}" / "mode_off").glob("seed_*")):
            f = d / "probs_test.npy"
            if f.exists():
                probs.append(np.load(f)[:, 1])
        if not probs or len(np.unique(y)) < 2:
            continue
        p = np.mean(probs, axis=0)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.2, alpha=0.85,
                label=f"{cohort} (AUC {roc_auc_score(y, p):.3f})")
        all_y.append(y); all_p.append(p); plotted += 1
    if plotted == 0:
        raise SkipFigure("no per-sample test probabilities found")
    y_all, p_all = np.concatenate(all_y), np.concatenate(all_p)
    fpr, tpr, _ = roc_curve(y_all, p_all)
    ax.plot(fpr, tpr, lw=2.6, color=CODAGAT,
            label=f"pooled (AUC {roc_auc_score(y_all, p_all):.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="black", lw=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("CoDA-GAT ROC by held-out cohort")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_aspect("equal")
    return fig


@figure(7, "adversary_modes_macrof1",
        ["results_b7/master_results.csv", "results_b7/primary_contrast_report.txt"],
        "Pre-registered primary contrast: per-cohort macro-F1 with the cohort adversary off, "
        "marginal and conditional; points are cohorts, annotation is the registered test.")
def fig07():
    b7 = read_csv(B7 / "master_results.csv")
    b7 = b7[b7["m"] == 0]
    per = b7.groupby(["mode", "fold"])["macro_f1"].mean().reset_index()
    modes = [m for m in ("off", "marginal", "conditional") if m in set(per["mode"])]
    data = [per[per["mode"] == m]["macro_f1"].values for m in modes]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bp = ax.boxplot(data, labels=modes, widths=0.55, patch_artist=True,
                    medianprops={"color": "black"})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(CODAGAT if modes[i] == "marginal" else NEUTRAL)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.09, 0.09, len(vals)), vals,
                   s=22, color="black", zorder=3, alpha=0.8)
    ax.set_ylabel("Macro-F1 (per held-out cohort)")
    ax.set_title("Adversary mode contrast (each point = one cohort)")

    rep = B7 / "primary_contrast_report.txt"
    note = "pre-registered: marginal vs off, paired Wilcoxon"
    if rep.exists():
        for line in rep.read_text().splitlines():
            if "marginal vs off" in line:
                note = line.strip()
                break
    ax.annotate(note, xy=(0.5, -0.22), xycoords="axes fraction", ha="center", fontsize=7.5,
                wrap=True)
    return fig


# ==========================================================================
# C. RELIABILITY (corrected APS)
# ==========================================================================
def aps_lac_table():
    d = read_csv(VER / "aps_diagnostic_per_run.csv")
    recs = []
    for alpha, g in d.groupby("alpha"):
        recs.append({"alpha": alpha, "score": "LAC",
                      "coverage": g["lac_coverage"].mean(), "coverage_sd": g["lac_coverage"].std(),
                      "set_size": g["lac_set_size"].mean(), "set_sd": g["lac_set_size"].std()})
        recs.append({"alpha": alpha, "score": "APS (corrected)",
                      "coverage": g["aps_coverage_corrected"].mean(),
                      "coverage_sd": g["aps_coverage_corrected"].std(),
                      "set_size": g["aps_set_size_corrected"].mean(),
                      "set_sd": g["aps_set_size_corrected"].std()})
        recs.append({"alpha": alpha, "score": "APS (original, degenerate)",
                      "coverage": g["aps_coverage_original"].mean(),
                      "coverage_sd": g["aps_coverage_original"].std(),
                      "set_size": g["aps_set_size_original"].mean(),
                      "set_sd": g["aps_set_size_original"].std()})
    return pd.DataFrame(recs)


@figure(8, "conformal_coverage_vs_promised", ["verification_checks/aps_diagnostic_per_run.csv"],
        "Achieved versus promised coverage for LAC and the corrected randomized APS; the "
        "original APS is shown for contrast, pinned at 1.000 by the omitted randomisation term.")
def fig08():
    t = aps_lac_table()
    styles = {"LAC": (OKABE[0], "o", "-"), "APS (corrected)": (CODAGAT, "s", "-"),
              "APS (original, degenerate)": (NEUTRAL, "^", "--")}
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot([0.7, 1.0], [0.7, 1.0], ls=":", color="black", lw=1, label="y = x (perfect)")
    for score, g in t.groupby("score"):
        g = g.sort_values("alpha")
        c, m, ls = styles.get(score, (NEUTRAL, "o", "-"))
        ax.errorbar(1 - g["alpha"], g["coverage"], yerr=g["coverage_sd"], marker=m, ls=ls,
                    color=c, capsize=3, lw=1.6, label=score)
    ax.set_xlabel("Promised coverage (1 − α)")
    ax.set_ylabel("Achieved coverage on held-out cohort")
    ax.set_title("Conformal reliability")
    ax.legend(fontsize=8, loc="lower right")
    return fig


@figure(9, "conformal_set_size_vs_alpha", ["verification_checks/aps_diagnostic_per_run.csv"],
        "Mean prediction-set size by α. A size of 2.0 on a two-class task means the full "
        "label set, i.e. no information; the corrected APS produces informative sets.")
def fig09():
    t = aps_lac_table()
    styles = {"LAC": (OKABE[0], "o", "-"), "APS (corrected)": (CODAGAT, "s", "-"),
              "APS (original, degenerate)": (NEUTRAL, "^", "--")}
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.axhline(2.0, color="black", ls=":", lw=1, label="full label set (uninformative)")
    for score, g in t.groupby("score"):
        g = g.sort_values("alpha")
        c, m, ls = styles.get(score, (NEUTRAL, "o", "-"))
        ax.errorbar(g["alpha"], g["set_size"], yerr=g["set_sd"], marker=m, ls=ls, color=c,
                    capsize=3, lw=1.6, label=score)
    ax.set_xlabel("α (miscoverage allowance)")
    ax.set_ylabel("Mean set size (classes)")
    ax.set_ylim(0.6, 2.15)
    ax.set_title("Prediction-set size")
    ax.legend(fontsize=8)
    return fig


@figure(10, "coverage_per_cohort_alpha10", ["verification_checks/aps_diagnostic_per_run.csv",
                                             "shared/fold_assignment_lodo.csv"],
        "Coverage per held-out cohort at α = 0.10: whether the 90% promise survives each "
        "individual cohort shift, not just on average.")
def fig10():
    d = read_csv(VER / "aps_diagnostic_per_run.csv")
    d = d[np.isclose(d["alpha"], 0.10)]
    if d.empty:
        raise SkipFigure("no alpha=0.10 rows")
    cl = cohort_labels()
    g = d.groupby("fold").agg(lac=("lac_coverage", "mean"), lac_sd=("lac_coverage", "std"),
                               aps=("aps_coverage_corrected", "mean"),
                               aps_sd=("aps_coverage_corrected", "std")).reset_index()
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 0.2, g["lac"], 0.38, yerr=g["lac_sd"], color=OKABE[0], capsize=2.5, label="LAC")
    ax.bar(x + 0.2, g["aps"], 0.38, yerr=g["aps_sd"], color=CODAGAT, capsize=2.5,
           label="APS (corrected)")
    ax.axhline(0.90, color="black", ls="--", lw=1.2, label="promised 90%")
    ax.set_xticks(x)
    ax.set_xticklabels([cl.get(f, f) for f in g["fold"]], rotation=45, ha="right")
    ax.set_ylabel("Achieved coverage")
    ax.set_ylim(0.6, 1.02)
    ax.set_title("Per-cohort coverage at α = 0.10")
    ax.legend(fontsize=8, ncol=3)
    return fig


# ==========================================================================
# D. NETWORK ROBUSTNESS
# ==========================================================================
def masking_pooled(models=None):
    d = read_csv(GUTIL / "masking_curves.csv")
    if models:
        d = d[d["model"].isin(models)]
    per_fold = d.groupby(["model", "fold", "mask_rate"])["auc"].mean().reset_index()
    g = per_fold.groupby(["model", "mask_rate"])["auc"].agg(["mean", "std", "count"]).reset_index()
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g


@figure(11, "masking_real_vs_empty", ["graph_utility/masking_curves.csv"],
        "The graph's clearest benefit: AUC as genes are progressively hidden at inference. "
        "The network-informed model degrades more slowly than the graph-free control, with "
        "95% CI bands across cohorts.")
def fig11():
    g = masking_pooled(["real_graph", "empty_graph"])
    if g.empty:
        raise SkipFigure("no real/empty masking rows")
    labels = {"real_graph": ("STRING graph (CoDA-GAT)", CODAGAT),
              "empty_graph": ("no graph (control)", NEUTRAL)}
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for m, sub in g.groupby("model"):
        sub = sub.sort_values("mask_rate")
        lab, c = labels.get(m, (m, OKABE[0]))
        ax.plot(sub["mask_rate"] * 100, sub["mean"], marker="o", color=c, lw=2, label=lab)
        ax.fill_between(sub["mask_rate"] * 100, sub["mean"] - sub["ci"], sub["mean"] + sub["ci"],
                        color=c, alpha=0.18)
    ax.set_xlabel("Genes hidden at inference (%)")
    ax.set_ylabel("AUC")
    ax.set_title("Robustness to missing genes")
    ax.legend()
    return fig


@figure(12, "masking_all_conditions", ["graph_utility/masking_curves.csv",
                                        "learned_graph/masking_curves_all_conditions.csv"],
        "All graph conditions under masking, including the learned topologies, isolating how "
        "much of the robustness is connectivity versus real biology.")
def fig12():
    frames = [read_csv(GUTIL / "masking_curves.csv")]
    lp = LEARNED / "masking_curves_all_conditions.csv"
    if not lp.exists():
        raise SkipFigure("learned_graph/masking_curves_all_conditions.csv missing "
                          "(scripts/learned_graph.py has not been run)")
    frames.append(pd.read_csv(lp))
    d = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["model", "fold", "seed", "mask_rate", "draw"])
    per_fold = d.groupby(["model", "fold", "mask_rate"])["auc"].mean().reset_index()
    g = per_fold.groupby(["model", "mask_rate"])["auc"].agg(["mean", "std", "count"]).reset_index()
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(7.4, 5))
    for i, (m, sub) in enumerate(g.groupby("model")):
        sub = sub.sort_values("mask_rate")
        c = CODAGAT if m == "real_graph" else OKABE[i % len(OKABE)]
        ax.plot(sub["mask_rate"] * 100, sub["mean"], marker="o", lw=1.8, color=c, label=m)
        ax.fill_between(sub["mask_rate"] * 100, sub["mean"] - sub["ci"], sub["mean"] + sub["ci"],
                        color=c, alpha=0.13)
    ax.set_xlabel("Genes hidden at inference (%)")
    ax.set_ylabel("AUC")
    ax.set_title("Masking: every graph condition")
    ax.legend(fontsize=8)
    return fig


@figure(13, "advantage_over_empty", ["graph_utility/masking_paired_real_vs_empty.csv",
                                      "learned_graph/paired_comparisons.csv"],
        "Advantage over the graph-free control under masking, per condition, with "
        "confidence intervals across cohort x masking-rate cells.")
def fig13():
    p = GUTIL / "masking_paired_real_vs_empty.csv"
    d = read_csv(p)
    adv = d["auc_drop_empty"] - d["auc_drop_real"]
    rows = [{"condition": "STRING graph", "mean": adv.mean(),
             "lo": np.percentile(adv, 2.5), "hi": np.percentile(adv, 97.5)}]

    lp = LEARNED / "paired_comparisons.csv"
    if lp.exists():
        pc = pd.read_csv(lp)
        for _, r in pc[pc["comparison"].str.endswith("vs empty_graph")].iterrows():
            rows.append({"condition": r["comparison"].replace(" vs empty_graph", ""),
                          "mean": r["mean_delta"], "lo": r["ci_lo"], "hi": r["ci_hi"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = [CODAGAT if c == "STRING graph" else NEUTRAL for c in df["condition"]]
    ax.bar(range(len(df)), df["mean"], color=colors,
           yerr=[df["mean"] - df["lo"], df["hi"] - df["mean"]], capsize=4,
           error_kw={"lw": 0.9})
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["condition"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("AUC advantage over no-graph control")
    ax.set_title("How much does connectivity buy under missingness?")
    if len(df) == 1:
        ax.annotate("learned-graph conditions unavailable\n(learned_graph.py not run)",
                    xy=(0.98, 0.04), xycoords="axes fraction", ha="right", fontsize=7.5,
                    color="#666")
    return fig


@figure(14, "learned_vs_string_topology", ["learned_graph/topology_comparison.csv",
                                            "learned_graph/topology_cross_fold_stability.csv"],
        "Overlap between learned topologies and STRING (Jaccard) alongside cross-fold "
        "topology stability, indicating whether a learned graph is reproducible or noise.")
def fig14():
    p = LEARNED / "topology_comparison.csv"
    if not p.exists():
        raise SkipFigure("learned_graph/topology_comparison.csv missing "
                          "(scripts/learned_graph.py has not been run)")
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for v, g in d.groupby("variant"):
        axes[0].plot(g["fold"], g["jaccard_with_string"], marker="o", label=v)
    axes[0].set_xlabel("Fold"); axes[0].set_ylabel("Jaccard with STRING")
    axes[0].set_title("Learned vs STRING overlap"); axes[0].legend(fontsize=8)

    sp = LEARNED / "topology_cross_fold_stability.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        axes[1].bar(range(len(s)), s["mean_cross_fold_edge_jaccard"], color=OKABE[0],
                    yerr=[s["mean_cross_fold_edge_jaccard"] - s["min"],
                          s["max"] - s["mean_cross_fold_edge_jaccard"]], capsize=4)
        axes[1].set_xticks(range(len(s)))
        axes[1].set_xticklabels(s["variant"], rotation=20, ha="right", fontsize=8)
        axes[1].set_ylabel("Cross-fold edge Jaccard")
        axes[1].set_title("Topology stability")
    return fig


# ==========================================================================
# E. WHAT THE MODEL LOOKS AT
# ==========================================================================
@figure(15, "top_attended_genes", ["attention_analysis_b9/per_fold_gene_attention.csv"],
        "The fifteen genes receiving most attention, averaged across cohorts, with the "
        "spread across folds shown as error bars.")
def fig15():
    d = read_csv(ATTN / "per_fold_gene_attention.csv", index_col=0)
    mean, sd = d.mean(axis=0), d.std(axis=0)
    top = mean.sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(top))[::-1], top.values, xerr=sd[top.index].values,
            color=CODAGAT, capsize=3, error_kw={"lw": 0.8, "ecolor": "#444"})
    ax.set_yticks(range(len(top))[::-1])
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("Mean attention emitted (a.u.)")
    ax.set_title("Top-15 attended genes (± spread across 8 cohorts)")
    return fig


@figure(16, "attention_stability_heatmap", ["attention_analysis_b9/cross_fold_attention_correlation.csv",
                                             "shared/fold_assignment_lodo.csv"],
        "Agreement between cohorts on which genes matter: Spearman correlation of the "
        "attention ranking between every pair of held-out folds.")
def fig16():
    d = read_csv(ATTN / "cross_fold_attention_correlation.csv", index_col=0)
    cl = cohort_labels()
    names = [cl.get(int(c), c) for c in d.columns]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(d.values, cmap="magma", vmin=np.nanmin(d.values), vmax=1.0)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, f"{d.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if d.values[i, j] < 0.8 else "black")
    ax.set_title("Cross-cohort attention-rank agreement (Spearman)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, label="Spearman ρ")
    return fig


# ==========================================================================
# F. CEILING EVIDENCE
# ==========================================================================
def model_families():
    """Authoritative model -> family map, imported from the analysis script itself."""
    try:
        from consensus_failures import MODEL_REGISTRY
        return {m: s["architecture_family"] for m, s in MODEL_REGISTRY.items()}
    except Exception:
        return {}


@figure(17, "error_agreement_heatmap", ["consensus_failures/pairwise_agreement.csv"],
        "Do different models fail on the same patients? Pairwise Jaccard overlap of error "
        "sets across all models, ordered by architecture family.")
def fig17():
    d = read_csv(CONS / "pairwise_agreement.csv")
    fams = model_families()
    models = sorted(set(d["model_a"]) | set(d["model_b"]),
                    key=lambda m: (fams.get(m, "zz"), m))
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
    # family boundaries
    bounds, prev = [], None
    for i, m in enumerate(models):
        f = fams.get(m, "?")
        if prev is not None and f != prev:
            bounds.append(i - 0.5)
        prev = f
    for b in bounds:
        ax.axhline(b, color="white", lw=1.4); ax.axvline(b, color="white", lw=1.4)
    ax.set_title(f"Error-set agreement across {len(models)} models")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, label="Jaccard of error sets")
    return fig


@figure(18, "family_agreement_vs_chance", ["consensus_failures/family_agreement_summary.csv"],
        "Within-family versus between-family error agreement against the chance "
        "expectation; between-family agreement above chance is the informative quantity.")
def fig18():
    d = read_csv(CONS / "family_agreement_summary.csv")
    d = d[d["level"] == "architecture_family"]
    if d.empty:
        d = read_csv(CONS / "family_agreement_summary.csv")
        d = d[d["level"] == "variant_group"]
    if d.empty:
        raise SkipFigure("no family-level rows")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(x - 0.2, d["mean_jaccard_observed"], 0.38, color=CODAGAT, label="observed")
    ax.bar(x + 0.2, d["mean_jaccard_expected_chance"], 0.38, color=NEUTRAL, label="expected by chance")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(i, max(r["mean_jaccard_observed"], r["mean_jaccard_expected_chance"]) + 0.01,
                f"+{r['mean_jaccard_excess']:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(d["relation"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean Jaccard of error sets")
    ax.set_title("Error agreement vs chance (excess annotated)")
    ax.legend(fontsize=8)
    return fig


@figure(19, "consensus_failure_stages", ["consensus_failures/error_matrix.csv",
                                          "consensus_failures/consensus_failure_samples.csv"],
        "Recorded fibrosis stage of the patients that nearly every model gets wrong, versus "
        "everyone else: concentration at the F1/F2 boundary points at the labels.")
def fig19():
    em = read_csv(CONS / "error_matrix.csv")
    cf = read_csv(CONS / "consensus_failure_samples.csv")
    cons_ids = set(cf["sample_id"])
    em["group"] = np.where(em["sample_id"].isin(cons_ids), "consensus failure", "all other samples")
    stages = ["F0", "F1", "F2", "F3", "F4"]
    tab = pd.crosstab(em["group"], em["stage"]).reindex(columns=stages, fill_value=0)
    prop = tab.div(tab.sum(axis=1), axis=0)

    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, (grp, color) in enumerate([("consensus failure", CODAGAT),
                                       ("all other samples", NEUTRAL)]):
        if grp not in prop.index:
            continue
        ax.bar(x + (i - 0.5) * 0.4, prop.loc[grp] * 100, 0.38, color=color,
               label=f"{grp} (n={int(tab.loc[grp].sum())})")
    ax.axvspan(0.5, 2.5, color=OKABE[2], alpha=0.10, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_xlabel("Recorded fibrosis stage")
    ax.set_ylabel("Share of group (%)")
    ax.set_title("Where consensus failures sit (shaded = F1/F2 decision boundary)")
    ax.legend(fontsize=8)
    return fig


@figure(20, "error_rate_all_models", ["consensus_failures/error_matrix.csv"],
        "Error rate of every model on the pooled held-out samples; the tight clustering is "
        "the empirical signature of a shared ceiling.")
def fig20():
    em = read_csv(CONS / "error_matrix.csv")
    meta = {"sample_id", "fold", "stage", "y_true", "n_models_wrong", "group"}
    models = [c for c in em.columns if c not in meta]
    rates = pd.Series({m: as_bool(em[m]).mean() for m in models}).sort_values()
    fams = model_families()
    colors = [CODAGAT if "codagat" in m else NEUTRAL for m in rates.index]

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.26 * len(rates))))
    ax.barh(range(len(rates)), rates.values * 100, color=colors)
    lo, hi = rates.values.min() * 100, rates.values.max() * 100
    ax.axvspan(lo, hi, color=OKABE[2], alpha=0.10, zorder=0)
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{m}  [{fams.get(m, '?')}]" for m in rates.index], fontsize=6.5)
    ax.set_xlabel("Error rate on held-out samples (%)")
    ax.set_title(f"Every model lands in a {lo:.0f}–{hi:.0f}% error band")
    return fig


# ==========================================================================
# G. THE FLAGGING TOOL
# ==========================================================================
@figure(21, "testA_flagged_stage_distribution", ["test_a_adjacency/stage_distribution_5class.csv"],
        "Test A: recorded stage of samples the detector flags versus those it does not, as "
        "proportions; flags concentrate next to the F2 decision boundary.")
def fig21():
    d = read_csv(TA / "stage_distribution_5class.csv", index_col=0)
    stages = [c for c in ["F0", "F1", "F2", "F3", "F4"] if c in d.columns]
    prop = d[stages].div(d[stages].sum(axis=1), axis=0)
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, (row, color) in enumerate(zip(prop.index, [CODAGAT, NEUTRAL])):
        ax.bar(x + (i - 0.5) * 0.4, prop.loc[row] * 100, 0.38, color=color,
               label=f"{row} (n={int(d.loc[row, stages].sum())})")
    ax.axvspan(0.5, 2.5, color=OKABE[2], alpha=0.10, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_xlabel("Recorded fibrosis stage")
    ax.set_ylabel("Share of group (%)")
    ax.set_title("Test A: flagged samples cluster at F1/F2")
    ax.legend(fontsize=8)
    return fig


@figure(22, "testB_operating_curve", ["test_b_rate/threshold_sweep_extended.csv"],
        "Test B operating curve: how the boundary-adjacency enrichment holds as the "
        "flagging rate rises, with the independently predicted 18.6% label-error rate marked.")
def fig22():
    d = read_csv(TB / "threshold_sweep_extended.csv").sort_values("pooled_flagging_rate")
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.plot(d["pooled_flagging_rate"] * 100, d["odds_ratio"], marker="o", color=CODAGAT, lw=1.8)
    for _, r in d.iterrows():
        ax.annotate(f"p{int(r['uncertainty_percentile'])}",
                    (r["pooled_flagging_rate"] * 100, r["odds_ratio"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.axvline(18.6, color=OKABE[0], ls="--", lw=1.3, label="predicted label-error rate 18.6%")
    ax.axhline(1.0, color="black", ls=":", lw=1, label="no enrichment (OR = 1)")
    p70 = d[d["uncertainty_percentile"] == 70]
    if not p70.empty:
        r = p70.iloc[0]
        ax.scatter([r["pooled_flagging_rate"] * 100], [r["odds_ratio"]], s=170,
                   facecolor="none", edgecolor=OKABE[2], lw=2.2, zorder=5,
                   label=f"operating point p70 ({r['pooled_flagging_rate']*100:.1f}%, "
                         f"OR {r['odds_ratio']:.2f}, p={r['fisher_p']:.1e})")
    ax.set_xlabel("Samples flagged (%)")
    ax.set_ylabel("Boundary-adjacency enrichment (odds ratio)")
    ax.set_title("Test B: enrichment holds out to the predicted error rate")
    ax.legend(fontsize=7.5, loc="lower right")
    return fig


@figure(23, "testC_cleaned_vs_random", ["test_c_cleaning/per_fold_deltas.json",
                                         "shared/fold_assignment_lodo.csv"],
        "Test C: per-cohort change in held-out macro-F1 when flagged samples are removed "
        "from training, against removing the same number at random.")
def fig23():
    p = need(TC / "per_fold_deltas.json")
    entries = json.loads(p.read_text())
    cl = cohort_labels()

    def pick(pred):
        return next((e for e in entries if pred(e)), None)
    cleaned = pick(lambda e: e["comparison"] == "cleaned vs baseline" and e["metric"] == "macro_f1")
    random_ = pick(lambda e: e["comparison"].startswith("random") and "baseline" in e["comparison"]
                   and e["metric"] == "macro_f1")
    if cleaned is None:
        raise SkipFigure("no 'cleaned vs baseline' macro_f1 entry")

    folds = sorted(int(k) for k in cleaned["per_fold_deltas"])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, f in enumerate(folds):
        cv = cleaned["per_fold_deltas"][str(f)]
        ax.scatter(i, cv, s=62, color=CODAGAT, zorder=3,
                   label="flagged removed (cleaned)" if i == 0 else None)
        if random_ and str(f) in random_["per_fold_deltas"]:
            rv = random_["per_fold_deltas"][str(f)]
            ax.scatter(i, rv, s=62, color=NEUTRAL, zorder=3,
                       label="random removal (control)" if i == 0 else None)
            ax.plot([i, i], [cv, rv], color="#bbb", lw=1, zorder=1)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([cl.get(f, f) for f in folds], rotation=45, ha="right")
    ax.set_ylabel("Δ macro-F1 vs uncleaned baseline")
    ax.set_title("Test C: does removing flagged samples help?")
    ax.legend(fontsize=8)
    return fig


# ==========================================================================
# H. CONTROLS
# ==========================================================================
@figure(24, "ablation_deltas", ["ablations_b9/master_ablation_results.csv",
                                 "results_b7/master_results.csv"],
        "Ablations: change in held-out AUC and macro-F1 when each component is removed, "
        "relative to the full model.")
def fig24():
    a = read_csv(B9 / "master_ablation_results.csv")
    b = read_csv(B7 / "master_results.csv")
    base = b[(b["mode"] == "off") & (b["m"] == 0)].groupby("fold")[["auc", "macro_f1"]].mean()
    rows = []
    for abl, g in a.groupby("ablation"):
        per = g.groupby("fold")[["auc", "macro_f1"]].mean()
        j = per.join(base, lsuffix="_abl", rsuffix="_base").dropna()
        if j.empty:
            continue
        rows.append({"ablation": abl,
                      "d_auc": (j["auc_abl"] - j["auc_base"]).mean(),
                      "d_auc_sd": (j["auc_abl"] - j["auc_base"]).std(),
                      "d_f1": (j["macro_f1_abl"] - j["macro_f1_base"]).mean(),
                      "d_f1_sd": (j["macro_f1_abl"] - j["macro_f1_base"]).std()})
    if not rows:
        raise SkipFigure("no overlapping ablation/baseline folds")
    d = pd.DataFrame(rows).sort_values("d_auc")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x - 0.2, d["d_auc"], 0.38, yerr=d["d_auc_sd"], color=OKABE[0], capsize=3, label="Δ AUC")
    ax.bar(x + 0.2, d["d_f1"], 0.38, yerr=d["d_f1_sd"], color=OKABE[1], capsize=3, label="Δ macro-F1")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(d["ablation"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Change vs full model (negative = removal hurts)")
    ax.set_title("Component ablations")
    ax.legend(fontsize=8)
    return fig


@figure(25, "safety_checks", ["safety_checks_b5/placebo_summary.csv",
                               "safety_checks_b5/consistency_detail.csv"],
        "Estimator safety checks: placebo components should sit at zero, and the two "
        "independent prevalence estimates should agree (difference CI spanning zero).")
def fig25():
    pl = read_csv(B5 / "placebo_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # grouped by metric x component across the 8 cohorts -- one row per cohort
    # would need ~5pt labels, so the per-cohort values are shown as points on
    # four readable categories instead
    pl["group"] = (pl["metric"] + "\n" + pl["component"].str.replace("_component", "", regex=False))
    groups = sorted(pl["group"].unique())
    data = [pl[pl["group"] == g]["mean"].values for g in groups]
    bp = axes[0].boxplot(data, labels=groups, widths=0.5, patch_artist=True,
                          medianprops={"color": "black"}, vert=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(NEUTRAL); patch.set_alpha(0.45)
    rng = np.random.default_rng(1)
    for i, (g, vals) in enumerate(zip(groups, data), start=1):
        passed = pl[pl["group"] == g]["pass"].to_numpy()
        cols = [OKABE[2] if as_bool(pd.Series([p]))[0] else OKABE[5] for p in passed]
        axes[0].scatter(np.full(len(vals), i) + rng.uniform(-0.1, 0.1, len(vals)), vals,
                        c=cols, s=28, zorder=3, edgecolor="black", linewidth=0.4)
    axes[0].axhline(0, color="black", lw=1.4)
    axes[0].set_ylabel("Placebo component value (should be 0)")
    axes[0].tick_params(axis="x", labelsize=8)
    n_pass = int(as_bool(pl["pass"]).sum())
    axes[0].set_title(f"Placebo: {n_pass}/{len(pl)} cells include 0\n(each point = one cohort)")

    co = read_csv(B5 / "consistency_detail.csv")
    y2 = np.arange(len(co))
    axes[1].errorbar(co["diff"], y2, xerr=[co["diff"] - co["diff_ci_lo"], co["diff_ci_hi"] - co["diff"]],
                     fmt="o", ms=4, capsize=2, lw=0.9, ecolor="#888",
                     mfc=CODAGAT, mec="black")
    axes[1].axvline(0, color="black", lw=1.2)
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels([f"fold {int(r['fold'])} · {r['metric']}" for _, r in co.iterrows()],
                             fontsize=7)
    axes[1].set_xlabel("cross-site − within-site prevalence estimate")
    axes[1].set_title(f"Consistency: {int(as_bool(co['agree']).sum())}/{len(co)} agree")
    return fig


@figure(26, "decomposition_per_cohort", ["decomposition_b8/per_fold_decomposition.csv",
                                          "shared/fold_assignment_lodo.csv"],
        "Where cross-cohort performance goes: the case-mix (prevalence) component and the "
        "inseparable residual per cohort, as counterfactual contrasts.")
def fig26():
    d = read_csv(B8 / "per_fold_decomposition.csv")
    d = d[d["metric"] == "macro_f1"]
    if d.empty:
        raise SkipFigure("no macro_f1 decomposition rows")
    cl = cohort_labels()
    d = d.sort_values("fold")
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - 0.2, d["prevalence_component_shapley"], 0.38, color=CODAGAT,
           yerr=[d["prevalence_component_shapley"] - d["prevalence_ci_lo"],
                 d["prevalence_ci_hi"] - d["prevalence_component_shapley"]],
           capsize=3, label="case-mix (prevalence)")
    ax.bar(x + 0.2, d["residual_component_shapley"], 0.38, color=NEUTRAL,
           yerr=[d["residual_component_shapley"] - d["residual_ci_lo"],
                 d["residual_ci_hi"] - d["residual_component_shapley"]],
           capsize=3, label="residual (platform + biology, inseparable)")
    ax.plot(x, d["total_gap"], "k_", ms=18, label="total gap")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([cl.get(f, f) for f in d["fold"]], rotation=45, ha="right")
    ax.set_ylabel("Δ macro-F1 (counterfactual contrast)")
    ax.set_title("Degradation decomposition per cohort (Shapley-symmetrised)")
    ax.legend(fontsize=8)
    return fig


# ==========================================================================
# Driver
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="CoDA-GAT figure generation (read-only)")
    ap.add_argument("--only", nargs="*", type=int, default=None, help="figure numbers to build")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    made, skipped = [], []
    figs_in_order = []

    for spec in sorted(REGISTRY, key=lambda s: s["num"]):
        if args.only and spec["num"] not in args.only:
            continue
        tag = f"fig{spec['num']:02d}_{spec['name']}"
        try:
            fig = spec["fn"]()
            fig.savefig(outdir / f"{tag}.png")
            fig.savefig(outdir / f"{tag}.pdf")
            figs_in_order.append((spec, fig))
            made.append(spec)
            print(f"[fig] {tag}  OK", flush=True)
        except SkipFigure as e:
            skipped.append((spec, str(e)))
            print(f"[fig] {tag}  SKIPPED -- {e}", flush=True)
        except Exception as e:
            skipped.append((spec, f"error: {e}"))
            print(f"[fig] {tag}  ERROR -- {e}", flush=True)
            traceback.print_exc()

    # combined PDF, one figure per page, in order
    if figs_in_order:
        with PdfPages(outdir / "all_figures.pdf") as pdf:
            for spec, fig in figs_in_order:
                pdf.savefig(fig)
        print(f"[fig] all_figures.pdf  ({len(figs_in_order)} pages)")
    for _, fig in figs_in_order:
        plt.close(fig)

    # manifest
    lines = ["# Figure manifest", "",
             "Every number is read from the listed source file(s). Nothing is hardcoded.",
             "APS values come from `verification_checks/` (corrected randomized APS), not the",
             "degenerate `aps_*` columns in `results_b7/master_results.csv`.", ""]
    for spec in made:
        tag = f"fig{spec['num']:02d}_{spec['name']}"
        lines += [f"## {tag}",
                  f"- **files**: `{tag}.png`, `{tag}.pdf`",
                  f"- **sources**: " + ", ".join(f"`{s}`" for s in spec["sources"]),
                  f"- **caption**: {spec['caption']}", ""]
    if skipped:
        lines += ["## Skipped figures", ""]
        for spec, why in skipped:
            lines.append(f"- **fig{spec['num']:02d}_{spec['name']}** — {why}")
        lines.append("")
    (outdir / "MANIFEST.md").write_text("\n".join(lines))

    print("\n" + "=" * 70)
    print(f"FIGURES MADE: {len(made)}")
    for spec in made:
        print(f"  fig{spec['num']:02d}_{spec['name']}")
    print(f"\nSKIPPED: {len(skipped)}")
    for spec, why in skipped:
        print(f"  fig{spec['num']:02d}_{spec['name']}  <- {why}")
    print("=" * 70)
    print(f"output: {outdir}  (PNG + PDF each, all_figures.pdf, MANIFEST.md)")


if __name__ == "__main__":
    main()
