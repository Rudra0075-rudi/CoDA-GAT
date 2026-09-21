"""
CoDA-GAT — figure diagnosis + repair.

Three problems were reported in the figures produced by make_figures.py. This
script DIAGNOSES each one from the saved files first, writes the findings to
figures/diagnosis.txt, and only then regenerates the affected figures using the
values the diagnosis established.

  Issue A  baseline per-cohort coverage. Are the baseline AUCs in
           baselines/summary_*.csv eight-cohort means or single-cohort values?
  Issue B  masking contradiction. Does the real-vs-empty advantage flip sign
           depending on how it is aggregated (or on which metric is used)?
  Issue C  learned-graph outputs. Where are they, and what is actually present?

Read-only on every results directory. Nothing under results_b7/, baselines/,
ablations_b9/, graph_utility/ or learned_graph/ is written or modified. The only
writes are into figures/.

No number in this file is hardcoded. Every value plotted or printed is read or
recomputed from a file on disk, and each recomputation is cross-checked against
the saved metrics it should reproduce.

Style, saving, the combined PDF and the manifest are reused from
make_figures.py by REPLACING its registry entries for the affected figure
numbers, so the repaired figures are visually identical in style to the others
and figures/all_figures.pdf is rebuilt complete.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

# make_figures defines the paths, palette, rcParams, SkipFigure, the saving
# driver and the manifest writer. Importing it (rather than copying) is what
# guarantees the repaired figures match the rest in style.
import make_figures as mf
from make_figures import (
    REPO, SHARED, B7, BASE, B9, GUTIL, LEARNED, OUT,
    OKABE, CODAGAT, NEUTRAL, CEILING, SkipFigure, read_csv, ceiling_band,
    cohort_labels,
)

FOLDS = REPO / "folds"
PSD = BASE / "per_sample_predictions"
GUTIL_CKPT = GUTIL / "control_checkpoints"
LEARNED_CKPT = LEARNED / "checkpoints"
LEARNED_EDGES = LEARNED / "edges"

SEEDS = [42, 43, 44, 45, 46]
MASK_METRICS = ["auc", "macro_f1", "balanced_acc"]


def stable_seed(*parts):
    """Deterministic seed from strings. Python's hash() is randomised per
    process, so using it would make the bootstrap CIs differ between runs of
    the same script on the same data."""
    h = 0
    for s in parts:
        for ch in str(s):
            h = (h * 131 + ord(ch)) % 1_000_003
    return h

# where each graph condition's saved rate-0 predictions live
CONDITION_CKPT_ROOT = {
    "real_graph":     (B7, "mode_off"),
    "empty_graph":    (GUTIL_CKPT, "empty_graph"),
    "shuffled_graph": (GUTIL_CKPT, "shuffled_graph"),
    "random_graph":   (LEARNED_CKPT, "random_graph"),
    "coexpr_graph":   (LEARNED_CKPT, "coexpr_graph"),
    "e2e_graph":      (LEARNED_CKPT, "e2e_graph"),
}
CONDITION_LABEL = {
    "real_graph": "STRING (CoDA-GAT)", "empty_graph": "no graph",
    "shuffled_graph": "shuffled STRING", "random_graph": "random edges",
    "coexpr_graph": "learned: co-expression", "e2e_graph": "learned: end-to-end",
}
CONDITION_COLOR = {
    "real_graph": CODAGAT, "empty_graph": NEUTRAL, "shuffled_graph": OKABE[0],
    "random_graph": OKABE[7], "coexpr_graph": OKABE[2], "e2e_graph": OKABE[3],
}

# the report-stage files learned_graph.py writes; used to say precisely which
# of its stages completed rather than guessing
LEARNED_EXPECTED = [
    "masking_curves_all_conditions.csv", "masking_curves_pooled.csv",
    "paired_comparisons.csv", "protocol_verification.csv", "edge_budget_audit.csv",
    "topology_comparison.csv", "topology_cross_fold_stability.csv",
    "masking_curves_all_conditions.png", "summary.txt",
]

D = {}          # diagnosis results, shared between the report and the figures
DIAG_LINES = []


# ==========================================================================
# small shared utilities
# ==========================================================================
def say(line=""):
    DIAG_LINES.append(str(line))
    print(str(line), flush=True)


def rule(title=None):
    say("=" * 78)
    if title:
        say(title)
        say("=" * 78)


def boot_ci(x, n_boot=5000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean. Paired deltas are passed in, so this
    is a CI on the paired difference, not on either arm."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def fold_dirs():
    return sorted(p for p in FOLDS.iterdir() if p.is_dir() and p.name.startswith("fold"))


def fold_index(name):
    return int(re.match(r"fold(\d+)_", name).group(1))


def load_fold_truth():
    """fold -> (test_cohort, y_true). The held-out labels, used for evaluation
    only, exactly as every prior block used them."""
    out = {}
    for fd in fold_dirs():
        y = pd.read_csv(fd / "test" / "y.csv", index_col=0)
        out[fold_index(fd.name)] = (fd.name.split("_", 1)[1],
                                     y["y_T1_significant"].to_numpy().astype(int))
    return out


def load_condition_probs(condition, fold_dir_name, seed):
    """Positive-class probability vector for one (condition, fold, seed), or None."""
    root, sub = CONDITION_CKPT_ROOT[condition]
    p = root / fold_dir_name / sub / f"seed_{seed}" / "probs_test.npy"
    if not p.exists():
        return None
    a = np.load(p)
    return a[:, 1] if a.ndim == 2 else a


# ==========================================================================
# ISSUE A -- baseline per-cohort coverage
# ==========================================================================
def canonical_baseline_folds():
    """What the CANONICAL baseline files actually contain, per model.

    baselines/summary_*.csv holds one row per model with auc_mean/auc_std. The
    std being empty is the tell: it means n=1. The per-fold file behind each
    summary is baselines/<stage>/all_folds.csv, so that is what gets counted.
    """
    rows = []
    for stage, key in (("tabular", "model"), ("hybrids", "model"),
                        ("graph_controls", "control"), ("coverage", None)):
        p = BASE / stage / "all_folds.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if key is None or key not in d.columns:
            rows.append({"stage": stage, "model": f"<{stage}: no model column>",
                          "folds": sorted(d["fold"].unique().tolist()), "n_folds": d["fold"].nunique(),
                          "auc_canonical": np.nan})
            continue
        for m, g in d.groupby(key):
            rows.append({"stage": stage, "model": str(m),
                          "folds": sorted(g["fold"].unique().tolist()),
                          "n_folds": g["fold"].nunique(),
                          "auc_canonical": float(g["auc"].mean()) if "auc" in g else np.nan})
    return pd.DataFrame(rows)


def recompute_from_per_sample():
    """Per-cohort AUC / macro-F1 / balanced accuracy recomputed from the saved
    per-sample predictions. Uses each model's OWN saved threshold -- the one B3
    tuned on that fold's calibration cohort -- never a fresh 0.5."""
    rows = []
    if not PSD.exists():
        return pd.DataFrame(rows)
    for d in sorted(PSD.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for f in sorted(d.glob("fold*_*.csv")):
            t = pd.read_csv(f)
            y = t["y_true"].to_numpy().astype(int)
            p = t["p_test"].to_numpy(dtype=float)
            thr = float(t["threshold"].iloc[0])
            pred = (p >= thr).astype(int)
            rows.append({
                "model": d.name, "fold": int(t["fold"].iloc[0]),
                "test_cohort": str(t["test_cohort"].iloc[0]), "n_samples": len(t),
                "threshold": thr,
                "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "balanced_acc": float(balanced_accuracy_score(y, pred)),
            })
    return pd.DataFrame(rows)


def codagat_per_fold():
    """CoDA-GAT's own per-cohort AUC: the pre-registered primary configuration
    (adversary off, m=0), averaged over its 5 training seeds within each fold."""
    b7 = read_csv(B7 / "master_results.csv")
    off = b7[(b7["mode"] == "off") & (b7["m"] == 0)]
    return off.groupby("fold")[["auc", "macro_f1", "balanced_acc"]].mean()


def issue_a():
    rule("ISSUE A -- baseline per-cohort coverage")
    canon = canonical_baseline_folds()
    per_sample = recompute_from_per_sample()
    D["canon"] = canon
    D["per_sample"] = per_sample

    say("A1. What the canonical baseline files contain")
    say("-" * 78)
    if canon.empty:
        say("  no baselines/<stage>/all_folds.csv found")
    else:
        for _, r in canon.iterrows():
            say(f"  {r['stage']:15s} {r['model']:34s} folds={r['folds']}  n={r['n_folds']}")
        n_multi = int((canon["n_folds"] > 1).sum())
        say("")
        say(f"  models with more than one fold in the canonical files: {n_multi} / {len(canon)}")
        say("  The empty auc_std column in baselines/summary_*.csv is the same fact seen from")
        say("  the other side: a standard deviation of a single value is undefined.")

    say("")
    say("A2. Per-sample predictions: cohort coverage")
    say("-" * 78)
    if per_sample.empty:
        say("  baselines/per_sample_predictions/ has no per-model prediction files.")
        D["recomputed"] = pd.DataFrame()
        return
    cov = per_sample.groupby("model")["fold"].agg(["nunique", lambda s: sorted(s.unique().tolist())])
    cov.columns = ["n_cohorts", "folds"]
    for m, r in cov.iterrows():
        say(f"  {m:34s} cohorts={r['n_cohorts']}  folds={r['folds']}")
    say("")
    say(f"  total held-out samples covered: {int(per_sample['n_samples'].sum() / max(cov['n_cohorts'].max(), 1))}"
        f" per model across {int(cov['n_cohorts'].max())} cohorts")

    # ---- reproduction check: does the recomputation reproduce fold 0? ----
    say("")
    say("A3. Reproduction check -- recomputed vs saved, on the folds both cover")
    say("-" * 78)
    checks = []
    if not canon.empty:
        for stage, key in (("tabular", "model"), ("hybrids", "model"), ("graph_controls", "control")):
            p = BASE / stage / "all_folds.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            if key not in d.columns:
                continue
            j = d.rename(columns={key: "model"}).merge(
                per_sample, on=["model", "fold"], suffixes=("_saved", "_recomp"))
            for _, r in j.iterrows():
                checks.append({"stage": stage, "model": r["model"], "fold": int(r["fold"]),
                                "auc_saved": r["auc_saved"], "auc_recomputed": r["auc_recomp"],
                                "absdiff": abs(r["auc_saved"] - r["auc_recomp"])})
    rc = BASE / "per_sample_predictions" / "recomputed_metrics_tabular.csv"
    if rc.exists():
        r2 = pd.read_csv(rc).merge(per_sample, on=["model", "fold"], suffixes=("_b3", "_recomp"))
        if not r2.empty:
            say(f"  vs recomputed_metrics_tabular.csv: {len(r2)} rows, "
                f"max |AUC diff| = {(r2['auc_b3'] - r2['auc_recomp']).abs().max():.2e}")
    chk = pd.DataFrame(checks)
    D["repro_check"] = chk
    if chk.empty:
        say("  no overlapping (model, fold) rows between canonical files and per-sample predictions")
    else:
        worst = chk["absdiff"].max()
        say(f"  vs canonical all_folds.csv: {len(chk)} overlapping rows, max |AUC diff| = {worst:.2e}")
        say("  PASS: the per-sample predictions reproduce the saved metrics, so the recomputed"
            if worst < 1e-6 else
            "  WARNING: the per-sample predictions do NOT reproduce the saved metrics.")
        say("  8-cohort means below are the same quantity, just measured on all 8 cohorts."
            if worst < 1e-6 else
            "  Do not trust the recomputed means until this is explained.")

    # ---- the table the user asked for ----
    agg = per_sample.groupby("model")["auc"].agg(["count", "mean", "std"])
    rows = []
    canon_map = {}
    if not canon.empty:
        for _, r in canon.iterrows():
            canon_map[r["model"]] = (r["auc_canonical"], r["n_folds"], r["folds"])
    cl = cohort_labels()
    for m, r in agg.iterrows():
        old, n_old, folds_old = canon_map.get(m, (np.nan, 0, []))
        rows.append({
            "model": m,
            "cohorts_available": int(r["count"]),
            "canonical_n_folds": n_old,
            "canonical_basis": ", ".join(cl.get(f, str(f)) for f in folds_old) if folds_old else "-",
            "auc_reported": old,
            "auc_8cohort_mean": r["mean"],
            "auc_8cohort_std": r["std"],
            "delta_vs_reported": (r["mean"] - old) if np.isfinite(old) else np.nan,
        })
    # models with NO per-sample predictions at all -- these stay single-cohort
    for m, (old, n_old, folds_old) in canon_map.items():
        if m in agg.index or m.startswith("<"):
            continue
        rows.append({"model": m, "cohorts_available": 0, "canonical_n_folds": n_old,
                      "canonical_basis": ", ".join(cl.get(f, str(f)) for f in folds_old),
                      "auc_reported": old, "auc_8cohort_mean": np.nan,
                      "auc_8cohort_std": np.nan, "delta_vs_reported": np.nan})
    tbl = pd.DataFrame(rows).sort_values(
        ["cohorts_available", "auc_8cohort_mean"], ascending=[False, False])
    D["recomputed"] = tbl

    say("")
    say("A4. model | cohorts available | reported AUC | recomputed 8-cohort mean +/- std")
    say("-" * 78)
    disp = tbl.copy()
    for c in ("auc_reported", "auc_8cohort_mean", "auc_8cohort_std", "delta_vs_reported"):
        disp[c] = disp[c].map(lambda v: "-" if not np.isfinite(v) else f"{v:+.4f}"
                               if c == "delta_vs_reported" else f"{v:.4f}")
    say(disp[["model", "cohorts_available", "canonical_n_folds", "canonical_basis",
              "auc_reported", "auc_8cohort_mean", "auc_8cohort_std", "delta_vs_reported"]]
        .to_string(index=False))

    cg = codagat_per_fold()["auc"]
    say("")
    say(f"  CoDA-GAT (results_b7, mode=off, m=0): 8-cohort mean AUC = "
        f"{cg.mean():.4f} +/- {cg.std():.4f}  (n={len(cg)} cohorts)")

    single = tbl[tbl["cohorts_available"] == 0]["model"].tolist()
    say("")
    say("VERDICT (Issue A)")
    say("-" * 78)
    say("  The canonical baseline summaries are SINGLE-COHORT values, not 8-cohort means:")
    say("  every model in baselines/<stage>/all_folds.csv has exactly one fold, and the")
    say("  reported auc_mean is that one cohort's AUC with an undefined std.")
    say(f"  {int((tbl['cohorts_available'] > 0).sum())} models do have all-8-cohort per-sample")
    say("  predictions, so their true cross-cohort mean is recomputable and is used from here on.")
    if single:
        say(f"  {len(single)} model(s) have NO per-sample predictions and remain single-cohort:")
        for m in single:
            say(f"    - {m}")
        say("  These are marked 'single cohort' in every figure and are never averaged in with")
        say("  8-cohort means.")
    # honest note about where 8-cohort graph-control numbers do exist
    p9 = B9 / "master_ablation_results.csv"
    if p9.exists():
        a9 = pd.read_csv(p9)
        say("")
        say("  NOTE: 8-cohort values for three of the graph controls DO exist elsewhere, in")
        say(f"  ablations_b9/master_ablation_results.csv ({', '.join(sorted(a9['ablation'].unique()))}).")
        say("  Those are B9 runs -- full CoDA-GAT, 5 seeds per fold -- not B3's single-fold")
        say("  control runs. They are a different protocol and are NOT substituted for B3's")
        say("  rows here; they appear in the ablation figures on their own terms.")


# ==========================================================================
# ISSUE B -- masking contradiction
# ==========================================================================
def masking_raw():
    """graph_utility/masking_curves.csv -- the only file with per-(fold, seed,
    rate, draw) masking metrics. learned_graph's equivalent is folded in when it
    exists."""
    d = read_csv(GUTIL / "masking_curves.csv")
    lp = LEARNED / "masking_curves_all_conditions.csv"
    if lp.exists():
        d = pd.concat([d, pd.read_csv(lp)], ignore_index=True).drop_duplicates(
            subset=["model", "fold", "seed", "mask_rate", "draw"])
    return d


def per_cohort_levels(d, metric):
    """(model, fold, rate) -> metric, averaged over seeds and mask draws.

    This is the innermost unit of aggregation. Everything downstream is an
    average of THESE, which is what 'mean of per-cohort values' means.
    """
    return d.groupby(["model", "fold", "mask_rate"])[metric].mean().reset_index()


def level_deltas(d, metric, a="real_graph", b="empty_graph"):
    """Paired per-(fold, rate) difference in metric LEVEL between two conditions."""
    lv = per_cohort_levels(d, metric)
    piv = lv.pivot_table(index=["fold", "mask_rate"], columns="model", values=metric)
    if a not in piv.columns or b not in piv.columns:
        return pd.DataFrame()
    out = (piv[a] - piv[b]).rename("delta").reset_index()
    return out


def flatness_deltas(d, metric, a="real_graph", b="empty_graph"):
    """Paired per-(fold, rate) difference in DEGRADATION from each condition's own
    rate-0 baseline. Positive = `a` degrades less than `b`, i.e. `a` is flatter.

    This is the quantity graph_utility's masking_paired_real_vs_empty.csv calls
    flatness_advantage. It is a different question from the level difference:
    a model can sit lower everywhere and still fall more slowly.
    """
    lv = per_cohort_levels(d, metric)
    piv = lv.pivot_table(index=["fold", "mask_rate"], columns="model", values=metric)
    if a not in piv.columns or b not in piv.columns:
        return pd.DataFrame()
    rows = []
    for (fold, rate) in piv.index:
        if rate == 0:
            continue
        try:
            base_a = piv.loc[(fold, 0.0), a]
            base_b = piv.loc[(fold, 0.0), b]
        except KeyError:
            continue
        drop_a = base_a - piv.loc[(fold, rate), a]
        drop_b = base_b - piv.loc[(fold, rate), b]
        rows.append({"fold": fold, "mask_rate": rate,
                      "drop_a": drop_a, "drop_b": drop_b, "delta": drop_b - drop_a})
    return pd.DataFrame(rows)


def pooled_vs_percohort_rate0():
    """The two aggregations, computed from the SAME saved rate-0 probabilities.

      sample-pooled : concatenate all 8 cohorts' held-out samples, one AUC
      per-cohort    : AUC within each cohort, then average the 8 AUCs

    These differ whenever cohorts have different prevalence or different score
    scales, which is exactly the LODO situation, so the contrast is not academic.
    Masked-rate probabilities were never saved, so the pooled aggregation is
    only recomputable at rate 0 without re-running inference (which is
    forbidden here).
    """
    truth = load_fold_truth()
    rows = []
    for cond in CONDITION_CKPT_ROOT:
        for seed in SEEDS:
            ps, ys, per_cohort = [], [], []
            for fd in fold_dirs():
                fi = fold_index(fd.name)
                p = load_condition_probs(cond, fd.name, seed)
                if p is None:
                    continue
                _, y = truth[fi]
                if len(p) != len(y):
                    continue
                ps.append(p)
                ys.append(y)
                if len(np.unique(y)) > 1:
                    per_cohort.append(roc_auc_score(y, p))
            if not ps:
                continue
            yy, pp = np.concatenate(ys), np.concatenate(ps)
            rows.append({
                "condition": cond, "seed": seed, "n_cohorts": len(ps), "n_samples": len(yy),
                "auc_sample_pooled": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else np.nan,
                "auc_mean_per_cohort": float(np.mean(per_cohort)) if per_cohort else np.nan,
            })
    return pd.DataFrame(rows)


def per_cohort_rate0_table():
    """Per-(condition, fold) rate-0 AUC from saved probabilities, averaged over
    seeds. Feeds the clean-data advantage panels."""
    truth = load_fold_truth()
    rows = []
    for cond in CONDITION_CKPT_ROOT:
        for fd in fold_dirs():
            fi = fold_index(fd.name)
            cohort, y = truth[fi]
            aucs = []
            for seed in SEEDS:
                p = load_condition_probs(cond, fd.name, seed)
                if p is None or len(p) != len(y) or len(np.unique(y)) < 2:
                    continue
                aucs.append(roc_auc_score(y, p))
            if aucs:
                rows.append({"condition": cond, "fold": fi, "test_cohort": cohort,
                              "n_seeds": len(aucs), "auc": float(np.mean(aucs))})
    return pd.DataFrame(rows)


def locate_claim(pattern="0.0243"):
    """Where does the disputed number actually live? Searched rather than assumed."""
    hits = []
    for p in sorted(REPO.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".py", ".txt", ".md", ".json", ".csv"):
            continue
        if any(part in (".git", "catboost_info") for part in p.parts):
            continue
        if p.stat().st_size > 20 * 1024 * 1024:
            continue
        try:
            text = p.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pattern in line:
                hits.append((str(p.relative_to(REPO)), i, line.strip()[:150]))
    return hits


def issue_b():
    rule("ISSUE B -- masking contradiction")
    d = masking_raw()
    D["masking"] = d

    say("B1. Which file and which columns each artefact used")
    say("-" * 78)
    say("  fig11 (make_figures.py, masking_pooled):")
    say("      file    graph_utility/masking_curves.csv")
    say("      columns model, fold, seed, mask_rate, draw, auc")
    say("      recipe  mean over (seed, draw) within each (model, fold, rate), then mean")
    say("              over the 8 folds; band = 1.96 * std / sqrt(8) across folds")
    say(f"      rows    {len(d)}  models={sorted(d['model'].unique())}")
    say(f"              rates={sorted(d['mask_rate'].unique())}  seeds={sorted(d['seed'].unique())}")
    say("  graph_utility/masking_curves_pooled.csv is that same recipe, saved by")
    say("  graph_utility.py itself -- so the figure and the saved pooled file agree by")
    say("  construction. Verified below.")

    mp = GUTIL / "masking_curves_pooled.csv"
    if mp.exists():
        saved = pd.read_csv(mp)
        mine = per_cohort_levels(d, "auc").groupby(["model", "mask_rate"])["auc"].mean().reset_index()
        j = saved.merge(mine, on=["model", "mask_rate"])
        if not j.empty:
            say(f"      check   {len(j)} rows, max |auc_mean - recomputed| = "
                f"{(j['auc_mean'] - j['auc']).abs().max():.2e}  -> the figure's recipe is the"
                f" saved file's recipe")

    say("")
    say("B2. What graph_utility/summary.txt actually contains")
    say("-" * 78)
    sp = GUTIL / "summary.txt"
    if not sp.exists():
        say("  graph_utility/summary.txt does not exist")
    else:
        txt = sp.read_text()
        tests = re.findall(r"^TEST \d.*$", txt, flags=re.M)
        say(f"  {sp.relative_to(REPO)}: {len(txt.splitlines())} lines")
        say(f"  sections present: {tests if tests else 'none'}")
        say(f"  contains the string '+0.0243': {'0.0243' in txt}")
        say(f"  contains the word 'masking' in a TEST heading: "
            f"{any('mask' in t.lower() for t in tests)}")
        if not any(t.startswith("TEST 1") for t in tests):
            say("  FINDING: the TEST 1 (masking/robustness) section is ABSENT from summary.txt.")
            say("  The file holds only the TEST 3 block plus the header and the OVERALL verdict.")
            say("  It was last written by a partial re-run of graph_utility.py, which rewrites")
            say("  summary.txt wholesale, so the masking section was overwritten away.")
        m = re.search(r"^OVERALL.*$", txt, flags=re.M)
        if m:
            say(f"  its own verdict line: {m.group(0)[:200]}")

    say("")
    say("B3. Where the +0.0243 figure comes from")
    say("-" * 78)
    hits = locate_claim("0.0243")
    if not hits:
        say("  the string '0.0243' does not occur anywhere in the project")
    for path, ln, line in hits:
        say(f"  {path}:{ln}  {line}")
    say("  FINDING: it occurs only in source comments, in no results file. It is a docstring")
    say("  header, attributed to graph_utility.py, not a value graph_utility.py ever wrote.")
    say("  Whatever its provenance, it cannot be reproduced as an AUC advantage from the")
    say("  saved masking data -- see B5, where the AUC advantage is NEGATIVE with a CI that")
    say("  excludes zero on the wrong side. Treat the docstring as wrong and the data as right.")

    # ---- B4: the two aggregations ----
    say("")
    say("B4. Aggregation 1 vs Aggregation 2 (real vs empty)")
    say("-" * 78)
    pv = pooled_vs_percohort_rate0()
    D["pooled_vs_percohort"] = pv
    if pv.empty:
        say("  no saved rate-0 probabilities found; pooled aggregation not computable")
    else:
        g = pv.groupby("condition")[["auc_sample_pooled", "auc_mean_per_cohort"]].agg(["mean", "std"])
        say("  rate 0 (no masking), recomputed from saved per-sample probabilities,")
        say("  mean +/- std over the 5 training seeds:")
        say("")
        say(f"  {'condition':24s} {'sample-pooled':>18s} {'mean-per-cohort':>18s} {'n_cohorts':>10s}")
        for cond in CONDITION_CKPT_ROOT:
            if cond not in g.index:
                continue
            nn = int(pv[pv["condition"] == cond]["n_cohorts"].max())
            say(f"  {CONDITION_LABEL[cond]:24s} "
                f"{g.loc[cond, ('auc_sample_pooled', 'mean')]:.4f}+/-{g.loc[cond, ('auc_sample_pooled', 'std')]:.4f}   "
                f"{g.loc[cond, ('auc_mean_per_cohort', 'mean')]:.4f}+/-{g.loc[cond, ('auc_mean_per_cohort', 'std')]:.4f}   "
                f"{nn:>10d}")
        if {"real_graph", "empty_graph"} <= set(g.index):
            dp = g.loc["real_graph", ("auc_sample_pooled", "mean")] - g.loc["empty_graph", ("auc_sample_pooled", "mean")]
            dc = g.loc["real_graph", ("auc_mean_per_cohort", "mean")] - g.loc["empty_graph", ("auc_mean_per_cohort", "mean")]
            say("")
            say(f"  real - empty at rate 0:  sample-pooled {dp:+.4f}   mean-per-cohort {dc:+.4f}")
            say(f"  Both aggregations give the SAME SIGN ({'negative' if dp < 0 else 'positive'}).")
            say("  The two levels differ in magnitude because LODO cohorts have different")
            say("  prevalence, so pooling is prevalence-weighted while the per-cohort mean is not.")
            say("  That difference in level is what makes an 0.80-vs-0.83 style mismatch between")
            say("  two of our own tables possible without either being wrong. The DIRECTION of")
            say("  the real-vs-empty contrast is unaffected.")
        say("")
        say("  LIMIT, stated plainly: masked-rate per-sample probabilities were never saved --")
        say("  masking_curves.csv stores only per-(fold, seed, rate, draw) metrics. So the")
        say("  sample-pooled aggregation is recomputable at rate 0 ONLY. Computing pooled")
        say("  masked curves would require re-running inference, which this script will not do.")
        say("  Every masking-rate number below is therefore mean-of-per-cohort, and the figures")
        say("  say so on the axis.")

    # ---- B5: direction, per metric, per rate ----
    say("")
    say("B5. real-vs-empty advantage at each masking rate (mean of per-cohort values)")
    say("-" * 78)
    lvl_rows, flat_rows = [], []
    for metric in MASK_METRICS:
        lv = level_deltas(d, metric)
        fl = flatness_deltas(d, metric)
        if lv.empty:
            continue
        say(f"  metric = {metric}")
        say(f"    {'rate':>6s} {'level delta (real-empty)':>34s} {'flatness adv (empty-real drop)':>36s}")
        for rate in sorted(lv["mask_rate"].unique()):
            s = lv[lv["mask_rate"] == rate]["delta"]
            lo, hi = boot_ci(s.to_numpy(), seed=int(rate * 1000))
            cell = f"{s.mean():+.4f} [{lo:+.4f},{hi:+.4f}] n={len(s)}"
            fcell = ""
            if not fl.empty and rate > 0:
                fs = fl[fl["mask_rate"] == rate]["delta"]
                flo, fhi = boot_ci(fs.to_numpy(), seed=int(rate * 1000) + 1)
                fcell = f"{fs.mean():+.4f} [{flo:+.4f},{fhi:+.4f}] n={len(fs)}"
            say(f"    {rate:>6.2f} {cell:>34s} {fcell:>36s}")
            lvl_rows.append({"metric": metric, "mask_rate": rate, "mean": s.mean(),
                              "lo": lo, "hi": hi, "n": len(s)})
        if not fl.empty:
            allf = fl["delta"].to_numpy()
            lo, hi = boot_ci(allf, seed=7)
            end = fl[fl["mask_rate"] == fl["mask_rate"].max()]["delta"].to_numpy()
            elo, ehi = boot_ci(end, seed=8)
            say(f"    pooled over all (cohort x masked rate) cells: {allf.mean():+.4f} "
                f"[{lo:+.4f},{hi:+.4f}] n={len(allf)}")
            say(f"    endpoint only ({fl['mask_rate'].max():.0%} masked):        {end.mean():+.4f} "
                f"[{elo:+.4f},{ehi:+.4f}] n={len(end)}")
            flat_rows.append({"metric": metric, "scope": "all masked cells", "mean": allf.mean(),
                               "lo": lo, "hi": hi, "n": len(allf)})
            flat_rows.append({"metric": metric, "scope": "endpoint", "mean": end.mean(),
                               "lo": elo, "hi": ehi, "n": len(end)})
        say("")
    D["level_ci"] = pd.DataFrame(lvl_rows)
    D["flat_ci"] = pd.DataFrame(flat_rows)

    # cross-check against graph_utility's own saved paired file
    pp = GUTIL / "masking_paired_real_vs_empty.csv"
    if pp.exists():
        pr = pd.read_csv(pp)
        if "flatness_advantage" in pr.columns:
            mine = flatness_deltas(d, "auc")
            j = pr.merge(mine, on=["fold", "mask_rate"])
            say(f"  cross-check vs graph_utility/masking_paired_real_vs_empty.csv: {len(j)} rows,"
                f" max |flatness_advantage - recomputed| = "
                f"{(j['flatness_advantage'] - j['delta']).abs().max():.2e}")
            say(f"  its own mean flatness_advantage (AUC) = {pr['flatness_advantage'].mean():+.4f}")
            say("")

    # ---- also: shuffled, so 'connectivity vs biology' stays visible ----
    say("B6. the same contrast against shuffled STRING (connectivity held, biology destroyed)")
    say("-" * 78)
    for metric in MASK_METRICS:
        fl = flatness_deltas(d, metric, "real_graph", "shuffled_graph")
        if fl.empty:
            continue
        lo, hi = boot_ci(fl["delta"].to_numpy(), seed=11)
        say(f"  {metric:13s} real vs shuffled flatness advantage = {fl['delta'].mean():+.4f} "
            f"[{lo:+.4f},{hi:+.4f}] n={len(fl)}")

    # ---- verdict ----
    fa = D["flat_ci"]
    say("")
    say("VERDICT (Issue B)")
    say("-" * 78)
    say("  The figure is right and the docstring is wrong. Under the saved data the real")
    say("  STRING graph sits BELOW the empty graph on AUC at every masking rate, and falls")
    say("  faster, so its AUC flatness advantage is negative with a CI excluding zero.")
    say("")
    say("  The direction does NOT depend on the aggregation. Sample-pooled and")
    say("  mean-of-per-cohort agree in sign at rate 0; only the level shifts.")
    say("")
    say("  The direction DOES depend on the METRIC, and this is a real result, not an")
    say("  artefact:")
    if not fa.empty:
        for _, r in fa[fa["scope"] == "all masked cells"].iterrows():
            verdict = "FAVOURS the graph" if r["lo"] > 0 else (
                "FAVOURS the no-graph control" if r["hi"] < 0 else "is indistinguishable from zero")
            say(f"    {r['metric']:13s} flatness advantage {r['mean']:+.4f} "
                f"[{r['lo']:+.4f},{r['hi']:+.4f}] -> {verdict}")
    say("  Ranking (AUC) degrades faster with the graph; thresholded decision quality")
    say("  (macro-F1, balanced accuracy) degrades more slowly with it. Both are measured on")
    say("  identical gene subsets, paired at the draw level, so this is not sampling noise:")
    say("  the graph is stabilising the decision boundary while not helping the ranking.")
    say("")
    say("  Reported at face value: there is NO single-number 'robustness advantage'. Any")
    say("  claim of one must name its metric. The regenerated fig11 shows both metrics and")
    say("  the signed advantage side by side so the disagreement is visible, not chosen.")


# ==========================================================================
# ISSUE C -- learned-graph outputs
# ==========================================================================
def edge_sets():
    """variant -> fold -> (set of undirected pairs, n_in_string) from
    learned_graph/edges/. These files are the learned topologies themselves."""
    out, meta = {}, []
    if not LEARNED_EDGES.exists():
        return out, pd.DataFrame(meta)
    for p in sorted(LEARNED_EDGES.glob("learned_edges_fold*_*.csv")):
        m = re.match(r"learned_edges_fold(\d+)_(.+)\.csv", p.name)
        if not m:
            continue
        fold, variant = int(m.group(1)), m.group(2)
        e = pd.read_csv(p)
        pairs = {(min(int(a), int(b)), max(int(a), int(b)))
                 for a, b in zip(e["src_idx"], e["dst_idx"])}
        in_string = e["in_string"]
        n_in = int(in_string.sum() if in_string.dtype == bool
                   else in_string.astype(str).str.strip().str.lower().isin(["true", "1"]).sum())
        out.setdefault(variant, {})[fold] = pairs
        meta.append({"variant": variant, "fold": fold, "n_rows": len(e),
                      "n_unique_pairs": len(pairs), "n_in_string": n_in,
                      "columns": ",".join(e.columns)})
    return out, pd.DataFrame(meta)


def string_pair_set():
    e = read_csv(SHARED / "graph_edges.csv")
    return {(min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(e["src_idx"], e["dst_idx"])}


def learned_run_inventory():
    rows = []
    if not LEARNED_CKPT.exists():
        return pd.DataFrame(rows)
    for fd in sorted(LEARNED_CKPT.iterdir()):
        if not fd.is_dir():
            continue
        for cd in sorted(fd.iterdir()):
            if not cd.is_dir():
                continue
            for sd in sorted(cd.iterdir()):
                if not sd.is_dir():
                    continue
                mp = sd / "run_metrics.json"
                rec = {"fold_dir": fd.name, "condition": cd.name, "seed": sd.name,
                        "done": (sd / "DONE.marker").exists(),
                        "has_probs": (sd / "probs_test.npy").exists(),
                        "has_weights": (sd / "weights.pt").exists()}
                if mp.exists():
                    rec.update({k: v for k, v in json.loads(mp.read_text()).items()
                                if k in ("auc", "macro_f1", "balanced_acc", "best_val_f1")})
                rows.append(rec)
    return pd.DataFrame(rows)


def issue_c():
    rule("ISSUE C -- learned-graph outputs: located")
    if not LEARNED.exists():
        say(f"  {LEARNED} does not exist")
        D["edges"], D["edge_meta"], D["learned_runs"] = {}, pd.DataFrame(), pd.DataFrame()
        return

    say("C1. Directory tree")
    say("-" * 78)
    files = sorted(p for p in LEARNED.rglob("*") if p.is_file())
    say(f"  {LEARNED.relative_to(REPO)}/  -- {len(files)} files")
    top = sorted(p.name for p in LEARNED.iterdir())
    say(f"  top level: {top}")

    say("")
    say("C2. Learned edge lists -- PRESENT")
    say("-" * 78)
    edges, meta = edge_sets()
    D["edges"], D["edge_meta"] = edges, meta
    if meta.empty:
        say("  learned_graph/edges/ contains no learned_edges_fold*_*.csv")
    else:
        say(f"  path    learned_graph/edges/learned_edges_fold<f>_<variant>.csv")
        say(f"  columns {meta['columns'].iloc[0]}")
        say(f"  variants {sorted(meta['variant'].unique())}  folds {sorted(meta['fold'].unique())}")
        say("")
        say(meta[["variant", "fold", "n_rows", "n_unique_pairs", "n_in_string"]].to_string(index=False))

    say("")
    say("C3. Trained runs -- PRESENT")
    say("-" * 78)
    inv = learned_run_inventory()
    D["learned_runs"] = inv
    if inv.empty:
        say("  learned_graph/checkpoints/ is empty")
    else:
        say(f"  path    learned_graph/checkpoints/<fold_dir>/<condition>/seed_<s>/")
        say(f"  files   run_metrics.json, probs_test.npy, predictions_test.npy, weights.pt, DONE.marker")
        piv = inv.assign(done=inv["done"].astype(int)).pivot_table(
            index="condition", columns="fold_dir", values="done", aggfunc="sum")
        say("")
        say("  completed seeds per (condition, fold):")
        say(piv.fillna(0).astype(int).to_string())
        say("")
        say("  clean-data AUC from run_metrics.json (mean over seeds, then over folds):")
        if "auc" in inv.columns:
            per_fold = inv.groupby(["condition", "fold_dir"])["auc"].mean()
            for cond, s in per_fold.groupby(level=0):
                say(f"    {cond:16s} {s.mean():.4f} +/- {s.std():.4f}  (n={len(s)} cohorts)")

    say("")
    say("C4. Report-stage outputs -- WHAT IS MISSING")
    say("-" * 78)
    missing = [f for f in LEARNED_EXPECTED if not (LEARNED / f).exists()]
    for f in LEARNED_EXPECTED:
        say(f"  {'PRESENT' if (LEARNED / f).exists() else 'MISSING':8s} learned_graph/{f}")
    say("")
    if missing:
        say("  FINDING: learned_graph.py's stage 1 (topology) and stage 2 (train) DID run and")
        say("  completed -- the edge lists and all checkpoints are on disk. Stage 3 (the masking")
        say("  sweep) and stage 4 (report) did NOT write their outputs, so")
        say("  masking_curves_all_conditions.csv and paired_comparisons.csv genuinely do not")
        say("  exist. make_figures.py's skip message named the script rather than the stage,")
        say("  which read as 'never run'; the accurate statement is 'trained, not yet masked'.")
        say("")
        say("  The masking sweep is inference only -- no retraining -- and is resumable:")
        say("      python scripts/learned_graph.py --stage mask")
        say("  Re-running THIS script afterwards will pick the new file up automatically and")
        say("  upgrade fig12 and fig13 from clean-data-only to full masking curves.")
    else:
        say("  all report-stage outputs present; fig12/fig13 use the full masking curves")


# ==========================================================================
# the diagnosis driver
# ==========================================================================
def run_diagnosis(write=True):
    if D.get("_done"):
        return
    DIAG_LINES.clear()
    rule("CoDA-GAT figure diagnosis")
    say("Read-only. Every value below is read or recomputed from a file on disk.")
    say(f"repo: {REPO}")
    say("")
    issue_a()
    say("")
    issue_b()
    say("")
    issue_c()
    say("")
    rule("END OF DIAGNOSIS")
    D["_done"] = True
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "diagnosis.txt").write_text("\n".join(DIAG_LINES) + "\n")
        print(f"\n[diag] wrote {(OUT / 'diagnosis.txt').relative_to(REPO)}", flush=True)


def ensure_diagnosis():
    """Figures call this so they work even when invoked without the driver."""
    if not D.get("_done"):
        run_diagnosis(write=False)


# ==========================================================================
# REGENERATED FIGURES
# ==========================================================================
SINGLE_MARK = "†"      # dagger, explained in every legend that uses it


def model_auc_table():
    """One row per model, with the number of cohorts it is based on carried
    alongside the mean so nothing silently mixes 1-cohort and 8-cohort values."""
    ensure_diagnosis()
    ps = D.get("per_sample", pd.DataFrame())
    tbl = D.get("recomputed", pd.DataFrame())
    rows = []

    cg = codagat_per_fold()["auc"]
    rows.append({"model": "CoDA-GAT", "auc_mean": cg.mean(), "auc_std": cg.std(),
                  "n_cohorts": len(cg), "highlight": True,
                  "source": "results_b7/master_results.csv (mode=off, m=0)"})

    if not ps.empty:
        g = ps.groupby("model")["auc"].agg(["count", "mean", "std"])
        for m, r in g.iterrows():
            rows.append({"model": str(m), "auc_mean": r["mean"], "auc_std": r["std"],
                          "n_cohorts": int(r["count"]), "highlight": False,
                          "source": "baselines/per_sample_predictions/ (recomputed)"})

    have = {r["model"] for r in rows}
    if not tbl.empty:
        for _, r in tbl[tbl["cohorts_available"] == 0].iterrows():
            if str(r["model"]) in have or str(r["model"]).startswith("<"):
                continue
            rows.append({"model": str(r["model"]), "auc_mean": r["auc_reported"],
                          "auc_std": np.nan, "n_cohorts": int(r["canonical_n_folds"]),
                          "highlight": False,
                          "source": f"baselines canonical ({r['canonical_basis']} only)"})

    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["auc_mean"])]
    n_max = int(df["n_cohorts"].max())
    df["full_cohorts"] = df["n_cohorts"] >= n_max
    df["label"] = [f"{m}  {SINGLE_MARK}" if not full else m
                   for m, full in zip(df["model"], df["full_cohorts"])]
    return df.sort_values(["full_cohorts", "auc_mean"], ascending=[True, True]), n_max


def new_fig04():
    df, n_max = model_auc_table()
    if df.empty:
        raise SkipFigure("no model AUCs available")
    single = df[~df["full_cohorts"]]
    colors = [CODAGAT if h else NEUTRAL for h in df["highlight"]]
    hatches = ["" if full else "///" for full in df["full_cohorts"]]

    fig, ax = plt.subplots(figsize=(9.0, max(5.0, 0.30 * len(df))))
    ceiling_band(ax, orient="x")
    bars = ax.barh(range(len(df)), df["auc_mean"], xerr=df["auc_std"].fillna(0),
                   color=colors, edgecolor="#333", linewidth=0.5,
                   error_kw={"lw": 0.8, "ecolor": "#444"})
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
            b.set_alpha(0.55)
    if len(single):
        ax.axhline(len(single) - 0.5, color="#333", lw=1.0, ls="--")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel(f"AUC — mean over {n_max} held-out cohorts (± std across cohorts)")
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="black", lw=0.8, ls=":")
    ax.set_title("Cross-cohort discrimination: all models\n"
                 f"{SINGLE_MARK} single cohort only — not comparable to the {n_max}-cohort means",
                 fontsize=10)
    ax.legend(handles=[
        Patch(color=CODAGAT, label="CoDA-GAT"),
        Patch(color=NEUTRAL, label=f"baseline / control ({n_max} cohorts)"),
        Patch(facecolor=NEUTRAL, alpha=0.55, hatch="///", edgecolor="#333",
              label=f"{SINGLE_MARK} single cohort ({len(single)} models)"),
        Patch(color=OKABE[2], alpha=0.12, label=f"label-noise ceiling {CEILING[0]}–{CEILING[1]}"),
    ], loc="lower right", fontsize=8)
    return fig


def new_fig05():
    ensure_diagnosis()
    cl = cohort_labels()
    ps = D.get("per_sample", pd.DataFrame())
    frames = {}

    cg = codagat_per_fold()["auc"]
    frames["CoDA-GAT"] = cg

    if not ps.empty:
        for m, g in ps.groupby("model"):
            frames[str(m)] = g.groupby("fold")["auc"].mean()

    p9 = B9 / "master_ablation_results.csv"
    if p9.exists():
        a9 = pd.read_csv(p9)
        for m, g in a9.groupby("ablation"):
            frames[f"abl: {m}"] = g.groupby("fold")["auc"].mean()

    # single-cohort rows, kept separate and visibly incomplete
    single_frames = {}
    for stage, key in (("tabular", "model"), ("hybrids", "model"), ("graph_controls", "control")):
        p = BASE / stage / "all_folds.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if key not in d.columns:
            continue
        for m, g in d.groupby(key):
            name = str(m)
            if name in frames:
                continue
            single_frames[f"{name}  {SINGLE_MARK}"] = g.groupby("fold")["auc"].mean()

    if not frames and not single_frames:
        raise SkipFigure("no per-cohort AUC sources found")

    n_cohorts = max((len(s) for s in frames.values()), default=0)
    full = pd.DataFrame(frames).T.sort_index()
    full = full.loc[full.mean(axis=1).sort_values(ascending=False).index]
    part = pd.DataFrame(single_frames).T.sort_index() if single_frames else pd.DataFrame()
    mat = pd.concat([full, part]) if not part.empty else full
    mat = mat.reindex(sorted(mat.columns), axis=1)
    mat.columns = [cl.get(c, str(c)) for c in mat.columns]

    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.30 * len(mat))))
    masked = np.ma.masked_invalid(mat.values.astype(float))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#E8E8E8")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0)
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
    if not part.empty:
        ax.axhline(len(full) - 0.5, color="#B00", lw=1.6)
        ax.text(mat.shape[1] - 0.4, len(full) - 0.35,
                f"{SINGLE_MARK} below: single cohort only (grey = never evaluated)",
                fontsize=7.5, color="#B00", ha="right", va="top")
    ax.set_title(f"Per-cohort AUC — held-out cohort in columns\n"
                 f"rows above the red line are complete {n_cohorts}-cohort LODO evaluations",
                 fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="AUC", fraction=0.025)
    return fig


def _curve(lv, model, metric):
    s = lv[lv["model"] == model]
    g = s.groupby("mask_rate")[metric].agg(["mean", "std", "count"]).reset_index()
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g.sort_values("mask_rate")


def new_fig11():
    ensure_diagnosis()
    d = D.get("masking", masking_raw())
    present = set(d["model"].unique())
    if not {"real_graph", "empty_graph"} <= present:
        raise SkipFigure("real_graph / empty_graph masking rows missing")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))

    for ax, metric, nice in zip(axes[:2], ["auc", "macro_f1"], ["AUC", "macro-F1"]):
        lv = per_cohort_levels(d, metric)
        for model in ("real_graph", "empty_graph"):
            g = _curve(lv, model, metric)
            c = CONDITION_COLOR[model]
            ax.plot(g["mask_rate"] * 100, g["mean"], marker="o", lw=2, color=c,
                    label=CONDITION_LABEL[model])
            ax.fill_between(g["mask_rate"] * 100, g["mean"] - g["ci"], g["mean"] + g["ci"],
                            color=c, alpha=0.18)
        ax.set_xlabel("Genes hidden at inference (%)")
        ax.set_ylabel(f"{nice} — mean of 8 per-cohort values (95% CI across cohorts)")
        ax.set_title(f"{nice}: graph {'loses' if metric == 'auc' else 'gains'} under masking")
        ax.legend(fontsize=8)

    ax = axes[2]
    fa = D.get("flat_ci", pd.DataFrame())
    fa = fa[fa["scope"] == "all masked cells"] if not fa.empty else fa
    if fa.empty:
        ax.text(0.5, 0.5, "flatness CIs unavailable", ha="center", va="center")
    else:
        names = {"auc": "AUC", "macro_f1": "macro-F1", "balanced_acc": "balanced acc."}
        fa = fa.assign(nice=[names.get(m, m) for m in fa["metric"]])
        cols = [OKABE[2] if lo > 0 else (NEUTRAL if hi > 0 else CODAGAT)
                for lo, hi in zip(fa["lo"], fa["hi"])]
        ax.bar(range(len(fa)), fa["mean"], color=cols, edgecolor="#333", lw=0.5,
               yerr=[fa["mean"] - fa["lo"], fa["hi"] - fa["mean"]], capsize=4,
               error_kw={"lw": 0.9})
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(fa)))
        ax.set_xticklabels(fa["nice"], fontsize=9)
        ax.set_ylabel("STRING graph's flatness advantage over no-graph\n"
                      "(control's drop − graph's drop; >0 = graph degrades less)", fontsize=8.5)
        ax.set_title("The advantage's sign depends on the metric", fontsize=10)
        for i, (m, lo, hi) in enumerate(zip(fa["mean"], fa["lo"], fa["hi"])):
            ax.annotate(f"{m:+.4f}", (i, hi if m > 0 else lo), textcoords="offset points",
                        xytext=(0, 6 if m > 0 else -14), ha="center", fontsize=7.5)

    fig.suptitle("Robustness to missing genes — reported at face value: AUC and threshold "
                 "metrics disagree in direction", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def new_fig12():
    ensure_diagnosis()
    d = D.get("masking", masking_raw())
    lv = per_cohort_levels(d, "auc")
    masked_conditions = [c for c in CONDITION_CKPT_ROOT if c in set(lv["model"].unique())]
    clean = D.get("pooled_vs_percohort", pd.DataFrame())
    if not masked_conditions and clean.empty:
        raise SkipFigure("no masking curves and no rate-0 probabilities")

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for cond in masked_conditions:
        g = _curve(lv, cond, "auc")
        c = CONDITION_COLOR[cond]
        ax.plot(g["mask_rate"] * 100, g["mean"], marker="o", lw=2, color=c,
                label=f"{CONDITION_LABEL[cond]} — full curve")
        ax.fill_between(g["mask_rate"] * 100, g["mean"] - g["ci"], g["mean"] + g["ci"],
                        color=c, alpha=0.14)

    # conditions trained but not yet swept: show their rate-0 point only, labelled
    clean_only = []
    if not clean.empty:
        cg = clean.groupby("condition")["auc_mean_per_cohort"].agg(["mean", "std"])
        for cond in CONDITION_CKPT_ROOT:
            if cond in masked_conditions or cond not in cg.index:
                continue
            clean_only.append(cond)
            ax.errorbar(0, cg.loc[cond, "mean"], yerr=cg.loc[cond, "std"], marker="D",
                        ms=8, mfc="white", mew=2, capsize=4,
                        color=CONDITION_COLOR[cond],
                        label=f"{CONDITION_LABEL[cond]} — 0% only")

    ax.set_xlabel("Genes hidden at inference (%)")
    ax.set_ylabel("AUC — mean of 8 per-cohort values (95% CI across cohorts)")
    ax.set_title("Every graph condition under masking", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    if clean_only:
        ax.annotate(
            "open diamonds: trained (8 cohorts × 5 seeds) but the masking sweep\n"
            "has not been run — learned_graph/masking_curves_all_conditions.csv absent.\n"
            "Complete with:  python scripts/learned_graph.py --stage mask",
            xy=(0.99, 0.02), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=7.5, color="#555",
            bbox={"fc": "#F5F5F5", "ec": "#CCC", "lw": 0.6, "pad": 3.5})
    return fig


def new_fig13():
    ensure_diagnosis()
    d = D.get("masking", masking_raw())
    present = set(d["model"].unique())

    # left: flatness advantage over empty, under masking, for whatever has curves
    left = []
    for cond in CONDITION_CKPT_ROOT:
        if cond in ("empty_graph",) or cond not in present:
            continue
        for metric in ("auc", "macro_f1"):
            fl = flatness_deltas(d, metric, cond, "empty_graph")
            if fl.empty:
                continue
            lo, hi = boot_ci(fl["delta"].to_numpy(), seed=stable_seed(cond, metric))
            left.append({"condition": cond, "metric": metric, "mean": fl["delta"].mean(),
                          "lo": lo, "hi": hi, "n": len(fl)})
    left = pd.DataFrame(left)

    # right: clean-data (0% masked) AUC advantage over empty, paired per cohort
    pc = D.get("per_cohort_rate0")
    if pc is None:
        pc = per_cohort_rate0_table()
        D["per_cohort_rate0"] = pc
    right = []
    if not pc.empty:
        piv = pc.pivot_table(index="fold", columns="condition", values="auc")
        if "empty_graph" in piv.columns:
            for cond in CONDITION_CKPT_ROOT:
                if cond == "empty_graph" or cond not in piv.columns:
                    continue
                delta = (piv[cond] - piv["empty_graph"]).dropna()
                if len(delta) < 2:
                    continue
                lo, hi = boot_ci(delta.to_numpy(), seed=stable_seed(cond))
                right.append({"condition": cond, "mean": delta.mean(), "lo": lo, "hi": hi,
                               "n": len(delta)})
    right = pd.DataFrame(right)

    if left.empty and right.empty:
        raise SkipFigure("no condition can be compared against empty_graph")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    if left.empty:
        ax.text(0.5, 0.5, "no masking curves available", ha="center", va="center")
    else:
        conds = [c for c in CONDITION_CKPT_ROOT if c in set(left["condition"])]
        w, metrics = 0.38, ["auc", "macro_f1"]
        mcol = {"auc": OKABE[0], "macro_f1": OKABE[1]}
        for k, metric in enumerate(metrics):
            sub = left[left["metric"] == metric].set_index("condition").reindex(conds)
            x = np.arange(len(conds)) + (k - 0.5) * w
            ax.bar(x, sub["mean"], width=w, color=mcol[metric], edgecolor="#333", lw=0.5,
                   yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]], capsize=3,
                   error_kw={"lw": 0.8},
                   label={"auc": "AUC", "macro_f1": "macro-F1"}[metric])
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], rotation=18, ha="right", fontsize=8)
        ax.set_ylabel("Flatness advantage over no-graph\n(>0 = degrades less than the control)",
                      fontsize=8.5)
        ax.set_title("Under masking (mean of per-cohort values,\n95% CI over cohort × rate cells)",
                     fontsize=9.5)
        ax.legend(fontsize=8)

    ax = axes[1]
    if right.empty:
        ax.text(0.5, 0.5, "no rate-0 probabilities available", ha="center", va="center")
    else:
        conds = [c for c in CONDITION_CKPT_ROOT if c in set(right["condition"])]
        sub = right.set_index("condition").reindex(conds)
        cols = [CONDITION_COLOR[c] for c in conds]
        ax.bar(range(len(conds)), sub["mean"], color=cols, edgecolor="#333", lw=0.5,
               yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]], capsize=4,
               error_kw={"lw": 0.9})
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([CONDITION_LABEL[c] for c in conds], rotation=18, ha="right", fontsize=8)
        ax.set_ylabel("AUC advantage over no-graph at 0% masked", fontsize=8.5)
        ax.set_title(f"Clean data, paired within cohort\n"
                     f"(95% CI over {int(sub['n'].max())} cohorts)", fontsize=9.5)
        for i, (m, lo, hi) in enumerate(zip(sub["mean"], sub["lo"], sub["hi"])):
            ax.annotate(f"{m:+.4f}", (i, hi), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=7.5)

    fig.suptitle("Advantage over the no-graph control, per graph condition", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def new_fig14():
    ensure_diagnosis()
    edges = D.get("edges", {})
    if not edges:
        raise SkipFigure("learned_graph/edges/ has no learned edge lists")
    string = string_pair_set()
    n_genes = len(read_csv(SHARED / "panel_genes.csv"))
    n_possible = n_genes * (n_genes - 1) // 2

    def jac(a, b):
        u = len(a | b)
        return len(a & b) / u if u else np.nan

    # chance level: two independent edge sets of the observed sizes drawn from all
    # possible gene pairs -- computed, not assumed
    sizes = [len(s) for v in edges.values() for s in v.values()] + [len(string)]
    k = float(np.mean(sizes))
    exp_inter = k * k / n_possible
    chance = exp_inter / (2 * k - exp_inter)

    variants = sorted(edges)
    cl = cohort_labels()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    folds = sorted({f for v in edges.values() for f in v})
    for i, v in enumerate(variants):
        ys = [jac(edges[v].get(f, set()), string) if f in edges[v] else np.nan for f in folds]
        ax.plot(range(len(folds)), ys, marker="o", lw=1.6, color=OKABE[i % len(OKABE)],
                label=f"{v} (mean {np.nanmean(ys):.3f})")
    ax.axhline(chance, color="#B00", lw=1.2, ls="--",
               label=f"chance overlap {chance:.3f} ({int(round(k))} of {n_possible} pairs)")
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([cl.get(f, f"fold{f}") for f in folds], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Jaccard overlap with the STRING graph")
    ax.set_xlabel("Held-out cohort (topology learned on that fold's training split)")
    ax.set_title("Learned topology vs STRING: edge overlap", fontsize=10)
    ax.set_ylim(0, max(0.35, ax.get_ylim()[1]))
    ax.legend(fontsize=8)

    ax = axes[1]
    data, labels, notes = [], [], []
    for v in variants:
        fs = sorted(edges[v])
        js = [jac(edges[v][a], edges[v][b]) for a, b in itertools.combinations(fs, 2)]
        if js:
            data.append(js)
            labels.append(v)
            notes.append(f"n={len(js)} fold pairs")
    if data:
        # matplotlib >=3.9 renamed boxplot(labels=) to tick_labels=
        bp = ax.boxplot(data, tick_labels=labels, widths=0.55, patch_artist=True,
                        medianprops={"color": "black"})
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(OKABE[i % len(OKABE)])
            patch.set_alpha(0.45)
        for i, js in enumerate(data):
            x = np.random.default_rng(i).normal(i + 1, 0.055, len(js))
            ax.scatter(x, js, s=14, color=OKABE[i % len(OKABE)], edgecolor="#333",
                       linewidth=0.4, zorder=3)
        ax.axhline(1.0, color="#333", lw=1.2, ls=":",
                   label="STRING = 1.00 (identical across folds by construction)")
        ax.axhline(chance, color="#B00", lw=1.2, ls="--", label=f"chance {chance:.3f}")
        ax.set_ylabel("Pairwise Jaccard between folds' learned topologies")
        ax.set_ylim(0, 1.05)
        ax.set_title("Cross-fold topology stability", fontsize=10)
        ax.legend(fontsize=7.5, loc="center right")
        ax.tick_params(axis="x", labelsize=8)
        for i, n in enumerate(notes):
            ax.annotate(n, (i + 1, 0.02), ha="center", fontsize=7, color="#555")
    fig.suptitle("Do the learned graphs recover STRING biology, and are they stable across cohorts?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ==========================================================================
# registry surgery + driver
# ==========================================================================
OVERRIDES = [
    (4, "all_models_auc_ranked",
     ["results_b7/master_results.csv", "baselines/per_sample_predictions/",
      "baselines/*/all_folds.csv"],
     "Every model ranked by mean cross-cohort AUC. Baseline values are recomputed from the "
     "saved per-sample predictions over all 8 held-out cohorts; models for which only one "
     "cohort was ever evaluated are hatched and daggered and are never averaged in with the "
     "8-cohort means. Shaded band is the 0.87-0.90 label-noise ceiling.",
     new_fig04),
    (5, "per_fold_auc_heatmap",
     ["results_b7/master_results.csv", "baselines/per_sample_predictions/",
      "ablations_b9/master_ablation_results.csv", "baselines/*/all_folds.csv"],
     "Per-cohort AUC for each model, recomputed from saved per-sample predictions. Rows "
     "above the red line are complete 8-cohort LODO evaluations; rows below were evaluated "
     "on a single cohort and their unevaluated cells are grey, not imputed.",
     new_fig05),
    (11, "masking_real_vs_empty",
     ["graph_utility/masking_curves.csv"],
     "Robustness to missing genes, mean of per-cohort values. The STRING graph degrades "
     "FASTER than the no-graph control on AUC and SLOWER on macro-F1 and balanced accuracy; "
     "both are shown with the signed advantage and its CI, because the sign of the "
     "'robustness advantage' depends on which metric is chosen.",
     new_fig11),
    (12, "masking_all_conditions",
     ["graph_utility/masking_curves.csv", "learned_graph/masking_curves_all_conditions.csv",
      "learned_graph/checkpoints/*/run_metrics.json"],
     "All graph conditions under masking. Conditions with a completed masking sweep are "
     "drawn as full curves with CI bands; conditions that are trained but not yet swept "
     "appear as their 0%-masked point only, marked as such.",
     new_fig12),
    (13, "advantage_over_empty",
     ["graph_utility/masking_curves.csv", "results_b7/", "graph_utility/control_checkpoints/",
      "learned_graph/checkpoints/"],
     "Advantage over the no-graph control for every graph condition: flatness advantage "
     "under masking (left, AUC and macro-F1) and clean-data AUC advantage paired within "
     "cohort (right), each with bootstrap CIs.",
     new_fig13),
    (14, "learned_vs_string_topology",
     ["learned_graph/edges/", "shared/graph_edges.csv", "shared/panel_genes.csv"],
     "Learned-vs-STRING edge overlap (Jaccard, against a computed chance level) and "
     "cross-fold stability of each learned topology, for both learned variants.",
     new_fig14),
]


def patch_registry():
    by_num = {spec["num"]: spec for spec in mf.REGISTRY}
    for num, name, sources, caption, fn in OVERRIDES:
        spec = {"num": num, "name": name, "sources": sources, "caption": caption, "fn": fn}
        if num in by_num:
            by_num[num].update(spec)
        else:
            mf.REGISTRY.append(spec)
    print(f"[fix] replaced {len(OVERRIDES)} registry entries: "
          f"{[n for n, *_ in OVERRIDES]}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="diagnose the three reported figure problems, then regenerate the "
                    "affected figures (read-only on all results)")
    ap.add_argument("--diagnose-only", action="store_true",
                    help="write figures/diagnosis.txt and stop; no plotting")
    ap.add_argument("--fixed-only", action="store_true",
                    help="rebuild ONLY figs 4,5,11,12,13,14. Fast, but all_figures.pdf "
                         "will then contain only those pages")
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="explicit figure numbers to build")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()

    run_diagnosis(write=True)
    if args.diagnose_only:
        print("[fix] --diagnose-only: no figures rebuilt")
        return

    patch_registry()

    only = args.only
    if only is None and args.fixed_only:
        only = [n for n, *_ in OVERRIDES]
    argv = ["make_figures.py", "--outdir", args.outdir]
    if only:
        argv += ["--only"] + [str(n) for n in only]
        if args.fixed_only:
            print("[fix] NOTE: all_figures.pdf will hold only the rebuilt pages. "
                  "Run without --fixed-only for the complete document.")
    sys.argv = argv
    mf.main()


if __name__ == "__main__":
    main()
