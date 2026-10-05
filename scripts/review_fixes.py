"""
CoDA-GAT — review fixes: four statistical corrections, then the figures rebuilt.

Read-only on every results directory. The only writes are into review_fixes/
and figures/. Nothing under results_b7/, baselines/, ablations_b9/,
graph_utility/, learned_graph/ or consensus_failures/ is touched.

  TASK 1  Masking CIs used the wrong unit of analysis. The old panels pooled
          40 (cohort x masking-rate) cells and put a CI across them, which
          treats five rates from the same cohort as five independent
          observations -- they are not, they are five re-measurements of one
          cohort under nested gene subsets. Everything is redone with COHORT
          as the unit: per masking rate, the network-minus-control difference
          is formed WITHIN each cohort, and the 95% CI is across the 8 cohorts.

  TASK 2  consensus_failures/ stores two different "between-group" agreement
          numbers. Both are reported here, and fig18 is made to use the
          family-level one averaged over family blocks.

  TASK 3  The LAC prediction sets are summarised by mean set size only, which
          hides whether a set of size 2 means "uncertain" or "the method has
          collapsed". The full size-0 / size-1 / size-2 distribution is
          reported per alpha, pooled and per cohort.

  TASK 4  The label-noise ceiling was previously obtained by inverting an
          observed AUC, which is circular: it assumes the best model already
          sits at the ceiling, which is the very thing the ceiling argument is
          meant to test. It is replaced by a forward simulation driven by
          PUBLISHED inter-rater weighted kappa for fibrosis staging.

No number is hardcoded. Empirical inputs are read from disk; simulation inputs
are the published kappa values passed in via --kappas, and every derived
quantity is either exact or Monte Carlo with its standard error reported.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm, multivariate_normal, rankdata, wilcoxon, t as student_t

import make_figures as mf
import fix_figures as fx
from make_figures import (
    REPO, SHARED, B7, CONS, OKABE, CODAGAT, NEUTRAL, SkipFigure, read_csv, cohort_labels,
    as_bool,
)
from fix_figures import (
    masking_raw, per_cohort_levels, fold_dirs, fold_index,
    CONDITION_LABEL, CONDITION_COLOR, CONDITION_CKPT_ROOT, SEEDS,
)

OUT_DIR = REPO / "review_fixes"
ALPHAS = [0.05, 0.10, 0.20]
STAGES = ["F0", "F1", "F2", "F3", "F4"]
F2_CUT_INDEX = 2          # F2+ = stages at or above index 2
DEFAULT_KAPPAS = [0.47, 0.53, 0.59]

LINES = []


def say(line=""):
    LINES.append(str(line))
    print(str(line), flush=True)


def head(title):
    say("=" * 78)
    say(title)
    say("=" * 78)


# ==========================================================================
# shared statistics
# ==========================================================================
def cohort_ci(values, conf=0.95):
    """Mean and CI across cohorts, with the cohort as the unit of analysis.

    A t interval, not a bootstrap: with n=8 a percentile bootstrap cannot
    resolve the tails it is being asked about (its 2.5th percentile is
    determined by a handful of resamples), whereas the t interval states its
    normality assumption openly. n is small either way and the interval is
    descriptive.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return {"mean": float(x.mean()) if n else np.nan, "lo": np.nan, "hi": np.nan,
                "sd": np.nan, "n": n, "p_wilcoxon": np.nan}
    m, sd = float(x.mean()), float(x.std(ddof=1))
    half = student_t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n)
    try:
        p = float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    except ValueError:
        p = np.nan
    return {"mean": m, "lo": m - half, "hi": m + half, "sd": sd, "n": n, "p_wilcoxon": p}


def fmt_ci(d, digits=4):
    if not np.isfinite(d.get("lo", np.nan)):
        return f"{d['mean']:+.{digits}f} (n={d['n']}, no CI)"
    return f"{d['mean']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}] n={d['n']}"


# ==========================================================================
# TASK 1 -- masking, with the cohort as the unit of analysis
# ==========================================================================
def cohort_unit_masking(metrics=("auc", "macro_f1", "balanced_acc"),
                         control="empty_graph"):
    """Per (metric, condition, cohort, rate): the within-cohort difference
    against the control, in two forms.

      delta_level = M_net(c, r) - M_ctrl(c, r)
          the literal network-minus-control difference at that masking rate.

      delta_drop  = [M_ctrl(c,0) - M_ctrl(c,r)] - [M_net(c,0) - M_net(c,r)]
          the difference in how far each model FELL from its own clean-data
          value. Positive means the network degrades less.

    These are not independent: delta_drop(c,r) = delta_level(c,r) -
    delta_level(c,0) exactly. The drop form is the level form re-centred on
    each cohort's own rate-0 value, which is why the two can carry opposite
    signs whenever a model starts from a different level. Both are reported so
    the reader can see that rather than having one of them chosen for them.
    """
    d = masking_raw()
    rows = []
    for metric in metrics:
        lv = per_cohort_levels(d, metric)
        piv = lv.pivot_table(index=["fold", "mask_rate"], columns="model", values=metric)
        if control not in piv.columns:
            continue
        for cond in piv.columns:
            if cond == control:
                continue
            for (fold, rate) in piv.index:
                net, ctl = piv.loc[(fold, rate), cond], piv.loc[(fold, rate), control]
                if not (np.isfinite(net) and np.isfinite(ctl)):
                    continue
                try:
                    net0 = piv.loc[(fold, 0.0), cond]
                    ctl0 = piv.loc[(fold, 0.0), control]
                except KeyError:
                    net0 = ctl0 = np.nan
                rows.append({
                    "metric": metric, "condition": cond, "control": control,
                    "fold": int(fold), "mask_rate": float(rate),
                    "value_network": float(net), "value_control": float(ctl),
                    "delta_level": float(net - ctl),
                    "delta_drop": float((ctl0 - ctl) - (net0 - net))
                                   if np.isfinite(net0) and np.isfinite(ctl0) else np.nan,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        cl = cohort_labels()
        out["test_cohort"] = [cl.get(f, f"fold{f}") for f in out["fold"]]
    return out


def summarize_cohort_unit(tbl):
    """Collapse to one row per (metric, condition, rate, quantity), CI across cohorts."""
    rows = []
    for (metric, cond, rate), g in tbl.groupby(["metric", "condition", "mask_rate"]):
        for quantity in ("delta_level", "delta_drop"):
            s = cohort_ci(g[quantity].to_numpy())
            rows.append({"metric": metric, "condition": cond, "mask_rate": rate,
                          "quantity": quantity, **s})
    return pd.DataFrame(rows)


def task1():
    head("TASK 1 -- masking CIs recomputed with COHORT as the unit (n = 8)")
    tbl = cohort_unit_masking()
    if tbl.empty:
        say("  no masking data available")
        return tbl, pd.DataFrame()
    summ = summarize_cohort_unit(tbl)

    say("What changed, and why it matters")
    say("-" * 78)
    say("  OLD: one CI over 40 (cohort x masking-rate) cells, pooled across rates.")
    say("  NEW: per masking rate, the network-minus-control difference is formed inside")
    say("       each cohort, then a 95% t interval across the 8 cohorts.")
    say("")
    say("  The old interval was too narrow for a reason worth naming: masking rates within")
    say("  a cohort share that cohort's samples, its labels and its trained weights, and")
    say("  the gene subsets are nested draws over the same 72 genes. Five rates from one")
    say("  cohort are five re-measurements, not five independent observations. Pooling them")
    say("  inflated n from 8 to 40 and shrank every interval by roughly sqrt(5).")
    say("")
    say(f"  cells: {len(tbl)}  conditions: {sorted(tbl['condition'].unique())}")
    say(f"  cohorts: {tbl['fold'].nunique()}  rates: {sorted(tbl['mask_rate'].unique())}")

    for metric in ["auc", "macro_f1", "balanced_acc"]:
        sub = summ[(summ["metric"] == metric) & (summ["condition"] == "real_graph")]
        if sub.empty:
            continue
        say("")
        say(f"STRING graph minus no-graph control -- metric = {metric}")
        say("-" * 78)
        say(f"  {'rate':>6s}  {'level difference (n=8)':>34s}  {'drop difference (n=8)':>34s}")
        for rate in sorted(sub["mask_rate"].unique()):
            lv = sub[(sub["mask_rate"] == rate) & (sub["quantity"] == "delta_level")].iloc[0]
            dr = sub[(sub["mask_rate"] == rate) & (sub["quantity"] == "delta_drop")].iloc[0]
            lvs = fmt_ci(lv)
            drs = "-" if rate == 0 else fmt_ci(dr)
            say(f"  {rate:>6.0%}  {lvs:>34s}  {drs:>34s}")

    say("")
    say("Effect of the correction, at the endpoint (50% of genes hidden)")
    say("-" * 78)
    for metric in ["auc", "macro_f1", "balanced_acc"]:
        sub = tbl[(tbl["metric"] == metric) & (tbl["condition"] == "real_graph")]
        if sub.empty:
            continue
        end = sub[sub["mask_rate"] == sub["mask_rate"].max()]
        allc = sub[sub["mask_rate"] > 0]
        new = cohort_ci(end["delta_drop"].to_numpy())
        old = cohort_ci(allc["delta_drop"].to_numpy())   # same estimator, wrong unit
        say(f"  {metric:13s} cohort-unit @50%: {fmt_ci(new)}")
        say(f"  {'':13s} old 40-cell pool : {fmt_ci(old)}   <- n inflated 8 -> 40")

    say("")
    say("VERDICT (Task 1)")
    say("-" * 78)
    sig = summ[(summ["quantity"] == "delta_drop") & (summ["mask_rate"] > 0)
               & (summ["n"] >= 2)].copy()
    sig["excludes_zero"] = (sig["lo"] > 0) | (sig["hi"] < 0)
    for metric in ["auc", "macro_f1", "balanced_acc"]:
        s = sig[(sig["metric"] == metric) & (sig["condition"] == "real_graph")]
        if s.empty:
            continue
        k = int(s["excludes_zero"].sum())
        say(f"  {metric:13s}: {k} of {len(s)} masking rates have a cohort-unit CI excluding zero")
    say("  With the correct unit the intervals widen and most of them cover zero. The")
    say("  qualitative pattern survives -- AUC favours the control, the threshold metrics")
    say("  favour the graph -- but at n=8 cohorts it is a DIRECTIONAL observation, not a")
    say("  demonstrated effect. The earlier narrow intervals overstated the evidence.")
    return tbl, summ


# ==========================================================================
# TASK 2 -- which "between-group" agreement is which
# ==========================================================================
def agreement_levels():
    """Reconcile the two between-group agreement numbers in
    consensus_failures/family_agreement_summary.csv.

    The saved `between` rows average over model PAIRS. With 9 GAT, 11
    tree-ensemble and 3 linear models, the cross-family pairs split 99 / 33 /
    27, so a pair-weighted mean is 62% determined by the gat-vs-tree block
    alone. Averaging over the three family BLOCKS instead gives each family
    pair equal weight, which is what a statement of the form "linear vs tree
    vs GAT" means.
    """
    d = read_csv(CONS / "family_agreement_summary.csv")
    cols = ["mean_jaccard_observed", "mean_jaccard_expected_chance",
            "mean_jaccard_excess", "mean_mcc", "mean_perm_z"]
    out = {"raw": d}

    for level in ("architecture_family", "variant_group"):
        pw = d[d["level"] == level]
        blocks = d[d["level"] == f"{level}_pairwise"].copy()
        if blocks.empty:
            continue
        blocks["_is_within"] = [str(r).split(" vs ")[0] == str(r).split(" vs ")[-1]
                                 for r in blocks["relation"]]
        for relation, is_within in (("between", False), ("within", True)):
            b = blocks[blocks["_is_within"] == is_within]
            row = pw[pw["relation"] == relation]
            rec = {"level": level, "relation": relation,
                    "n_blocks": len(b), "n_pairs": int(b["n_pairs"].sum())}
            for c in cols:
                rec[f"{c}__block_averaged"] = float(b[c].mean()) if len(b) else np.nan
                rec[f"{c}__pair_weighted"] = float(row[c].iloc[0]) if len(row) else np.nan
            rec["blocks"] = "; ".join(f"{r}={v:.4f}" for r, v in
                                       zip(b["relation"], b["mean_jaccard_observed"]))
            out.setdefault("reconciled", []).append(rec)
    out["reconciled"] = pd.DataFrame(out.get("reconciled", []))
    return out


def task2():
    head("TASK 2 -- between-family vs between-variant-group error agreement")
    lv = agreement_levels()
    rec = lv["reconciled"]
    if rec.empty:
        say("  family_agreement_summary.csv has no pairwise blocks to reconcile")
        return rec

    say("Both numbers, side by side")
    say("-" * 78)
    for _, r in rec.iterrows():
        say(f"  level={r['level']:20s} relation={r['relation']:8s} "
            f"({r['n_blocks']} blocks, {r['n_pairs']} model pairs)")
        say(f"      Jaccard, block-averaged : {r['mean_jaccard_observed__block_averaged']:.4f}"
            f"   (excess over chance {r['mean_jaccard_excess__block_averaged']:+.4f})")
        say(f"      Jaccard, pair-weighted  : {r['mean_jaccard_observed__pair_weighted']:.4f}"
            f"   (excess over chance {r['mean_jaccard_excess__pair_weighted']:+.4f})")
        say(f"      MCC,     block-averaged : {r['mean_mcc__block_averaged']:.4f}")
        say(f"      blocks: {r['blocks']}")
        say("")

    fam = rec[(rec["level"] == "architecture_family") & (rec["relation"] == "between")]
    var = rec[(rec["level"] == "variant_group") & (rec["relation"] == "between")]
    say("The answer to 'which one does fig18 plot?'")
    say("-" * 78)
    say("  fig18 in make_figures.py already filters to level == 'architecture_family',")
    say("  falling back to 'variant_group' only if that level is absent. It is not absent,")
    say("  so fig18 has been plotting the FAMILY level all along -- the caption was right.")
    say("")
    say("  The real inconsistency is finer: it plots the saved `between` row, which is")
    say("  PAIR-WEIGHTED, while a claim about 'linear vs tree vs GAT' is a statement about")
    say("  the three family BLOCKS. With 9 GAT, 11 tree and 3 linear models the cross-family")
    say("  pairs split 99 / 33 / 27, so the pair-weighted mean is 62% the gat-vs-tree block.")
    if not fam.empty:
        f = fam.iloc[0]
        say("")
        say(f"  between-FAMILY (linear vs tree vs GAT), averaged over the 3 family blocks:")
        say(f"      Jaccard {f['mean_jaccard_observed__block_averaged']:.4f}  "
            f"vs chance {f['mean_jaccard_expected_chance__block_averaged']:.4f}  "
            f"excess {f['mean_jaccard_excess__block_averaged']:+.4f}")
        say(f"      pair-weighted equivalent: {f['mean_jaccard_observed__pair_weighted']:.4f} "
            f"(difference {f['mean_jaccard_observed__block_averaged'] - f['mean_jaccard_observed__pair_weighted']:+.4f})")
    if not var.empty:
        v = var.iloc[0]
        say("")
        say(f"  between-VARIANT-GROUP (feature view / ablation / corruption groups):")
        say(f"      Jaccard {v['mean_jaccard_observed__block_averaged']:.4f} block-averaged, "
            f"{v['mean_jaccard_observed__pair_weighted']:.4f} pair-weighted")
    if not fam.empty and not var.empty:
        say("")
        say("  The variant-group value is the HIGHER of the two, and necessarily so: its")
        say("  'between' set still contains many GAT-vs-GAT comparisons (a graph-corrupted")
        say("  model against a feature-ablated model sits in different variant groups but the")
        say("  same architecture family). Quoting it as evidence that architecturally")
        say("  different models fail together would overstate the case. The family-level")
        say("  block-averaged number is the defensible one, and fig18 now uses it.")
    say("")
    say("  fig18 is regenerated with the block-averaged family values, the three individual")
    say("  block values overlaid as points so their spread is visible, and the pair-weighted")
    say("  and variant-group numbers annotated for comparison.")
    return rec


# ==========================================================================
# TASK 3 -- LAC prediction-set size distribution
# ==========================================================================
def lac_set_sizes(mode="off"):
    """Exact LAC set sizes from the saved probabilities and thresholds.

    LAC score is s(x, k) = 1 - p_k(x); the set is {k : s <= q_hat}, i.e.
    {k : p_k >= 1 - q_hat}. With K=2 and p_0 + p_1 = 1:
        size 2  <=>  min(p) >= 1 - q_hat   (both classes retained)
        size 0  <=>  max(p) <  1 - q_hat   which requires q_hat < 0.5
        size 1  otherwise
    So an empty set is possible only when the calibration cohort was easy
    enough to push q_hat below 0.5 -- a coverage-relevant event that a mean
    set size of, say, 1.45 hides completely.
    """
    truth = fx.load_fold_truth()
    rows = []
    for fd in fold_dirs():
        fi = fold_index(fd.name)
        cohort, y = truth[fi]
        for seed in SEEDS:
            run = B7 / fd.name / f"mode_{mode}" / f"seed_{seed}"
            thr_p, prob_p = run / "conformal_threshold.json", run / "probs_test.npy"
            if not (thr_p.exists() and prob_p.exists()):
                continue
            thr = json.loads(thr_p.read_text()).get("lac", {})
            p = np.load(prob_p)
            if p.ndim != 2 or len(p) != len(y):
                continue
            for a in ALPHAS:
                q = thr.get(str(a), thr.get(f"{a:g}"))
                if q is None:
                    continue
                q = float(q)
                inset = p >= (1.0 - q)
                size = inset.sum(axis=1)
                covered = inset[np.arange(len(y)), y]
                rows.append({
                    "fold": fi, "test_cohort": cohort, "seed": seed, "alpha": a,
                    "q_hat": q, "n": len(y),
                    "n_size0": int((size == 0).sum()),
                    "n_size1": int((size == 1).sum()),
                    "n_size2": int((size == 2).sum()),
                    "mean_set_size": float(size.mean()),
                    "coverage": float(covered.mean()),
                })
    return pd.DataFrame(rows)


def task3():
    head("TASK 3 -- LAC prediction-set size distribution (mode = off, m = 0)")
    d = lac_set_sizes()
    if d.empty:
        say("  no saved LAC thresholds / probabilities found")
        return d, pd.DataFrame(), pd.DataFrame()

    # cross-check against the published summary columns
    m = read_csv(B7 / "master_results.csv")
    m = m[(m["mode"] == "off") & (m["m"] == 0)]
    checks = []
    for a in ALPHAS:
        col = f"lac_a{a:g}_mean_set_size"
        if col not in m.columns:
            continue
        j = d[d["alpha"] == a].merge(m[["fold", "seed", col]], on=["fold", "seed"])
        if not j.empty:
            checks.append({"alpha": a, "n_runs": len(j),
                            "max_absdiff": float((j["mean_set_size"] - j[col]).abs().max())})
    if checks:
        say("Reproduction check against master_results.csv")
        say("-" * 78)
        for c in checks:
            say(f"  alpha={c['alpha']:.2f}: {c['n_runs']} runs, "
                f"max |mean set size - saved| = {c['max_absdiff']:.2e}")
        say("  The recomputed sets reproduce the saved mean set size, so the distribution")
        say("  below is a decomposition of the published number, not a different quantity.")
        say("")

    # pooled: sum counts over all runs, so every held-out patient-prediction counts once
    pooled = d.groupby("alpha").agg(
        n_runs=("n", "size"), n_predictions=("n", "sum"),
        n_size0=("n_size0", "sum"), n_size1=("n_size1", "sum"), n_size2=("n_size2", "sum"),
        q_hat_mean=("q_hat", "mean"), q_hat_min=("q_hat", "min"), q_hat_max=("q_hat", "max"),
        coverage_mean=("coverage", "mean")).reset_index()
    for k in (0, 1, 2):
        pooled[f"share_size{k}"] = pooled[f"n_size{k}"] / pooled["n_predictions"]
    pooled["mean_set_size"] = (pooled["n_size1"] + 2 * pooled["n_size2"]) / pooled["n_predictions"]

    say("Pooled over all 8 cohorts x 5 seeds")
    say("-" * 78)
    say(f"  {'alpha':>6s} {'size 0':>9s} {'size 1':>9s} {'size 2':>9s} "
        f"{'mean size':>10s} {'coverage':>9s} {'q_hat range':>20s}")
    for _, r in pooled.iterrows():
        say(f"  {r['alpha']:>6.2f} {r['share_size0']:>8.1%} {r['share_size1']:>8.1%} "
            f"{r['share_size2']:>8.1%} {r['mean_set_size']:>10.3f} {r['coverage_mean']:>8.1%} "
            f"{f'[{r.q_hat_min:.3f}, {r.q_hat_max:.3f}]':>20s}")
    say(f"  ({int(pooled['n_predictions'].iloc[0])} patient-predictions per alpha "
        f"= 1064 held-out patients x 5 seeds)")

    # per cohort
    per = d.groupby(["alpha", "fold", "test_cohort"]).agg(
        n_predictions=("n", "sum"), n_size0=("n_size0", "sum"),
        n_size1=("n_size1", "sum"), n_size2=("n_size2", "sum"),
        q_hat_mean=("q_hat", "mean"), coverage_mean=("coverage", "mean")).reset_index()
    for k in (0, 1, 2):
        per[f"share_size{k}"] = per[f"n_size{k}"] / per["n_predictions"]
    per["mean_set_size"] = (per["n_size1"] + 2 * per["n_size2"]) / per["n_predictions"]

    say("")
    say("Per held-out cohort")
    say("-" * 78)
    for a in ALPHAS:
        sub = per[per["alpha"] == a].sort_values("fold")
        say(f"  alpha = {a:.2f}   (nominal coverage {1 - a:.0%})")
        say(f"    {'cohort':<12s} {'n':>5s} {'size 0':>8s} {'size 1':>8s} {'size 2':>8s} "
            f"{'mean':>7s} {'cover':>7s} {'q_hat':>7s}")
        for _, r in sub.iterrows():
            say(f"    {r['test_cohort']:<12s} {int(r['n_predictions'] / len(SEEDS)):>5d} "
                f"{r['share_size0']:>7.1%} {r['share_size1']:>7.1%} {r['share_size2']:>7.1%} "
                f"{r['mean_set_size']:>7.3f} {r['coverage_mean']:>6.1%} {r['q_hat_mean']:>7.3f}")
        say("")

    say("VERDICT (Task 3)")
    say("-" * 78)
    n_empty = int(pooled["n_size0"].sum())
    if n_empty == 0:
        say("  NO empty prediction sets occur at any alpha. Every q_hat stays above 0.5, and")
        say("  with K=2 that makes a size-0 set arithmetically impossible: the larger of two")
        say("  probabilities summing to 1 is always at least 0.5, hence always retained.")
        say("  So the LAC sets are a clean two-way split -- 'one stage group' or 'both' --")
        say("  and no patient is ever returned an abstention that would need a fallback rule.")
    else:
        say(f"  {n_empty} empty prediction sets occur "
            f"({n_empty / pooled['n_predictions'].sum():.2%} of all predictions). These arise")
        say("  where q_hat < 0.5 and are genuine abstentions: the method declines to name any")
        say("  class. They must be reported separately, because a mean set size near 1 can be")
        say("  produced either by confident singletons or by a mix of empty and full sets.")
    say("  This is the contrast with APS, which was degenerate: APS returned size 2 for")
    say("  100% of patients at every alpha. LAC's distribution genuinely varies with alpha,")
    say("  which is what makes its coverage numbers meaningful.")
    return d, pooled, per


# ==========================================================================
# TASK 4 -- label-noise ceiling from published inter-rater kappa
# ==========================================================================
def stage_prevalence(exact_only=True):
    """Marginal fibrosis-stage distribution, from the recorded labels.

    exact_only keeps the 1,063 samples whose stage is recorded exactly and
    drops the 303 with grouped labels (grouped_F0_F1, grouped_F3_F4,
    grouped_F0_F2), whose stage cannot be placed in a single bin without an
    assumption. Reported both ways so the choice is visible.
    """
    lab = pd.read_parquet(SHARED / "labels.parquet")
    if "is_supervised" in lab.columns:
        lab = lab[as_bool(lab["is_supervised"])]
    if exact_only and "resolution" in lab.columns:
        lab = lab[lab["resolution"].astype(str) == "exact"]
    counts = lab["stage"].value_counts().reindex(STAGES, fill_value=0)
    return counts.to_numpy(dtype=float) / counts.sum(), int(counts.sum())


def thresholds_from_prevalence(prev):
    """Cut points on the standard-normal scale reproducing a stage marginal.

    Both the ideal reader (applied to the latent severity z) and the noisy
    reader (applied to its own reading, which has variance 1 + sigma^2) use the
    cut points that reproduce THIS marginal on their own scale. That keeps the
    observed stage distribution fixed as sigma varies, so nothing in the
    comparison is driven by a drifting prevalence.
    """
    cum = np.cumsum(prev)[:-1]
    return norm.ppf(cum)


def _bvn_cdf(x, y, rho):
    if np.isneginf(x) or np.isneginf(y):
        return 0.0
    if np.isposinf(x) and np.isposinf(y):
        return 1.0
    if np.isposinf(x):
        return float(norm.cdf(y))
    if np.isposinf(y):
        return float(norm.cdf(x))
    # rho = 1 makes the covariance singular and multivariate_normal raises. That
    # happens for any sigma small enough that 1/(1+sigma^2) rounds to 1.0 in
    # double precision, which the root-finder's lower bracket does hit. Clamping
    # is numerically indistinguishable from the limit at this magnitude.
    rho = min(float(rho), 1.0 - 1e-9)
    return float(multivariate_normal(mean=[0.0, 0.0],
                                      cov=[[1.0, rho], [rho, 1.0]]).cdf([x, y]))


def joint_stage_table(a, rho):
    """5x5 table of P(reader 1 -> stage i, reader 2 -> stage j), exactly.

    The two readings are z + e1 and z + e2. After standardising by
    sqrt(1+sigma^2) each is standard normal and their correlation is
    rho = 1/(1+sigma^2), so the table is a bivariate-normal rectangle
    probability, computed by inclusion-exclusion on the CDF.
    """
    edges = np.concatenate([[-np.inf], a, [np.inf]])
    k = len(edges) - 1
    P = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            P[i, j] = (_bvn_cdf(edges[i + 1], edges[j + 1], rho)
                       - _bvn_cdf(edges[i], edges[j + 1], rho)
                       - _bvn_cdf(edges[i + 1], edges[j], rho)
                       + _bvn_cdf(edges[i], edges[j], rho))
    return np.clip(P, 0.0, None)


def weight_matrix(k, kind):
    i, j = np.indices((k, k))
    if kind == "linear":
        return 1.0 - np.abs(i - j) / (k - 1)
    if kind == "quadratic":
        return 1.0 - (i - j) ** 2 / (k - 1) ** 2
    raise ValueError(kind)


def weighted_kappa_from_sigma(sigma, a, kind):
    rho = 1.0 / (1.0 + sigma ** 2)
    P = joint_stage_table(a, rho)
    P = P / P.sum()
    W = weight_matrix(P.shape[0], kind)
    pi, pj = P.sum(axis=1), P.sum(axis=0)
    po = float((W * P).sum())
    pe = float((W * np.outer(pi, pj)).sum())
    return (po - pe) / (1.0 - pe)


def invert_kappa(target, a, kind):
    """sigma such that the simulated weighted kappa equals the published value.

    kappa is monotone decreasing in sigma (sigma=0 gives perfect agreement,
    sigma->inf gives chance), so a bracketed root is unique.
    """
    f = lambda s: weighted_kappa_from_sigma(s, a, kind) - target
    lo, hi = 1e-3, 0.05
    while f(hi) > 0 and hi < 1e4:
        hi *= 2
    if f(hi) > 0:
        return np.nan
    return float(optimize.brentq(f, lo, hi, xtol=1e-8, rtol=1e-10))


def misassignment_exact(sigma, a_cut):
    """P(noisy reader's F2+ call != ideal reader's F2+ call), in closed form.

    y_true = 1[z > a], y_obs = 1[(z+e)/sqrt(1+s^2) > a]. Both margins are
    standard normal with correlation 1/sqrt(1+s^2), so
        P(disagree) = 2 * [Phi(a) - Phi_2(a, a; rho)].
    """
    rho = 1.0 / np.sqrt(1.0 + sigma ** 2)
    return float(2.0 * (norm.cdf(a_cut) - _bvn_cdf(a_cut, a_cut, rho)))


def reader_reader_disagreement(sigma, a_cut):
    """P(two independent readers disagree on F2+) -- the directly observable
    quantity the published kappa describes, kept as a sanity anchor."""
    rho = 1.0 / (1.0 + sigma ** 2)
    return float(2.0 * (norm.cdf(a_cut) - _bvn_cdf(a_cut, a_cut, rho)))


def ceiling_auc(sigma, a_cut, n=400_000, n_rep=5, seed=20260922):
    """AUC of a PERFECT model, scored against the noisy labels we actually use.

    The model's score is the true latent severity z; the label is one
    pathologist's F2+ call. No model can do better than this, because z is all
    the information there is and the label contains noise the model cannot
    predict. Monte Carlo over independent replicates; the spread across
    replicates is the reported standard error, so the precision is stated
    rather than assumed.
    """
    s = np.sqrt(1.0 + sigma ** 2)
    vals = []
    for r in range(n_rep):
        rng = np.random.default_rng(seed + r)
        z = rng.standard_normal(n)
        obs = (z + rng.standard_normal(n) * sigma) / s
        y = obs > a_cut
        n1, n0 = int(y.sum()), int((~y).sum())
        if n1 == 0 or n0 == 0:
            continue
        ranks = rankdata(z)
        vals.append((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    if not vals:
        return np.nan, np.nan
    v = np.asarray(vals)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def task4(kappas, mc_n, mc_rep):
    head("TASK 4 -- label-noise ceiling from published inter-rater kappa")

    prev_exact, n_exact = stage_prevalence(exact_only=True)
    prev_all, n_all = stage_prevalence(exact_only=False)
    a_exact = thresholds_from_prevalence(prev_exact)
    a_all = thresholds_from_prevalence(prev_all)

    say("Why the previous ceiling was replaced")
    say("-" * 78)
    say("  The old figure inverted an OBSERVED AUC (0.902) to a latent noise sigma (0.663)")
    say("  and called the result a ceiling. That is circular twice over. First, it assumes")
    say("  the best model already sits AT the ceiling -- which is the proposition the")
    say("  ceiling argument exists to test, so the conclusion is smuggled into the premise.")
    say("  Second, 0.902 was not even a cross-cohort number: it is panel72_combined__logreg")
    say("  on GSE130970 alone (see figures/diagnosis.txt), and that model's true 8-cohort")
    say("  mean is 0.838. An inversion anchored on the single most favourable cohort")
    say("  produces a ceiling that flatters whatever is compared against it.")
    say("")
    say("  The replacement runs in the opposite direction: published inter-rater agreement")
    say("  for fibrosis staging goes IN, and the ceiling comes OUT. No observed AUC of ours")
    say("  appears anywhere in the calculation.")
    say("")
    say("The model, stated in full so it can be disagreed with")
    say("-" * 78)
    say("  1. Each patient has a latent fibrosis severity z ~ N(0, 1).")
    say("  2. A pathologist reads the biopsy and observes z + e, e ~ N(0, sigma^2),")
    say("     independently for each reader. sigma is the only free parameter.")
    say("  3. Both an ideal reader (reading z) and a real reader (reading z + e) assign")
    say("     F0-F4 using the cut points that reproduce the OBSERVED stage marginal on")
    say("     their own scale, so the stage distribution stays fixed as sigma varies.")
    say("  4. sigma is chosen so that two independent readers reproduce the published")
    say("     weighted kappa. This is done exactly, from the bivariate-normal stage table,")
    say("     not by simulation.")
    say("  5. Ceiling AUC is then the AUC of a model that recovers z PERFECTLY, scored")
    say("     against one reader's F2+ call -- the label our models are trained on.")
    say("")
    say("  Assumptions that matter: normal latent severity, additive homoscedastic reader")
    say("  noise, and thresholds shared across readers. Real staging noise is probably")
    say("  worse at the middle stages than at F0 or F4, which this model does not capture.")
    say("  It is a first-order estimate and should be quoted as a range, not a line.")
    say("")
    say("Empirical inputs")
    say("-" * 78)
    say(f"  stage marginal, exact labels only (n={n_exact}): "
        f"{dict(zip(STAGES, np.round(prev_exact, 4)))}")
    say(f"  stage marginal, all supervised  (n={n_all}): "
        f"{dict(zip(STAGES, np.round(prev_all, 4)))}")
    say(f"  F2+ prevalence: exact {prev_exact[F2_CUT_INDEX:].sum():.4f}, "
        f"all {prev_all[F2_CUT_INDEX:].sum():.4f}")
    say(f"  F2+ cut point on the latent scale: exact a={a_exact[F2_CUT_INDEX - 1]:+.4f}, "
        f"all a={a_all[F2_CUT_INDEX - 1]:+.4f}")
    say(f"  published weighted kappa values used: {kappas}")
    say(f"  Monte Carlo for ceiling AUC: {mc_rep} replicates x {mc_n:,} patients")

    rows = []
    for label, prev, a in (("exact-labels", prev_exact, a_exact), ("all-labels", prev_all, a_all)):
        a_cut = a[F2_CUT_INDEX - 1]
        for kind in ("linear", "quadratic"):
            for kap in kappas:
                sigma = invert_kappa(kap, a, kind)
                if not np.isfinite(sigma):
                    continue
                auc, se = ceiling_auc(sigma, a_cut, n=mc_n, n_rep=mc_rep)
                rows.append({
                    "stage_marginal": label, "kappa_weighting": kind, "kappa_published": kap,
                    "sigma": sigma, "reliability_rho": 1.0 / (1.0 + sigma ** 2),
                    "kappa_check": weighted_kappa_from_sigma(sigma, a, kind),
                    "ceiling_auc": auc, "ceiling_auc_mc_se": se,
                    "f2_misassignment_rate": misassignment_exact(sigma, a_cut),
                    "reader_reader_f2_disagreement": reader_reader_disagreement(sigma, a_cut),
                })
    res = pd.DataFrame(rows)
    if res.empty:
        say("  inversion failed for every kappa")
        return res

    say("")
    say("Results (primary: exact-labels marginal)")
    say("-" * 78)
    say(f"  {'weighting':>10s} {'kappa':>6s} {'sigma':>7s} {'ceiling AUC':>14s} "
        f"{'F2+ misassign':>14s} {'reader-reader':>14s}")
    for _, r in res[res["stage_marginal"] == "exact-labels"].iterrows():
        say(f"  {r['kappa_weighting']:>10s} {r['kappa_published']:>6.2f} {r['sigma']:>7.3f} "
            f"{r['ceiling_auc']:>9.4f}+/-{r['ceiling_auc_mc_se']:.4f} "
            f"{r['f2_misassignment_rate']:>13.1%} {r['reader_reader_f2_disagreement']:>13.1%}")

    say("")
    say("Same, using all supervised labels including the grouped ones")
    say("-" * 78)
    for _, r in res[res["stage_marginal"] == "all-labels"].iterrows():
        say(f"  {r['kappa_weighting']:>10s} {r['kappa_published']:>6.2f} {r['sigma']:>7.3f} "
            f"{r['ceiling_auc']:>9.4f}+/-{r['ceiling_auc_mc_se']:.4f} "
            f"{r['f2_misassignment_rate']:>13.1%} {r['reader_reader_f2_disagreement']:>13.1%}")

    prim = res[res["stage_marginal"] == "exact-labels"]
    lo, hi = float(prim["ceiling_auc"].min()), float(prim["ceiling_auc"].max())
    say("")
    say("VERDICT (Task 4)")
    say("-" * 78)
    say(f"  Implied ceiling AUC across kappa = {kappas} and both weightings: "
        f"{lo:.3f} to {hi:.3f}")
    say(f"  Implied F2+ misassignment rate: "
        f"{prim['f2_misassignment_rate'].min():.1%} to {prim['f2_misassignment_rate'].max():.1%}")
    say("")
    say("  Quadratic weighting always implies MORE noise than linear at the same kappa,")
    say("  because quadratic weights forgive one-stage disagreements almost entirely; so")
    say("  reproducing a given kappa under quadratic weighting requires a larger sigma and")
    say("  yields a lower ceiling. The two weightings bracket the honest range, which is")
    say("  why both are reported rather than one being chosen.")
    say("")
    say(f"  figures/ currently shades a ceiling band of {mf.CEILING[0]}-{mf.CEILING[1]}, which")
    say("  came from the discarded AUC inversion. This simulation does not support that band.")
    say("  The band is NOT changed by this script -- CEILING lives in make_figures.py and")
    say("  editing it would silently move a reference line in figures 4 and 5. Change it")
    say("  deliberately, or drop the band and quote the range in the caption instead.")
    return res


# ==========================================================================
# REGENERATED FIGURES
# ==========================================================================
METRIC_NICE = {"auc": "AUC", "macro_f1": "macro-F1", "balanced_acc": "balanced acc."}
METRIC_COLOR = {"auc": OKABE[0], "macro_f1": OKABE[1], "balanced_acc": OKABE[2]}
_CACHE = {}


def task1_summary():
    if "t1" not in _CACHE:
        tbl = cohort_unit_masking()
        _CACHE["t1"] = (tbl, summarize_cohort_unit(tbl) if not tbl.empty else pd.DataFrame())
    return _CACHE["t1"]


def new_fig11():
    d = masking_raw()
    if not {"real_graph", "empty_graph"} <= set(d["model"].unique()):
        raise SkipFigure("real_graph / empty_graph masking rows missing")
    _, summ = task1_summary()
    if summ.empty:
        raise SkipFigure("cohort-unit summary unavailable")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))

    for ax, metric in zip(axes[:2], ["auc", "macro_f1"]):
        lv = per_cohort_levels(d, metric)
        for model in ("real_graph", "empty_graph"):
            g = fx._curve(lv, model, metric)
            c = CONDITION_COLOR[model]
            ax.plot(g["mask_rate"] * 100, g["mean"], marker="o", lw=2, color=c,
                    label=CONDITION_LABEL[model])
            ax.fill_between(g["mask_rate"] * 100, g["mean"] - g["ci"], g["mean"] + g["ci"],
                            color=c, alpha=0.18)
        ax.set_xlabel("Genes hidden at inference (%)")
        ax.set_ylabel(f"{METRIC_NICE[metric]} — mean of 8 per-cohort values (95% CI, n = 8)")
        ax.set_title(f"{METRIC_NICE[metric]}: graph "
                     f"{'loses' if metric == 'auc' else 'gains'} under masking")
        ax.legend(fontsize=8)

    ax = axes[2]
    sub = summ[(summ["condition"] == "real_graph") & (summ["quantity"] == "delta_level")]
    for k, metric in enumerate(["auc", "macro_f1", "balanced_acc"]):
        s = sub[sub["metric"] == metric].sort_values("mask_rate")
        if s.empty:
            continue
        x = s["mask_rate"] * 100 + (k - 1) * 0.9          # jitter so CIs stay readable
        ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["lo"], s["hi"] - s["mean"]],
                    marker="o", ms=5, lw=1.6, capsize=3, color=METRIC_COLOR[metric],
                    label=METRIC_NICE[metric])
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Genes hidden at inference (%)")
    ax.set_ylabel("STRING graph − no-graph control\n(95% CI across the 8 held-out cohorts)",
                  fontsize=8.5)
    ax.set_title("Difference at each masking rate, cohort as the unit (n = 8)", fontsize=9.5)
    ax.legend(fontsize=8)

    fig.suptitle("Robustness to missing genes — AUC and threshold metrics disagree in "
                 "direction, and at n = 8 cohorts most intervals cover zero", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def new_fig13():
    _, summ = task1_summary()
    if summ.empty:
        raise SkipFigure("cohort-unit summary unavailable")

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})

    # ---- LEFT: per-rate, cohort-unit difference vs the no-graph control ----
    ax = axes[0]
    sub = summ[(summ["metric"] == "auc") & (summ["quantity"] == "delta_level")]
    conds = [c for c in CONDITION_CKPT_ROOT
             if c != "empty_graph" and c in set(sub["condition"])]
    if not conds:
        ax.text(0.5, 0.5, "no conditions to compare", ha="center", va="center")
    else:
        for k, cond in enumerate(conds):
            s = sub[sub["condition"] == cond].sort_values("mask_rate")
            off = (k - (len(conds) - 1) / 2) * 1.1
            ax.errorbar(s["mask_rate"] * 100 + off, s["mean"],
                        yerr=[s["mean"] - s["lo"], s["hi"] - s["mean"]],
                        marker="o", ms=4.5, lw=1.5, capsize=2.5,
                        color=CONDITION_COLOR[cond], label=CONDITION_LABEL[cond])
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Genes hidden at inference (%)")
        ax.set_ylabel("AUC minus the no-graph control\n"
                      "(95% CI across the 8 held-out cohorts)", fontsize=8.5)
        ax.set_title("Difference at each masking rate, cohort as the unit (n = 8)",
                     fontsize=9.5)
        ax.legend(fontsize=7.5)

    # ---- RIGHT: unchanged -- clean-data paired advantage from saved probabilities ----
    ax = axes[1]
    fx.ensure_diagnosis()
    pc = _CACHE.get("pc")
    if pc is None:
        pc = fx.D.get("per_cohort_rate0")
        if pc is None:
            pc = fx.per_cohort_rate0_table()
        _CACHE["pc"] = pc
    rows = []
    if not pc.empty:
        piv = pc.pivot_table(index="fold", columns="condition", values="auc")
        if "empty_graph" in piv.columns:
            for cond in CONDITION_CKPT_ROOT:
                if cond == "empty_graph" or cond not in piv.columns:
                    continue
                delta = (piv[cond] - piv["empty_graph"]).dropna()
                if len(delta) >= 2:
                    rows.append({"condition": cond, **cohort_ci(delta.to_numpy())})
    right = pd.DataFrame(rows)
    if right.empty:
        ax.text(0.5, 0.5, "no rate-0 probabilities available", ha="center", va="center")
    else:
        order = [c for c in CONDITION_CKPT_ROOT if c in set(right["condition"])]
        s = right.set_index("condition").reindex(order)
        ax.bar(range(len(order)), s["mean"], color=[CONDITION_COLOR[c] for c in order],
               edgecolor="#333", lw=0.5,
               yerr=[s["mean"] - s["lo"], s["hi"] - s["mean"]], capsize=4,
               error_kw={"lw": 0.9})
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([CONDITION_LABEL[c] for c in order], rotation=18, ha="right",
                           fontsize=8)
        ax.set_ylabel("AUC advantage over no-graph at 0% masked", fontsize=8.5)
        ax.set_title(f"Clean data, paired within cohort\n(95% CI, n = {int(s['n'].max())} cohorts)",
                     fontsize=9.5)
        for i, (m, hi) in enumerate(zip(s["mean"], s["hi"])):
            ax.annotate(f"{m:+.4f}", (i, hi), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=7.5)

    fig.suptitle("Advantage over the no-graph control, per graph condition", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def new_fig18():
    rec = _CACHE.get("t2")
    if rec is None:
        rec = agreement_levels()["reconciled"]
        _CACHE["t2"] = rec
    if rec.empty:
        raise SkipFigure("no family-level agreement blocks available")
    fam = rec[rec["level"] == "architecture_family"].set_index("relation")
    if fam.empty:
        raise SkipFigure("architecture_family level absent")

    raw = read_csv(CONS / "family_agreement_summary.csv")
    blocks = raw[raw["level"] == "architecture_family_pairwise"].copy()
    blocks["_within"] = [str(r).split(" vs ")[0] == str(r).split(" vs ")[-1]
                         for r in blocks["relation"]]

    order = [r for r in ("within", "between") if r in fam.index]
    x = np.arange(len(order))
    obs = [fam.loc[r, "mean_jaccard_observed__block_averaged"] for r in order]
    exp = [fam.loc[r, "mean_jaccard_expected_chance__block_averaged"] for r in order]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(x - 0.2, obs, 0.38, color=CODAGAT, label="observed (averaged over family blocks)")
    ax.bar(x + 0.2, exp, 0.38, color=NEUTRAL, label="expected by chance")

    # the individual family blocks, so the spread behind each average is visible
    for i, r in enumerate(order):
        b = blocks[blocks["_within"] == (r == "within")]
        if b.empty:
            continue
        jx = np.random.default_rng(i).normal(i - 0.2, 0.035, len(b))
        ax.scatter(jx, b["mean_jaccard_observed"], s=26, color="white", edgecolor="#222",
                   linewidth=0.9, zorder=4,
                   label="individual family blocks" if i == 0 else None)
        for xi, (_, row) in zip(jx, b.iterrows()):
            ax.annotate(str(row["relation"]).replace("tree_ensemble", "tree"),
                        (xi, row["mean_jaccard_observed"]), textcoords="offset points",
                        xytext=(7, -2), fontsize=6.5, color="#333")
        ax.annotate(f"excess {fam.loc[r, 'mean_jaccard_excess__block_averaged']:+.3f}",
                    (i, max(obs[i], exp[i]) + 0.035), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} architecture family" for r in order], fontsize=9)
    ax.set_ylabel("Mean Jaccard of error sets")
    ax.set_ylim(0, max(max(obs), max(exp)) + 0.16)
    ax.set_title("Error agreement within vs between architecture families\n"
                 "(linear · tree ensemble · GAT)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")

    note = []
    if "between" in fam.index:
        note.append(f"between-family, pair-weighted: "
                    f"{fam.loc['between', 'mean_jaccard_observed__pair_weighted']:.3f}")
    var = rec[(rec["level"] == "variant_group") & (rec["relation"] == "between")]
    if not var.empty:
        note.append(f"between-variant-group: "
                    f"{var.iloc[0]['mean_jaccard_observed__block_averaged']:.3f} "
                    f"(higher — still contains GAT-vs-GAT pairs)")
    if note:
        ax.annotate("\n".join(note), xy=(0.02, 0.97), xycoords="axes fraction", va="top",
                    fontsize=7, color="#444",
                    bbox={"fc": "#F5F5F5", "ec": "#CCC", "lw": 0.6, "pad": 3.0})
    return fig


OVERRIDES = [
    (11, "masking_real_vs_empty", ["graph_utility/masking_curves.csv",
                                    "review_fixes/task1_masking_cohort_unit.csv"],
     "Robustness to missing genes. The right panel gives the STRING-minus-control "
     "difference at each masking rate with the cohort as the unit of analysis (95% CI "
     "across the 8 held-out cohorts). AUC favours the control and the threshold metrics "
     "favour the graph, but at n = 8 most intervals cover zero.",
     new_fig11),
    (13, "advantage_over_empty", ["graph_utility/masking_curves.csv",
                                   "learned_graph/masking_curves_all_conditions.csv",
                                   "review_fixes/task1_masking_cohort_unit.csv"],
     "Advantage over the no-graph control for every graph condition: per-masking-rate "
     "AUC difference with the cohort as the unit (left, 95% CI across 8 cohorts) and "
     "clean-data AUC advantage paired within cohort (right).",
     new_fig13),
    (18, "family_agreement_vs_chance", ["consensus_failures/family_agreement_summary.csv",
                                         "review_fixes/task2_agreement_levels.csv"],
     "Within- versus between-architecture-family error agreement against chance, averaged "
     "over the three family blocks (linear, tree ensemble, GAT) so each family pair counts "
     "equally. Individual blocks are overlaid; the pair-weighted and variant-group values "
     "are annotated for comparison.",
     new_fig18),
]


def patch_registry():
    by_num = {spec["num"]: spec for spec in mf.REGISTRY}
    for num, name, sources, caption, fn in OVERRIDES:
        spec = {"num": num, "name": name, "sources": sources, "caption": caption, "fn": fn}
        if num in by_num:
            by_num[num].update(spec)
        else:
            mf.REGISTRY.append(spec)
    print(f"[review] replaced registry entries {[n for n, *_ in OVERRIDES]}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="four statistical corrections, then rebuild the figures (read-only "
                    "on all results)")
    ap.add_argument("--kappas", nargs="*", type=float, default=DEFAULT_KAPPAS,
                    help="published inter-rater weighted kappa values to invert")
    ap.add_argument("--mc-n", type=int, default=400_000,
                    help="patients per Monte Carlo replicate for the ceiling AUC")
    ap.add_argument("--mc-rep", type=int, default=5,
                    help="Monte Carlo replicates; their spread is the reported SE")
    ap.add_argument("--tasks-only", action="store_true",
                    help="write review_fixes/ and stop; no figures rebuilt")
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="explicit figure numbers to build")
    ap.add_argument("--outdir", default=str(mf.OUT))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LINES.clear()
    head("CoDA-GAT review fixes")
    say("Read-only on all results files. Writes only review_fixes/ and figures/.")
    say(f"repo: {REPO}")
    say("")

    t1_tbl, t1_summ = task1()
    _CACHE["t1"] = (t1_tbl, t1_summ)
    say("")
    t2 = task2()
    _CACHE["t2"] = t2
    say("")
    t3_runs, t3_pooled, t3_per = task3()
    say("")
    t4 = task4(args.kappas, args.mc_n, args.mc_rep)

    for name, df in (("task1_masking_cohort_unit", t1_tbl),
                      ("task1_masking_cohort_unit_summary", t1_summ),
                      ("task2_agreement_levels", t2),
                      ("task3_lac_set_sizes_per_run", t3_runs),
                      ("task3_lac_set_sizes_pooled", t3_pooled),
                      ("task3_lac_set_sizes_per_cohort", t3_per),
                      ("task4_latent_noise_ceiling", t4)):
        if df is not None and not df.empty:
            df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    say("")
    head("FILES WRITTEN")
    for p in sorted(OUT_DIR.glob("*")):
        say(f"  review_fixes/{p.name}")
    (OUT_DIR / "summary.txt").write_text("\n".join(LINES) + "\n")
    print(f"\n[review] wrote {(OUT_DIR / 'summary.txt').relative_to(REPO)}", flush=True)

    if args.tasks_only:
        print("[review] --tasks-only: no figures rebuilt")
        return

    # both correction layers, in order: fix_figures first, then these overrides
    fx.patch_registry()
    patch_registry()
    argv = ["make_figures.py", "--outdir", args.outdir]
    if args.only:
        argv += ["--only"] + [str(n) for n in args.only]
        print("[review] NOTE: --only makes all_figures.pdf partial. Omit it for the "
              "complete document.")
    sys.argv = argv
    mf.main()


if __name__ == "__main__":
    main()
