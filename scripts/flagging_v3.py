"""
CoDA-GAT — flagging v3: a NON-CIRCULAR test of the label-noise estimate.

WHAT WAS WRONG WITH THE OLD CLAIM
  flagging_v2 flagged 18.0% of patients and the forward noise model implied an
  18.6% F2+ misassignment rate, and those two numbers were reported as
  agreeing. They cannot agree, because 18% was not measured -- it was CHOSEN.
  MATCHED_COUNTS in flagging_v2 contains the literal string "18%", and
  rank_to_count() takes exactly that many patients by construction. A quantity
  fixed by fiat cannot corroborate a quantity predicted by a model; the match
  was an artefact of the target, and would have appeared at any target.

WHAT REPLACES IT
  Confident Learning (Northcutt, Jiang & Chuang, JAIR 2021) ESTIMATES the noise
  rate from the data. Nothing sets it. The estimate is then compared against an
  independent reference computed in review_fixes/ from published inter-rater
  kappa, which uses no model output at all. Two quantities derived from
  disjoint evidence either overlap or they do not, and this script reports
  which at face value.

  reference  (review_fixes/task4_latent_noise_ceiling.csv) <- published kappa
  estimate   (this script, Confident Learning)             <- model + labels
                       ^ no shared input, so overlap is informative

METHOD DISCIPLINE
  - The ordinal stage F0-F4 is used ONLY to evaluate flags, never to produce
    them. Enforced by flagging_v2.assert_no_stage_leak(), reused unchanged.
  - Per-class thresholds come from each fold's CALIBRATION cohort only, never
    from the test cohort whose labels are in question.
  - NO THRESHOLD TUNING. Confident Learning's thresholds are computed from a
    formula, not selected. This script exposes no knob that changes the
    estimate; --n-boot only affects CI resolution.
  - Everything is pre-registered in flagging_v3/PREREG.txt, written with a
    timestamp before any result is computed.

Inference only -- no model is loaded and nothing is retrained. Reads saved
probabilities from results_b7/, graph_utility/control_checkpoints/ and
flagging_v2/cache/. Read-only on every results directory; writes flagging_v3/.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import binomtest

# the evaluation metrics are IMPORTED, not re-implemented, so v2 and v3 flag
# sets are scored by identical code and are directly comparable
from flagging_v2 import (
    ADJACENT_STAGES, TWO_STAGE_STAGES, ALL_STAGES, ALLOWED_FLAG_INPUTS,
    assert_no_stage_leak, describe_variant, per_cohort_table, fisher_or,
    boot_indices, OKABE, CODAGAT, NEUTRAL, SEEDS, fold_index, fold_dirs,
)

REPO = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO / "folds"
RESULTS_B7 = REPO / "results_b7"
GUTIL_CKPT = REPO / "graph_utility" / "control_checkpoints"
V2_CACHE = REPO / "flagging_v2" / "cache"
REVIEW = REPO / "review_fixes"
OUT_DIR = REPO / "flagging_v3"

PRIMARY_MODE = "off"
CONDITIONS = ["real_graph", "empty_graph"]
CONDITION_LABEL = {"real_graph": "CoDA-GAT (STRING graph)", "empty_graph": "empty-graph control"}
N_BOOT = 2000
BOOT_SEED = 20260923
N_CLASSES = 2
REFERENCE_MARGINAL = "exact-labels"

LINES = []


def log(msg):
    print(f"[flag-v3] {msg}", flush=True)


def say(line=""):
    LINES.append(str(line))
    print(str(line), flush=True)


def head(title):
    say("=" * 78)
    say(title)
    say("=" * 78)


# ==========================================================================
# reference range -- read from disk, never hardcoded
# ==========================================================================
def load_reference_range(marginal=REFERENCE_MARGINAL):
    """The forward noise model's F2+ misassignment range from review_fixes/.

    Produced by inverting PUBLISHED inter-rater weighted kappa into a latent
    reader-noise sigma and computing P(noisy F2+ call != ideal F2+ call) in
    closed form. No AUC, no probability, no output of any model of ours enters
    it -- which is the entire point of using it as the reference.
    """
    p = REVIEW / "task4_latent_noise_ceiling.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run scripts/review_fixes.py first -- flagging_v3's reference "
            f"range comes from it and must not be typed in by hand.")
    d = pd.read_csv(p)
    sub = d[d["stage_marginal"] == marginal]
    if sub.empty:
        sub = d
    return {
        "lo": float(sub["f2_misassignment_rate"].min()),
        "hi": float(sub["f2_misassignment_rate"].max()),
        "marginal": marginal,
        "n_settings": len(sub),
        "kappas": sorted(sub["kappa_published"].unique().tolist()),
        "weightings": sorted(sub["kappa_weighting"].unique().tolist()),
        "all_range": (float(d["f2_misassignment_rate"].min()),
                       float(d["f2_misassignment_rate"].max())),
    }


# ==========================================================================
# PRE-REGISTRATION -- written before anything is computed
# ==========================================================================
PREREG = """CoDA-GAT flagging v3 -- PRE-REGISTRATION
written: {stamp}
script:  scripts/flagging_v3.py

This file is written BEFORE any result in this analysis is computed, and is
never modified afterwards. If it already exists, the script leaves it alone.

--------------------------------------------------------------------------------
WHY THIS ANALYSIS EXISTS
--------------------------------------------------------------------------------
flagging_v2 reported that its 18.0% flag rate agreed with the forward noise
model's 18.6% F2+ misassignment rate. That comparison is CIRCULAR: 18% was a
pre-set target in flagging_v2 (MATCHED_COUNTS contains "18%", and
rank_to_count() selects exactly that many patients). The flag rate was chosen,
not estimated, so it cannot corroborate anything. It would have "agreed" with
whatever number it had been set to.

This analysis replaces it with a rate that is ESTIMATED from data and compared
against a reference derived from evidence that shares no input with it.

--------------------------------------------------------------------------------
INDEPENDENT REFERENCE (fixed before this analysis runs)
--------------------------------------------------------------------------------
Source: review_fixes/task4_latent_noise_ceiling.csv, column
f2_misassignment_rate, over all published weighted kappa values and BOTH the
linear and quadratic weightings, on the '{marginal}' stage marginal.

    reference range = {lo:.1%} to {hi:.1%}
    kappa values    = {kappas}
    weightings      = {weightings}
    settings        = {n_settings}

That range is produced by inverting published inter-rater agreement into a
latent reader-noise sigma and evaluating P(noisy F2+ call != ideal F2+ call)
in closed form. It uses NO model output: no AUC, no predicted probability, no
flag. This independence is what makes an overlap meaningful.

--------------------------------------------------------------------------------
PRIMARY TEST
--------------------------------------------------------------------------------
The Confident Learning (Northcutt, Jiang & Chuang 2021) estimated noise rate
for CoDA-GAT (mode = {mode}, seed-averaged p(F2+) over {n_seeds} seeds), pooled
over the 8 held-out cohorts.

DECISION RULE, fixed now: the estimate AGREES with the reference iff its 95%
bootstrap CI overlaps [{lo:.4f}, {hi:.4f}]. Overlap of the intervals, not of
the point estimates. A point estimate inside the range with a CI that also
extends far outside it is reported as agreement with wide uncertainty, and the
width is stated. Non-overlap is reported as disagreement, plainly, and is NOT
followed by an adjustment of either quantity.

--------------------------------------------------------------------------------
SECONDARY TESTS
--------------------------------------------------------------------------------
(a) The same Confident Learning estimate for the EMPTY-GRAPH model. Reported
    beside the real-graph estimate. No claim about which is "right" is made in
    advance; both are shown against the same reference.

(b) DIRECTION TEST. Among Confident-Learning-flagged patients, the share whose
    recorded label is "too high" (recorded F2+, model confidently says not).
    Two-sided exact binomial test against 0.5. The noise model predicts
    symmetry when both scales carry the same F2+ prevalence, so a significant
    departure indicates the noise is directional -- a systematic reading bias
    rather than symmetric reader disagreement.

(c) PER-COHORT rates for all 8 held-out cohorts. n = 8 is underpowered;
    per-cohort numbers are descriptive.

--------------------------------------------------------------------------------
CONFIDENT LEARNING -- exact specification
--------------------------------------------------------------------------------
1. PER-CLASS THRESHOLDS, from each fold's CALIBRATION cohort only:
       t_j = mean of p_hat(y = j | x) over calibration samples LABELLED j
   Computed per fold. Never from the test cohort. These are computed from a
   formula; they are not tuned, searched, or selected.

2. CONFIDENT JOINT on the test cohort (Northcutt et al. eq. 2):
       C[i][j] = #{{ x : label(x) = i,  p_j(x) >= t_j,
                     j = argmax over classes k with p_k(x) >= t_k }}
   Samples exceeding no threshold are excluded, as in the original method.
   Per-fold joints are SUMMED to pool the 8 cohorts.

3. CALIBRATION of the joint to a distribution:
       Q[i][j] = C[i][j] / sum_j C[i][j] * n_i,  then normalised to sum to 1
   Q is CL's estimate of P(recorded label = i, true label = j).

4. ESTIMATED NOISE RATE = sum of the off-diagonal of Q = 1 - trace(Q).

5. FLAGGED PATIENTS, both variants reported:
     (i)  confident-joint off-diagonal members -- every sample whose confident
          class differs from its recorded label;
     (ii) prune_by_noise_rate -- cleanlab's default: within each off-diagonal
          cell, rank candidates by margin p_j - p_i and keep round(Q[i][j] * n)
          of them, so the flagged count matches the estimated rate.

6. CROSS-CHECK: cleanlab.filter.find_label_issues, if cleanlab is installed.
   It computes its own thresholds from the test split rather than from
   calibration, so it is an independent implementation, NOT a replication.
   Both are reported. If cleanlab is absent this is recorded as unavailable.

--------------------------------------------------------------------------------
UNCERTAINTY
--------------------------------------------------------------------------------
95% CIs by bootstrap over PATIENTS WITHIN COHORT: {n_boot} resamples, each
cohort resampled with replacement to its own original size. Per-fold thresholds
are held fixed, since they derive from calibration cohorts that are not
resampled. The confident joint is rebuilt on each resample and the rate
recomputed, so the CI propagates the joint's own sampling variability.

--------------------------------------------------------------------------------
COMPARABILITY WITH flagging_v2
--------------------------------------------------------------------------------
CL-flagged patients are scored on exactly the metrics flagging_v2 used --
stage distribution F0-F4, F1/F2 boundary enrichment odds ratio, two-stage
share (F0 or F3), and direction split -- by IMPORTING flagging_v2's functions
rather than reimplementing them, so no difference between v2 and v3 numbers can
come from the scoring code.

--------------------------------------------------------------------------------
HARD RULES
--------------------------------------------------------------------------------
  1. The ordinal stage F0-F4 is used ONLY to evaluate flags, NEVER to create
     them. Enforced by assert_no_stage_leak() against the whitelist:
       {allowed}
  2. Thresholds come from calibration cohorts only.
  3. No threshold tuning of any kind. No knob in this script changes the
     estimate.
  4. The result is reported at face value, INCLUDING if the estimated rate
     falls outside the reference range. A non-overlap will not be followed by
     re-deriving either number.
  5. Inference only. No model is loaded; no retraining.
"""


def write_prereg(ref):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "PREREG.txt"
    if path.exists():
        log(f"PREREG.txt already exists ({path.stat().st_size} bytes) -- left untouched.")
        return path
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    path.write_text(PREREG.format(
        stamp=stamp, marginal=ref["marginal"], lo=ref["lo"], hi=ref["hi"],
        kappas=ref["kappas"], weightings=ref["weightings"], n_settings=ref["n_settings"],
        mode=PRIMARY_MODE, n_seeds=len(SEEDS), n_boot=N_BOOT,
        allowed=", ".join(sorted(ALLOWED_FLAG_INPUTS)),
    ))
    log(f"wrote {path.relative_to(REPO)} at {stamp}")
    return path


# ==========================================================================
# saved probabilities
# ==========================================================================
def run_dir_for(condition, fold_dir_name, seed):
    if condition == "real_graph":
        return RESULTS_B7 / fold_dir_name / f"mode_{PRIMARY_MODE}" / f"seed_{seed}"
    return GUTIL_CKPT / fold_dir_name / condition / f"seed_{seed}"


def seed_averaged_probs(condition, fold_dir, split):
    """Seed-averaged P(class) for one split, from saved arrays only.

    Averaging probabilities across seeds before running CL (rather than running
    CL per seed) is deliberate: CL's thresholds are means of predicted
    probabilities, and the quantity of interest is the ensemble's confidence,
    not the spread of five independently initialised runs.
    """
    mats = []
    for seed in SEEDS:
        p = run_dir_for(condition, fold_dir.name, seed) / f"probs_{split}.npy"
        if p.exists():
            a = np.load(p)
            mats.append(a if a.ndim == 2 else np.column_stack([1 - a, a]))
    if not mats and split == "test":
        # documented fallback: flagging_v2 cached seed-level test probabilities
        fid = fold_index(fold_dir.name)
        mode = PRIMARY_MODE if condition == "real_graph" else "-"
        for seed in SEEDS:
            c = V2_CACHE / f"{condition}_{mode}_fold{fid}_seed{seed}.npz"
            if c.exists():
                p1 = np.load(c, allow_pickle=True)["p_test"]
                mats.append(np.column_stack([1 - p1, p1]))
    if not mats:
        return None, 0
    return np.mean(mats, axis=0), len(mats)


def split_labels(fold_dir, split):
    y = pd.read_csv(fold_dir / split / "y.csv", index_col=0)
    ids = pd.read_csv(fold_dir / split / "sample_ids.csv")["sample_id"].tolist()
    assert list(y.index) == ids, f"{fold_dir.name}/{split}: label order does not match ids"
    return y["y_T1_significant"].to_numpy().astype(int), ids, y


# ==========================================================================
# Confident Learning
# ==========================================================================
def per_class_thresholds(p_cal, y_cal, n_classes=N_CLASSES):
    """t_j = mean predicted probability of class j among samples LABELLED j.

    Northcutt et al. eq. 1. A class absent from a calibration cohort has no
    threshold; 0.5 is used and the event is recorded, because inventing a
    threshold silently would change the joint.
    """
    t = np.full(n_classes, np.nan)
    for j in range(n_classes):
        m = y_cal == j
        if m.any():
            t[j] = float(p_cal[m, j].mean())
    missing = [j for j in range(n_classes) if not np.isfinite(t[j])]
    t = np.where(np.isfinite(t), t, 0.5)
    return t, missing


def confident_class(p, t, n_classes=N_CLASSES):
    """Per sample: the class it is CONFIDENTLY assigned to, or -1 for none.

    A sample counts only if some class exceeds its threshold; among those that
    do, the highest probability wins. Samples exceeding nothing are excluded
    from the joint, exactly as in the original method -- that exclusion is what
    makes the joint 'confident'.
    """
    above = p >= t[None, :]
    any_above = above.any(axis=1)
    masked = np.where(above, p, -np.inf)
    j = masked.argmax(axis=1)
    return np.where(any_above, j, -1), any_above


def confident_joint(y, j_conf, n_classes=N_CLASSES):
    C = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(n_classes):
            C[i, j] = np.sum((y == i) & (j_conf == j))
    return C


def calibrate_joint(C, y, n_classes=N_CLASSES):
    """Scale each row of the confident joint to that class's true label count,
    then normalise -- cleanlab's calibrate_confident_joint. This is what turns
    a count of confident assignments into an estimate of
    P(recorded label = i, true label = j)."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    rows = C.sum(axis=1)
    Q = np.zeros_like(C)
    for i in range(n_classes):
        if rows[i] > 0:
            Q[i] = C[i] / rows[i] * counts[i]
    total = Q.sum()
    return Q / total if total > 0 else Q


def noise_rate_from_joint(Q):
    return float(1.0 - np.trace(Q)) if Q.sum() > 0 else np.nan


def prune_by_noise_rate(df, Q, n_classes=N_CLASSES):
    """cleanlab's default selection: within each off-diagonal cell take the
    round(Q[i][j] * n) candidates with the largest margin p_j - p_i, so the
    flagged COUNT matches the ESTIMATED rate instead of being set by hand."""
    n = len(df)
    flags = np.zeros(n, dtype=bool)
    probs = np.column_stack([df["p_class0"].to_numpy(), df["p_class1"].to_numpy()])
    y = df["y_true"].to_numpy()
    jc = df["confident_class"].to_numpy()
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                continue
            k = int(round(Q[i, j] * n))
            if k <= 0:
                continue
            cand = np.where((y == i) & (jc == j))[0]
            if cand.size == 0:
                continue
            margin = probs[cand, j] - probs[cand, i]
            flags[cand[np.argsort(-margin)[:k]]] = True
    return flags


def run_cl(df, thresholds_by_fold):
    """Pooled Confident Learning over the 8 cohorts, each fold scored with its
    own calibration-derived thresholds."""
    assert_no_stage_leak("confident_learning", ["p_f2plus", "y_true", "fold"])
    jc = np.full(len(df), -1, dtype=int)
    for fold_id, t in thresholds_by_fold.items():
        m = (df["fold"] == fold_id).to_numpy()
        if not m.any():
            continue
        p = np.column_stack([df.loc[m, "p_class0"].to_numpy(),
                             df.loc[m, "p_class1"].to_numpy()])
        jc[m], _ = confident_class(p, t)
    out = df.copy()
    out["confident_class"] = jc
    y = out["y_true"].to_numpy()
    C = confident_joint(y, jc)
    Q = calibrate_joint(C, y)
    out["cl_offdiag"] = (jc >= 0) & (jc != y)
    out["cl_pruned"] = prune_by_noise_rate(out, Q)
    return out, C, Q


def bootstrap_noise_rate(df, thresholds_by_fold, n_boot, seed):
    """Resample patients within cohort, rebuild the joint, recompute the rate.

    The confident class of each patient is fixed (it depends only on that
    patient's probabilities and its fold's thresholds), so resampling can reuse
    it -- but the JOINT and its calibration are rebuilt every time, which is
    where the variability that matters lives.
    """
    y = df["y_true"].to_numpy()
    jc = df["confident_class"].to_numpy()
    folds = df["fold"].to_numpy()
    vals = []
    for idx in boot_indices(folds, n_boot, seed):
        Cb = confident_joint(y[idx], jc[idx])
        Qb = calibrate_joint(Cb, y[idx])
        vals.append(noise_rate_from_joint(Qb))
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan, np.nan, np.nan
    return (float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def cleanlab_crosscheck(df):
    """Independent implementation, not a replication: cleanlab derives its own
    thresholds from the data it is given, where this script derives them from
    the calibration cohorts. Disagreement between the two is informative about
    that choice, not about a bug."""
    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.count import compute_confident_joint, estimate_joint
        import cleanlab
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}
    probs = np.column_stack([df["p_class0"].to_numpy(), df["p_class1"].to_numpy()])
    y = df["y_true"].to_numpy()
    try:
        issues = find_label_issues(labels=y, pred_probs=probs,
                                    filter_by="prune_by_noise_rate")
        cj = compute_confident_joint(labels=y, pred_probs=probs)
        Q = estimate_joint(labels=y, pred_probs=probs, confident_joint=cj)
    except Exception as exc:                                   # noqa: BLE001
        return {"available": False, "reason": f"cleanlab raised: {exc}"}
    return {"available": True, "version": getattr(cleanlab, "__version__", "?"),
            "n_issues": int(np.asarray(issues).sum()),
            "rate": float(np.asarray(issues).mean()),
            "joint": np.asarray(Q), "confident_joint": np.asarray(cj),
            "issues": np.asarray(issues).astype(bool)}


# ==========================================================================
# assembly
# ==========================================================================
def build_condition_frame(condition, folds):
    rows, thresholds, notes = [], {}, []
    for fold_dir in folds:
        fid = fold_index(fold_dir.name)
        p_cal, n_cal_seeds = seed_averaged_probs(condition, fold_dir, "calibration")
        p_test, n_test_seeds = seed_averaged_probs(condition, fold_dir, "test")
        if p_cal is None or p_test is None:
            notes.append(f"fold{fid}: missing saved probabilities -- skipped")
            continue
        y_cal, _, _ = split_labels(fold_dir, "calibration")
        y_test, test_ids, meta = split_labels(fold_dir, "test")
        if len(p_cal) != len(y_cal) or len(p_test) != len(y_test):
            notes.append(f"fold{fid}: probability/label length mismatch -- skipped")
            continue
        t, missing = per_class_thresholds(p_cal, y_cal)
        if missing:
            notes.append(f"fold{fid}: calibration cohort has no samples of class(es) "
                          f"{missing}; threshold defaulted to 0.5 for those")
        thresholds[fid] = t
        rows.append(pd.DataFrame({
            "condition": condition, "fold": fid,
            "test_cohort": fold_dir.name.split("_", 1)[1],
            "sample_id": test_ids, "y_true": y_test,
            "stage": meta["stage"].to_numpy(),
            "p_class0": p_test[:, 0], "p_class1": p_test[:, 1],
            "p_f2plus": p_test[:, 1],
            "t0": t[0], "t1": t[1],
            "n_cal_seeds": n_cal_seeds, "n_test_seeds": n_test_seeds,
        }))
    if not rows:
        return None, {}, notes
    df = pd.concat(rows, ignore_index=True)
    df["disagreement"] = np.abs(df["p_f2plus"] - df["y_true"])
    return df, thresholds, notes


def direction_test(df, flags):
    """Among flagged patients: is 'recorded too high' as common as 'too low'?"""
    y = df["y_true"].to_numpy()
    too_high = int((flags & (y == 1)).sum())    # recorded F2+, model says not
    too_low = int((flags & (y == 0)).sum())
    n = too_high + too_low
    if n == 0:
        return {"n": 0, "too_high": 0, "too_low": 0, "share_too_high": np.nan, "p": np.nan}
    bt = binomtest(too_high, n, 0.5, alternative="two-sided")
    ci = bt.proportion_ci(confidence_level=0.95)
    return {"n": n, "too_high": too_high, "too_low": too_low,
            "share_too_high": too_high / n, "p": float(bt.pvalue),
            "ci_lo": float(ci.low), "ci_hi": float(ci.high)}


def overlaps(lo_a, hi_a, lo_b, hi_b):
    return (lo_a <= hi_b) and (lo_b <= hi_a)


# ==========================================================================
# plots
# ==========================================================================
def plot_rate_vs_reference(results, ref, path_png, path_pdf):
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.axhspan(ref["lo"] * 100, ref["hi"] * 100, color=OKABE[2], alpha=0.16, zorder=0,
               label=f"independent reference {ref['lo']:.1%}–{ref['hi']:.1%}\n"
                     f"(published κ, no model output)")
    names, x = [], []
    for k, (cond, r) in enumerate(results.items()):
        for variant, key in (("confident joint", "rate"), ("pruned", "rate_pruned")):
            if not np.isfinite(r.get(key, np.nan)):
                continue
            x.append(len(names))
            names.append(f"{CONDITION_LABEL[cond]}\n{variant}")
            lo, hi = r["ci_lo"] * 100, r["ci_hi"] * 100
            val = r[key] * 100
            ax.errorbar(len(names) - 1, val, yerr=[[val - lo], [hi - val]],
                        marker="o", ms=9, capsize=5, lw=2,
                        color=CODAGAT if cond == "real_graph" else OKABE[0])
            ax.annotate(f"{val:.1f}%", (len(names) - 1, hi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Estimated label noise rate (%)")
    ax.set_title("Confident Learning estimate vs the independent noise-model reference\n"
                 "(estimate and reference share no input)", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_stage_distributions(frames, path_png, path_pdf):
    items = [(n, d, f) for n, (d, f) in frames.items() if f.sum() > 0]
    if not items:
        return False
    ncol = min(3, len(items))
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.4 * nrow), squeeze=False)
    x = np.arange(len(ALL_STAGES))
    for ax, (name, d, f) in zip(axes.ravel(), items):
        bg = d["stage"].value_counts(normalize=True).reindex(ALL_STAGES, fill_value=0)
        fg = d.loc[f, "stage"].value_counts(normalize=True).reindex(ALL_STAGES, fill_value=0)
        ax.bar(x - 0.2, bg * 100, 0.38, color=NEUTRAL, label="all held-out patients")
        ax.bar(x + 0.2, fg * 100, 0.38, color=CODAGAT, label="CL-flagged")
        ax.axvspan(0.5, 2.5, color=OKABE[2], alpha=0.10, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_STAGES)
        ax.set_ylabel("Share (%)")
        ax.set_title(f"{name}\nn={int(f.sum())}", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7.5)
    fig.suptitle("Stage distribution of Confident-Learning-flagged patients "
                 "(shaded = F1/F2, one stage from the F2+ boundary)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return True


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Confident Learning noise-rate estimate vs an independent "
                    "noise-model reference (inference only, read-only on results)")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help="bootstrap resamples; affects CI resolution only, never the estimate")
    ap.add_argument("--prereg-only", action="store_true",
                    help="write PREREG.txt and stop, so it can be committed before results")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = load_reference_range()
    prereg_path = write_prereg(ref)
    if args.prereg_only:
        log("--prereg-only: stopping before any result is computed")
        return

    LINES.clear()
    head("CoDA-GAT flagging v3 -- non-circular label-noise test")
    say(f"pre-registration: {prereg_path.relative_to(REPO)}")
    say("Inference only; no model loaded; read-only on all results directories.")
    say("")
    say("WHY THE OLD CLAIM WAS WITHDRAWN")
    say("-" * 78)
    say("  flagging_v2 compared its 18.0% flag rate with the noise model's 18.6% and")
    say("  called them consistent. 18% was a PRE-SET TARGET in flagging_v2 -- its")
    say("  MATCHED_COUNTS list contains the string '18%' and rank_to_count() takes exactly")
    say("  that many patients. The rate was chosen, not measured, so the agreement was")
    say("  guaranteed in advance and carried no evidence. It is replaced below by a rate")
    say("  that nothing in this analysis sets.")
    say("")

    head("INDEPENDENT REFERENCE")
    say(f"  source   review_fixes/task4_latent_noise_ceiling.csv ({ref['marginal']} marginal)")
    say(f"  range    {ref['lo']:.2%} to {ref['hi']:.2%}   "
        f"({ref['n_settings']} settings: kappa {ref['kappas']} x {ref['weightings']})")
    say(f"  all-marginal range for comparison: {ref['all_range'][0]:.2%} to "
        f"{ref['all_range'][1]:.2%}")
    say("  Derived from published inter-rater kappa alone. No AUC, no predicted")
    say("  probability and no flag of ours enters it -- which is what makes an overlap")
    say("  with the Confident Learning estimate informative rather than tautological.")
    say("")

    folds = fold_dirs(args.folds)
    results, frames, cohort_tables, summary_rows = {}, {}, [], []

    for cond in CONDITIONS:
        head(f"CONFIDENT LEARNING -- {CONDITION_LABEL[cond]}")
        df, thresholds, notes = build_condition_frame(cond, folds)
        for n in notes:
            say(f"  NOTE {n}")
        if df is None:
            say("  no usable saved probabilities -- condition skipped")
            continue

        say(f"  patients {len(df)} across {df['fold'].nunique()} cohorts; "
            f"seeds averaged: calibration {df['n_cal_seeds'].min()}-{df['n_cal_seeds'].max()}, "
            f"test {df['n_test_seeds'].min()}-{df['n_test_seeds'].max()}")
        say("")
        say("  per-fold thresholds from the CALIBRATION cohort only "
            "(t_j = mean p(class j) among calibration samples labelled j):")
        say(f"    {'cohort':<12s} {'t0':>8s} {'t1':>8s}")
        for fid, t in sorted(thresholds.items()):
            coh = df[df["fold"] == fid]["test_cohort"].iloc[0]
            say(f"    {coh:<12s} {t[0]:>8.4f} {t[1]:>8.4f}")

        scored, C, Q = run_cl(df, thresholds)
        rate = noise_rate_from_joint(Q)
        mean_b, lo_b, hi_b = bootstrap_noise_rate(scored, thresholds, args.n_boot, BOOT_SEED)
        n_off = int(scored["cl_offdiag"].sum())
        n_pruned = int(scored["cl_pruned"].sum())

        say("")
        say("  confident joint C (rows = recorded label, cols = confident class):")
        say(f"    {'':>14s} {'conf=0':>10s} {'conf=1':>10s}   (excluded, no class above "
            f"threshold: {int((scored['confident_class'] < 0).sum())})")
        for i in range(N_CLASSES):
            say(f"    recorded={i:<5d} {C[i, 0]:>10.0f} {C[i, 1]:>10.0f}")
        say("")
        say("  calibrated joint Q = P(recorded label, true label):")
        for i in range(N_CLASSES):
            say(f"    recorded={i:<5d} {Q[i, 0]:>10.4f} {Q[i, 1]:>10.4f}")
        say(f"    trace (agreement) = {np.trace(Q):.4f}")
        say("")
        say(f"  ESTIMATED NOISE RATE = 1 - trace(Q) = {rate:.4f} ({rate:.2%})")
        say(f"  95% CI (bootstrap over patients within cohort, {args.n_boot} resamples): "
            f"[{lo_b:.4f}, {hi_b:.4f}] = [{lo_b:.2%}, {hi_b:.2%}]")
        say(f"  flagged, confident-joint off-diagonal: {n_off} ({n_off / len(scored):.2%})")
        say(f"  flagged, prune_by_noise_rate:          {n_pruned} "
            f"({n_pruned / len(scored):.2%})")

        agree = overlaps(lo_b, hi_b, ref["lo"], ref["hi"])
        results[cond] = {"rate": rate, "rate_pruned": n_pruned / len(scored),
                          "ci_lo": lo_b, "ci_hi": hi_b, "boot_mean": mean_b,
                          "n_offdiag": n_off, "n_pruned": n_pruned,
                          "n_patients": len(scored), "agrees": agree,
                          "Q": Q, "C": C}
        frames[CONDITION_LABEL[cond]] = (scored, scored["cl_pruned"].to_numpy())

        say("")
        say(f"  vs reference [{ref['lo']:.2%}, {ref['hi']:.2%}]: "
            f"{'OVERLAPS -> AGREES' if agree else 'NO OVERLAP -> DISAGREES'}")

        # ---- the flagging_v2 metric set, computed by v2's own code ----
        for variant, col in (("confident joint", "cl_offdiag"), ("pruned", "cl_pruned")):
            flags = scored[col].to_numpy()
            r = describe_variant(f"CL {variant} [{cond}]", scored, flags, None, None,
                                  extra={"condition": cond, "variant_kind": variant,
                                          "cl_rate": rate, "cl_ci_lo": lo_b, "cl_ci_hi": hi_b,
                                          "agrees_with_reference": agree})
            summary_rows.append(r)
            cohort_tables.append(per_cohort_table(f"CL {variant} [{cond}]", scored, flags))

        # ---- direction test ----
        say("")
        say("  DIRECTION TEST (exact binomial vs 0.5, on prune_by_noise_rate flags)")
        dt = direction_test(scored, scored["cl_pruned"].to_numpy())
        results[cond]["direction"] = dt
        if dt["n"]:
            say(f"    recorded too high (label F2+, model says not): {dt['too_high']}")
            say(f"    recorded too low  (label not-F2+, model says F2+): {dt['too_low']}")
            say(f"    share too high = {dt['share_too_high']:.3f} "
                f"[{dt['ci_lo']:.3f}, {dt['ci_hi']:.3f}]  p = {dt['p']:.4g}")
            say(f"    -> {'SYMMETRIC: consistent with the noise model' if dt['p'] >= 0.05 else 'DIRECTIONAL: the noise model predicts symmetry; this departs from it'}")
        else:
            say("    no flags -- direction not testable")

        # ---- cleanlab cross-check ----
        say("")
        say("  CROSS-CHECK: cleanlab.filter.find_label_issues")
        cc = cleanlab_crosscheck(scored)
        results[cond]["cleanlab"] = cc
        if not cc["available"]:
            say(f"    NOT AVAILABLE -- {cc['reason']}")
            say("    Install with:  /opt/anaconda3/envs/dagat/bin/pip install cleanlab")
            say("    then re-run. The primary result above does not depend on it.")
        else:
            say(f"    cleanlab {cc['version']}: {cc['n_issues']} issues ({cc['rate']:.2%})")
            say(f"    this script (pruned): {n_pruned} ({n_pruned / len(scored):.2%})")
            ours = scored["cl_pruned"].to_numpy()
            inter = int((ours & cc["issues"]).sum())
            union = int((ours | cc["issues"]).sum())
            say(f"    overlap {inter}, Jaccard {inter / max(union, 1):.3f}")
            say("    Exact equality is not expected: cleanlab derives its thresholds from the")
            say("    data it is handed, while this script derives them from the calibration")
            say("    cohorts, as pre-registered. The difference measures that choice.")
        say("")

    # ---------------- primary verdict ----------------
    head("PRIMARY TEST -- does the estimate agree with the independent reference?")
    if "real_graph" not in results:
        say("  real-graph condition unavailable; primary test not evaluable")
    else:
        r = results["real_graph"]
        say(f"  Confident Learning estimate (CoDA-GAT, mode={PRIMARY_MODE}, pooled over "
            f"{r['n_patients']} patients in 8 cohorts)")
        say(f"      {r['rate']:.2%}  95% CI [{r['ci_lo']:.2%}, {r['ci_hi']:.2%}]")
        say(f"  Independent reference (published kappa, no model output)")
        say(f"      {ref['lo']:.2%} to {ref['hi']:.2%}")
        say("")
        if r["agrees"]:
            say("  VERDICT: AGREES. The 95% CI overlaps the reference range.")
            say("  Two quantities built from disjoint evidence -- one from published reader")
            say("  agreement, one from the model's own confident disagreements -- land in the")
            say("  same region. Unlike the withdrawn 18.3% vs 18.6% comparison, neither was")
            say("  set to match the other, so this is evidence rather than arithmetic.")
        else:
            say("  VERDICT: DISAGREES. The 95% CI does not overlap the reference range.")
            say("  Reported at face value, as pre-registered, with no adjustment to either")
            say("  quantity. Candidate explanations, none tested here: the latent-noise model")
            say("  assumes normal severity and homoscedastic reader noise; Confident Learning")
            say("  assumes the classifier is well calibrated out-of-sample; and either could")
            say("  be wrong without the other being wrong.")
        say("")
        say(f"  {'width of the CI: ':>22s}{(r['ci_hi'] - r['ci_lo']):.2%}  "
            f"(reference width {(ref['hi'] - ref['lo']):.2%})")

    if "empty_graph" in results:
        say("")
        say("SECONDARY (a) -- empty-graph control")
        say("-" * 78)
        e = results["empty_graph"]
        say(f"  {e['rate']:.2%}  95% CI [{e['ci_lo']:.2%}, {e['ci_hi']:.2%}]  "
            f"-> {'agrees' if e['agrees'] else 'disagrees'} with the reference")
        if "real_graph" in results:
            d = results["real_graph"]["rate"] - e["rate"]
            say(f"  real minus empty: {d:+.2%}. Both are estimates of the SAME label noise,")
            say("  so a large gap would indicate the estimate is tracking the model rather")
            say("  than the labels. They are shown side by side without a claim that either")
            say("  is the correct one.")

    # ---------------- comparability table ----------------
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        head("FLAG QUALITY -- same metrics as flagging_v2, computed by v2's own code")
        say(f"  {'variant':34s} {'n':>5s} {'rate':>6s} {'OR':>7s} {'OR 95% CI':>15s} "
            f"{'2-stage':>8s} {'low/high':>10s}")
        for _, r in summary.iterrows():
            ci = (f"[{np.exp(r['log_or_ci_lo']):.2f},{np.exp(r['log_or_ci_hi']):.2f}]"
                  if np.isfinite(r["log_or_ci_lo"]) else "-")
            orr = f"{r['or_f1f2']:.3f}" if np.isfinite(r["or_f1f2"]) else "-"
            ts = f"{r['two_stage_share']:.1%}" if np.isfinite(r["two_stage_share"]) else "-"
            say(f"  {r['variant']:34s} {r['n_flagged']:>5d} {r['flag_rate']:>6.1%} "
                f"{orr:>7s} {ci:>15s} {ts:>8s} "
                f"{f'{int(r.dir_recorded_too_low)}/{int(r.dir_recorded_too_high)}':>10s}")
        say("")
        say("  Stage distribution of flagged patients:")
        say(f"  {'variant':34s} " + " ".join(f"{s:>6s}" for s in ALL_STAGES))
        for _, r in summary.iterrows():
            say(f"  {r['variant']:34s} " + " ".join(f"{int(r[f'n_{s}']):>6d}" for s in ALL_STAGES))
        say("")
        say("  Directly comparable with flagging_v2/variant_summary.csv: the odds ratio,")
        say("  two-stage share and direction split are produced by the imported v2 functions,")
        say("  so any difference is a difference in the FLAGS, not in the scoring.")

    cohort_df = pd.concat(cohort_tables, ignore_index=True) if cohort_tables else pd.DataFrame()
    if not cohort_df.empty:
        head("SECONDARY (c) -- PER-COHORT (n = 8; descriptive, underpowered)")
        for name, g in cohort_df.groupby("variant", sort=False):
            say(f"  {name}")
            say(f"    {'cohort':<12s} {'n':>5s} {'flags':>6s} {'rate':>7s} {'OR':>8s}")
            for _, r in g.sort_values("fold").iterrows():
                orr = f"{r['or_f1f2']:.3f}" if np.isfinite(r["or_f1f2"]) else "n/a"
                say(f"    {r['test_cohort']:<12s} {int(r['n_patients']):>5d} "
                    f"{int(r['n_flagged']):>6d} {r['flag_rate']:>7.2%} {orr:>8s}")
            say("")

    # ---------------- outputs ----------------
    rate_rows = []
    for cond, r in results.items():
        rr = {"condition": cond, "n_patients": r["n_patients"],
              "cl_noise_rate": r["rate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
              "boot_mean": r["boot_mean"], "n_offdiag": r["n_offdiag"],
              "n_pruned": r["n_pruned"], "pruned_rate": r["rate_pruned"],
              "reference_lo": ref["lo"], "reference_hi": ref["hi"],
              "agrees_with_reference": r["agrees"]}
        rr.update({f"Q_{i}{j}": r["Q"][i, j] for i in range(N_CLASSES) for j in range(N_CLASSES)})
        rr.update({f"C_{i}{j}": r["C"][i, j] for i in range(N_CLASSES) for j in range(N_CLASSES)})
        d = r.get("direction", {})
        rr.update({f"dir_{k}": v for k, v in d.items()})
        cc = r.get("cleanlab", {})
        rr["cleanlab_available"] = cc.get("available", False)
        rr["cleanlab_n_issues"] = cc.get("n_issues", np.nan)
        rate_rows.append(rr)
    pd.DataFrame(rate_rows).to_csv(OUT_DIR / "noise_rate_estimates.csv", index=False)

    for cond in results:
        d, _ = frames[CONDITION_LABEL[cond]]
        d.to_csv(OUT_DIR / f"per_patient_{cond}.csv", index=False)
        d.loc[d["cl_pruned"], ["fold", "test_cohort", "sample_id", "stage", "y_true",
                                "p_f2plus", "confident_class"]].to_csv(
            OUT_DIR / f"flagged_patients_{cond}.csv", index=False)
    if not summary.empty:
        summary.to_csv(OUT_DIR / "variant_summary.csv", index=False)
    if not cohort_df.empty:
        cohort_df.to_csv(OUT_DIR / "per_cohort_values.csv", index=False)
    with open(OUT_DIR / "reference_range.json", "w") as f:
        json.dump(ref, f, indent=2)

    if results:
        plot_rate_vs_reference(results, ref, OUT_DIR / "noise_rate_vs_reference.png",
                                OUT_DIR / "noise_rate_vs_reference.pdf")
        plot_stage_distributions(frames, OUT_DIR / "stage_distribution_cl.png",
                                  OUT_DIR / "stage_distribution_cl.pdf")

    head("FILES WRITTEN")
    for p in sorted(OUT_DIR.glob("*")):
        if p.is_file():
            say(f"  flagging_v3/{p.name}")
    (OUT_DIR / "summary.txt").write_text("\n".join(LINES) + "\n")
    log(f"wrote {(OUT_DIR / 'summary.txt').relative_to(REPO)}")


if __name__ == "__main__":
    main()
