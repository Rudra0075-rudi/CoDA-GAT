"""
CoDA-GAT — flagging v4: ensemble stability, a planted-error ground truth, and
direction bias.

The previous rounds established WHAT the flags look like. None of them
established whether the flagging procedure actually works, because until now
there has been no case where the right answer was known. This adds three
things:

  TASK 1  ENSEMBLE + SPLIT-HALF STABILITY. Flag from all 15 CoDA-GAT runs
          (5 seeds x 3 adversary modes) instead of 5. Then measure stability
          honestly: split the runs in half, flag independently from each half
          at the same count, and see how much the two agree. A rule that
          cannot reproduce itself across halves of its own ensemble is not
          measuring a property of the patient.

  TASK 2  PLANTED-ERROR TEST. Every earlier result was an association: flags
          land at F1/F2, which is where label error is plausible. That is not
          the same as showing the flags FIND label errors. Here errors are
          planted -- boundary labels are flipped in known places -- so recall
          and precision are measurable against a ground truth we control.
          This is the first test in the project that can fail outright.

  TASK 3  DIRECTION BIAS. flagging_v3's direction test asked whether flags are
          symmetric. If they are not, the cause could be genuine asymmetric
          label noise, or simply a model whose operating point sits off the
          cohort's prevalence. Those are distinguished here by prior-correcting
          the threshold on the calibration cohort and re-running the test.

DISCIPLINE
  - The ordinal stage F0-F4 is used ONLY to evaluate flags and to CHOOSE WHICH
    LABELS TO FLIP in Task 2. It never enters a flagging rule. Planting is an
    evaluation construct: it changes the BINARY label a rule sees, never gives
    a rule access to the stage. Enforced by assert_no_stage_leak().
  - All thresholds come from calibration cohorts only.
  - Inference only. No model is loaded and nothing is retrained: everything
    runs on saved probabilities and the flagging_v2 MC-dropout cache.
  - Pre-registered in flagging_v4/PREREG.txt before any result is computed.

Read-only on every results directory; writes only flagging_v4/.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import binomtest

import flagging_v2 as v2
import flagging_v3 as v3
from flagging_v2 import (
    ALL_STAGES, ALLOWED_FLAG_INPUTS, assert_no_stage_leak, describe_variant,
    per_cohort_table, fisher_or, boot_indices, rank_to_count,
    UNCERTAINTY_PERCENTILE, DISAGREEMENT_THRESHOLD, SEEDS, MODES,
    fold_index, fold_dirs, OKABE, CODAGAT, NEUTRAL,
)
from flagging_v3 import (
    per_class_thresholds, confident_class, confident_joint, calibrate_joint,
    noise_rate_from_joint, prune_by_noise_rate, split_labels, load_reference_range,
)

REPO = Path(__file__).resolve().parent.parent
V2_CACHE = REPO / "flagging_v2" / "cache"
OUT_DIR = REPO / "flagging_v4"

CONDITIONS = ["real_graph", "empty_graph", "shuffled_graph"]
CONDITION_LABEL = {"real_graph": "CoDA-GAT (STRING graph)",
                   "empty_graph": "empty-graph control",
                   "shuffled_graph": "shuffled-STRING control"}
CONDITION_COLOR = {"real_graph": CODAGAT, "empty_graph": NEUTRAL, "shuffled_graph": OKABE[0]}
PRIMARY_MODE = "off"

# ---- Task 1 ----
N_SPLITS = 200
SPLIT_SIZES_ENSEMBLE = (8, 7)     # 15 runs
SPLIT_SIZES_SINGLE = (3, 2)       # 5 runs, mode=off only
FLAG_COUNTS = [52, 192]           # v2's baseline count and its 18% operating point

# ---- Task 2 ----
PLANT_RATES = [0.05, 0.10, 0.15]
N_REPEATS = 100
PLANT_SEED_BASE = 20260924
PLANT_UP_STAGE = "F1"             # recorded F1 -> F2+     (label 0 -> 1)
PLANT_DOWN_STAGE = "F2"           # recorded F2 -> not-F2+ (label 1 -> 0)

# ---- shared ----
N_BOOT = 2000
BOOT_SEED = 20260925
N_CLASSES = 2

LINES = []


def log(msg):
    print(f"[flag-v4] {msg}", flush=True)


def say(line=""):
    LINES.append(str(line))
    print(str(line), flush=True)


def head(title):
    say("=" * 78)
    say(title)
    say("=" * 78)


def pct_ci(values, conf=95):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v.mean()) if v.size else np.nan, np.nan, np.nan)
    lo = (100 - conf) / 2
    return float(v.mean()), float(np.percentile(v, lo)), float(np.percentile(v, 100 - lo))


def fmt(m, lo, hi, p=3, pctfmt=False):
    if not np.isfinite(m):
        return "-"
    f = (lambda x: f"{x:.1%}") if pctfmt else (lambda x: f"{x:.{p}f}")
    if not np.isfinite(lo):
        return f(m)
    return f"{f(m)} [{f(lo)}, {f(hi)}]"


# ==========================================================================
# PRE-REGISTRATION
# ==========================================================================
PREREG = """CoDA-GAT flagging v4 -- PRE-REGISTRATION
written: {stamp}
script:  scripts/flagging_v4.py

Written BEFORE any result in this analysis is computed, and never modified
afterwards. If it already exists the script leaves it alone.

================================================================================
TASK 1 -- ENSEMBLE FLAGS AND SPLIT-HALF STABILITY
================================================================================
ENSEMBLE. Average p(F2+) and MC-dropout uncertainty over ALL 15 CoDA-GAT runs
(5 seeds x 3 adversary modes: {modes}). Calibration uncertainty is pooled over
the same 15 runs, and the threshold is its {pct}th percentile, per fold.

RULES APPLIED to the ensemble:
  (a) flagging_v2 V1: disagreement > {dis} AND uncertainty <= calibration p{pct}
  (b) flagging_v3 Confident Learning, thresholds from the calibration cohort

STABILITY BY SPLIT-HALF. {n_splits} random partitions of the 15 runs into two
disjoint groups of sizes {se}. Each group produces its own ensemble average,
its own calibration threshold, and its own flag set at the SAME flag count
(ranked by the flagging_v2 rule: disagreement > {dis}, then lowest uncertainty
first). Statistic: Jaccard overlap between the two groups' flag sets. Reported
as mean with a 95% percentile interval over the {n_splits} splits.

Flag counts, both pre-specified: {counts}.

COMPARISON. The same split-half procedure on the single-mode (off) pool of
{n_seeds} runs, split {ss}.

CONFOUND CONTROL, declared now because the comparison above is not like for
like: {se} halves contain more runs than {ss} halves, so the 15-run ensemble
would look more stable purely from averaging more runs, regardless of whether
mode diversity helps. A size-matched control is therefore also run: the 15-run
pool split into disjoint groups of sizes {ss}, identical in size to the
single-mode split. Any stability difference that survives THAT comparison is
attributable to diversity rather than to count.

ALSO REPORTED for the ensemble flags: F1/F2 boundary enrichment odds ratio,
two-stage share (F0 or F3), and direction split -- computed by the imported
flagging_v2 functions so the numbers are comparable with v2 and v3.

DECISION RULE: the ensemble is MORE STABLE than single-mode iff its mean
split-half Jaccard exceeds the single-mode value AND the two 95% intervals do
not overlap, in the size-matched comparison. Otherwise: "not more stable".

================================================================================
TASK 2 -- PLANTED-ERROR TEST
================================================================================
The first test in this project with a known ground truth. Models are UNCHANGED;
only held-out test labels are altered.

PLANTING. Flip binary F2+ labels of boundary patients in equal numbers:
    recorded {up_stage} -> F2+        (binary label 0 -> 1)
    recorded {down_stage} -> not-F2+  (binary label 1 -> 0)
Total planting rates: {rates} of all test patients, split equally between the
two directions. {n_repeats} repeats per rate, seeds fixed and derived from
{seed_base}, so the identical planted set is given to every condition -- the
real/empty/shuffled comparison is paired by construction.

The ordinal stage selects WHICH labels to flip. That is an evaluation
construct: flipping changes the BINARY label a rule sees, and gives no rule
access to the stage. assert_no_stage_leak() still guards every rule.

APPLIED PER REPEAT:
  (a) the flagging_v2 rule at a matched flag count, identical across the three
      conditions. Two pre-specified counts:
        N = number of planted errors at that rate. Note in advance: with this
            count precision and recall are arithmetically EQUAL, so it measures
            ranking quality only, and will be reported as one number, not two.
        N = {fixed_n} (the v2 18% operating point), where precision and recall
            differ and are separately informative. This is the PRIMARY setting.
  (b) Confident Learning with calibration-derived thresholds. Calibration
      labels are NEVER flipped, so the thresholds are fixed across repeats;
      only the confident joint and its calibration change.

METRICS
  recall     = share of planted errors that are flagged
  precision  = share of flags that are planted errors
  slope      = increase in CL's estimated noise rate per unit planted rate,
               by OLS of CL rate on planted rate over {{0, {rates}}} within each
               repeat, giving {n_repeats} slopes; reported as mean with a 95%
               percentile interval. A perfect detector with additive background
               gives slope 1; slope 0 means CL is blind to planted error.
  background = flag rate and CL noise rate at 0% planting.

PAIRED DIFFERENCES: real minus empty for recall and precision, computed within
repeat (same planted labels both arms), reported as mean with a 95% percentile
interval over repeats. Shuffled-STRING is reported alongside as the control
that separates connectivity from biology.

DECISION RULE: the flagging rule DETECTS planted label error iff recall at the
primary setting exceeds the background flag rate with non-overlapping 95%
intervals. The graph HELPS iff the paired real-minus-empty difference in recall
has a 95% interval strictly above zero.

================================================================================
TASK 3 -- DIRECTION BIAS
================================================================================
DESCRIPTIVE, per cohort: mean predicted p(F2+); predicted-positive share at
threshold 0.5; recorded F2+ share. A gap between the second and third is an
operating-point offset, not necessarily label noise.

PRIOR CORRECTION. Per fold, choose tau so the CALIBRATION cohort's
predicted-positive share equals its recorded F2+ share (tau is the
(1 - recorded positive rate) quantile of the calibration probabilities). Apply
that fold's tau to its test cohort. To feed this into Confident Learning, which
takes probabilities rather than a threshold, apply the equivalent monotone
prior shift
        p' = w*p / (w*p + (1-p)),   w = (1 - tau) / tau
which maps tau to exactly 0.5 and preserves ranking. Per-class CL thresholds
are recomputed on the prior-shifted CALIBRATION probabilities.

TEST: the flagging_v3 direction test -- exact two-sided binomial against 0.5 on
the share of flags whose recorded label is "too high" -- before and after prior
correction, pooled and per cohort with exact binomial intervals.

DECISION RULE: the direction bias is an OPERATING-POINT ARTEFACT iff the
pooled test is significant before correction and non-significant after. It is
GENUINE ASYMMETRIC NOISE iff it remains significant after correction. If it is
non-significant both before and after, there is no bias to explain.

================================================================================
HARD RULES
================================================================================
  1. The ordinal stage is used ONLY to evaluate flags and to choose which
     labels to flip in Task 2. It NEVER creates a flag. Whitelist enforced by
     assert_no_stage_leak():
       {allowed}
  2. Thresholds come from calibration cohorts only.
  3. Every result is reported at face value -- including if the ensemble is NOT
     more stable, if recall is no better than background, or if the direction
     bias persists after correction. No test is re-run with different settings
     after its result is seen.
  4. Inference only. No retraining; no model is loaded.
"""


def write_prereg():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "PREREG.txt"
    if path.exists():
        log(f"PREREG.txt already exists ({path.stat().st_size} bytes) -- left untouched.")
        return path
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    path.write_text(PREREG.format(
        stamp=stamp, modes=MODES, pct=UNCERTAINTY_PERCENTILE, dis=DISAGREEMENT_THRESHOLD,
        n_splits=N_SPLITS, se=SPLIT_SIZES_ENSEMBLE, ss=SPLIT_SIZES_SINGLE,
        counts=FLAG_COUNTS, n_seeds=len(SEEDS),
        rates=", ".join(f"{r:.0%}" for r in PLANT_RATES), n_repeats=N_REPEATS,
        seed_base=PLANT_SEED_BASE, up_stage=PLANT_UP_STAGE, down_stage=PLANT_DOWN_STAGE,
        fixed_n=FLAG_COUNTS[-1],
        allowed=", ".join(sorted(ALLOWED_FLAG_INPUTS)),
    ))
    log(f"wrote {path.relative_to(REPO)} at {stamp}")
    return path


# ==========================================================================
# loading saved per-run outputs (no model is constructed)
# ==========================================================================
def load_run(condition, mode, fold_dir, seed):
    """One run's saved outputs. p_test/u_test/u_cal come from the flagging_v2
    MC-dropout cache; calibration PROBABILITIES come from the checkpoint
    directory, since the cache stored only uncertainty."""
    fid = fold_index(fold_dir.name)
    cache_mode = mode if condition == "real_graph" else "-"
    c = V2_CACHE / f"{condition}_{cache_mode}_fold{fid}_seed{seed}.npz"
    if not c.exists():
        return None
    z = np.load(c, allow_pickle=True)
    p1 = np.asarray(z["p_test"], dtype=float)
    rd = v2.run_dir_for(condition, mode, fold_dir.name, seed)
    pc = rd / "probs_calibration.npy"
    if not pc.exists():
        return None
    p_cal = np.load(pc)
    if p_cal.ndim == 1:
        p_cal = np.column_stack([1 - p_cal, p_cal])
    return {"p_test1": p1, "u_test": np.asarray(z["u_test"], dtype=float),
            "u_cal": np.asarray(z["u_cal"], dtype=float), "p_cal": p_cal}


def run_keys(condition):
    """The (mode, seed) pairs that exist for a condition. Only real_graph has
    the three adversary modes; the controls were trained without an adversary."""
    modes = MODES if condition == "real_graph" else ["-"]
    return [(m, s) for m in modes for s in SEEDS]


def build_pool(condition, folds):
    """All runs for a condition, aligned per fold, plus the evaluation metadata.

    Nothing here is averaged yet -- Task 1 needs to average over arbitrary
    SUBSETS of runs, so the per-run arrays are kept and combined on demand.
    """
    per_fold, meta_rows, missing = {}, [], []
    for fold_dir in folds:
        fid = fold_index(fold_dir.name)
        y_test, test_ids, meta = split_labels(fold_dir, "test")
        y_cal, _, _ = split_labels(fold_dir, "calibration")
        runs = {}
        for (mode, seed) in run_keys(condition):
            r = load_run(condition, mode, fold_dir, seed)
            if r is None:
                missing.append(f"{condition}/{mode}/fold{fid}/seed{seed}")
                continue
            if len(r["p_test1"]) != len(y_test) or len(r["p_cal"]) != len(y_cal):
                missing.append(f"{condition}/{mode}/fold{fid}/seed{seed} (length mismatch)")
                continue
            runs[(mode, seed)] = r
        if not runs:
            continue
        per_fold[fid] = {"runs": runs, "y_test": y_test, "y_cal": y_cal,
                          "sample_ids": test_ids}
        meta_rows.append(pd.DataFrame({
            "fold": fid, "test_cohort": fold_dir.name.split("_", 1)[1],
            "sample_id": test_ids, "y_true": y_test, "stage": meta["stage"].to_numpy(),
        }))
    meta = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame()
    return per_fold, meta, missing


def ensemble_frame(per_fold, meta, keys):
    """Average p(F2+) and MC-dropout uncertainty over a SUBSET of runs, and
    derive that subset's own calibration-p40 threshold per fold."""
    p_all, u_all, p_cal_all, thr = [], [], {}, {}
    for fid in sorted(per_fold):
        d = per_fold[fid]
        use = [d["runs"][k] for k in keys if k in d["runs"]]
        if not use:
            return None, None, None
        p_all.append(np.mean([r["p_test1"] for r in use], axis=0))
        u_all.append(np.mean([r["u_test"] for r in use], axis=0))
        p_cal_all[fid] = np.mean([r["p_cal"] for r in use], axis=0)
        thr[fid] = float(np.percentile(np.concatenate([r["u_cal"] for r in use]),
                                        UNCERTAINTY_PERCENTILE))
    df = meta.copy()
    df["p_f2plus"] = np.concatenate(p_all)
    df["uncertainty"] = np.concatenate(u_all)
    df["p_class1"] = df["p_f2plus"]
    df["p_class0"] = 1.0 - df["p_f2plus"]
    df["disagreement"] = np.abs(df["p_f2plus"] - df["y_true"])
    return df, thr, p_cal_all


def v1_flags(df, thr):
    """flagging_v2's V1 rule, applied to whatever ensemble average is passed in."""
    assert_no_stage_leak("v1_ensemble", ["disagreement", "uncertainty", "fold"])
    f = np.zeros(len(df), dtype=bool)
    for fid, t in thr.items():
        m = (df["fold"] == fid).to_numpy()
        f[m] = ((df.loc[m, "disagreement"].to_numpy() > DISAGREEMENT_THRESHOLD)
                & (df.loc[m, "uncertainty"].to_numpy() <= t))
    return f


def cl_on_frame(df, p_cal_by_fold, y_cal_by_fold):
    """Confident Learning on a frame, thresholds from calibration only."""
    assert_no_stage_leak("cl_ensemble", ["p_f2plus", "y_true", "fold"])
    jc = np.full(len(df), -1, dtype=int)
    thr = {}
    for fid, p_cal in p_cal_by_fold.items():
        t, _ = per_class_thresholds(p_cal, y_cal_by_fold[fid])
        thr[fid] = t
        m = (df["fold"] == fid).to_numpy()
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
    out["cl_pruned"] = prune_by_noise_rate(out, Q)
    out["cl_offdiag"] = (jc >= 0) & (jc != y)
    return out, C, Q, thr


# ==========================================================================
# TASK 1 -- ensemble + split-half stability
# ==========================================================================
def jaccard(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else np.nan


def split_half_stability(per_fold, meta, pool_keys, sizes, n_splits, n_target, seed):
    """Flag independently from two disjoint groups of runs, at the same count,
    and measure agreement. This is the question 'would a different set of runs
    have flagged the same patients?', which no single-ensemble number answers.
    """
    rng = np.random.default_rng(seed)
    keys = list(pool_keys)
    k1, k2 = sizes
    if len(keys) < k1 + k2:
        return None
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(len(keys))
        g1 = [keys[i] for i in perm[:k1]]
        g2 = [keys[i] for i in perm[k1:k1 + k2]]
        f = []
        for g in (g1, g2):
            df, _, _ = ensemble_frame(per_fold, meta, g)
            if df is None:
                break
            fl, _ = rank_to_count(df, n_target)
            f.append(fl)
        if len(f) == 2:
            vals.append(jaccard(f[0], f[1]))
    return np.asarray(vals, dtype=float)


def task1(per_fold_by_cond, meta_by_cond, args):
    head("TASK 1 -- ensemble flags and split-half stability")
    cond = "real_graph"
    per_fold, meta = per_fold_by_cond[cond], meta_by_cond[cond]
    keys15 = [k for k in run_keys(cond) if any(k in d["runs"] for d in per_fold.values())]
    keys_off = [k for k in keys15 if k[0] == PRIMARY_MODE]
    say(f"  runs available: {len(keys15)} of 15 (modes {sorted({k[0] for k in keys15})})")
    say(f"  single-mode pool (off): {len(keys_off)} runs")
    say("")

    ens, thr, p_cal = ensemble_frame(per_fold, meta, keys15)
    y_cal = {fid: d["y_cal"] for fid, d in per_fold.items()}
    f_v1 = v1_flags(ens, thr)
    cl_ens, C, Q, _ = cl_on_frame(ens, p_cal, y_cal)
    f_cl = cl_ens["cl_pruned"].to_numpy()
    rate_cl = noise_rate_from_joint(Q)

    say("  ENSEMBLE FLAGS (all 15 runs averaged)")
    say("-" * 78)
    say(f"    v2 V1 rule:            {int(f_v1.sum()):4d} flags ({f_v1.mean():.2%})")
    say(f"    Confident Learning:    {int(f_cl.sum()):4d} flags ({f_cl.mean():.2%}), "
        f"estimated noise rate {rate_cl:.2%}")
    say(f"    per-fold calibration p{UNCERTAINTY_PERCENTILE} thresholds: "
        f"{ {k: round(v, 5) for k, v in sorted(thr.items())} }")

    rows = []
    for name, fl in (("ensemble V1 (15 runs)", f_v1), ("ensemble CL (15 runs)", f_cl)):
        r = describe_variant(name, ens, fl, None, None, extra={"n_runs": len(keys15)})
        rows.append(r)
        say(f"    {name:26s} OR={r['or_f1f2']:.3f} "
            f"[{np.exp(r['log_or_ci_lo']):.2f},{np.exp(r['log_or_ci_hi']):.2f}]  "
            f"two-stage {r['two_stage_share']:.1%}  "
            f"low/high {int(r['dir_recorded_too_low'])}/{int(r['dir_recorded_too_high'])}"
            if np.isfinite(r["or_f1f2"]) else f"    {name}: too few flags to test")

    say("")
    say("  SPLIT-HALF STABILITY (Jaccard between two disjoint run-groups, same flag count)")
    say("-" * 78)
    stab_rows = []
    for n_target in FLAG_COUNTS:
        say(f"    at {n_target} flags:")
        specs = [
            (f"15-run ensemble, split {SPLIT_SIZES_ENSEMBLE}", keys15, SPLIT_SIZES_ENSEMBLE),
            (f"single-mode off, split {SPLIT_SIZES_SINGLE}", keys_off, SPLIT_SIZES_SINGLE),
            (f"15-run pool, SIZE-MATCHED split {SPLIT_SIZES_SINGLE}", keys15, SPLIT_SIZES_SINGLE),
        ]
        got = {}
        for label, keys, sizes in specs:
            v = split_half_stability(per_fold, meta, keys, sizes, args.n_splits,
                                      n_target, BOOT_SEED + n_target)
            if v is None or v.size == 0:
                say(f"      {label:44s} not evaluable (too few runs)")
                continue
            m, lo, hi = pct_ci(v)
            got[label] = (m, lo, hi)
            say(f"      {label:44s} Jaccard {fmt(m, lo, hi)}")
            stab_rows.append({"n_target": n_target, "pool": label, "mean_jaccard": m,
                               "ci_lo": lo, "ci_hi": hi, "n_splits": int(v.size)})
        # pre-registered decision, on the SIZE-MATCHED comparison only
        a = got.get(f"15-run pool, SIZE-MATCHED split {SPLIT_SIZES_SINGLE}")
        b = got.get(f"single-mode off, split {SPLIT_SIZES_SINGLE}")
        if a and b:
            better = a[0] > b[0] and a[1] > b[2]
            say(f"      -> size-matched verdict: mode diversity "
                f"{'INCREASES' if better else 'does NOT increase'} stability "
                f"({a[0]:.3f} vs {b[0]:.3f}; intervals "
                f"{'do not overlap' if a[1] > b[2] or b[1] > a[2] else 'overlap'})")
        say("")

    say("  Read the size-matched row, not the first one: an 8/7 split averages more runs")
    say("  than a 3/2 split, so it would look steadier even if mode diversity added")
    say("  nothing. Only the size-matched contrast isolates diversity from count.")
    return pd.DataFrame(rows), pd.DataFrame(stab_rows), ens, thr, p_cal, y_cal


# ==========================================================================
# TASK 2 -- planted-error test
# ==========================================================================
def plant_labels(meta, rate, seed):
    """Flip boundary labels in equal numbers, both directions.

    The stage picks WHICH patients are eligible. The rules downstream see only
    the flipped BINARY label, never the stage, so this creates a ground truth
    without handing any rule the answer.
    """
    rng = np.random.default_rng(seed)
    n = len(meta)
    per_side = int(round(n * rate)) // 2
    up_pool = np.where((meta["stage"] == PLANT_UP_STAGE).to_numpy()
                       & (meta["y_true"].to_numpy() == 0))[0]
    down_pool = np.where((meta["stage"] == PLANT_DOWN_STAGE).to_numpy()
                         & (meta["y_true"].to_numpy() == 1))[0]
    k = min(per_side, len(up_pool), len(down_pool))
    up = rng.choice(up_pool, size=k, replace=False) if k else np.array([], dtype=int)
    down = rng.choice(down_pool, size=k, replace=False) if k else np.array([], dtype=int)
    y = meta["y_true"].to_numpy().copy()
    y[up] = 1
    y[down] = 0
    planted = np.zeros(n, dtype=bool)
    planted[up] = True
    planted[down] = True
    return y, planted, k, per_side


def task2(per_fold_by_cond, meta_by_cond, args):
    head("TASK 2 -- planted-error test (the first test here with a ground truth)")
    say("  Boundary labels are flipped in known places; models are unchanged. Recall and")
    say("  precision are therefore measurable, not inferred from where flags happen to")
    say("  land. The identical planted set is given to every condition, so real vs empty")
    say("  is paired by construction.")
    say("")

    base = {}
    for cond in CONDITIONS:
        if cond not in per_fold_by_cond:
            continue
        per_fold, meta = per_fold_by_cond[cond], meta_by_cond[cond]
        keys = [k for k in run_keys(cond) if any(k in d["runs"] for d in per_fold.values())]
        keys = [k for k in keys if k[0] in (PRIMARY_MODE, "-")]
        df, thr, p_cal = ensemble_frame(per_fold, meta, keys)
        if df is None:
            continue
        base[cond] = {"df": df, "thr": thr, "p_cal": p_cal,
                       "y_cal": {fid: d["y_cal"] for fid, d in per_fold.items()},
                       "cl_thr": {fid: per_class_thresholds(p_cal[fid],
                                                             per_fold[fid]["y_cal"])[0]
                                   for fid in p_cal}}
    if "real_graph" not in base:
        say("  real-graph pool unavailable -- Task 2 not evaluable")
        return pd.DataFrame(), pd.DataFrame()

    meta = meta_by_cond["real_graph"]
    n_fixed = FLAG_COUNTS[-1]
    records = []

    for rate in [0.0] + list(PLANT_RATES):
        n_rep = 1 if rate == 0.0 else args.n_repeats
        for rep in range(n_rep):
            seed = PLANT_SEED_BASE + int(rate * 10000) * 1000 + rep
            y_p, planted, k, want = plant_labels(meta, rate, seed)
            n_planted = int(planted.sum())
            for cond, b in base.items():
                df = b["df"].copy()
                df["y_true"] = y_p
                df["p_class1"] = df["p_f2plus"]
                df["p_class0"] = 1.0 - df["p_f2plus"]
                df["disagreement"] = np.abs(df["p_f2plus"] - y_p)

                # (a) v2 rule at matched flag counts
                for tag, n_t in (("oracle_count", max(n_planted, 1)), ("fixed_192", n_fixed)):
                    fl, n_cand = rank_to_count(df, n_t)
                    nf = int(fl.sum())
                    rec = (planted & fl).sum() / n_planted if n_planted else np.nan
                    prec = (planted & fl).sum() / nf if nf else np.nan
                    records.append({"task": "v2_rule", "setting": tag, "condition": cond,
                                     "rate": rate, "rep": rep, "n_planted": n_planted,
                                     "n_flagged": nf, "recall": rec, "precision": prec,
                                     "flag_rate": nf / len(df)})

                # (b) Confident Learning; calibration labels are never flipped,
                #     so thresholds are fixed and only the joint changes
                jc = np.full(len(df), -1, dtype=int)
                for fid, t in b["cl_thr"].items():
                    m = (df["fold"] == fid).to_numpy()
                    if not m.any():
                        continue
                    p = np.column_stack([df.loc[m, "p_class0"].to_numpy(),
                                         df.loc[m, "p_class1"].to_numpy()])
                    jc[m], _ = confident_class(p, t)
                df["confident_class"] = jc
                C = confident_joint(y_p, jc)
                Q = calibrate_joint(C, y_p)
                fl_cl = prune_by_noise_rate(df, Q)
                nf = int(fl_cl.sum())
                records.append({
                    "task": "confident_learning", "setting": "pruned", "condition": cond,
                    "rate": rate, "rep": rep, "n_planted": n_planted, "n_flagged": nf,
                    "recall": (planted & fl_cl).sum() / n_planted if n_planted else np.nan,
                    "precision": (planted & fl_cl).sum() / nf if nf else np.nan,
                    "flag_rate": nf / len(df), "cl_noise_rate": noise_rate_from_joint(Q)})
        log(f"  planting rate {rate:.0%}: {n_rep} repeats done")

    rec = pd.DataFrame(records)

    # ---- background ----
    say("  BACKGROUND at 0% planting (nothing flipped)")
    say("-" * 78)
    bg = rec[rec["rate"] == 0.0]
    for cond in CONDITIONS:
        s = bg[(bg["condition"] == cond)]
        if s.empty:
            continue
        v2r = s[(s["task"] == "v2_rule") & (s["setting"] == "fixed_192")]
        clr = s[s["task"] == "confident_learning"]
        say(f"    {CONDITION_LABEL[cond]:28s} v2 flag rate "
            f"{v2r['flag_rate'].mean():.2%}   CL noise rate "
            f"{clr['cl_noise_rate'].mean():.2%}   CL flag rate {clr['flag_rate'].mean():.2%}")
    say("")

    # ---- recall / precision ----
    say(f"  RECALL and PRECISION  (primary setting: v2 rule at {n_fixed} flags)")
    say("-" * 78)
    say(f"    {'condition':28s} {'rate':>6s} {'recall':>22s} {'precision':>22s}")
    summ_rows = []
    for task, setting in (("v2_rule", "fixed_192"), ("v2_rule", "oracle_count"),
                           ("confident_learning", "pruned")):
        say(f"    -- {task} / {setting} --")
        for cond in CONDITIONS:
            for rate in PLANT_RATES:
                s = rec[(rec["task"] == task) & (rec["setting"] == setting)
                        & (rec["condition"] == cond) & (rec["rate"] == rate)]
                if s.empty:
                    continue
                rm, rl, rh = pct_ci(s["recall"])
                pm, pl, ph = pct_ci(s["precision"])
                same = setting == "oracle_count"
                say(f"    {CONDITION_LABEL[cond]:28s} {rate:>6.0%} "
                    f"{fmt(rm, rl, rh, pctfmt=True):>22s} "
                    f"{'(= recall)' if same else fmt(pm, pl, ph, pctfmt=True):>22s}")
                summ_rows.append({"task": task, "setting": setting, "condition": cond,
                                   "rate": rate, "recall": rm, "recall_lo": rl,
                                   "recall_hi": rh, "precision": pm, "precision_lo": pl,
                                   "precision_hi": ph, "n_repeats": len(s)})
        say("")
    say(f"    At 'oracle_count' the flag count equals the planted count, so precision and")
    say(f"    recall are arithmetically identical -- declared in advance, reported once.")
    say("")

    # ---- CL slope ----
    say("  CONFIDENT LEARNING SENSITIVITY: slope of estimated noise rate on planted rate")
    say("-" * 78)
    slope_rows = []
    for cond in CONDITIONS:
        s = rec[(rec["task"] == "confident_learning") & (rec["condition"] == cond)]
        if s.empty:
            continue
        base0 = s[s["rate"] == 0.0]["cl_noise_rate"].mean()
        slopes = []
        for rep in range(args.n_repeats):
            pts = [(0.0, base0)]
            for rate in PLANT_RATES:
                g = s[(s["rate"] == rate) & (s["rep"] == rep)]
                if not g.empty:
                    pts.append((rate, float(g["cl_noise_rate"].iloc[0])))
            if len(pts) >= 3:
                x, yv = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                slopes.append(float(np.polyfit(x, yv, 1)[0]))
        m, lo, hi = pct_ci(slopes)
        say(f"    {CONDITION_LABEL[cond]:28s} slope {fmt(m, lo, hi)}   "
            f"(1.0 = every planted error shows up in the estimate; 0 = blind)")
        slope_rows.append({"condition": cond, "slope": m, "ci_lo": lo, "ci_hi": hi,
                            "background_rate": base0, "n_slopes": len(slopes)})
    say("")

    # ---- paired differences ----
    say("  PAIRED DIFFERENCES, real minus control (same planted labels in both arms)")
    say("-" * 78)
    paired_rows = []
    for task, setting in (("v2_rule", "fixed_192"), ("confident_learning", "pruned")):
        for ctrl in ("empty_graph", "shuffled_graph"):
            for rate in PLANT_RATES:
                a = rec[(rec.task == task) & (rec.setting == setting)
                        & (rec.condition == "real_graph") & (rec.rate == rate)]
                b = rec[(rec.task == task) & (rec.setting == setting)
                        & (rec.condition == ctrl) & (rec.rate == rate)]
                if a.empty or b.empty:
                    continue
                j = a.merge(b, on="rep", suffixes=("_a", "_b"))
                dr = pct_ci(j["recall_a"] - j["recall_b"])
                dp = pct_ci(j["precision_a"] - j["precision_b"])
                verdict = ("real BETTER" if dr[1] > 0 else
                           "real WORSE" if dr[2] < 0 else "no difference")
                say(f"    {task:20s} vs {ctrl:15s} @{rate:>4.0%}  "
                    f"drecall {fmt(*dr, p=4):>26s}  -> {verdict}")
                paired_rows.append({"task": task, "setting": setting, "control": ctrl,
                                     "rate": rate, "d_recall": dr[0], "d_recall_lo": dr[1],
                                     "d_recall_hi": dr[2], "d_precision": dp[0],
                                     "d_precision_lo": dp[1], "d_precision_hi": dp[2],
                                     "verdict": verdict})
        say("")

    # ---- pre-registered verdicts ----
    say("  VERDICTS (Task 2)")
    say("-" * 78)
    prim = rec[(rec.task == "v2_rule") & (rec.setting == "fixed_192")
               & (rec.condition == "real_graph")]
    bg_rate = float(bg[(bg.condition == "real_graph") & (bg.task == "v2_rule")
                       & (bg.setting == "fixed_192")]["flag_rate"].mean())
    for rate in PLANT_RATES:
        s = prim[prim["rate"] == rate]
        if s.empty:
            continue
        rm, rl, rh = pct_ci(s["recall"])
        detects = np.isfinite(rl) and rl > bg_rate
        say(f"    @{rate:.0%}: recall {rm:.1%} [{rl:.1%}, {rh:.1%}] vs background flag rate "
            f"{bg_rate:.1%}  -> {'DETECTS planted error' if detects else 'NOT distinguishable from background'}")
    pr = pd.DataFrame(paired_rows)
    if not pr.empty:
        real_better = pr[(pr.control == "empty_graph") & (pr.verdict == "real BETTER")]
        say("")
        if real_better.empty:
            say("    GRAPH: no matched-count setting shows the real graph recovering planted")
            say("    errors better than the empty graph with a CI excluding zero. Reported as")
            say("    no difference, per the pre-registration -- not as a trend.")
        else:
            say(f"    GRAPH: real beats empty at {len(real_better)} setting(s):")
            for _, r in real_better.iterrows():
                say(f"      {r['task']} @{r['rate']:.0%}: drecall {r['d_recall']:+.4f} "
                    f"[{r['d_recall_lo']:+.4f}, {r['d_recall_hi']:+.4f}]")
    return rec, pd.DataFrame(summ_rows + slope_rows + paired_rows)


# ==========================================================================
# TASK 3 -- direction bias
# ==========================================================================
def prior_shift(p1, tau):
    """Monotone prior shift mapping tau to 0.5: p' = w p / (w p + 1 - p),
    w = (1-tau)/tau. Ranking is preserved, so discrimination is untouched and
    only the operating point moves."""
    tau = float(np.clip(tau, 1e-6, 1 - 1e-6))
    w = (1.0 - tau) / tau
    return w * p1 / (w * p1 + (1.0 - p1))


def task3(per_fold_by_cond, meta_by_cond, ens, thr, p_cal, y_cal, args):
    head("TASK 3 -- direction bias: real asymmetry, or a mis-set operating point?")
    per_fold = per_fold_by_cond["real_graph"]

    say("  DESCRIPTIVE, per cohort")
    say("-" * 78)
    say(f"    {'cohort':<12s} {'n':>5s} {'mean p(F2+)':>12s} {'pred +ve @0.5':>14s} "
        f"{'recorded F2+':>13s} {'gap':>8s}")
    desc = []
    for fid in sorted(per_fold):
        m = (ens["fold"] == fid).to_numpy()
        g = ens[m]
        pp = float((g["p_f2plus"] >= 0.5).mean())
        rec_share = float(g["y_true"].mean())
        desc.append({"fold": fid, "test_cohort": g["test_cohort"].iloc[0],
                      "n": int(m.sum()), "mean_p": float(g["p_f2plus"].mean()),
                      "pred_pos_share": pp, "recorded_share": rec_share,
                      "gap": pp - rec_share})
        say(f"    {desc[-1]['test_cohort']:<12s} {desc[-1]['n']:>5d} "
            f"{desc[-1]['mean_p']:>12.3f} {pp:>14.1%} {rec_share:>13.1%} "
            f"{desc[-1]['gap']:>+8.1%}")
    desc_df = pd.DataFrame(desc)
    say(f"    mean |gap| across cohorts: {desc_df['gap'].abs().mean():.1%}")
    say("")

    # ---- prior correction ----
    say("  PRIOR CORRECTION: tau chosen so the CALIBRATION cohort's predicted-positive")
    say("  share equals its recorded F2+ share; applied to the test cohort.")
    say("-" * 78)
    say(f"    {'cohort':<12s} {'tau':>8s} {'cal recorded +ve':>17s} {'test +ve @0.5':>14s} "
        f"{'test +ve @tau':>14s}")
    taus, p_cal_shift = {}, {}
    for fid in sorted(per_fold):
        pc1 = p_cal[fid][:, 1]
        yc = y_cal[fid]
        target = float(yc.mean())
        tau = float(np.quantile(pc1, 1.0 - target)) if 0 < target < 1 else 0.5
        taus[fid] = tau
        m = (ens["fold"] == fid).to_numpy()
        pt = ens.loc[m, "p_f2plus"].to_numpy()
        shifted = prior_shift(pc1, tau)
        p_cal_shift[fid] = np.column_stack([1 - shifted, shifted])
        coh = ens.loc[m, "test_cohort"].iloc[0]
        say(f"    {coh:<12s} {tau:>8.4f} {target:>17.1%} "
            f"{(pt >= 0.5).mean():>14.1%} {(pt >= tau).mean():>14.1%}")

    shifted_test = np.concatenate([
        prior_shift(ens.loc[(ens["fold"] == fid).to_numpy(), "p_f2plus"].to_numpy(), taus[fid])
        for fid in sorted(per_fold)])
    ens_pc = ens.copy()
    ens_pc["p_f2plus"] = shifted_test
    ens_pc["p_class1"] = shifted_test
    ens_pc["p_class0"] = 1.0 - shifted_test
    ens_pc["disagreement"] = np.abs(shifted_test - ens_pc["y_true"])

    say("")
    say("  DIRECTION TEST (exact two-sided binomial vs 0.5, on CL pruned flags)")
    say("-" * 78)
    out_rows = []
    for label, frame, pcal in (("before prior correction", ens, p_cal),
                                ("after prior correction", ens_pc, p_cal_shift)):
        scored, C, Q, _ = cl_on_frame(frame, pcal, y_cal)
        flags = scored["cl_pruned"].to_numpy()
        dt = v3.direction_test(scored, flags)
        say(f"    {label}: CL noise rate {noise_rate_from_joint(Q):.2%}, "
            f"{int(flags.sum())} flags")
        if dt["n"]:
            say(f"      too high {dt['too_high']} / too low {dt['too_low']}   "
                f"share too high {dt['share_too_high']:.3f} "
                f"[{dt['ci_lo']:.3f}, {dt['ci_hi']:.3f}]   p = {dt['p']:.4g}")
        out_rows.append({"stage": label, "n_flags": int(flags.sum()),
                          "cl_noise_rate": noise_rate_from_joint(Q), **dt})

        say(f"      per cohort:")
        say(f"        {'cohort':<12s} {'flags':>6s} {'high':>5s} {'low':>5s} "
            f"{'share high':>22s} {'p':>9s}")
        for fid in sorted(per_fold):
            m = (scored["fold"] == fid).to_numpy()
            sub, fl = scored[m], flags[m]
            d = v3.direction_test(sub, fl)
            coh = sub["test_cohort"].iloc[0]
            if d["n"] == 0:
                say(f"        {coh:<12s} {0:>6d} {'-':>5s} {'-':>5s} {'-':>22s} {'-':>9s}")
                continue
            share_cell = f"{d['share_too_high']:.3f} [{d['ci_lo']:.2f}, {d['ci_hi']:.2f}]"
            say(f"        {coh:<12s} {int(fl.sum()):>6d} {d['too_high']:>5d} "
                f"{d['too_low']:>5d} {share_cell:>22s} {d['p']:>9.3g}")
            out_rows.append({"stage": label, "fold": fid, "test_cohort": coh,
                              "n_flags": int(fl.sum()), **d})
        say("")

    before = next((r for r in out_rows if r["stage"] == "before prior correction"
                   and "fold" not in r), None)
    after = next((r for r in out_rows if r["stage"] == "after prior correction"
                  and "fold" not in r), None)
    say("  VERDICT (Task 3)")
    say("-" * 78)
    if before and after and np.isfinite(before.get("p", np.nan)):
        sig_b, sig_a = before["p"] < 0.05, after["p"] < 0.05
        if sig_b and not sig_a:
            say("    OPERATING-POINT ARTEFACT. The asymmetry is significant before prior")
            say("    correction and not after, so it reflects a decision threshold sitting")
            say("    off the cohort's prevalence rather than asymmetric label noise.")
        elif sig_b and sig_a:
            say("    GENUINE ASYMMETRY. The bias survives prior correction, so it is not")
            say("    explained by the operating point. Recorded labels are more often too")
            say("    high than too low, or the reverse -- see the sign above.")
        elif not sig_b and not sig_a:
            say("    NO BIAS TO EXPLAIN. The direction split is consistent with 0.5 both")
            say("    before and after correction, which is what the symmetric noise model")
            say("    predicts when both scales carry the same F2+ prevalence.")
        else:
            say("    Non-significant before correction and significant after -- an unusual")
            say("    pattern. The correction has moved the operating point far enough to")
            say("    induce an asymmetry; treat the corrected test as the informative one")
            say("    and report both, as done above.")
    else:
        say("    not evaluable (no flags)")
    return desc_df, pd.DataFrame(out_rows), taus


# ==========================================================================
# plots
# ==========================================================================
def plot_task2(rec, path_png, path_pdf):
    if rec.empty:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    prim = rec[(rec.task == "v2_rule") & (rec.setting == "fixed_192")]
    for ax, metric, nice in zip(axes[:2], ("recall", "precision"), ("Recall", "Precision")):
        for cond in CONDITIONS:
            s = prim[prim.condition == cond]
            if s.empty:
                continue
            g = s[s.rate > 0].groupby("rate")[metric]
            m = g.mean()
            lo = g.quantile(0.025)
            hi = g.quantile(0.975)
            ax.errorbar(m.index * 100, m * 100,
                        yerr=[(m - lo) * 100, (hi - m) * 100], marker="o", capsize=3,
                        lw=1.8, color=CONDITION_COLOR[cond], label=CONDITION_LABEL[cond])
        ax.set_xlabel("Planted error rate (% of test patients)")
        ax.set_ylabel(f"{nice} (%)")
        ax.set_title(f"{nice} of planted errors", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5)

    ax = axes[2]
    cl = rec[rec.task == "confident_learning"]
    for cond in CONDITIONS:
        s = cl[cl.condition == cond]
        if s.empty:
            continue
        g = s.groupby("rate")["cl_noise_rate"]
        m = g.mean()
        ax.errorbar(m.index * 100, m * 100,
                    yerr=[(m - g.quantile(0.025)) * 100, (g.quantile(0.975) - m) * 100],
                    marker="o", capsize=3, lw=1.8, color=CONDITION_COLOR[cond],
                    label=CONDITION_LABEL[cond])
    lim = np.array([0, max(PLANT_RATES) * 100])
    base = cl[(cl.rate == 0) & (cl.condition == "real_graph")]["cl_noise_rate"].mean() * 100
    ax.plot(lim, base + lim, ls="--", color="black", lw=1,
            label="slope 1 (perfect sensitivity)")
    ax.set_xlabel("Planted error rate (%)")
    ax.set_ylabel("Confident Learning estimated noise rate (%)")
    ax.set_title("Does the estimate track planted error?", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.suptitle("Planted-error test: can the flagging rules recover errors we put there?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_task1(stab, path_png, path_pdf):
    if stab.empty:
        return False
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    pools = list(dict.fromkeys(stab["pool"]))
    targets = sorted(stab["n_target"].unique())
    w = 0.8 / max(len(pools), 1)
    for k, pool in enumerate(pools):
        s = stab[stab["pool"] == pool].set_index("n_target").reindex(targets)
        x = np.arange(len(targets)) + (k - (len(pools) - 1) / 2) * w
        ax.bar(x, s["mean_jaccard"], w, label=pool, color=OKABE[k % len(OKABE)],
               yerr=[s["mean_jaccard"] - s["ci_lo"], s["ci_hi"] - s["mean_jaccard"]],
               capsize=3, error_kw={"lw": 0.8})
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([f"{t} flags" for t in targets])
    ax.set_ylabel("Split-half Jaccard overlap")
    ax.set_ylim(0, 1)
    ax.set_title("Would a different half of the runs flag the same patients?", fontsize=10.5)
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return True


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="flagging v4: ensemble stability, planted-error ground truth, "
                    "direction bias (inference only, read-only on results)")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--n-repeats", type=int, default=N_REPEATS)
    ap.add_argument("--tasks", nargs="*", type=int, default=[1, 2, 3],
                    choices=[1, 2, 3])
    ap.add_argument("--prereg-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prereg = write_prereg()
    if args.prereg_only:
        log("--prereg-only: stopping before any result is computed")
        return

    LINES.clear()
    head("CoDA-GAT flagging v4")
    say(f"pre-registration: {prereg.relative_to(REPO)}")
    say("Inference only -- no model is constructed, nothing is retrained. Sources:")
    say("  flagging_v2/cache/ (MC-dropout), results_b7/ and graph_utility/ (probabilities).")
    say("")
    ref = load_reference_range()
    say(f"independent noise reference (review_fixes/): {ref['lo']:.2%} to {ref['hi']:.2%}")
    say("")

    folds = fold_dirs(args.folds)
    per_fold_by_cond, meta_by_cond = {}, {}
    for cond in CONDITIONS:
        pf, meta, missing = build_pool(cond, folds)
        if not pf:
            log(f"  {cond}: no usable runs -- skipped")
            continue
        per_fold_by_cond[cond], meta_by_cond[cond] = pf, meta
        n_runs = sum(len(d["runs"]) for d in pf.values())
        say(f"  {CONDITION_LABEL[cond]:28s} {len(pf)} folds, {n_runs} run-folds loaded"
            + (f", {len(missing)} missing" if missing else ""))
        for m in missing[:5]:
            say(f"      missing: {m}")
    say("")

    if "real_graph" not in per_fold_by_cond:
        say("real-graph pool unavailable -- nothing to do")
        (OUT_DIR / "summary.txt").write_text("\n".join(LINES) + "\n")
        return

    ens = thr = p_cal = y_cal = None
    if 1 in args.tasks:
        t1_variants, t1_stab, ens, thr, p_cal, y_cal = task1(
            per_fold_by_cond, meta_by_cond, args)
        if not t1_variants.empty:
            t1_variants.to_csv(OUT_DIR / "task1_ensemble_variants.csv", index=False)
        if not t1_stab.empty:
            t1_stab.to_csv(OUT_DIR / "task1_split_half_stability.csv", index=False)
            plot_task1(t1_stab, OUT_DIR / "task1_split_half_stability.png",
                       OUT_DIR / "task1_split_half_stability.pdf")
        say("")

    if 2 in args.tasks:
        t2_raw, t2_summ = task2(per_fold_by_cond, meta_by_cond, args)
        if not t2_raw.empty:
            t2_raw.to_csv(OUT_DIR / "task2_planted_raw.csv", index=False)
            plot_task2(t2_raw, OUT_DIR / "task2_planted_error.png",
                       OUT_DIR / "task2_planted_error.pdf")
        if not t2_summ.empty:
            t2_summ.to_csv(OUT_DIR / "task2_planted_summary.csv", index=False)
        say("")

    if 3 in args.tasks:
        if ens is None:
            per_fold = per_fold_by_cond["real_graph"]
            keys = [k for k in run_keys("real_graph")
                    if any(k in d["runs"] for d in per_fold.values())]
            ens, thr, p_cal = ensemble_frame(per_fold, meta_by_cond["real_graph"], keys)
            y_cal = {fid: d["y_cal"] for fid, d in per_fold.items()}
        t3_desc, t3_dir, taus = task3(per_fold_by_cond, meta_by_cond, ens, thr,
                                       p_cal, y_cal, args)
        t3_desc.to_csv(OUT_DIR / "task3_per_cohort_descriptive.csv", index=False)
        t3_dir.to_csv(OUT_DIR / "task3_direction_tests.csv", index=False)
        with open(OUT_DIR / "task3_prior_thresholds.json", "w") as f:
            json.dump({str(k): v for k, v in taus.items()}, f, indent=2)

    head("FILES WRITTEN")
    for p in sorted(OUT_DIR.glob("*")):
        if p.is_file():
            say(f"  flagging_v4/{p.name}")
    (OUT_DIR / "summary.txt").write_text("\n".join(LINES) + "\n")
    log(f"wrote {(OUT_DIR / 'summary.txt').relative_to(REPO)}")


if __name__ == "__main__":
    main()
