"""
CoDA-GAT — label-error flagging, version 2: better rules, and does the graph help?

TWO QUESTIONS
  1. Can the Test A flagging rule be improved? Test A flagged 52 of 1,064
     held-out patients (4.9%) with an F1/F2 enrichment odds ratio of 2.73.
     4.9% is below the 15-25% band that was pre-declared as "useful", so the
     flagged set is small and possibly over-strict.
  2. Does the gene graph produce BETTER flags than graph-free models? Every
     accuracy analysis so far says the graph contributes nothing. Flag quality
     is a different axis and has not been tested.

METHOD DISCIPLINE, enforced in code rather than promised in prose
  - The recorded ordinal stage (F0-F4) is used ONLY to evaluate flags. It never
    enters a flagging rule. assert_no_stage_leak() checks this on every variant
    by confirming each rule's inputs are drawn from a whitelist of model
    outputs plus the recorded BINARY label.
  - The MC-dropout uncertainty threshold is taken from each fold's CALIBRATION
    cohort only, never from the test cohort whose labels are being questioned.
  - Every variant is pre-registered in flagging_v2/PREREG.txt, written with a
    timestamp BEFORE any result is computed, and every variant is reported
    whether it helps or not.

Inference only. No retraining. torch.load(weights_only=False) on existing
checkpoints. Read-only on every results directory; writes only flagging_v2/.

REPRODUCIBILITY NOTE, stated up front because it bears on the baseline check:
Test A computed its calibration MC-dropout WITHOUT seeding the torch RNG, so
its 52-flag result is not bit-reproducible -- not by this script and not by
re-running Test A itself. This script seeds every MC-dropout pass by
(condition, mode, fold, seed), so its own numbers ARE reproducible, and the
baseline variant is checked against Test A within a stated tolerance rather
than for exact equality.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact
from torch_geometric.loader import DataLoader

from train_fold0_b1 import (
    CoDAGAT, get_device, set_seed, N_FEATURES, N_NODES, BATCH_SIZE, SHARED_DIR,
)
from run_full_experiment_b7 import (
    mc_dropout_probs, load_edge_index_generic, load_fold_info, build_model,
)
from ablations_b9 import load_split_for_ablation, build_edge_index_and_perm, ABLATIONS

REPO = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO / "folds"
RESULTS_B7 = REPO / "results_b7"
GUTIL_CKPT = REPO / "graph_utility" / "control_checkpoints"
BASE_PSD = REPO / "baselines" / "per_sample_predictions"
TEST_A = REPO / "test_a_adjacency"
OUT_DIR = REPO / "flagging_v2"
CACHE = OUT_DIR / "cache"

# ---- pre-specified constants (mirrored into PREREG.txt) ----
SEEDS = [42, 43, 44, 45, 46]
MODES = ["off", "marginal", "conditional"]
PRIMARY_MODE = "off"
UNCERTAINTY_PERCENTILE = 40      # Test A's EXECUTED threshold (its summary.txt says p40)
DISAGREEMENT_THRESHOLD = 0.5
CONF_HI, CONF_LO = 0.8, 0.2      # two-model agreement confidence band
STABLE_MIN_RUNS = 12             # of 15 (5 seeds x 3 modes)
STABILITY_FRACTION = 0.80        # "flagged in >= 80% of runs" = 12/15
N_MC_PASSES = 30
MATCHED_COUNTS = [52, "18%"]     # Test A's count, and the pre-declared 18% target
N_BOOT = 2000
BOOT_SEED = 20260922
HALDANE = 0.5                    # continuity correction, CIs and paired contrasts only

ADJACENT_STAGES = {"F1", "F2"}       # one stage from the F2+ boundary
TWO_STAGE_STAGES = {"F0", "F3"}      # >=2 stages from the boundary
ALL_STAGES = ["F0", "F1", "F2", "F3", "F4"]

GRAPH_CONDITIONS = ["real_graph", "empty_graph", "shuffled_graph"]
TABULAR_MODELS = [
    "panel72_combined__catboost", "panel72_combined__logreg",
    "panel72_full__catboost", "panel72_full__logreg",
]
PRIMARY_TABULAR = ["panel72_combined__catboost", "panel72_combined__logreg"]

# the ONLY fields any flagging rule may read. Enforced, not documented.
ALLOWED_FLAG_INPUTS = {
    "p_f2plus", "uncertainty", "y_true", "disagreement", "fold", "sample_id",
    "p_tabular", "n_runs_flagged",
}

OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
CODAGAT, NEUTRAL = "#D55E00", "#9AA0A6"

LINES = []


def log(msg):
    print(f"[flag-v2] {msg}", flush=True)


def say(line=""):
    LINES.append(str(line))
    print(str(line), flush=True)


def head(title):
    say("=" * 78)
    say(title)
    say("=" * 78)


# ==========================================================================
# PRE-REGISTRATION -- written before anything is computed
# ==========================================================================
PREREG = """CoDA-GAT flagging v2 -- PRE-REGISTRATION
written: {stamp}
script:  scripts/flagging_v2.py
git-free timestamp; this file is written BEFORE any result in this analysis is
computed, and is not modified afterwards.

--------------------------------------------------------------------------------
BACKGROUND (already known, not re-tested here)
--------------------------------------------------------------------------------
Test A flagged 52/1064 held-out patients (4.9%) using: disagreement > 0.5 AND
MC-dropout uncertainty <= the 40th percentile of that fold's calibration
cohort's uncertainty. Enrichment at F1/F2 vs F0/F3/F4: one-sided Fisher
OR = 2.731, p = 0.0005. The 4.9% rate is below the 15-25% band declared useful.

--------------------------------------------------------------------------------
PRIMARY COMPARISON
--------------------------------------------------------------------------------
Real-graph CoDA-GAT flags vs empty-graph CoDA-GAT flags, at MATCHED flag counts
of {counts}. Matching is by construction, not by luck: see RANKING below.

PRIMARY METRIC
  One-sided Fisher exact odds ratio for flags falling at F1/F2 rather than
  F0/F3/F4. H1: flagged patients are more likely to sit adjacent to the F2+
  boundary. Reported on raw counts.

SECONDARY METRICS
  1. two-stage share  -- share of flags at F0 or F3, i.e. flags that imply a
     >=2-stage annotation error. Lower is better; these are less plausible as
     ordinary reader disagreement.
  2. stability        -- share of a variant's flagged patients that are also
     flagged in >= {stable_frac:.0%} of the 15 CoDA-GAT runs (5 seeds x 3
     adversary modes), i.e. >= {stable_runs} of 15.
  3. direction split  -- "recorded too low" (model says F2+, label says not)
     vs "recorded too high" (model says not F2+, label says F2+).

--------------------------------------------------------------------------------
FLAG VARIANTS (all pre-specified; all will be reported)
--------------------------------------------------------------------------------
V1  baseline
      disagreement > {dis} AND uncertainty <= calibration p{pct}
      Seed-averaged p(F2+) and uncertainty over the {n_seeds} seeds of mode
      '{mode}'. This is Test A's rule, unchanged.

V2  two-model agreement (one variant per tabular partner)
      Flag iff CoDA-GAT AND the tabular model BOTH place the patient in the
      opposite class with confidence beyond {conf_hi}/{conf_lo}:
        label says not-F2+  ->  p_gat > {conf_hi} AND p_tab > {conf_hi}
        label says F2+      ->  p_gat < {conf_lo} AND p_tab < {conf_lo}
      MC-dropout uncertainty is NOT used in this variant.
      Partners, all reported: {tabular}
      Primary partners (declared now, before results): {primary_tabular}

V3  stable
      Flag iff the patient meets V1's rule in >= {stable_runs} of the 15 runs.
      Each run uses its OWN calibration-derived p{pct} threshold.

V4  graph comparison
      V1's rule applied to real_graph, empty_graph and shuffled_graph. Weights
      come from results_b7 (real) and graph_utility/control_checkpoints
      (empty, shuffled) -- the checkpoints graph_utility.py retrained. MC-dropout
      is recomputed by inference; nothing is retrained.

--------------------------------------------------------------------------------
RANKING, so flag counts match exactly
--------------------------------------------------------------------------------
To flag exactly N patients under any condition: restrict to patients with
disagreement > {dis} (the model contradicts the recorded label), then rank by
MC-dropout uncertainty ASCENDING -- most confident contradiction first -- and
take the top N. Ties broken by disagreement descending, then sample_id.
Declared now because it decides the comparison. If a condition has fewer than N
patients with disagreement > {dis}, the shortfall is reported, not padded.

--------------------------------------------------------------------------------
INFERENCE AND UNCERTAINTY
--------------------------------------------------------------------------------
  - 95% CIs by bootstrap over PATIENTS WITHIN COHORT: {n_boot} resamples, each
    cohort resampled with replacement to its own original size, so cohort
    composition is preserved. Flag sets are held FIXED across resamples; the
    CI is on the enrichment measured over a resampled patient population.
  - Odds ratios in CIs and paired contrasts use a Haldane-Anscombe +{haldane}
    correction so an empty cell gives a finite value. Headline ORs and p-values
    are exact Fisher on raw counts, uncorrected.
  - Graph comparison: paired difference in log odds ratio, real minus control,
    computed on the SAME resampled patients for both conditions. Verdict is
    stated as better / equal / worse by whether the CI excludes zero.
  - Per-cohort values reported for all 8 held-out cohorts. n = 8 is
    underpowered; cross-cohort statements are descriptive.

--------------------------------------------------------------------------------
HARD RULES
--------------------------------------------------------------------------------
  1. The ordinal stage F0-F4 is used ONLY to evaluate flags, NEVER to create
     them. Enforced by assert_no_stage_leak(), which fails the run if any rule
     reads a field outside this whitelist:
       {allowed}
  2. The uncertainty threshold comes from calibration cohorts only.
  3. Every variant above is reported, including ones that do worse. No variant
     is added, dropped or re-tuned after results are seen.
  4. Inference only. No retraining.

--------------------------------------------------------------------------------
WHAT WOULD COUNT AS THE GRAPH HELPING
--------------------------------------------------------------------------------
Declared now to prevent a post-hoc reading: the real graph produces better
flags iff the paired difference in log OR (real minus empty) has a 95% CI
strictly above zero at a matched flag count. A CI containing zero is reported
as "no difference detected", NOT as a trend in either direction.
"""


def write_prereg():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    text = PREREG.format(
        stamp=stamp,
        counts=", ".join(str(c) for c in MATCHED_COUNTS),
        stable_frac=STABILITY_FRACTION, stable_runs=STABLE_MIN_RUNS,
        dis=DISAGREEMENT_THRESHOLD, pct=UNCERTAINTY_PERCENTILE,
        n_seeds=len(SEEDS), mode=PRIMARY_MODE,
        conf_hi=CONF_HI, conf_lo=CONF_LO,
        tabular=", ".join(TABULAR_MODELS),
        primary_tabular=", ".join(PRIMARY_TABULAR),
        n_boot=N_BOOT, haldane=HALDANE,
        allowed=", ".join(sorted(ALLOWED_FLAG_INPUTS)),
    )
    path = OUT_DIR / "PREREG.txt"
    if path.exists():
        log(f"PREREG.txt already exists ({path.stat().st_size} bytes) -- left untouched. "
            f"A pre-registration that gets rewritten after results is not one.")
    else:
        path.write_text(text)
        log(f"wrote {path.relative_to(REPO)} at {stamp}")
    return path


def assert_no_stage_leak(rule_name, fields_used):
    """Fails the run if a flagging rule touches anything outside the whitelist.

    This is the guard that makes the central claim checkable: flags are built
    from model outputs and the recorded BINARY label, so any concentration at
    F1/F2 is a finding about the flags rather than a consequence of having
    looked at the stage while building them.
    """
    leaked = set(fields_used) - ALLOWED_FLAG_INPUTS
    if leaked:
        raise AssertionError(
            f"STAGE LEAK in rule '{rule_name}': reads {sorted(leaked)}, which is outside "
            f"the allowed set {sorted(ALLOWED_FLAG_INPUTS)}. The ordinal stage may only be "
            f"used to EVALUATE flags. Aborting rather than producing a circular result.")


# ==========================================================================
# inference: per-run probabilities and MC-dropout uncertainty
# ==========================================================================
def fold_index(name):
    return int(re.match(r"fold(\d+)_", name).group(1))


def fold_dirs(selected=None):
    ds = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir() and p.name.startswith("fold"))
    return [d for d in ds if selected is None or d.name in selected]


def run_dir_for(condition, mode, fold_dir_name, seed):
    if condition == "real_graph":
        return RESULTS_B7 / fold_dir_name / f"mode_{mode}" / f"seed_{seed}"
    return GUTIL_CKPT / fold_dir_name / condition / f"seed_{seed}"


def mc_seed(condition, mode, fold_id, seed):
    """Deterministic MC-dropout seed. Test A left this to the global RNG, which
    is why its result cannot be reproduced exactly; here it is pinned."""
    key = f"{condition}|{mode}|{fold_id}|{seed}"
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) % 1_000_003
    return 900_000 + h


def build_model_for_checkpoint(condition, mode, ckpt, device):
    """Construct the architecture the checkpoint was actually trained with.

    mode='conditional' is NOT plain CoDAGAT: run_full_experiment_b7 trains it as
    CoDAGATConditional, whose CDAN-style adversary consumes the outer product of
    the 16-dim embedding with the 2-class H1 softmax, so h4.0.weight is
    [32, 32] rather than [32, 16]. Building the base class for those
    checkpoints is a shape mismatch, not a loadable model. b7's own
    build_model() is the single source of truth for that dispatch, so it is
    imported rather than re-derived.

    The graph controls (empty_graph, shuffled_graph) were trained by
    ablations_b9 / graph_utility as plain CoDAGAT with no adversary, which is
    the 'off' architecture.

    Fallback: if a checkpoint still mismatches, the h4.* keys are dropped and
    the rest is loaded with strict=False -- H4 is the cohort adversary and is
    used for neither predictions nor MC-dropout, so discarding it changes no
    output. The assertion makes that narrow: if ANY missing or unexpected key
    does not start with 'h4.', the run fails rather than silently proceeding
    with a partially-initialised encoder.
    """
    build_mode = mode if condition == "real_graph" else "off"
    model = build_model(build_mode, N_FEATURES, ckpt["n_ordinal_classes"],
                        ckpt["n_cohorts"]).to(device)
    state = ckpt["model_state_dict"]
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        pruned = {k: v for k, v in state.items() if not k.startswith("h4.")}
        result = model.load_state_dict(pruned, strict=False)
        offenders = [k for k in list(result.missing_keys) + list(result.unexpected_keys)
                     if not k.startswith("h4.")]
        assert not offenders, (
            f"state_dict mismatch beyond the H4 adversary for {condition}/{mode}: "
            f"{offenders}. Only h4.* may be skipped; anything else means the encoder or a "
            f"prediction head is not loaded and every downstream number would be wrong. "
            f"Original error: {exc}")
        log(f"  {condition}/{mode}: loaded without H4 (adversary unused for prediction "
            f"and MC-dropout); skipped {sorted(k for k in state if k.startswith('h4.'))}")
    model.eval()
    return model


def compute_run_outputs(condition, mode, fold_dir, seed, edge_index, dev_lookup,
                         n_passes, recompute=False):
    """p(F2+) and MC-dropout uncertainty for one (condition, mode, fold, seed),
    on BOTH the calibration and test splits.

    Calibration MC-dropout was never saved by any earlier block, and it is
    exactly what the threshold needs, so it is computed here. Test-split
    quantities are loaded from disk where a previous block saved them
    (results_b7 saved probs and MC-dropout; graph_utility saved probs only) and
    computed otherwise. Cached under flagging_v2/cache/.
    """
    fold_id = fold_index(fold_dir.name)
    tag = f"{condition}_{mode}_fold{fold_id}_seed{seed}"
    cpath = CACHE / f"{tag}.npz"
    if cpath.exists() and not recompute:
        z = np.load(cpath, allow_pickle=True)
        return {k: z[k] for k in z.files}

    rd = run_dir_for(condition, mode, fold_dir.name, seed)
    wpath = rd / "weights.pt"
    if not wpath.exists():
        return None

    device = get_device()
    set_seed(mc_seed(condition, mode, fold_id, seed))

    ckpt = torch.load(wpath, weights_only=False, map_location=device)
    model = build_model_for_checkpoint(condition, mode, ckpt, device)

    if condition == "real_graph":
        ei, node_perm = edge_index, None
    else:
        ei, node_perm = build_edge_index_and_perm(ABLATIONS[condition], edge_index, fold_id)

    info = load_fold_info(fold_dir)
    cohort_to_idx = {c: i for i, c in enumerate(sorted(info["train_cohorts"]))}

    cal_data, cal_ids, _ = load_split_for_ablation(
        fold_dir, "calibration", ei, dev_lookup, cohort_to_idx, node_perm=node_perm)
    cal_loader = DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False)
    cal_mc = mc_dropout_probs(model, cal_loader, device, n_passes=n_passes)
    u_cal = cal_mc.std(axis=0)

    p_saved = rd / "probs_test.npy"
    mc_saved = rd / "mc_dropout_probs_test.npy"
    if p_saved.exists() and mc_saved.exists():
        p_test = np.load(p_saved)[:, 1]
        u_test = np.load(mc_saved).std(axis=0)
        source = "saved"
    else:
        test_data, _, _ = load_split_for_ablation(
            fold_dir, "test", ei, dev_lookup, cohort_to_idx, node_perm=node_perm)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        if p_saved.exists():
            p_test = np.load(p_saved)[:, 1]
        else:
            model.eval()
            with torch.no_grad():
                ps = []
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch, grl_lambd=0.0)
                    ps.append(torch.softmax(out["h1_logits"], dim=1)[:, 1].cpu().numpy())
            p_test = np.concatenate(ps)
        test_mc = mc_dropout_probs(model, test_loader, device, n_passes=n_passes)
        u_test = test_mc.std(axis=0)
        source = "computed"

    out = {"p_test": p_test, "u_test": u_test, "u_cal": u_cal,
           "source": np.array(source), "fold": np.array(fold_id), "seed": np.array(seed)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cpath, **out)
    return out


def build_master(conditions, modes, folds, n_passes, recompute):
    """One tidy frame of per-patient values, plus the per-run detail the stable
    variant needs. Test sample order is identical across conditions because the
    test split is the same, so a single patient index aligns everything."""
    edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")
    labels = pd.read_parquet(SHARED_DIR / "labels.parquet")

    rows, per_run, cal_pools = [], [], {}
    t0 = time.time()
    total = sum(len(modes if c == "real_graph" else ["-"]) * len(SEEDS)
                for c in conditions) * len(folds)
    done = 0

    for fold_dir in folds:
        fold_id = fold_index(fold_dir.name)
        test_ids = pd.read_csv(fold_dir / "test" / "sample_ids.csv")["sample_id"].tolist()
        meta = labels.loc[test_ids]
        y_true = meta["y_T1_significant"].to_numpy().astype(int)
        stage = meta["stage"].to_numpy()
        cohort = fold_dir.name.split("_", 1)[1]

        for condition in conditions:
            cond_modes = modes if condition == "real_graph" else ["-"]
            for mode in cond_modes:
                p_seeds, u_seeds, u_cal_seeds = [], [], []
                for seed in SEEDS:
                    r = compute_run_outputs(condition, mode, fold_dir, seed, edge_index,
                                             dev_lookup, n_passes, recompute)
                    done += 1
                    if r is None:
                        log(f"  MISSING {condition} {mode} fold{fold_id} seed{seed} -- skipped")
                        continue
                    p_seeds.append(r["p_test"])
                    u_seeds.append(r["u_test"])
                    u_cal_seeds.append(r["u_cal"])
                    # per-run flags, for the stable variant
                    thr = float(np.percentile(r["u_cal"], UNCERTAINTY_PERCENTILE))
                    dis = np.abs(r["p_test"] - y_true)
                    flg = (dis > DISAGREEMENT_THRESHOLD) & (r["u_test"] <= thr)
                    per_run.append(pd.DataFrame({
                        "condition": condition, "mode": mode, "seed": seed,
                        "fold": fold_id, "sample_id": test_ids, "run_flag": flg,
                    }))
                    if done % 20 == 0:
                        log(f"  {done}/{total} runs  ({time.time() - t0:.0f}s)")
                if not p_seeds:
                    continue
                cal_pools[(condition, mode, fold_id)] = np.concatenate(u_cal_seeds)
                rows.append(pd.DataFrame({
                    "condition": condition, "mode": mode, "fold": fold_id,
                    "test_cohort": cohort, "sample_id": test_ids,
                    "p_f2plus": np.mean(p_seeds, axis=0),
                    "uncertainty": np.mean(u_seeds, axis=0),
                    "y_true": y_true, "stage": stage,
                }))

    master = pd.concat(rows, ignore_index=True)
    master["disagreement"] = np.abs(master["p_f2plus"] - master["y_true"])
    runs = pd.concat(per_run, ignore_index=True) if per_run else pd.DataFrame()
    return master, runs, cal_pools


def attach_tabular(master):
    """Per-patient tabular probabilities, joined on sample_id. One fit per fold,
    so there is no seed axis to average."""
    out = {}
    for name in TABULAR_MODELS:
        d = BASE_PSD / name
        if not d.is_dir():
            log(f"  tabular model {name} has no per-sample predictions -- variant skipped")
            continue
        parts = [pd.read_csv(f)[["sample_id", "p_test"]] for f in sorted(d.glob("fold*_*.csv"))]
        if not parts:
            continue
        t = pd.concat(parts, ignore_index=True).rename(columns={"p_test": "p_tabular"})
        out[name] = t.drop_duplicates("sample_id").set_index("sample_id")["p_tabular"]
    return out


# ==========================================================================
# flag rules
# ==========================================================================
def rule_baseline(df, cal_pools, condition, mode):
    assert_no_stage_leak("baseline", ["disagreement", "uncertainty", "fold"])
    flags = np.zeros(len(df), dtype=bool)
    thresholds = {}
    for fold_id, g in df.groupby("fold"):
        pool = cal_pools.get((condition, mode, fold_id))
        if pool is None:
            continue
        thr = float(np.percentile(pool, UNCERTAINTY_PERCENTILE))
        thresholds[fold_id] = thr
        m = (df["fold"] == fold_id).to_numpy()
        flags[m] = ((df.loc[m, "disagreement"].to_numpy() > DISAGREEMENT_THRESHOLD)
                    & (df.loc[m, "uncertainty"].to_numpy() <= thr))
    return flags, thresholds


def rule_two_model(df, p_tab):
    assert_no_stage_leak("two_model", ["p_f2plus", "p_tabular", "y_true"])
    pt = df["sample_id"].map(p_tab).to_numpy(dtype=float)
    pg, y = df["p_f2plus"].to_numpy(), df["y_true"].to_numpy()
    both_high = (pg > CONF_HI) & (pt > CONF_HI) & (y == 0)     # recorded too low
    both_low = (pg < CONF_LO) & (pt < CONF_LO) & (y == 1)       # recorded too high
    return (both_high | both_low) & np.isfinite(pt), np.isfinite(pt).sum()


def rule_stable(df, runs, condition, min_runs=STABLE_MIN_RUNS):
    assert_no_stage_leak("stable", ["n_runs_flagged", "sample_id"])
    r = runs[runs["condition"] == condition]
    if r.empty:
        return np.zeros(len(df), dtype=bool), pd.Series(dtype=float), 0
    counts = r.groupby("sample_id")["run_flag"].sum()
    n_runs = r.groupby("sample_id")["run_flag"].size().max()
    c = df["sample_id"].map(counts).fillna(0).to_numpy()
    return c >= min_runs, counts, int(n_runs)


def rank_to_count(df, n_target):
    """Take exactly n_target flags, by the ranking pre-registered above:
    disagreement > 0.5, then most-confident (lowest uncertainty) first."""
    assert_no_stage_leak("matched_count", ["disagreement", "uncertainty", "sample_id"])
    cand = df[df["disagreement"] > DISAGREEMENT_THRESHOLD].copy()
    cand = cand.sort_values(["uncertainty", "disagreement", "sample_id"],
                             ascending=[True, False, True])
    chosen = set(cand.head(n_target)["sample_id"])
    flags = df["sample_id"].isin(chosen).to_numpy()
    return flags, len(cand)


# ==========================================================================
# metrics
# ==========================================================================
def or_counts(flags, adjacent):
    a = int((flags & adjacent).sum())         # flagged, F1/F2
    b = int((flags & ~adjacent).sum())        # flagged, F0/F3/F4
    c = int((~flags & adjacent).sum())
    d = int((~flags & ~adjacent).sum())
    return a, b, c, d


def fisher_or(flags, adjacent):
    a, b, c, d = or_counts(flags, adjacent)
    if (a + b) == 0:
        return np.nan, np.nan, (a, b, c, d)
    orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return float(orr), float(p), (a, b, c, d)


def log_or_haldane(flags, adjacent):
    a, b, c, d = or_counts(flags, adjacent)
    a, b, c, d = a + HALDANE, b + HALDANE, c + HALDANE, d + HALDANE
    return float(np.log((a * d) / (b * c)))


def boot_indices(folds, n_boot, seed):
    rng = np.random.default_rng(seed)
    by_fold = [np.where(folds == f)[0] for f in np.unique(folds)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(ix, size=ix.size, replace=True) for ix in by_fold])


def bootstrap_ci(flags, adjacent, folds, n_boot=N_BOOT, seed=BOOT_SEED, statistic="log_or"):
    vals = []
    for idx in boot_indices(folds, n_boot, seed):
        f, a = flags[idx], adjacent[idx]
        vals.append(log_or_haldane(f, a) if statistic == "log_or" else f.mean())
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired_log_or_ci(flags_a, flags_b, adjacent, folds, n_boot=N_BOOT, seed=BOOT_SEED):
    """Real minus control, on the SAME resampled patients each time -- so the
    interval is on the difference, not the difference of two intervals."""
    diffs = []
    for idx in boot_indices(folds, n_boot, seed):
        a = adjacent[idx]
        diffs.append(log_or_haldane(flags_a[idx], a) - log_or_haldane(flags_b[idx], a))
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return np.nan, np.nan, np.nan
    return (float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def describe_variant(name, df, flags, runs=None, condition=None, extra=None):
    """Every pre-registered metric for one variant, in one row."""
    adjacent = df["stage"].isin(ADJACENT_STAGES).to_numpy()
    two_stage = df["stage"].isin(TWO_STAGE_STAGES).to_numpy()
    folds = df["fold"].to_numpy()
    n_flag = int(flags.sum())
    orr, p, (a, b, c, d) = fisher_or(flags, adjacent)
    lo, hi = bootstrap_ci(flags, adjacent, folds) if n_flag else (np.nan, np.nan)

    stability = np.nan
    if runs is not None and not runs.empty and condition is not None and n_flag:
        r = runs[runs["condition"] == condition]
        if not r.empty:
            cnt = r.groupby("sample_id")["run_flag"].sum()
            tot = r.groupby("sample_id")["run_flag"].size().max()
            need = int(np.ceil(STABILITY_FRACTION * tot))
            ids = df.loc[flags, "sample_id"]
            stability = float((ids.map(cnt).fillna(0) >= need).mean())

    y = df["y_true"].to_numpy()
    too_low = int((flags & (y == 0)).sum())     # label says not-F2+, model says F2+
    too_high = int((flags & (y == 1)).sum())

    row = {
        "variant": name, "n_flagged": n_flag,
        "flag_rate": n_flag / len(df) if len(df) else np.nan,
        "or_f1f2": orr, "or_p_onesided": p,
        "log_or_ci_lo": lo, "log_or_ci_hi": hi,
        "n_flag_adjacent": a, "n_flag_distant": b,
        "two_stage_share": float(two_stage[flags].mean()) if n_flag else np.nan,
        "stability_share": stability,
        "dir_recorded_too_low": too_low, "dir_recorded_too_high": too_high,
        "dir_split_low_share": too_low / n_flag if n_flag else np.nan,
    }
    for s in ALL_STAGES:
        row[f"n_{s}"] = int(((df["stage"] == s).to_numpy() & flags).sum())
    if extra:
        row.update(extra)
    return row


def per_cohort_table(name, df, flags):
    adjacent = df["stage"].isin(ADJACENT_STAGES).to_numpy()
    rows = []
    for (fold, cohort), g in df.groupby(["fold", "test_cohort"]):
        m = (df["fold"] == fold).to_numpy()
        f, adj = flags[m], adjacent[m]
        orr, p, (a, b, c, d) = fisher_or(f, adj)
        rows.append({"variant": name, "fold": fold, "test_cohort": cohort,
                      "n_patients": int(m.sum()), "n_flagged": int(f.sum()),
                      "flag_rate": float(f.mean()), "or_f1f2": orr, "or_p_onesided": p,
                      "n_flag_adjacent": a, "n_flag_distant": b})
    return pd.DataFrame(rows)


# ==========================================================================
# plots
# ==========================================================================
def plot_stage_distributions(df_primary, variant_flags, path_png, path_pdf):
    """Flagged stage distribution per variant, against the background rate.

    Every panel shares the same y axis and the same background bars, so the
    comparison across variants is a visual read rather than an arithmetic one.
    """
    items = [(k, v) for k, v in variant_flags.items() if v.sum() > 0]
    if not items:
        return False
    ncol = 3
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow), squeeze=False)
    bg = df_primary["stage"].value_counts(normalize=True).reindex(ALL_STAGES, fill_value=0)
    x = np.arange(len(ALL_STAGES))

    for ax, (name, flags) in zip(axes.ravel(), items):
        sub = df_primary.loc[flags, "stage"]
        share = sub.value_counts(normalize=True).reindex(ALL_STAGES, fill_value=0)
        ax.bar(x - 0.2, bg * 100, 0.38, color=NEUTRAL, label="all held-out patients")
        ax.bar(x + 0.2, share * 100, 0.38, color=CODAGAT, label="flagged")
        ax.axvspan(0.5, 2.5, color=OKABE[2], alpha=0.10, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_STAGES)
        ax.set_ylabel("Share (%)")
        ax.set_title(f"{name}\nn={int(flags.sum())}", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7.5)
    fig.suptitle("Stage distribution of flagged patients per variant "
                 "(shaded = F1/F2, one stage from the F2+ boundary)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_graph_comparison(rows, path_png, path_pdf):
    d = pd.DataFrame(rows)
    if d.empty:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    targets = sorted(d["target_n"].unique())
    w = 0.26
    for k, cond in enumerate(GRAPH_CONDITIONS):
        s = d[d["condition"] == cond].set_index("target_n").reindex(targets)
        xx = np.arange(len(targets)) + (k - 1) * w
        ax.bar(xx, s["or_f1f2"], w, label=cond,
               color=CODAGAT if cond == "real_graph" else OKABE[k],
               yerr=[s["or_f1f2"] - np.exp(s["log_or_ci_lo"]),
                     np.exp(s["log_or_ci_hi"]) - s["or_f1f2"]],
               capsize=3, error_kw={"lw": 0.8})
    ax.axhline(1.0, color="black", lw=1, ls=":")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([f"{t} flags" for t in targets])
    ax.set_ylabel("F1/F2 enrichment odds ratio")
    ax.set_title("Flag quality at matched flag counts", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    pair = d[d["condition"] != "empty_graph"].dropna(subset=["paired_diff_mean"])
    if not pair.empty:
        labs = [f"{r.condition}\n{r.target_n} flags" for r in pair.itertuples()]
        xx = np.arange(len(pair))
        ax.bar(xx, pair["paired_diff_mean"],
               color=[CODAGAT if c == "real_graph" else OKABE[2] for c in pair["condition"]],
               yerr=[pair["paired_diff_mean"] - pair["paired_diff_lo"],
                     pair["paired_diff_hi"] - pair["paired_diff_mean"]],
               capsize=4, error_kw={"lw": 0.9})
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(xx)
        ax.set_xticklabels(labs, fontsize=7.5)
        ax.set_ylabel("log OR minus empty-graph log OR\n(paired, 95% CI)", fontsize=8.5)
        ax.set_title("Paired difference against the no-graph control", fontsize=10)
    fig.suptitle("Does the gene graph produce better label-error flags?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return True


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="CoDA-GAT flagging v2 -- improved rules + does the graph help? "
                    "(inference only, read-only on results)")
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--n-mc-passes", type=int, default=N_MC_PASSES)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--recompute", action="store_true",
                    help="ignore flagging_v2/cache/ and redo every MC-dropout pass")
    ap.add_argument("--prereg-only", action="store_true",
                    help="write PREREG.txt and stop, so the pre-registration can be "
                         "committed before any result exists")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prereg_path = write_prereg()
    if args.prereg_only:
        log("--prereg-only: stopping before any result is computed")
        return

    LINES.clear()
    head("CoDA-GAT flagging v2")
    say(f"pre-registration: {prereg_path.relative_to(REPO)} "
        f"(written {datetime.fromtimestamp(prereg_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
    say("Inference only; read-only on all results directories.")
    say("")

    folds = fold_dirs(args.folds)
    log(f"folds: {[d.name for d in folds]}")
    master, runs, cal_pools = build_master(
        GRAPH_CONDITIONS, MODES, folds, args.n_mc_passes, args.recompute)
    tabular = attach_tabular(master)

    prim = master[(master["condition"] == "real_graph")
                  & (master["mode"] == PRIMARY_MODE)].reset_index(drop=True)
    say(f"held-out patients: {len(prim)} across {prim['fold'].nunique()} cohorts")
    say(f"CoDA-GAT runs available for the stable variant: "
        f"{runs[runs['condition'] == 'real_graph'].groupby('sample_id')['run_flag'].size().max()} "
        f"(5 seeds x 3 modes = 15 expected)")
    say("")

    summary_rows, cohort_rows, variant_flags = [], [], {}

    # ---------------- V1 baseline ----------------
    head("V1 -- baseline (Test A's rule, unchanged)")
    f_base, thr = rule_baseline(prim, cal_pools, "real_graph", PRIMARY_MODE)
    variant_flags["V1 baseline"] = f_base
    row = describe_variant("V1 baseline", prim, f_base, runs, "real_graph")
    summary_rows.append(row)
    cohort_rows.append(per_cohort_table("V1 baseline", prim, f_base))
    say(f"  flagged {row['n_flagged']}/{len(prim)} ({row['flag_rate']:.1%})   "
        f"OR={row['or_f1f2']:.3f} p={row['or_p_onesided']:.2e}")
    say(f"  calibration p{UNCERTAINTY_PERCENTILE} thresholds per fold: "
        f"{ {k: round(v, 5) for k, v in thr.items()} }")

    # reproduction check against Test A
    ta = TEST_A / "per_sample_flags.csv"
    if ta.exists():
        old = pd.read_csv(ta)
        col = next((c for c in ("flagged", "flag", "is_flagged") if c in old.columns), None)
        if col is not None:
            old_ids = set(old.loc[old[col].astype(str).str.lower().isin(["true", "1"]),
                                   "sample_id"])
            new_ids = set(prim.loc[f_base, "sample_id"])
            inter = len(old_ids & new_ids)
            say("")
            say(f"  Test A reproduction: Test A flagged {len(old_ids)}, this run flags "
                f"{len(new_ids)}, overlap {inter} "
                f"(Jaccard {inter / max(len(old_ids | new_ids), 1):.3f})")
            say("  Exact equality is NOT expected and its absence is not a bug: Test A did")
            say("  not seed its calibration MC-dropout, so its p40 threshold came from an")
            say("  unreproducible draw. This run pins the seed, so ITS numbers reproduce.")

    # ---------------- V2 two-model agreement ----------------
    head("V2 -- two-model agreement (CoDA-GAT AND a tabular model)")
    if not tabular:
        say("  no tabular per-sample predictions available -- variant not evaluated")
    for name in TABULAR_MODELS:
        if name not in tabular:
            say(f"  {name}: no per-sample predictions -- skipped")
            continue
        flags, n_cov = rule_two_model(prim, tabular[name])
        label = f"V2 two-model: {name}"
        variant_flags[label] = flags
        r = describe_variant(label, prim, flags, runs, "real_graph",
                              extra={"tabular_partner": name, "n_with_tabular": int(n_cov),
                                     "is_primary_partner": name in PRIMARY_TABULAR})
        summary_rows.append(r)
        cohort_rows.append(per_cohort_table(label, prim, flags))
        say(f"  {name:32s} flagged {r['n_flagged']:4d} ({r['flag_rate']:5.1%})  "
            f"OR={r['or_f1f2']:.3f} p={r['or_p_onesided']:.2e}  "
            f"two-stage {r['two_stage_share']:.1%}"
            if np.isfinite(r["or_f1f2"]) else
            f"  {name:32s} flagged {r['n_flagged']:4d} -- too few to test")

    # ---------------- V3 stable ----------------
    head(f"V3 -- stable (baseline rule in >= {STABLE_MIN_RUNS} of 15 runs)")
    f_stab, counts, n_runs = rule_stable(prim, runs, "real_graph")
    variant_flags[f"V3 stable (>={STABLE_MIN_RUNS}/{n_runs})"] = f_stab
    r = describe_variant(f"V3 stable (>={STABLE_MIN_RUNS}/{n_runs})", prim, f_stab,
                          runs, "real_graph", extra={"n_runs_considered": n_runs})
    summary_rows.append(r)
    cohort_rows.append(per_cohort_table(r["variant"], prim, f_stab))
    say(f"  runs per patient: {n_runs}")
    say(f"  flagged {r['n_flagged']}/{len(prim)} ({r['flag_rate']:.1%})   "
        f"OR={r['or_f1f2']:.3f} p={r['or_p_onesided']:.2e}"
        if np.isfinite(r["or_f1f2"]) else f"  flagged {r['n_flagged']} -- too few to test")
    if len(counts):
        say(f"  distribution of runs-flagged among patients ever flagged: "
            f"{dict(counts[counts > 0].value_counts().sort_index())}")

    # ---------------- V4 graph comparison, matched counts ----------------
    head("V4 -- graph comparison at matched flag counts")
    targets = []
    for c in MATCHED_COUNTS:
        targets.append(int(round(len(prim) * float(str(c).rstrip("%")) / 100))
                       if str(c).endswith("%") else int(c))
    say(f"  matched flag counts: {targets}  "
        f"({'; '.join(f'{c} -> {t}' for c, t in zip(MATCHED_COUNTS, targets))})")
    say("  ranking: disagreement > 0.5, then lowest MC-dropout uncertainty first")
    say("")

    adjacent = prim["stage"].isin(ADJACENT_STAGES).to_numpy()
    folds_arr = prim["fold"].to_numpy()
    graph_rows, matched_flags = [], {}

    for n_target in targets:
        for cond in GRAPH_CONDITIONS:
            sub = master[(master["condition"] == cond)
                         & (master["mode"] == (PRIMARY_MODE if cond == "real_graph" else "-"))]
            if sub.empty:
                say(f"  {cond}: no runs available -- skipped")
                continue
            sub = sub.set_index("sample_id").reindex(prim["sample_id"]).reset_index()
            flags, n_cand = rank_to_count(sub, n_target)
            matched_flags[(cond, n_target)] = flags
            label = f"V4 {cond} @{n_target}"
            variant_flags[label] = flags
            r = describe_variant(label, prim, flags, runs,
                                  "real_graph" if cond == "real_graph" else cond,
                                  extra={"condition": cond, "target_n": n_target,
                                          "n_candidates": n_cand,
                                          "shortfall": max(0, n_target - int(flags.sum()))})
            summary_rows.append(r)
            cohort_rows.append(per_cohort_table(label, prim, flags))
            graph_rows.append(r)
            short = f"  SHORTFALL {r['shortfall']}" if r["shortfall"] else ""
            say(f"  {cond:15s} @{n_target:4d}: flagged {r['n_flagged']:4d}  "
                f"OR={r['or_f1f2']:.3f} [{np.exp(r['log_or_ci_lo']):.2f}, "
                f"{np.exp(r['log_or_ci_hi']):.2f}]  two-stage {r['two_stage_share']:.1%}  "
                f"stability {r['stability_share']:.1%}{short}"
                if np.isfinite(r["or_f1f2"]) else
                f"  {cond:15s} @{n_target:4d}: flagged {r['n_flagged']} -- too few to test")

    # paired contrasts against the empty graph
    say("")
    say("Paired difference in log odds ratio, real/shuffled minus empty (same resampled")
    say("patients for both arms, 95% bootstrap CI over patients within cohort)")
    say("-" * 78)
    graph_df = pd.DataFrame(graph_rows)
    verdicts = []
    for n_target in targets:
        fe = matched_flags.get(("empty_graph", n_target))
        if fe is None:
            continue
        for cond in ("real_graph", "shuffled_graph"):
            fa = matched_flags.get((cond, n_target))
            if fa is None:
                continue
            mean, lo, hi = paired_log_or_ci(fa, fe, adjacent, folds_arr, args.n_boot)
            if not np.isfinite(lo):
                continue
            verdict = ("BETTER than no graph" if lo > 0 else
                       "WORSE than no graph" if hi < 0 else
                       "NO DIFFERENCE DETECTED")
            say(f"  {cond:15s} @{n_target:4d}: dlogOR = {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                f"   -> {verdict}")
            verdicts.append({"condition": cond, "target_n": n_target, "paired_diff_mean": mean,
                              "paired_diff_lo": lo, "paired_diff_hi": hi, "verdict": verdict})
    vdf = pd.DataFrame(verdicts)
    if not graph_df.empty and not vdf.empty:
        graph_df = graph_df.merge(vdf, on=["condition", "target_n"], how="left")

    say("")
    say("VERDICT -- does the graph produce better flags?")
    say("-" * 78)
    real_v = vdf[vdf["condition"] == "real_graph"] if not vdf.empty else pd.DataFrame()
    if real_v.empty:
        say("  not evaluable from the available runs")
    else:
        uniq = set(real_v["verdict"])
        if uniq == {"NO DIFFERENCE DETECTED"}:
            say("  EQUAL. At every matched flag count the paired CI for real minus empty")
            say("  contains zero. Per the pre-registration this is reported as no difference")
            say("  detected, not as a trend: the graph does not produce better label-error")
            say("  flags than a graph-free model with identical architecture and training.")
            say("  That extends the existing finding -- no accuracy gain, no robustness gain")
            say("  on AUC -- to a fourth axis, flag quality.")
        elif "BETTER than no graph" in uniq:
            say("  The real graph is BETTER at one or more matched counts:")
            for _, r in real_v[real_v["verdict"].str.startswith("BETTER")].iterrows():
                say(f"    @{r['target_n']}: dlogOR {r['paired_diff_mean']:+.4f} "
                    f"[{r['paired_diff_lo']:+.4f}, {r['paired_diff_hi']:+.4f}]")
            say("  This would be the first axis on which the graph earns its place, so it")
            say("  deserves a replication before it is claimed.")
        else:
            say("  The real graph is WORSE at one or more matched counts:")
            for _, r in real_v[real_v["verdict"].str.startswith("WORSE")].iterrows():
                say(f"    @{r['target_n']}: dlogOR {r['paired_diff_mean']:+.4f} "
                    f"[{r['paired_diff_lo']:+.4f}, {r['paired_diff_hi']:+.4f}]")
        sh = vdf[vdf["condition"] == "shuffled_graph"]
        if not sh.empty:
            say("")
            say("  Shuffled STRING is the control that separates 'connectivity' from")
            say("  'biology': it keeps the edge count and destroys which genes connect.")
            for _, r in sh.iterrows():
                say(f"    shuffled @{r['target_n']}: dlogOR {r['paired_diff_mean']:+.4f} "
                    f"[{r['paired_diff_lo']:+.4f}, {r['paired_diff_hi']:+.4f}] -> {r['verdict']}")

    # ---------------- the pre-registered table ----------------
    summary = pd.DataFrame(summary_rows)
    head("VARIANT TABLE (every pre-registered variant, including the ones that do worse)")
    say(f"  {'variant':34s} {'n':>5s} {'rate':>6s} {'OR':>7s} {'OR 95% CI':>16s} "
        f"{'2-stage':>8s} {'stable':>7s} {'low/high':>10s}")
    for _, r in summary.iterrows():
        ci = (f"[{np.exp(r['log_or_ci_lo']):.2f},{np.exp(r['log_or_ci_hi']):.2f}]"
              if np.isfinite(r["log_or_ci_lo"]) else "-")
        orr = f"{r['or_f1f2']:.3f}" if np.isfinite(r["or_f1f2"]) else "-"
        ts = f"{r['two_stage_share']:.1%}" if np.isfinite(r["two_stage_share"]) else "-"
        st = f"{r['stability_share']:.1%}" if np.isfinite(r["stability_share"]) else "-"
        say(f"  {r['variant']:34s} {r['n_flagged']:>5d} {r['flag_rate']:>6.1%} {orr:>7s} "
            f"{ci:>16s} {ts:>8s} {st:>7s} "
            f"{f'{int(r.dir_recorded_too_low)}/{int(r.dir_recorded_too_high)}':>10s}")
    say("")
    say("  OR       = one-sided Fisher odds ratio, flags at F1/F2 vs F0/F3/F4 (raw counts)")
    say("  95% CI   = bootstrap over patients within cohort, Haldane-corrected log OR")
    say("  2-stage  = share of flags at F0 or F3, implying a >=2-stage error (lower better)")
    say("  stable   = share of flagged patients also flagged in >=80% of the 15 runs")
    say("  low/high = recorded too low / recorded too high")

    cohort_df = pd.concat(cohort_rows, ignore_index=True) if cohort_rows else pd.DataFrame()
    head("PER-COHORT VALUES (n = 8; descriptive, underpowered)")
    for name, g in cohort_df.groupby("variant", sort=False):
        say(f"  {name}")
        say(f"    {'cohort':<12s} {'n':>5s} {'flags':>6s} {'rate':>6s} {'OR':>8s}")
        for _, r in g.sort_values("fold").iterrows():
            orr = f"{r['or_f1f2']:.3f}" if np.isfinite(r["or_f1f2"]) else "n/a"
            say(f"    {r['test_cohort']:<12s} {int(r['n_patients']):>5d} "
                f"{int(r['n_flagged']):>6d} {r['flag_rate']:>6.1%} {orr:>8s}")
        say("")

    # ---------------- outputs ----------------
    master.to_csv(OUT_DIR / "per_patient_values.csv", index=False)
    if not runs.empty:
        runs.to_csv(OUT_DIR / "per_run_flags.csv", index=False)
    summary.to_csv(OUT_DIR / "variant_summary.csv", index=False)
    if not cohort_df.empty:
        cohort_df.to_csv(OUT_DIR / "per_cohort_values.csv", index=False)
    if not graph_df.empty:
        graph_df.to_csv(OUT_DIR / "graph_comparison.csv", index=False)
    flag_wide = prim[["fold", "test_cohort", "sample_id", "stage", "y_true",
                      "p_f2plus", "uncertainty", "disagreement"]].copy()
    for name, f in variant_flags.items():
        flag_wide[name] = f
    flag_wide.to_csv(OUT_DIR / "per_patient_flags.csv", index=False)

    made = plot_stage_distributions(prim, variant_flags,
                                    OUT_DIR / "stage_distribution_per_variant.png",
                                    OUT_DIR / "stage_distribution_per_variant.pdf")
    made_g = plot_graph_comparison(graph_rows, OUT_DIR / "graph_comparison.png",
                                    OUT_DIR / "graph_comparison.pdf")

    head("FILES WRITTEN")
    for p in sorted(OUT_DIR.glob("*")):
        if p.is_file():
            say(f"  flagging_v2/{p.name}")
    if not made:
        say("  (no stage-distribution plot: every variant flagged zero patients)")
    if not made_g:
        say("  (no graph-comparison plot: the contrast was not evaluable)")

    (OUT_DIR / "summary.txt").write_text("\n".join(LINES) + "\n")
    log(f"wrote {(OUT_DIR / 'summary.txt').relative_to(REPO)}")


if __name__ == "__main__":
    main()
