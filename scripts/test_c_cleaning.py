"""
CoDA-GAT — Test C: the cleaning experiment.

The decisive (quasi-)causal test of whether flagged samples carry label
errors rather than model errors. If they are label errors, removing them
from training cleans the signal -> held-out performance holds or improves.
If they were genuinely hard but correctly labelled, removing them discards
real signal -> performance degrades.

Three conditions per fold/seed/operating-point:
  BASELINE  -- B7's existing mode=off, m=0 results. NOT retrained.
  CLEANED   -- training set minus the flagged samples. Retrained.
  RANDOM    -- training set minus an equal number of randomly chosen
               samples. Retrained. Without this control, "maybe removing
               any N samples helps" is unanswerable.

Everything reused, unchanged, by import:
  - the flagging rule            : test_a_adjacency.compute_flags
  - the calibration uncertainty  : test_b_rate.get_calibration_uncertainty_pools
  - the model, training loop,
    conformal layer, evaluation  : run_full_experiment_b7.setup_fold / run_single
No flagging logic and no training/evaluation logic is redefined here. The
CLEANED and RANDOM conditions are produced purely by handing B7's own
run_single() a fold_setup whose idx_fit has had rows removed.

IN-SAMPLE CAVEAT (repeated in the output): Tests A/B validated the flagging
rule on HELD-OUT predictions. Here the same rule is applied to TRAINING-split
predictions, which are in-sample -- the model was fit to those labels. A
training sample with |p - y| > 0.5 is one the model refused to memorise
despite training on it. That is arguably a stronger label-error signal, but
the achieved flagging RATE will not match Test B's 18.3% held-out rate. The
rule is carried over unchanged; the realised rate is reported as observed.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import wilcoxon

from train_fold0_b1 import CoDAGAT, get_device, N_FEATURES, SHARED_DIR
from run_full_experiment_b7 import (
    setup_fold, run_single, extract_full, mc_dropout_probs,
    load_edge_index_generic, load_fold_info, parse_fold_index, fmt_hms,
    flatten_run_metrics,
)
from test_a_adjacency import compute_flags, MODE, SEEDS, RESULTS_B7, FOLDS_DIR
from test_b_rate import get_calibration_uncertainty_pools

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "test_c_cleaning"
FLAGS_DIR = OUT_ROOT / "train_flags"

PRIMARY_PERCENTILE = 70    # Test B: matches the predicted 18.6% misassignment rate, enrichment intact
SECONDARY_PERCENTILE = 40  # Test B: the conservative high-precision point
OPERATING_POINTS = [PRIMARY_PERCENTILE, SECONDARY_PERCENTILE]

RANDOM_REMOVAL_BASE_SEED = 20260911  # fixed + documented, independent of the training seeds
PRIMARY_METRIC = "macro_f1"


def log(msg):
    print(f"[TestC] {msg}", flush=True)


def random_removal_seed(fold_id, percentile, draw_num):
    """Single definition of the random-removal seed, used both to draw the
    removal set and to record it in condition_meta.json -- so the two can
    never drift apart."""
    return RANDOM_REMOVAL_BASE_SEED + fold_id * 1000 + percentile * 10 + draw_num


def draw_number(condition):
    return int(condition.replace("random_draw", "")) if condition.startswith("random_draw") else 0


def as_bool_array(series):
    """Flag columns round-trip through CSV. If pandas ever reads them back as
    object dtype, a bare .astype(bool) would make EVERY row True (non-empty
    strings are truthy) and silently corrupt the removal sets. Coerce explicitly."""
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()


# ==========================================================================
# FATAL GUARD -- flag computation must touch training-split info only
# ==========================================================================
def assert_train_only_flagging(fold_id, train_ids, cal_ids, test_ids, train_cohorts,
                                test_cohort, n_flag_rows, cal_pool_len, n_seeds):
    """
    Hard, multi-pronged assertion (NOT a comment) that this fold's flags were
    computed without touching the held-out test cohort's labels, predictions,
    or uncertainty distribution. Raises RuntimeError on any violation -- if
    this fires, the cleaning experiment would be circular and must stop.
    """
    train_set, cal_set, test_set = set(train_ids), set(cal_ids), set(test_ids)

    overlap_train_test = train_set & test_set
    if overlap_train_test:
        raise RuntimeError(
            f"FATAL [fold {fold_id}]: {len(overlap_train_test)} sample_id(s) appear in BOTH the "
            f"training split and the held-out test cohort, e.g. {sorted(overlap_train_test)[:5]}. "
            f"Flags would be contaminated by test data. Refusing to continue."
        )

    overlap_cal_test = cal_set & test_set
    if overlap_cal_test:
        raise RuntimeError(
            f"FATAL [fold {fold_id}]: {len(overlap_cal_test)} sample_id(s) appear in BOTH the "
            f"calibration cohort and the held-out test cohort. The uncertainty threshold would be "
            f"derived from test data. Refusing to continue."
        )

    if test_cohort in set(train_cohorts):
        raise RuntimeError(
            f"FATAL [fold {fold_id}]: the held-out test cohort '{test_cohort}' appears in the "
            f"training split's cohorts {sorted(set(train_cohorts))}. LODO is violated. "
            f"Refusing to continue."
        )

    if n_flag_rows != len(train_ids):
        raise RuntimeError(
            f"FATAL [fold {fold_id}]: the flag table has {n_flag_rows} rows but the training split "
            f"has {len(train_ids)} samples (test has {len(test_ids)}). The wrong split was used. "
            f"Refusing to continue."
        )

    expected = len(cal_ids) * n_seeds
    if cal_pool_len != expected:
        raise RuntimeError(
            f"FATAL [fold {fold_id}]: the uncertainty-threshold pool has {cal_pool_len} entries but "
            f"the calibration cohort ({len(cal_ids)} samples x {n_seeds} seeds) implies {expected}. "
            f"The pool did not come from calibration. Refusing to continue."
        )


def read_split_ids(fold_dir, split):
    return pd.read_csv(fold_dir / split / "sample_ids.csv")["sample_id"].tolist()


# ==========================================================================
# Step 2: train-split flags (the one genuinely new computation)
# ==========================================================================
def compute_train_side_values(fold_id, test_cohort, train_loader_full, n_mc_passes):
    """Per-sample P(F2+) and MC-dropout uncertainty on the TRAINING split,
    averaged over the fold's 5 checkpoints. B7 saved train embeddings but
    never train probabilities or train MC-dropout, so both are recomputed
    here (forward passes only, no training)."""
    device = get_device()
    p_seeds, u_seeds = [], []
    for seed in SEEDS:
        run_dir = RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / f"mode_{MODE}" / f"seed_{seed}"
        ckpt = torch.load(run_dir / "weights.pt", weights_only=False, map_location=device)
        model = CoDAGAT(N_FEATURES, ckpt["n_ordinal_classes"], ckpt["n_cohorts"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        _, probs_train, _, _ = extract_full(model, train_loader_full, device)
        mc_train = mc_dropout_probs(model, train_loader_full, device, n_passes=n_mc_passes)
        p_seeds.append(probs_train[:, 1])
        u_seeds.append(mc_train.std(axis=0))

    return np.mean(p_seeds, axis=0), np.mean(u_seeds, axis=0)


def build_fold_flags(fold_dir, edge_index, dev_lookup, labels_df, cal_pool,
                     operating_points, n_mc_passes, recompute):
    """Returns this fold's training-split flag table. Cached to disk -- the
    checkpoint reloads are the slow part. The fatal guard runs on BOTH the
    fresh and the cached path, so it can never be skipped by resuming."""
    fold_id = parse_fold_index(fold_dir.name)
    fold_info = load_fold_info(fold_dir)
    test_cohort = fold_info["test_cohort"]
    cache_path = FLAGS_DIR / f"fold{fold_id}_train_flags.csv"

    train_ids = read_split_ids(fold_dir, "train")
    cal_ids = read_split_ids(fold_dir, "calibration")
    test_ids = read_split_ids(fold_dir, "test")
    train_cohorts = pd.read_csv(fold_dir / "train" / "y.csv", index_col=0)["cohort"].tolist()

    if cache_path.exists() and not recompute:
        df_fold = pd.read_csv(cache_path)
        log(f"    cached train flags loaded ({cache_path.name}); use --recompute-flags to redo")
    else:
        log(f"    recomputing train-split probs + MC-dropout from {len(SEEDS)} checkpoints "
            f"(forward passes only, no training)...")
        fold_setup = setup_fold(fold_dir, edge_index, dev_lookup)
        p_train, u_train = compute_train_side_values(
            fold_id, test_cohort, fold_setup["train_loader_full"], n_mc_passes)
        y_train = fold_setup["y1_train"].astype(int)
        del fold_setup  # release before the training pass rebuilds it

        df_fold = pd.DataFrame({
            "fold": fold_id, "test_cohort": test_cohort, "sample_id": train_ids,
            "train_position": np.arange(len(train_ids)), "cohort": train_cohorts,
            "p_f2plus": p_train, "uncertainty": u_train, "y_true": y_train,
            "stage": labels_df.loc[train_ids, "stage"].to_numpy(),
            "disagreement": np.abs(p_train - y_train),
        })
        # Test A's rule, unchanged -- same function, fed train rows instead of test rows
        for q in operating_points:
            flagged, _ = compute_flags(df_fold, {fold_id: cal_pool}, q)
            df_fold[f"flagged_p{q}"] = flagged
        df_fold.to_csv(cache_path, index=False)

    # ---- FATAL GUARD (runs on cached and fresh paths alike) ----
    assert_train_only_flagging(
        fold_id=fold_id, train_ids=train_ids, cal_ids=cal_ids, test_ids=test_ids,
        train_cohorts=train_cohorts, test_cohort=test_cohort,
        n_flag_rows=len(df_fold), cal_pool_len=len(cal_pool), n_seeds=len(SEEDS),
    )
    log(f"    GUARD PASSED: flags use train({len(train_ids)}) + calibration({len(cal_ids)}) only; "
        f"test cohort '{test_cohort}' ({len(test_ids)} samples) untouched.")
    for q in operating_points:
        mask = as_bool_array(df_fold[f"flagged_p{q}"])
        log(f"    p{q}: {int(mask.sum())}/{len(df_fold)} train samples flagged ({mask.mean():.1%})")
    return df_fold


# ==========================================================================
# Step 3: condition construction
# ==========================================================================
def removed_positions_for_condition(df_fold, fold_id, idx_fit, percentile, condition, remove_from):
    """Positional train-row indices to remove for this condition."""
    flagged_positions = df_fold.loc[as_bool_array(df_fold[f"flagged_p{percentile}"]),
                                     "train_position"].to_numpy()
    if remove_from == "fit":
        flagged_positions = flagged_positions[np.isin(flagged_positions, idx_fit)]

    if condition == "cleaned":
        return flagged_positions

    if condition.startswith("random_draw"):
        n_remove = int(np.isin(idx_fit, flagged_positions).sum())
        if n_remove == 0:
            return np.array([], dtype=int)
        rng = np.random.default_rng(random_removal_seed(fold_id, percentile, draw_number(condition)))
        return rng.choice(idx_fit, size=min(n_remove, len(idx_fit)), replace=False)

    raise ValueError(f"unknown condition: {condition}")


def make_modified_fold_setup(fold_setup, removed_positions):
    """Shallow copy of B7's fold_setup with idx_fit filtered. val/cal/test
    loaders, class_weights and model config are shared by reference and are
    therefore identical to the baseline -- the ONLY thing that differs across
    conditions is which rows contribute gradient updates.

    NOTE: class_weights are deliberately NOT recomputed on the reduced set.
    Keeping B7's original full-train weights means exactly one thing varies
    across BASELINE / CLEANED / RANDOM. CLEANED and RANDOM are treated
    identically either way, so the control remains valid.
    """
    new_setup = dict(fold_setup)
    orig_idx_fit = fold_setup["idx_fit"]
    new_idx_fit = orig_idx_fit[~np.isin(orig_idx_fit, removed_positions)]
    new_setup["idx_fit"] = new_idx_fit
    # keep the augmenter's parent pool consistent (a no-op at m=0, correct if m>0 is ever used)
    new_setup["augmenter"].set_parent_pool(new_idx_fit)
    return new_setup


# ==========================================================================
# Result collection
# ==========================================================================
def load_baseline_rows():
    rows = []
    for marker in RESULTS_B7.glob(f"fold*/mode_{MODE}/seed_*/DONE.marker"):
        with open(marker.parent / "run_metrics.json") as f:
            rm = json.load(f)
        if rm.get("m", 0) != 0:
            continue
        row = flatten_run_metrics(rm)
        row["condition"] = "baseline"
        row["operating_point"] = "n/a"
        row["n_removed"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def collect_testc_rows():
    rows = []
    for marker in OUT_ROOT.glob("fold*/op_p*/*/seed_*/DONE.marker"):
        run_dir = marker.parent
        with open(run_dir / "run_metrics.json") as f:
            rm = json.load(f)
        row = flatten_run_metrics(rm)
        row["condition"] = run_dir.parent.name
        row["operating_point"] = run_dir.parent.parent.name.replace("op_", "")
        meta_path = run_dir / "condition_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                row["n_removed"] = json.load(f).get("n_removed", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# Step 5: paired comparisons
# ==========================================================================
def paired_comparison(df, condition_a, condition_b, metric, op_point=None):
    """Seeds averaged within fold first, then paired Wilcoxon across the folds."""
    a = df[df["condition"] == condition_a]
    b = df[df["condition"] == condition_b]
    if op_point is not None:
        a = a[a["operating_point"] == op_point]
        if condition_b != "baseline":
            b = b[b["operating_point"] == op_point]
    if a.empty or b.empty:
        return None

    joined = pd.concat([a.groupby("fold")[metric].mean(), b.groupby("fold")[metric].mean()],
                        axis=1, keys=["a", "b"]).dropna()
    if len(joined) < 2:
        return None

    deltas = joined["a"] - joined["b"]
    try:
        _, p = wilcoxon(joined["a"], joined["b"])
    except ValueError:
        p = float("nan")

    return {
        "comparison": f"{condition_a} vs {condition_b}", "metric": metric,
        "operating_point": op_point if op_point else "n/a", "n_folds": len(joined),
        "mean_a": float(joined["a"].mean()), "mean_b": float(joined["b"].mean()),
        "mean_delta": float(deltas.mean()), "std_delta": float(deltas.std()),
        "min_delta": float(deltas.min()), "max_delta": float(deltas.max()),
        "n_folds_improved": int((deltas > 0).sum()), "n_folds_degraded": int((deltas < 0).sum()),
        "wilcoxon_p": p,
        "per_fold_deltas": {int(f): float(d) for f, d in deltas.items()},
    }


def coverage_comparison(df):
    rows = []
    cov_cols = [c for c in df.columns if c.endswith("_coverage")]
    for op in sorted(df["operating_point"].astype(str).unique()):
        for cond in sorted(df[df["operating_point"].astype(str) == op]["condition"].unique()):
            sub = df[(df["operating_point"].astype(str) == op) & (df["condition"] == cond)]
            row = {"operating_point": op, "condition": cond, "n_runs": len(sub)}
            for c in cov_cols:
                row[c] = float(sub[c].mean())
            rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# Reporting
# ==========================================================================
def aggregate_and_report(args, quiet=False):
    baseline_df = load_baseline_rows()
    testc_df = collect_testc_rows()
    if testc_df.empty:
        if not quiet:
            log("no Test C runs completed yet -- nothing to aggregate.")
        return

    all_df = pd.concat([baseline_df, testc_df], ignore_index=True)
    all_df.to_csv(OUT_ROOT / "per_run_results.csv", index=False)

    comparisons = []
    for op in sorted(testc_df["operating_point"].unique()):
        conds = sorted(testc_df[testc_df["operating_point"] == op]["condition"].unique())
        randoms = [c for c in conds if c.startswith("random")]
        for metric in ("macro_f1", "auc", "balanced_acc"):
            for cond in conds:
                c = paired_comparison(all_df, cond, "baseline", metric, op_point=op)
                if c:
                    comparisons.append(c)
            for rnd in randoms:
                c = paired_comparison(all_df, "cleaned", rnd, metric, op_point=op)
                if c:
                    comparisons.append(c)

    comp_df = pd.DataFrame(comparisons)
    if not comp_df.empty:
        comp_df.drop(columns=["per_fold_deltas"]).to_csv(OUT_ROOT / "comparison_table.csv", index=False)
        with open(OUT_ROOT / "per_fold_deltas.json", "w") as f:
            json.dump([{k: c[k] for k in ("comparison", "metric", "operating_point", "per_fold_deltas")}
                       for c in comparisons], f, indent=2)

    coverage_comparison(all_df).to_csv(OUT_ROOT / "conformal_coverage_comparison.csv", index=False)

    if quiet:
        return

    primary_op = f"p{args.operating_points[0]}"

    def find(pred):
        return next((c for c in comparisons if pred(c)), None)

    cleaned_vs_base = find(lambda c: c["comparison"] == "cleaned vs baseline"
                           and c["metric"] == PRIMARY_METRIC and c["operating_point"] == primary_op)
    random_vs_base = find(lambda c: c["comparison"].startswith("random")
                          and c["comparison"].endswith("vs baseline")
                          and c["metric"] == PRIMARY_METRIC and c["operating_point"] == primary_op)
    cleaned_vs_random = find(lambda c: c["comparison"].startswith("cleaned vs random")
                             and c["metric"] == PRIMARY_METRIC and c["operating_point"] == primary_op)

    lines = ["=" * 78, "Test C: The Cleaning Experiment", "=" * 78, ""]
    lines += [
        "IN-SAMPLE CAVEAT: Tests A/B validated the flagging rule on HELD-OUT predictions.",
        "Here the identical rule (no retuning) is applied to TRAINING-split predictions,",
        "which are in-sample -- the model was fit to those labels. A flagged training sample",
        "is one the model refused to memorise despite training on it. The achieved training",
        "flagging rate therefore need NOT match Test B's 18.3% held-out rate; realised rates",
        "are in train_flags/ and were printed at run time, as observed.",
        "",
        "n = 8 folds: p-floor 0.008, MDE ~1.15 sigma_d. Everything below is DESCRIPTIVE with",
        "fold-level intervals, not a confirmatory significance test.",
        "",
    ]

    if cleaned_vs_base is None:
        lines.append("Not enough completed runs to form a verdict yet.")
    else:
        d = cleaned_vs_base
        lines.append(f"PRIMARY operating point {primary_op}, metric {PRIMARY_METRIC}:")
        lines.append(f"  CLEANED vs BASELINE: mean delta={d['mean_delta']:+.4f} "
                     f"(sd={d['std_delta']:.4f}, range {d['min_delta']:+.4f} to {d['max_delta']:+.4f}), "
                     f"{d['n_folds_improved']}/{d['n_folds']} folds improved, "
                     f"Wilcoxon p={d['wilcoxon_p']:.4f}")
        if random_vs_base:
            r = random_vs_base
            lines.append(f"  RANDOM  vs BASELINE: mean delta={r['mean_delta']:+.4f} "
                         f"(sd={r['std_delta']:.4f}, range {r['min_delta']:+.4f} to {r['max_delta']:+.4f}), "
                         f"{r['n_folds_improved']}/{r['n_folds']} folds improved, "
                         f"Wilcoxon p={r['wilcoxon_p']:.4f}")
        if cleaned_vs_random:
            cr = cleaned_vs_random
            lines.append(f"  CLEANED vs RANDOM  : mean delta={cr['mean_delta']:+.4f} "
                         f"(sd={cr['std_delta']:.4f}), {cr['n_folds_improved']}/{cr['n_folds']} folds "
                         f"favour cleaned, Wilcoxon p={cr['wilcoxon_p']:.4f}")
        lines.append("")

        delta, p = d["mean_delta"], d["wilcoxon_p"]
        noise = d["std_delta"] if d["std_delta"] > 0 else 1e-9
        if delta > 0 and p < 0.05:
            verdict = ("TEST C: cleaning IMPROVED held-out performance (strongest -- evidence flags "
                       "caught real label noise)")
        elif delta < 0 and p < 0.05:
            verdict = ("TEST C: cleaning DEGRADED held-out performance (flagged samples carried real "
                       "signal; label-error interpretation weakens)")
        elif abs(delta) < 0.5 * noise:
            verdict = ("TEST C: cleaning HELD performance (consistent with label error -- removing "
                       "them cost nothing)")
        else:
            verdict = ("TEST C: cleaning HELD performance (consistent with label error -- removing "
                       f"them cost nothing); direction {'positive' if delta > 0 else 'negative'} "
                       "but not separable from noise at n=8")
        lines += [verdict, ""]

        lines.append("CONTROL INTERPRETATION (the verdict is only meaningful relative to this):")
        if cleaned_vs_random and random_vs_base:
            if cleaned_vs_random["mean_delta"] > 0:
                lines.append(f"  Removing the FLAGGED samples beat removing an equal number of RANDOM "
                             f"samples by {cleaned_vs_random['mean_delta']:+.4f} {PRIMARY_METRIC} on "
                             f"average -- the flagged set is specifically informative, not just 'any N rows'.")
            else:
                lines.append(f"  Removing FLAGGED samples did NOT beat removing an equal number of RANDOM "
                             f"samples ({cleaned_vs_random['mean_delta']:+.4f} {PRIMARY_METRIC}) -- any "
                             f"apparent benefit is not specific to the flagged set. This materially "
                             f"weakens the label-error claim and must be reported as such.")
            lines.append(f"  Random removal alone moved {PRIMARY_METRIC} by "
                         f"{random_vs_base['mean_delta']:+.4f}; if that is comparable to the cleaned "
                         f"delta, 'removing any N samples helps' is not excluded.")
        else:
            lines.append("  Random-removal control not yet complete -- the verdict above cannot be "
                         "interpreted without it.")
        lines.append("")

    lines += [
        "Per-fold deltas: per_fold_deltas.json  |  all comparisons: comparison_table.csv",
        "Conformal coverage (does cleaning change calibration as well as accuracy?): "
        "conformal_coverage_comparison.csv",
        "=" * 78,
    ]

    summary = "\n".join(lines)
    with open(OUT_ROOT / "summary.txt", "w") as f:
        f.write(summary + "\n")
    for line in lines:
        log(line)
    log(f"outputs saved to {OUT_ROOT}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT Test C -- the cleaning experiment")
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--operating-points", nargs="*", type=int, default=OPERATING_POINTS)
    parser.add_argument("--n-random-draws", type=int, default=1,
                         help="independent random-removal control draws per fold/operating point")
    parser.add_argument("--n-mc-passes", type=int, default=30)
    parser.add_argument("--remove-from", choices=["fit", "train"], default="fit",
                         help="'fit' (default) removes flagged rows only from the gradient-update set, "
                              "leaving the early-stopping validation set identical across all "
                              "conditions; 'train' removes them from the whole train split")
    parser.add_argument("--recompute-flags", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        log("aggregate-only: rebuilding tables from existing runs...")
        aggregate_and_report(args)
        return

    log("Test C: the cleaning experiment")
    log(f"operating points: {[f'p{q}' for q in args.operating_points]}  seeds: {args.seeds}  "
        f"random draws: {args.n_random_draws}  remove-from: {args.remove_from}")
    log("Reusing unchanged: Test A's flagging rule, Test B's calibration pools, B7's setup_fold/run_single.")

    edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")
    labels_df = pd.read_parquet(SHARED_DIR / "labels.parquet")

    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()) if args.folds is None \
        else [FOLDS_DIR / f for f in args.folds]

    log("recomputing calibration uncertainty pools (Test B's function, unchanged)...")
    cal_pools = get_calibration_uncertainty_pools(fold_dirs, edge_index, dev_lookup, args.n_mc_passes)

    log("Step 2: computing training-split flags (one fold resident at a time)...")
    flag_frames = {}
    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        log(f"=== fold {fold_id} ({fold_dir.name}) ===")
        flag_frames[fold_id] = build_fold_flags(
            fold_dir, edge_index, dev_lookup, labels_df, cal_pools[fold_id],
            args.operating_points, args.n_mc_passes, args.recompute_flags)

    flags_df = pd.concat(flag_frames.values(), ignore_index=True)
    flags_df.to_csv(OUT_ROOT / "train_flags_all_folds.csv", index=False)

    log("")
    log("ACHIEVED TRAINING-SPLIT FLAGGING RATES (in-sample -- see the caveat in summary.txt):")
    for q in args.operating_points:
        col = f"flagged_p{q}"
        rates = flags_df.assign(_f=as_bool_array(flags_df[col]))
        by_fold = rates.groupby("fold")["_f"].mean()
        log(f"    p{q}: pooled {rates['_f'].mean():.1%}  per-fold {[f'{r:.1%}' for r in by_fold]}")
    log("")

    conditions = ["cleaned"] + [f"random_draw{d}" for d in range(args.n_random_draws)]
    total_runs = len(fold_dirs) * len(args.operating_points) * len(conditions) * len(args.seeds)
    log(f"Step 3: {total_runs} retraining runs planned ({len(fold_dirs)} folds x "
        f"{len(args.operating_points)} op points x {len(conditions)} conditions x "
        f"{len(args.seeds)} seeds). BASELINE is reused from B7, not retrained.")
    log("Resumable via DONE.marker; a failed run is logged and skipped, not fatal.")

    start_time, run_counter, completed = time.time(), 0, 0

    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        df_fold = flag_frames[fold_id]
        test_cohort = df_fold["test_cohort"].iloc[0]

        # skip the expensive setup entirely if every run for this fold is already done
        pending = [
            (q, cond, seed)
            for q in args.operating_points for cond in conditions for seed in args.seeds
            if not (OUT_ROOT / f"fold{fold_id}_{test_cohort}" / f"op_p{q}" / cond /
                    f"seed_{seed}" / "DONE.marker").exists()
        ]
        run_counter += (len(args.operating_points) * len(conditions) * len(args.seeds)) - len(pending)
        if not pending:
            log(f"=== fold {fold_id} ({test_cohort}): all runs already done, skipping ===")
            continue

        log(f"=== fold {fold_id} ({test_cohort}): building fold setup for {len(pending)} pending runs ===")
        fold_setup = setup_fold(fold_dir, edge_index, dev_lookup)

        for q, condition, seed in pending:
            run_counter += 1
            run_dir = OUT_ROOT / f"fold{fold_id}_{test_cohort}" / f"op_p{q}" / condition / f"seed_{seed}"

            removed = removed_positions_for_condition(
                df_fold, fold_id, fold_setup["idx_fit"], q, condition, args.remove_from)
            n_removed_from_fit = int(np.isin(fold_setup["idx_fit"], removed).sum())

            elapsed = time.time() - start_time
            eta = f"  ETA={fmt_hms(elapsed / completed * (total_runs - run_counter + 1))}" if completed else ""
            log(f"[{run_counter}/{total_runs}] fold={fold_id} p{q} {condition} seed={seed} "
                f"-- removing {n_removed_from_fit}/{len(fold_setup['idx_fit'])} fit rows"
                f"  elapsed={fmt_hms(elapsed)}{eta}")

            try:
                modified_setup = make_modified_fold_setup(fold_setup, removed)
                rm = run_single(modified_setup, fold_id, MODE, seed, 0, run_dir, args.n_mc_passes, 7)
                with open(run_dir / "condition_meta.json", "w") as f:
                    json.dump({
                        "condition": condition, "operating_point": f"p{q}",
                        "n_removed": n_removed_from_fit,
                        "n_fit_before": int(len(fold_setup["idx_fit"])),
                        "n_fit_after": int(len(modified_setup["idx_fit"])),
                        "remove_from": args.remove_from,
                        "random_removal_seed": (random_removal_seed(fold_id, q, draw_number(condition))
                                                 if condition.startswith("random_draw") else None),
                    }, f, indent=2)
                completed += 1
                log(f"    done: AUC={rm['auc']:.4f}  macro_f1={rm['macro_f1']:.4f}  "
                    f"balanced_acc={rm['balanced_acc']:.4f}")
            except Exception:
                log(f"    FAILED fold={fold_id} p{q} {condition} seed={seed} -- continuing.")
                traceback.print_exc()

            aggregate_and_report(args, quiet=True)

        del fold_setup  # release before the next fold

    log(f"all runs attempted. total elapsed: {fmt_hms(time.time() - start_time)}  "
        f"completed this session: {completed}/{total_runs}")
    aggregate_and_report(args)


if __name__ == "__main__":
    main()
