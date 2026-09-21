"""
CoDA-GAT — Block B5: the two mandatory safety checks that GATE B7.

Check 1 (placebo decomposition): split ONE cohort into two random halves.
Since both halves are the same cohort, there is no true distribution
shift, so every decomposition component should be ~0. A systematic
non-zero component means the estimator is biased.

Check 2 (internal consistency): the cross-site prevalence component (the
decomposition B8 will use) must agree with an INDEPENDENT ground-truth
estimate obtained by actually varying prevalence within a single cohort
via bootstrap subsampling and fitting the resulting dose-response slope.

Check 3 (order-permutation): the 2-factor decomposition is computed in
both possible orderings; if the shares swing a lot depending on order,
only the Shapley-symmetrized (averaged) version is valid, and this is
flagged explicitly.

CRITICAL: decompose_drop() below is THE decomposition function. It is
used unchanged for the placebo check, for the real per-fold cross-site
estimate in the consistency check, and is the function B8 will reuse --
there is no separate "simplified" version anywhere in this file.

This block trains its own fold models (real data only, no B2
augmentation, seed 42) because B1 only ever produced fold0 outputs and
B3's per-fold results were not persisted at sample-level granularity --
sample-level (sample_id, y_true, prob) pairs for every fold's
calibration/test cohort are required here to allow arbitrary re-splitting
and resampling. The model/training step itself (CoDAGAT, run_epoch,
extract_outputs) is imported from train_fold0_b1.py unchanged.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train_fold0_b1 import (
    CoDAGAT, run_epoch, extract_outputs,
    get_device, set_seed,
    N_NODES, N_FEATURES, N_EDGES_EXPECTED,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY, VAL_FRACTION, SEED,
    SHARED_DIR,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
OUT_DIR = REPO_ROOT / "safety_checks_b5"

METRICS = {
    "macro_f1": lambda y, pred: f1_score(y, pred, average="macro", zero_division=0),
    "accuracy": lambda y, pred: float(np.mean(y == pred)),
}


def log(msg):
    print(f"[B5] {msg}", flush=True)


def parse_fold_index(dir_name: str) -> int:
    m = re.match(r"fold(\d+)_", dir_name)
    if not m:
        raise ValueError(f"cannot parse fold index from directory name: {dir_name}")
    return int(m.group(1))


def percentile_ci(values, alpha=0.05):
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ==========================================================================
# Fold-agnostic data loading + training (same pattern as B3/B4; B1 is
# hardcoded to fold0). Model/loss/training-step imported unchanged.
# ==========================================================================
def load_edge_index_generic():
    edges = pd.read_csv(SHARED_DIR / "graph_edges.csv")
    assert len(edges) == N_EDGES_EXPECTED
    return torch.tensor(edges[["src_idx", "dst_idx"]].to_numpy().T, dtype=torch.long)


def load_fold_info(fold_dir: Path):
    with open(fold_dir / "fold_info.json") as f:
        return json.load(f)


def load_split_for_fold(fold_dir, split, edge_index, dev_lookup, cohort_to_idx):
    split_dir = fold_dir / split
    X = np.load(split_dir / "X.npy")
    y = pd.read_csv(split_dir / "y.csv", index_col=0)
    sample_ids = pd.read_csv(split_dir / "sample_ids.csv")["sample_id"].tolist()
    assert list(y.index) == sample_ids

    dev_scores = dev_lookup.loc[sample_ids, "deviation_score"].to_numpy(dtype=np.float32)
    y1 = y["y_T1_significant"].to_numpy(dtype=np.float32)
    y2_raw = y["y_T2_ordinal"].to_numpy(dtype=np.float32)
    cohorts = y["cohort"].tolist()

    data_list = []
    for i in range(X.shape[0]):
        x = torch.tensor(X[i], dtype=torch.float32)
        y1_i = torch.tensor(int(y1[i]), dtype=torch.long)
        y2_valid = not np.isnan(y2_raw[i])
        y2_i = torch.tensor(y2_raw[i] if y2_valid else -1.0, dtype=torch.float32)
        y2_mask_i = torch.tensor(y2_valid, dtype=torch.bool)
        y3_i = torch.tensor(dev_scores[i], dtype=torch.float32)
        if cohorts[i] in cohort_to_idx:
            cohort_i = torch.tensor(cohort_to_idx[cohorts[i]], dtype=torch.long)
            cohort_valid = torch.tensor(True)
        else:
            cohort_i = torch.tensor(-1, dtype=torch.long)
            cohort_valid = torch.tensor(False)
        data_list.append(Data(
            x=x, edge_index=edge_index,
            y1=y1_i.unsqueeze(0), y2=y2_i.unsqueeze(0), y2_mask=y2_mask_i.unsqueeze(0),
            y3=y3_i.unsqueeze(0), cohort=cohort_i.unsqueeze(0), cohort_valid=cohort_valid.unsqueeze(0),
            sample_id=sample_ids[i],
        ))
    return data_list, sample_ids, y


def train_fold_model(fold_dir, edge_index, dev_lookup, seed):
    """Trains CoDAGAT on one fold (real data, no augmentation), returns
    sample-level (y, prob) for calibration and test -- everything below
    is built from this."""
    fold_info = load_fold_info(fold_dir)
    cohort_to_idx = {c: i for i, c in enumerate(sorted(fold_info["train_cohorts"]))}
    n_cohorts = len(cohort_to_idx)

    train_data, train_ids, train_y = load_split_for_fold(fold_dir, "train", edge_index, dev_lookup, cohort_to_idx)
    cal_data, cal_ids, cal_y = load_split_for_fold(fold_dir, "calibration", edge_index, dev_lookup, cohort_to_idx)
    test_data, test_ids, test_y = load_split_for_fold(fold_dir, "test", edge_index, dev_lookup, cohort_to_idx)

    n_ordinal_classes = int(np.nanmax(
        pd.concat([train_y["y_T2_ordinal"], cal_y["y_T2_ordinal"], test_y["y_T2_ordinal"]])
    )) + 1

    y1_train = train_y["y_T1_significant"].to_numpy()
    counts = np.bincount(y1_train.astype(int))
    weights = len(y1_train) / (len(counts) * counts)
    device = get_device()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    strat_key = [f"{c}_{int(y)}" for c, y in zip(train_y["cohort"], y1_train)]
    idx_all = np.arange(len(train_data))
    idx_fit, idx_val = train_test_split(idx_all, test_size=VAL_FRACTION, random_state=seed, stratify=strat_key)
    fit_loader = DataLoader([train_data[i] for i in idx_fit], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader([train_data[i] for i in idx_val], batch_size=BATCH_SIZE, shuffle=False)
    cal_loader = DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    set_seed(seed)
    model = CoDAGAT(N_FEATURES, n_ordinal_classes, n_cohorts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        p = epoch / MAX_EPOCHS
        run_epoch(model, fit_loader, device, optimizer, class_weights, n_ordinal_classes, p, train=True)
        _, val_f1 = run_epoch(model, val_loader, device, optimizer, class_weights, n_ordinal_classes, p, train=False)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)

    _, probs_cal, _, _ = extract_outputs(model, cal_loader, device)
    _, probs_test, _, _ = extract_outputs(model, test_loader, device)

    return dict(
        y_cal=cal_y["y_T1_significant"].to_numpy().astype(int), prob_cal=probs_cal,
        y_test=test_y["y_T1_significant"].to_numpy().astype(int), prob_test=probs_test,
        cal_cohort=fold_info["calibration_source"], test_cohort=fold_info["test_cohort"],
        best_val_f1=best_val_f1,
    )


# ==========================================================================
# THE decomposition function -- reused unchanged by every check, and the
# function B8 will use for the real analysis.
# ==========================================================================
def synth_metric_at_prevalence(y, pred, target_prevalence, metric_fn, rng, n_boot):
    """Bootstrap-resample (y, pred) to hit target_prevalence, recompute
    metric_fn, average over n_boot draws. This is the shared low-level
    primitive both decompose_drop() and within_site_slope() build on --
    it is NOT itself "the decomposition"; it just answers "what would this
    metric look like at prevalence p, holding this cohort's own per-class
    outcomes fixed."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n = len(y)
    n_pos = int(round(target_prevalence * n))
    n_pos = min(max(n_pos, 0), n)
    n_neg = n - n_pos
    vals = np.empty(n_boot)
    for b in range(n_boot):
        samp_pos = rng.choice(pos_idx, size=n_pos, replace=True) if n_pos > 0 else np.array([], dtype=int)
        samp_neg = rng.choice(neg_idx, size=n_neg, replace=True) if n_neg > 0 else np.array([], dtype=int)
        idx = np.concatenate([samp_pos, samp_neg])
        vals[b] = metric_fn(y[idx], pred[idx])
    return float(vals.mean())


def decompose_drop(y_cal, prob_cal, y_test, prob_test, metric_fn, rng, threshold=0.5, n_boot=50):
    """
    Two-factor (prevalence x residual) decomposition of the metric drop
    between a calibration cohort and a test cohort, Shapley-symmetrized
    over both possible orderings. Uses TEST LABELS -- this is the
    post-hoc analysis step (never training/threshold-tuning), which the
    project's own rules explicitly allow.

    Ordering 1 (prevalence attributed first, using TEST's own rates):
      prevalence_share_o1 = metric(test rates @ cal prevalence) - metric(test @ test prevalence)
      residual_share_o1   = metric(cal)                          - metric(test rates @ cal prevalence)
    Ordering 2 (rates attributed first, using CAL's own rates):
      residual_share_o2   = metric(cal rates @ test prevalence)  - metric(test @ test prevalence)
      prevalence_share_o2 = metric(cal)                          - metric(cal rates @ test prevalence)

    Each ordering sums exactly to total_drop = metric(cal) - metric(test);
    the Shapley value for 2 factors is the average across both orderings,
    which therefore also sums exactly to total_drop.
    """
    pred_cal = (prob_cal >= threshold).astype(int)
    pred_test = (prob_test >= threshold).astype(int)

    metric_cal = float(metric_fn(y_cal, pred_cal))
    metric_test_raw = float(metric_fn(y_test, pred_test))
    pi_cal = float(y_cal.mean())
    pi_test = float(y_test.mean())

    metric_test_at_cal_prev = synth_metric_at_prevalence(y_test, pred_test, pi_cal, metric_fn, rng, n_boot)
    prevalence_share_o1 = metric_test_at_cal_prev - metric_test_raw
    residual_share_o1 = metric_cal - metric_test_at_cal_prev

    metric_cal_at_test_prev = synth_metric_at_prevalence(y_cal, pred_cal, pi_test, metric_fn, rng, n_boot)
    residual_share_o2 = metric_cal_at_test_prev - metric_test_raw
    prevalence_share_o2 = metric_cal - metric_cal_at_test_prev

    prevalence_component = 0.5 * (prevalence_share_o1 + prevalence_share_o2)
    residual_component = 0.5 * (residual_share_o1 + residual_share_o2)
    total_drop = metric_cal - metric_test_raw

    return {
        "metric_cal": metric_cal, "metric_test_raw": metric_test_raw, "total_drop": total_drop,
        "pi_cal": pi_cal, "pi_test": pi_test,
        "prevalence_share_o1": prevalence_share_o1, "residual_share_o1": residual_share_o1,
        "prevalence_share_o2": prevalence_share_o2, "residual_share_o2": residual_share_o2,
        "prevalence_component": prevalence_component, "residual_component": residual_component,
        "order_swing_prevalence": abs(prevalence_share_o1 - prevalence_share_o2),
        "order_swing_residual": abs(residual_share_o1 - residual_share_o2),
    }


def within_site_slope(y, prob, threshold, metric_fn, prevalence_grid, rng, n_boot=50):
    """Ground-truth prevalence sensitivity: bootstrap-subsample ONE
    cohort's own predictions across a grid of target prevalences (pure
    label-marginal shift, nothing else changes) and fit a linear slope.
    Deliberately independent of decompose_drop -- this is what Check 2
    compares the decomposition against."""
    pred = (prob >= threshold).astype(int)
    metrics_at_p = [synth_metric_at_prevalence(y, pred, p, metric_fn, rng, n_boot) for p in prevalence_grid]
    slope, intercept = np.polyfit(prevalence_grid, metrics_at_p, deg=1)
    return float(slope), metrics_at_p


# ==========================================================================
# Check 1: placebo decomposition
# ==========================================================================
def run_placebo_check(fold_results, n_splits, seed):
    log("  Check 1: placebo decomposition (random within-cohort halves)")
    detail_rows = []
    for fold_id, res in fold_results.items():
        cohort = res["test_cohort"]
        y, prob = res["y_test"], res["prob_test"]
        n = len(y)
        log(f"    cohort {cohort} (fold {fold_id} test set, n={n}): {n_splits} random half-splits")
        for split_seed in range(n_splits):
            rng = np.random.default_rng(seed * 100000 + fold_id * 1000 + split_seed)
            idx = rng.permutation(n)
            half = n // 2
            idxA, idxB = idx[:half], idx[half:]
            for metric_name, metric_fn in METRICS.items():
                decomp_rng = np.random.default_rng(seed * 100000 + fold_id * 1000 + split_seed + 500)
                d = decompose_drop(y[idxA], prob[idxA], y[idxB], prob[idxB], metric_fn, decomp_rng)
                detail_rows.append({
                    "fold": fold_id, "cohort": cohort, "split_seed": split_seed, "metric": metric_name,
                    "prevalence_component": d["prevalence_component"], "residual_component": d["residual_component"],
                    "order_swing_prevalence": d["order_swing_prevalence"], "order_swing_residual": d["order_swing_residual"],
                    "total_drop": d["total_drop"],
                })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(OUT_DIR / "placebo_detail.csv", index=False)

    summary_rows = []
    fail_list = []
    for (cohort, metric_name), grp in detail_df.groupby(["cohort", "metric"]):
        for comp in ("prevalence_component", "residual_component"):
            mean_v = grp[comp].mean()
            lo, hi = percentile_ci(grp[comp].to_numpy())
            passed = lo <= 0 <= hi
            if not passed:
                fail_list.append(f"{cohort}/{metric_name}/{comp} (mean={mean_v:.4f}, CI=[{lo:.4f},{hi:.4f}])")
            summary_rows.append({
                "cohort": cohort, "metric": metric_name, "component": comp,
                "mean": mean_v, "ci_lo": lo, "ci_hi": hi, "pass": passed,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "placebo_summary.csv", index=False)

    check1_pass = len(fail_list) == 0
    log(f"    Check 1 result: {'PASS' if check1_pass else 'FAIL'} "
        f"({len(fail_list)} of {len(summary_df)} cohort/metric/component cells non-zero)")
    return check1_pass, fail_list, detail_df, summary_df


# ==========================================================================
# Check 2: internal consistency (within-site slope vs cross-site component)
# ==========================================================================
def run_consistency_check(fold_results, n_grid, n_boot_ci, pass_threshold, seed):
    log("  Check 2: internal consistency (within-site slope vs cross-site prevalence component)")
    rows = []
    for fold_id, res in fold_results.items():
        y_cal, prob_cal = res["y_cal"], res["prob_cal"]
        y_test, prob_test = res["y_test"], res["prob_test"]
        pi_cal, pi_test = float(y_cal.mean()), float(y_test.mean())
        grid = np.linspace(min(pi_cal, pi_test), max(pi_cal, pi_test), n_grid)

        for metric_name, metric_fn in METRICS.items():
            rng = np.random.default_rng(seed * 7919 + fold_id * 31 + hash(metric_name) % 1000)

            decomp = decompose_drop(y_cal, prob_cal, y_test, prob_test, metric_fn, rng)
            cross_site_component = decomp["prevalence_component"]

            slope_test, _ = within_site_slope(y_test, prob_test, 0.5, metric_fn, grid, rng)
            slope_cal, _ = within_site_slope(y_cal, prob_cal, 0.5, metric_fn, grid, rng)
            avg_slope = 0.5 * (slope_test + slope_cal)
            within_site_implied = avg_slope * (pi_cal - pi_test)
            diff = cross_site_component - within_site_implied

            log(f"    fold {fold_id} / {metric_name}: cross_site={cross_site_component:+.4f}  "
                f"within_site_implied={within_site_implied:+.4f}  diff={diff:+.4f}  "
                f"({n_boot_ci} bootstrap draws for CI)")

            diffs_boot = np.empty(n_boot_ci)
            n_cal, n_test = len(y_cal), len(y_test)
            for b in range(n_boot_ci):
                brng = np.random.default_rng(seed * 500000 + fold_id * 5000 + b)
                idx_cal_b = brng.integers(0, n_cal, size=n_cal)
                idx_test_b = brng.integers(0, n_test, size=n_test)
                y_cal_b, prob_cal_b = y_cal[idx_cal_b], prob_cal[idx_cal_b]
                y_test_b, prob_test_b = y_test[idx_test_b], prob_test[idx_test_b]

                decomp_b = decompose_drop(y_cal_b, prob_cal_b, y_test_b, prob_test_b, metric_fn, brng, n_boot=30)
                pi_cal_b, pi_test_b = float(y_cal_b.mean()), float(y_test_b.mean())
                grid_b = np.linspace(min(pi_cal_b, pi_test_b), max(pi_cal_b, pi_test_b), n_grid)
                slope_test_b, _ = within_site_slope(y_test_b, prob_test_b, 0.5, metric_fn, grid_b, brng, n_boot=30)
                slope_cal_b, _ = within_site_slope(y_cal_b, prob_cal_b, 0.5, metric_fn, grid_b, brng, n_boot=30)
                implied_b = 0.5 * (slope_test_b + slope_cal_b) * (pi_cal_b - pi_test_b)
                diffs_boot[b] = decomp_b["prevalence_component"] - implied_b

            ci_lo, ci_hi = percentile_ci(diffs_boot)
            agree = ci_lo <= 0 <= ci_hi

            rows.append({
                "fold": fold_id, "metric": metric_name,
                "pi_cal": pi_cal, "pi_test": pi_test,
                "cross_site_component": cross_site_component,
                "slope_test": slope_test, "slope_cal": slope_cal,
                "within_site_implied": within_site_implied,
                "diff": diff, "diff_ci_lo": ci_lo, "diff_ci_hi": ci_hi, "agree": agree,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "consistency_detail.csv", index=False)

    agree_frac = df["agree"].mean()
    check2_pass = agree_frac > pass_threshold
    log(f"    Check 2 result: {'PASS' if check2_pass else 'FAIL'} "
        f"({df['agree'].sum()}/{len(df)} fold/metric cells agree within CI, "
        f"fraction={agree_frac:.2f}, threshold={pass_threshold})")
    return check2_pass, agree_frac, df


# ==========================================================================
# Check 3: order-permutation sanity
# ==========================================================================
def run_order_check(placebo_detail_df, swing_threshold=0.2):
    log("  Check 3: order-permutation sanity")
    # relative swing = |share_o1 - share_o2| / |total_drop|, measured across the placebo draws
    # (8 real cohorts x many random splits) -- a large, cheap sample of how much the ordering
    # matters for this decomposition in general.
    swings = placebo_detail_df["order_swing_prevalence"].to_numpy()
    drops = placebo_detail_df["total_drop"].abs().to_numpy()
    denom = np.clip(drops, 1e-3, None)
    rel_swing = swings / denom
    frac_large_swing = float(np.mean(rel_swing > swing_threshold))
    shapley_required = frac_large_swing > 0.1

    log(f"    relative order-swing (|share_o1 - share_o2| / |total_drop|): "
        f"median={np.median(rel_swing):.3f}, fraction > {swing_threshold} = {frac_large_swing:.3f}")
    log(f"    Shapley symmetrization required: {'YES' if shapley_required else 'NO (orderings ~agree)'}")
    return shapley_required, frac_large_swing


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT Block B5 -- safety checks gating B7")
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-placebo-splits", type=int, default=15)
    parser.add_argument("--n-prevalence-grid", type=int, default=7)
    parser.add_argument("--n-boot-ci", type=int, default=100,
                         help="outer bootstrap draws for Check 2's difference CI (this is the slow part)")
    parser.add_argument("--consistency-pass-threshold", type=float, default=0.5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()) if args.folds is None \
        else [FOLDS_DIR / f for f in args.folds]

    log("CoDA-GAT Block B5 -- safety checks gating B7")
    log(f"folds: {[d.name for d in fold_dirs]}  seed={args.seed}")
    log("NOTE: README.txt has no pre-specified 'prevalence dose-response' protocol to confirm against "
        "(checked -- only the augmentation-multiplier sweep is documented there). The protocol below "
        "(bootstrap subsampling to a target prevalence, dose-response slope) was designed from the "
        "execution plan's description of B7/B8, using the real cross-cohort prevalence gaps in "
        "fold_assignment_lodo.csv.")

    edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")

    log("training one model per fold (real data, no augmentation, seed=%d) -- needed for sample-level "
        "calibration/test predictions, which no prior block persisted to disk..." % args.seed)
    fold_results = {}
    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        log(f"  fold {fold_id}: {fold_dir.name}")
        res = train_fold_model(fold_dir, edge_index, dev_lookup, args.seed)
        log(f"    best_val_f1={res['best_val_f1']:.4f}  "
            f"cal_cohort={res['cal_cohort']} (n={len(res['y_cal'])}, pos_rate={res['y_cal'].mean():.3f})  "
            f"test_cohort={res['test_cohort']} (n={len(res['y_test'])}, pos_rate={res['y_test'].mean():.3f})")
        fold_results[fold_id] = res

    check1_pass, check1_fails, placebo_detail_df, placebo_summary_df = run_placebo_check(
        fold_results, args.n_placebo_splits, args.seed
    )
    check2_pass, agree_frac, consistency_df = run_consistency_check(
        fold_results, args.n_prevalence_grid, args.n_boot_ci, args.consistency_pass_threshold, args.seed
    )
    shapley_required, frac_large_swing = run_order_check(placebo_detail_df)

    overall_pass = check1_pass and check2_pass

    lines = []
    lines.append("=" * 78)
    lines.append("CoDA-GAT Block B5 -- Safety Check Report")
    lines.append("=" * 78)
    lines.append("")
    lines.append("n = 8 LODO sites: all effect sizes and CIs below are UNDERPOWERED / SUGGESTIVE,")
    lines.append("not confirmatory, consistent with the project's own stated 8-site limitation.")
    lines.append("")
    lines.append(f"CHECK 1 (placebo decomposition): {'PASS' if check1_pass else 'FAIL'}")
    lines.append(f"  {len(placebo_summary_df)} cohort/metric/component cells checked; "
                 f"{len(check1_fails)} non-zero.")
    if check1_fails:
        lines.append("  Non-zero components (estimator bias suspected):")
        for f in check1_fails:
            lines.append(f"    - {f}")
    lines.append("  Per-cohort/metric detail: safety_checks_b5/placebo_summary.csv")
    lines.append("")
    lines.append(f"CHECK 2 (internal consistency): {'PASS' if check2_pass else 'FAIL'}")
    lines.append(f"  Cross-site vs within-site prevalence estimates agree within CI on "
                 f"{agree_frac:.0%} of fold/metric cells (threshold: {args.consistency_pass_threshold:.0%}).")
    if not check2_pass:
        lines.append("  Systematic divergence detected -- per the spec, this is itself a reportable")
        lines.append("  finding: confounding (platform/biology/annotation tangled with prevalence) is present.")
    lines.append("  Per-fold/metric detail: safety_checks_b5/consistency_detail.csv")
    lines.append("")
    lines.append(f"CHECK 3 (order-permutation): Shapley symmetrization required = "
                 f"{'YES' if shapley_required else 'NO'}")
    lines.append(f"  Fraction of placebo draws with |order swing|/|total drop| > 0.2: {frac_large_swing:.2f}")
    lines.append("")
    lines.append("=" * 78)
    verdict = "B5 GATE: PASS — safe to proceed to B7" if overall_pass else "B5 GATE: FAIL — do NOT run B7; diagnose first"
    lines.append(verdict)
    if not overall_pass:
        failed_checks = []
        if not check1_pass:
            failed_checks.append("Check 1 (placebo) — the decomposition estimator shows non-zero "
                                  "components on data with no true shift; it is biased.")
        if not check2_pass:
            failed_checks.append("Check 2 (consistency) — the cross-site prevalence estimate diverges "
                                  "from the independent within-site slope estimate; confounding is present.")
        for fc in failed_checks:
            lines.append(f"  FAILED: {fc}")
    lines.append("=" * 78)

    report = "\n".join(lines)
    log("")
    for line in lines:
        log(line)

    with open(OUT_DIR / "b5_summary.txt", "w") as f:
        f.write(report + "\n")

    log("")
    log(f"all outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
