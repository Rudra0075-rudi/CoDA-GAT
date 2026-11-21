"""
CoDA-GAT — Test B: the rate-consistency check.

Asks whether Test A's detector flags at a rate of the same order as the
~18.6% latent-noise-simulation-predicted label misassignment rate, and
whether adjacency enrichment (the F1/F2 concentration Test A found)
survives as the flagging rate is pushed higher via a looser uncertainty
gate.

Reuses Test A's flagging and enrichment logic UNCHANGED, imported
directly from test_a_adjacency.py -- compute_flags() and enrichment_test()
are not redefined or retuned anywhere in this file. The only new
computation is the calibration-cohort MC-dropout pools needed to derive
uncertainty thresholds beyond p40 (see module docstring context / the
Step 1 note in the accompanying report -- Test A computed these in
memory but never wrote them to disk).

STAGE RESOLUTION (2026-11 audit): enrichment_test now scores exact-stage
records only (see stage_labels.py), so the resolution column is attached to
Test A's saved values below. The historical test_b_rate/ outputs counted
grouped F0-F1 / F3-F4 records under their stored stage code and could not be
recomputed without re-running calibration MC-dropout, which was never saved.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader

from train_fold0_b1 import CoDAGAT, get_device, N_FEATURES, SHARED_DIR, BATCH_SIZE
from run_full_experiment_b7 import (
    mc_dropout_probs, load_edge_index_generic, load_fold_info, load_split_for_fold,
)
from test_a_adjacency import (
    compute_flags, enrichment_test, parse_fold_index,
    DISAGREEMENT_THRESHOLD, MODE, SEEDS, RESULTS_B7, FOLDS_DIR,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_A_DIR = REPO_ROOT / "test_a_adjacency"
OUT_DIR = REPO_ROOT / "test_b_rate"

EXTENDED_PERCENTILES = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
DISAGREEMENT_SWEEP = [0.4, 0.5, 0.6, 0.7]
PREDICTED_MISASSIGNMENT_RATE = 0.186
REFERENCE_PERCENTILE = 40  # Test A's actual reported operating point (4.9% flagging rate)


def log(msg):
    print(f"[TestB] {msg}", flush=True)


# ==========================================================================
# The one thing Test A didn't save: calibration-cohort MC-dropout pools
# ==========================================================================
def get_calibration_uncertainty_pools(fold_dirs, edge_index, dev_lookup, n_mc_passes):
    device = get_device()
    pools = {}
    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        fold_info = load_fold_info(fold_dir)
        test_cohort = fold_info["test_cohort"]
        cohort_to_idx = {c: i for i, c in enumerate(sorted(fold_info["train_cohorts"]))}

        cal_data, cal_ids, cal_y = load_split_for_fold(fold_dir, "calibration", edge_index, dev_lookup, cohort_to_idx)
        cal_loader = DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False)

        cal_u_seeds = []
        for seed in SEEDS:
            run_dir = RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / f"mode_{MODE}" / f"seed_{seed}"
            ckpt = torch.load(run_dir / "weights.pt", weights_only=False, map_location=device)
            model = CoDAGAT(N_FEATURES, ckpt["n_ordinal_classes"], ckpt["n_cohorts"]).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            cal_mc = mc_dropout_probs(model, cal_loader, device, n_passes=n_mc_passes)
            cal_u_seeds.append(cal_mc.std(axis=0))

        pools[fold_id] = np.concatenate(cal_u_seeds)
        log(f"    fold {fold_id}: calibration uncertainty pool recomputed (n={len(pools[fold_id])})")
    return pools


# ==========================================================================
# Step 2+3: extended sweep with adjacency enrichment at every threshold
# ==========================================================================
def extended_sweep(df, cal_pools, percentiles, predicted_rate):
    rows = []
    for q in percentiles:
        flagged, _ = compute_flags(df, cal_pools, q)  # Test A's rule, unchanged
        _, (chi2, p_chi2), _, (odds_ratio, p_fisher) = enrichment_test(df, flagged)  # Test A's test, unchanged

        n_flag = int(flagged.sum())
        rate = n_flag / len(df)
        rate_by_fold = df.assign(flagged=flagged).groupby("fold")["flagged"].mean()
        flagged_exact = flagged & (df["resolution"] == "exact").to_numpy()
        prop_f2 = float(df.loc[flagged_exact, "stage"].eq("F2").mean()) if flagged_exact.any() else float("nan")

        row = {
            "uncertainty_percentile": q, "n_flagged": n_flag, "pooled_flagging_rate": rate,
            "odds_ratio": odds_ratio, "fisher_p": p_fisher, "chi2": chi2, "chi2_p": p_chi2,
            "proportion_flags_on_F2": prop_f2,
            "abs_diff_from_predicted_rate": abs(rate - predicted_rate),
        }
        row.update({f"fold{f}_rate": r for f, r in rate_by_fold.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# Step 5: disagreement-threshold sensitivity (uncertainty gate held fixed)
# ==========================================================================
def disagreement_sensitivity(df, cal_pools, reference_percentile, disagreement_values):
    rows = []
    for d in disagreement_values:
        flagged, _ = compute_flags(df, cal_pools, reference_percentile, disagreement_thresh=d)
        _, _, _, (odds_ratio, p_fisher) = enrichment_test(df, flagged)
        n_flag = int(flagged.sum())
        rows.append({
            "disagreement_threshold": d,
            "is_nonarbitrary_default": (d == DISAGREEMENT_THRESHOLD),
            "uncertainty_percentile_held_fixed": reference_percentile,
            "n_flagged": n_flag, "pooled_flagging_rate": n_flag / len(df),
            "odds_ratio": odds_ratio, "fisher_p": p_fisher,
        })
    return pd.DataFrame(rows)


# ==========================================================================
# Step 4: operating curve plot
# ==========================================================================
def make_operating_curve_plot(sweep_df, predicted_rate, reference_percentile, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = sweep_df["pooled_flagging_rate"] * 100
    y = sweep_df["odds_ratio"]
    ax.plot(x, y, marker="o", color="#2b6cb0", zorder=3)
    for _, row in sweep_df.iterrows():
        ax.annotate(f"p{int(row['uncertainty_percentile'])}\np={row['fisher_p']:.3f}",
                     (row["pooled_flagging_rate"] * 100, row["odds_ratio"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=7)

    ax.axvline(predicted_rate * 100, color="#c53030", linestyle="--",
               label=f"predicted misassignment rate ({predicted_rate * 100:.1f}%)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="odds ratio = 1 (no enrichment)")

    ref_row = sweep_df[sweep_df["uncertainty_percentile"] == reference_percentile]
    if not ref_row.empty:
        ax.scatter(ref_row["pooled_flagging_rate"] * 100, ref_row["odds_ratio"], color="#dd6b20",
                   s=140, zorder=5, label=f"current operating point (p{reference_percentile})")

    ax.set_xlabel("Pooled flagging rate (%)")
    ax.set_ylabel("Adjacency enrichment (Fisher odds ratio)")
    ax.set_title("Test B: Flagging Rate vs Adjacency Enrichment")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ==========================================================================
# Verdict
# ==========================================================================
def determine_verdict(sweep_df, reference_percentile, predicted_rate, tolerance, alpha=0.05):
    ref = sweep_df[sweep_df["uncertainty_percentile"] == reference_percentile].iloc[0]
    current_rate, current_or, current_p = ref["pooled_flagging_rate"], ref["odds_ratio"], ref["fisher_p"]

    if current_rate > predicted_rate + tolerance:
        verdict = "IMPLAUSIBLY HIGH"
    elif abs(current_rate - predicted_rate) <= tolerance and current_p < alpha and current_or > 1:
        verdict = "CONSISTENT"
    else:
        verdict = "CONSERVATIVE"

    # recommended operating point: the loosest threshold (highest rate) that
    # STILL shows significant, positive-direction enrichment
    significant = sweep_df[(sweep_df["fisher_p"] < alpha) & (sweep_df["odds_ratio"] > 1)]
    if not significant.empty:
        recommended = significant.loc[significant["pooled_flagging_rate"].idxmax()]
    else:
        recommended = ref

    # does enrichment survive up to/near the predicted rate, or break down first?
    at_or_above_predicted = sweep_df[sweep_df["pooled_flagging_rate"] >= predicted_rate - tolerance]
    enrichment_survives_near_predicted = (
        not at_or_above_predicted.empty
        and (at_or_above_predicted["fisher_p"] < alpha).any()
        and (at_or_above_predicted["odds_ratio"] > 1).any()
    )

    return verdict, ref, recommended, enrichment_survives_near_predicted


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT Test B -- rate-consistency check")
    parser.add_argument("--n-mc-passes", type=int, default=30)
    parser.add_argument("--reference-percentile", type=int, default=REFERENCE_PERCENTILE)
    parser.add_argument("--predicted-rate", type=float, default=PREDICTED_MISASSIGNMENT_RATE)
    parser.add_argument("--tolerance", type=float, default=0.05,
                         help="absolute rate tolerance (e.g. 0.05 = 5pp) used to judge 'near' the predicted rate")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="where to write results; the historical test_b_rate/ is never overwritten")
    args = parser.parse_args()

    out_dir = args.out_dir
    if (out_dir / "threshold_sweep_extended.csv").exists():
        raise SystemExit(f"{out_dir / 'threshold_sweep_extended.csv'} already exists (historical record). "
                         f"Refusing to overwrite it; pass --out-dir <new directory>.")
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Test B: the rate-consistency check")
    log(f"Predicted latent misassignment rate: {args.predicted_rate:.1%}")
    log("Reusing Test A's flagging rule and enrichment test UNCHANGED (imported, not redefined).")

    log("loading Test A's saved per-sample values (test-side; already computed, not recomputed)...")
    df = pd.read_csv(TEST_A_DIR / "per_sample_values.csv")
    if "resolution" not in df.columns:
        labels_df = pd.read_parquet(SHARED_DIR / "labels.parquet")
        df["resolution"] = labels_df.loc[df["sample_id"], "resolution"].to_numpy()
    log(f"    {len(df)} test samples across {df['fold'].nunique()} folds")

    log("recomputing calibration-cohort MC-dropout uncertainty pools -- the one piece Test A "
        "computed but did not persist to disk. Forward passes only, no training.")
    edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")
    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir())
    cal_pools = get_calibration_uncertainty_pools(fold_dirs, edge_index, dev_lookup, args.n_mc_passes)

    log("Steps 2+3: extended threshold sweep with adjacency enrichment at every threshold...")
    sweep_df = extended_sweep(df, cal_pools, EXTENDED_PERCENTILES, args.predicted_rate)
    sweep_df.to_csv(out_dir / "threshold_sweep_extended.csv", index=False)
    for _, row in sweep_df.iterrows():
        log(f"    p{int(row['uncertainty_percentile']):>2}: rate={row['pooled_flagging_rate']:6.1%}  "
            f"OR={row['odds_ratio']:6.3f}  fisher_p={row['fisher_p']:.4f}  "
            f"%flags_on_F2={row['proportion_flags_on_F2']:.1%}")

    nearest = sweep_df.loc[sweep_df["abs_diff_from_predicted_rate"].idxmin()]
    log(f"nearest to predicted {args.predicted_rate:.1%}: p{int(nearest['uncertainty_percentile'])} "
        f"(rate={nearest['pooled_flagging_rate']:.1%}, OR={nearest['odds_ratio']:.3f}, "
        f"p={nearest['fisher_p']:.4f})")

    log("Step 4: building operating curve...")
    operating_curve_df = sweep_df[["uncertainty_percentile", "pooled_flagging_rate",
                                    "odds_ratio", "fisher_p", "n_flagged"]].copy()
    operating_curve_df.to_csv(out_dir / "operating_curve.csv", index=False)
    make_operating_curve_plot(sweep_df, args.predicted_rate, args.reference_percentile,
                               out_dir / "operating_curve.png")
    log(f"    saved {out_dir / 'operating_curve.png'}")

    log(f"Step 5: disagreement-threshold sensitivity (uncertainty held at p{args.reference_percentile})...")
    disagree_df = disagreement_sensitivity(df, cal_pools, args.reference_percentile, DISAGREEMENT_SWEEP)
    disagree_df.to_csv(out_dir / "disagreement_sensitivity.csv", index=False)
    for _, row in disagree_df.iterrows():
        tag = "non-arbitrary default" if row["is_nonarbitrary_default"] else "sensitivity check only"
        log(f"    disagreement>{row['disagreement_threshold']}: rate={row['pooled_flagging_rate']:.1%}  "
            f"OR={row['odds_ratio']:.3f}  p={row['fisher_p']:.4f}  ({tag})")

    verdict, ref_row, recommended_row, enrichment_survives = determine_verdict(
        sweep_df, args.reference_percentile, args.predicted_rate, args.tolerance)

    verdict_text = {
        "CONSISTENT": "TEST B: CONSISTENT -- flagging rate is of the same order as the predicted "
                      "~19% misassignment rate",
        "CONSERVATIVE": "TEST B: CONSERVATIVE -- the detector flags well below the predicted rate "
                        "(high precision, low recall); acceptable for triage, stated plainly",
        "IMPLAUSIBLY HIGH": "TEST B: IMPLAUSIBLY HIGH -- the flagging rate far exceeds the predicted "
                            "misassignment rate, suggesting the detector is measuring model error",
    }[verdict]

    summary_lines = [
        "=" * 78, "Test B: The Rate-Consistency Check", "=" * 78, "",
        f"Predicted latent-noise misassignment rate: {args.predicted_rate:.1%}",
        f"Current operating point: p{args.reference_percentile} calibration-uncertainty threshold "
        f"-> {ref_row['pooled_flagging_rate']:.1%} flagging rate, "
        f"odds_ratio={ref_row['odds_ratio']:.3f}, fisher_p={ref_row['fisher_p']:.4f}", "",
        "Extended threshold sweep (rate + adjacency enrichment at every threshold):",
        sweep_df[["uncertainty_percentile", "pooled_flagging_rate", "odds_ratio", "fisher_p",
                  "proportion_flags_on_F2"]].to_string(index=False), "",
        f"Nearest threshold to the predicted {args.predicted_rate:.1%} rate: "
        f"p{int(nearest['uncertainty_percentile'])} "
        f"(rate={nearest['pooled_flagging_rate']:.1%}, OR={nearest['odds_ratio']:.3f}, "
        f"p={nearest['fisher_p']:.4f})", "",
        f"Does adjacency enrichment survive at/near the predicted rate? "
        f"{'YES' if enrichment_survives else 'NO'} -- "
        + (f"enrichment (p<0.05, OR>1) is still present among thresholds reaching within "
           f"{args.tolerance:.0%} of {args.predicted_rate:.1%}, so the detector's signal does not "
           f"appear to be pure model-error noise even out there."
           if enrichment_survives else
           f"no threshold that reaches within {args.tolerance:.0%} of {args.predicted_rate:.1%} still "
           f"shows significant, positive-direction enrichment -- the extra flags picked up by "
           f"loosening the threshold that far look like ordinary model error, not label error."),
        "",
        f"Recommended operating point: p{int(recommended_row['uncertainty_percentile'])} "
        f"(rate={recommended_row['pooled_flagging_rate']:.1%}, OR={recommended_row['odds_ratio']:.3f}, "
        f"p={recommended_row['fisher_p']:.4f}) -- the loosest threshold tested that still shows "
        f"significant (p<0.05), positive-direction (OR>1) adjacency enrichment. This balances rate "
        f"against enrichment: looser thresholds than this either lose significance or invert direction.",
        "",
        "Disagreement-threshold sensitivity (uncertainty gate held fixed; 0.5 is the non-arbitrary "
        "default -- exactly where the model's argmax flips relative to the recorded label -- the "
        "others are sensitivity checks only, not alternative operating choices):",
        disagree_df.to_string(index=False), "",
        f"n=8 folds throughout: per-fold rates are in threshold_sweep_extended.csv; all cross-fold "
        f"statements here are descriptive and underpowered, not confirmatory.", "",
        verdict_text,
        "=" * 78,
    ]
    summary = "\n".join(summary_lines)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary + "\n")
    for line in summary_lines:
        log(line)

    log(f"outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
