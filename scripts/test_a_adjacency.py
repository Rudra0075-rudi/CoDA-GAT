"""
CoDA-GAT — Test A: the adjacency check.

Decides whether "the model confidently disagrees with a recorded
fibrosis stage" behaves like a LABEL-ERROR signal (flagged samples
cluster at the F1/F2 boundary, where real pathologist disagreement
lives) or a MODEL-ERROR signal (flags scatter across distant stages
F0/F4). This is a cheap test on data that already exists -- no
retraining, no new model code. It reuses B7's saved checkpoints and test
predictions directly, and runs one extra cheap (forward-pass-only)
MC-dropout pass per checkpoint on the CALIBRATION cohort, since B7 only
ever ran MC-dropout on test.

CRITICAL: the flag (disagreement AND low uncertainty) is computed from:
  - p_i: the model's own P(F2+) on the test sample's features (the model
    never saw test labels during training)
  - u_i: MC-dropout uncertainty on the test sample's features
  - a "low uncertainty" threshold derived from the CALIBRATION cohort's
    own MC-dropout distribution, per fold -- never from the held-out
    test cohort's own uncertainty distribution
The test cohort's fine-grained ordinal STAGE label (F0-F4) -- the thing
this test checks for enrichment -- is never used to construct the flag,
only to evaluate it afterward.

STAGE RESOLUTION (2026-11 audit): grouped F0-F1 records are stored as "F0"
and grouped F3-F4 records as "F3". The enrichment test now uses exact-stage
records only (stage_labels.enrichment_test); grouped records keep their valid
binary label for flagging and are tabulated separately. The historical
summary in test_a_adjacency/ counted them under their stored stage code.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader

from train_fold0_b1 import CoDAGAT, get_device, N_FEATURES, SHARED_DIR, BATCH_SIZE
from run_full_experiment_b7 import (
    mc_dropout_probs, load_edge_index_generic, load_fold_info, load_split_for_fold,
)
from stage_labels import enrichment_test, grouped_table, ADJACENT_STAGES, ALL_STAGES  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
RESULTS_B7 = REPO_ROOT / "results_b7"
OUT_DIR = REPO_ROOT / "test_a_adjacency"

MODE = "off"
SEEDS = [42, 43, 44, 45, 46]
THRESHOLD_PERCENTILES = [10, 15, 20, 25, 30, 40]
PRIMARY_PERCENTILE = 20
DISAGREEMENT_THRESHOLD = 0.5  # |p - y| > 0.5 <=> the model's argmax prediction flips the label
SENSIBLE_RATE_LO, SENSIBLE_RATE_HI = 0.15, 0.25


def log(msg):
    print(f"[TestA] {msg}", flush=True)


def parse_fold_index(dir_name: str) -> int:
    m = re.match(r"fold(\d+)_", dir_name)
    if not m:
        raise ValueError(f"cannot parse fold index from directory name: {dir_name}")
    return int(m.group(1))


# ==========================================================================
# Per-fold: reload checkpoints, get calibration MC-dropout + saved test outputs
# ==========================================================================
def process_fold(fold_dir, edge_index, dev_lookup, labels_df, n_mc_passes):
    fold_id = parse_fold_index(fold_dir.name)
    fold_info = load_fold_info(fold_dir)
    cohort_to_idx = {c: i for i, c in enumerate(sorted(fold_info["train_cohorts"]))}
    test_cohort = fold_info["test_cohort"]

    cal_data, cal_ids, cal_y = load_split_for_fold(fold_dir, "calibration", edge_index, dev_lookup, cohort_to_idx)
    cal_loader = DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False)
    test_sample_ids = pd.read_csv(fold_dir / "test" / "sample_ids.csv")["sample_id"].tolist()

    device = get_device()
    p_test_seeds, u_test_seeds, cal_u_seeds = [], [], []

    for seed in SEEDS:
        run_dir = RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / f"mode_{MODE}" / f"seed_{seed}"
        ckpt = torch.load(run_dir / "weights.pt", weights_only=False, map_location=device)
        model = CoDAGAT(N_FEATURES, ckpt["n_ordinal_classes"], ckpt["n_cohorts"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        # Calibration-only MC-dropout -- B7 never saved this, and per the
        # critical rule it must come from calibration, never from test.
        cal_mc = mc_dropout_probs(model, cal_loader, device, n_passes=n_mc_passes)  # (n_passes, n_cal)
        cal_u_seeds.append(cal_mc.std(axis=0))

        probs_test = np.load(run_dir / "probs_test.npy")           # (n_test, 2) -- already saved by B7
        mc_test = np.load(run_dir / "mc_dropout_probs_test.npy")   # (n_passes, n_test) -- already saved by B7
        p_test_seeds.append(probs_test[:, 1])
        u_test_seeds.append(mc_test.std(axis=0))

    p_test = np.mean(p_test_seeds, axis=0)          # seed-averaged P(F2+)
    u_test = np.mean(u_test_seeds, axis=0)           # seed-averaged MC-dropout std
    cal_uncertainty_pool = np.concatenate(cal_u_seeds)  # pooled across the 5 seeds' calibration samples

    fold_labels = labels_df.loc[test_sample_ids]
    y_test = fold_labels["y_T1_significant"].to_numpy().astype(int)
    stage_test = fold_labels["stage"].to_numpy()
    resolution_test = fold_labels["resolution"].to_numpy()
    disagreement = np.abs(p_test - y_test)

    df = pd.DataFrame({
        "fold": fold_id, "test_cohort": test_cohort, "sample_id": test_sample_ids,
        "p_f2plus": p_test, "uncertainty": u_test, "y_true": y_test, "stage": stage_test,
        "resolution": resolution_test, "disagreement": disagreement,
    })
    return df, cal_uncertainty_pool


# ==========================================================================
# Flagging + enrichment test
# ==========================================================================
def compute_flags(df, cal_uncertainty_by_fold, percentile, disagreement_thresh=DISAGREEMENT_THRESHOLD):
    """Applies a per-fold, calibration-derived uncertainty threshold (never
    derived from the test cohort's own distribution) to flag samples."""
    flagged = np.zeros(len(df), dtype=bool)
    thresholds = {}
    for fold_id, cal_pool in cal_uncertainty_by_fold.items():
        thresh = float(np.percentile(cal_pool, percentile))
        thresholds[fold_id] = thresh
        mask = (df["fold"] == fold_id).to_numpy()
        flagged[mask] = (df.loc[mask, "disagreement"].to_numpy() > disagreement_thresh) & \
                        (df.loc[mask, "uncertainty"].to_numpy() <= thresh)
    return flagged, thresholds


def threshold_sensitivity(df, cal_uncertainty_by_fold):
    rows = []
    for q in THRESHOLD_PERCENTILES:
        flagged, _ = compute_flags(df, cal_uncertainty_by_fold, q)
        rate_by_fold = df.assign(flagged=flagged).groupby("fold")["flagged"].mean()
        row = {"uncertainty_percentile": q, "pooled_flagging_rate": float(flagged.mean())}
        row.update({f"fold{f}_rate": r for f, r in rate_by_fold.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT Test A -- the adjacency check")
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--primary-percentile", type=int, default=PRIMARY_PERCENTILE, choices=THRESHOLD_PERCENTILES)
    parser.add_argument("--n-mc-passes", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="where to write results; the historical test_a_adjacency/ is never overwritten")
    args = parser.parse_args()

    out_dir = args.out_dir
    if (out_dir / "per_sample_flags.csv").exists():
        raise SystemExit(f"{out_dir / 'per_sample_flags.csv'} already exists (historical record). "
                         f"Refusing to overwrite it; pass --out-dir <new directory>. To re-test saved "
                         f"flags without MC-dropout inference use corrections_2026-11/run_analysis_only.sh.")
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Test A: the adjacency check")
    log("Hypothesis: confident model/label disagreement clusters at F1/F2 (real pathologist")
    log("disagreement is adjacent-stage), not scattered across F0/F3/F4 (which would mean the")
    log("flag is catching model error, not label error).")

    edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")
    labels_df = pd.read_parquet(SHARED_DIR / "labels.parquet")

    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()) if args.folds is None \
        else [FOLDS_DIR / f for f in args.folds]

    all_rows, cal_uncertainty_by_fold = [], {}
    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        log(f"=== fold {fold_id}: {fold_dir.name} ===")
        df_fold, cal_pool = process_fold(fold_dir, edge_index, dev_lookup, labels_df, args.n_mc_passes)
        all_rows.append(df_fold)
        cal_uncertainty_by_fold[fold_id] = cal_pool
        log(f"    n_test={len(df_fold)}  mean disagreement={df_fold['disagreement'].mean():.4f}  "
            f"calibration uncertainty pool size={len(cal_pool)}")

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out_dir / "per_sample_values.csv", index=False)
    log(f"per-sample values saved ({len(df)} test samples across {len(fold_dirs)} folds)")

    log("Step 4: threshold sensitivity...")
    sens_df = threshold_sensitivity(df, cal_uncertainty_by_fold)
    sens_df.to_csv(out_dir / "threshold_sensitivity.csv", index=False)
    for _, row in sens_df.iterrows():
        log(f"    uncertainty <= calibration p{int(row['uncertainty_percentile'])}: "
            f"pooled flagging rate = {row['pooled_flagging_rate']:.1%}")

    log(f"primary threshold: calibration uncertainty percentile {args.primary_percentile}")
    flagged, thresholds = compute_flags(df, cal_uncertainty_by_fold, args.primary_percentile)
    df["flagged"] = flagged
    df.to_csv(out_dir / "per_sample_flags.csv", index=False)
    n_flagged = int(flagged.sum())
    rate = n_flagged / len(df)
    log(f"    {n_flagged}/{len(df)} samples flagged ({rate:.1%})")

    table_5, (chi2, p_chi2), table_2, (odds_ratio, p_fisher) = enrichment_test(df, flagged)
    n_exact = int((df["resolution"] == "exact").sum())
    g_table = grouped_table(df, flagged)
    table_5.to_csv(out_dir / "stage_distribution_5class.csv")
    g_table.to_csv(out_dir / "grouped_label_flags.csv")
    pd.DataFrame(table_2, index=["flagged", "unflagged"], columns=["adjacent_F1F2", "distant_F0F3F4"]).to_csv(
        out_dir / "adjacency_2x2_table.csv")

    log("")
    log("STAGE DISTRIBUTION (flagged vs unflagged):")
    log(table_5.to_string())
    log(f"2x5 chi-square test of independence: chi2={chi2:.3f}, p={p_chi2:.4f}")
    log(f"2x2 Fisher's exact (flagged enriched at adjacent F1/F2 vs distant F0/F3/F4, one-sided): "
        f"odds_ratio={odds_ratio:.3f}, p={p_fisher:.4f}")

    verdict_pass = (p_fisher < 0.05) and (odds_ratio > 1)
    if SENSIBLE_RATE_LO <= rate <= SENSIBLE_RATE_HI:
        rate_note = f"Flagging rate {rate:.1%} is within the sensible 15-25% band."
    elif rate > SENSIBLE_RATE_HI:
        rate_note = (f"CAVEAT: flagging rate {rate:.1%} is ABOVE the sensible 15-25% band -- "
                     f"at this threshold the flag may be catching ordinary model error, not "
                     f"specifically label error. Interpret the verdict below with that in mind, "
                     f"or re-run with a lower --primary-percentile.")
    else:
        rate_note = (f"CAVEAT: flagging rate {rate:.1%} is BELOW the sensible 15-25% band -- "
                     f"the flagged set may be too small/strict to draw a reliable conclusion. "
                     f"Check threshold_sensitivity.csv or re-run with a higher --primary-percentile.")

    verdict = ("TEST A: PASS -- flags concentrate at F1/F2, consistent with label error"
               if verdict_pass else
               "TEST A: FAIL -- flags do not significantly concentrate at F1/F2; "
               "label-error interpretation not supported")

    summary_lines = [
        "=" * 78, "Test A: The Adjacency Check", "=" * 78, "",
        f"Model: mode='{MODE}', {len(SEEDS)} seeds averaged, {len(fold_dirs)} folds.",
        f"Flag definition: disagreement=|p(F2+) - y_true| > {DISAGREEMENT_THRESHOLD} AND "
        f"MC-dropout uncertainty <= the {args.primary_percentile}th percentile of that fold's "
        f"CALIBRATION cohort's own MC-dropout distribution (never the test cohort's).", "",
        f"Flagged {n_flagged}/{len(df)} test samples ({rate:.1%}). {rate_note}", "",
        f"Stage tests use the {n_exact} exact-stage records only; the {len(df) - n_exact} grouped-label",
        "records (stored as F0 for F0-F1, F3 for F3-F4) are excluded from them and tabulated here:",
        g_table.to_string(), "",
        "Stage distribution, exact-stage records (flagged vs unflagged):", table_5.to_string(), "",
        f"2x5 chi-square (all 5 stages): chi2={chi2:.3f}, p={p_chi2:.4f}",
        f"2x2 Fisher's exact (adjacent F1/F2 vs distant F0/F3/F4, one-sided, "
        f"H1: flagged more likely adjacent): odds_ratio={odds_ratio:.3f}, p={p_fisher:.4f}", "",
        verdict,
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
