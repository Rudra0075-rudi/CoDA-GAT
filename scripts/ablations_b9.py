"""
CoDA-GAT — Block B9: ablations (does each piece earn its place?).

Six training-based ablation rows, each trained under the same 8-fold LODO
protocol as B7 (real data, m=0, adversary OFF -- isolating one factor at a
time against the plain baseline model), reusing CoDAGAT / run_epoch /
extract_full unchanged from train_fold0_b1.py and run_full_experiment_b7.py:

  shuffled_graph          -- real edges, node identities randomly permuted
  random_graph            -- Erdos-Renyi graph, same edge count as real
  empty_graph             -- no edges (GATConv's own self-loops only)
  no_h3_deviation         -- L_H3 (deviation-from-healthy head) zero-weighted
  no_pathway_features     -- feature columns 2-7 (6 pathway z-scores) zeroed
  no_fold_fitted_features -- feature columns 17-25 (Group C) zeroed

The reference point for every comparison is B7's already-completed
mode="off", m=0 result (results_b7/master_results.csv) -- NOT retrained
here, since it already exists.

The 7th row (augmentation m-sweep {0,1,3,7}) is NOT retrained by this
script -- m=0 already exists in results_b7/, and m in {1,3,7} is produced
by re-running run_full_experiment_b7.py --m {1,3,7} (its output-path bug
was fixed alongside this block; see that file's main()). This script's
aggregation step reads whichever of those exist and reports "not yet run"
for the rest, rather than silently omitting the row.

All ablation-vs-baseline comparisons here are DESCRIPTIVE, per the
pre-registration (preregistration_b6.md): the only confirmatory,
pre-registered test is the B7 adversary-mode contrast, already reported.
Wilcoxon p-values are still shown alongside the effect sizes (same
convention as Block B4's H2 decision), explicitly labeled indicative, not
confirmatory, at n=8.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import train_fold0_b1 as b1
from train_fold0_b1 import (
    CoDAGAT, run_epoch, get_device, set_seed,
    N_NODES, N_FEATURES, N_EDGES_EXPECTED,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY, VAL_FRACTION, SEED,
    SHARED_DIR,
)
from run_full_experiment_b7 import extract_full, load_edge_index_generic, load_fold_info

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
RESULTS_B7 = REPO_ROOT / "results_b7"
OUT_ROOT = REPO_ROOT / "ablations_b9"

SEEDS = [42, 43, 44, 45, 46]

ABLATIONS = {
    "shuffled_graph": {"graph": "shuffled"},
    "random_graph": {"graph": "random"},
    "empty_graph": {"graph": "empty"},
    "no_h3_deviation": {"h3_weight": 0.0},
    "no_pathway_features": {"zero_cols": (2, 8)},
    "no_fold_fitted_features": {"zero_cols": (17, 26)},
}


def log(msg):
    print(f"[B9] {msg}", flush=True)


def parse_fold_index(dir_name: str) -> int:
    m = re.match(r"fold(\d+)_", dir_name)
    if not m:
        raise ValueError(f"cannot parse fold index from directory name: {dir_name}")
    return int(m.group(1))


def fmt_hms(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ==========================================================================
# Fold-agnostic loader with graph/feature-masking hooks (extends the
# established B3/B4/B5/B7 pattern by two optional parameters)
# ==========================================================================
def load_split_for_ablation(fold_dir, split, edge_index, dev_lookup, cohort_to_idx,
                             node_perm=None, zero_cols=None):
    split_dir = fold_dir / split
    X = np.load(split_dir / "X.npy")
    if node_perm is not None:
        X = X[:, node_perm, :]
    if zero_cols is not None:
        start, end = zero_cols
        X = X.copy()
        X[:, :, start:end] = 0.0

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


def make_random_edge_index(n_nodes, n_directed_edges, rng):
    edges = set()
    while len(edges) < n_directed_edges:
        i, j = rng.integers(0, n_nodes, size=2)
        if i != j:
            edges.add((int(i), int(j)))
    return torch.tensor(np.array(sorted(edges)).T, dtype=torch.long)


def make_empty_edge_index():
    return torch.zeros((2, 0), dtype=torch.long)


def build_edge_index_and_perm(cfg, real_edge_index, fold_id):
    graph = cfg.get("graph")
    if graph is None:
        return real_edge_index, None
    rng = np.random.default_rng(2000 + fold_id)
    if graph == "shuffled":
        return real_edge_index, rng.permutation(N_NODES)
    if graph == "random":
        return make_random_edge_index(N_NODES, real_edge_index.shape[1], rng), None
    if graph == "empty":
        return make_empty_edge_index(), None
    raise ValueError(f"unknown graph ablation: {graph}")


# ==========================================================================
# Per-(fold, ablation) setup -- built once, reused across the 5 seeds
# ==========================================================================
def setup_fold_ablation(fold_dir, cfg, real_edge_index, dev_lookup):
    fold_id = parse_fold_index(fold_dir.name)
    fold_info = load_fold_info(fold_dir)
    cohort_to_idx = {c: i for i, c in enumerate(sorted(fold_info["train_cohorts"]))}
    n_cohorts = len(cohort_to_idx)

    edge_index, node_perm = build_edge_index_and_perm(cfg, real_edge_index, fold_id)
    zero_cols = cfg.get("zero_cols")

    train_data, train_ids, train_y = load_split_for_ablation(
        fold_dir, "train", edge_index, dev_lookup, cohort_to_idx, node_perm, zero_cols)
    cal_data, cal_ids, cal_y = load_split_for_ablation(
        fold_dir, "calibration", edge_index, dev_lookup, cohort_to_idx, node_perm, zero_cols)
    test_data, test_ids, test_y = load_split_for_ablation(
        fold_dir, "test", edge_index, dev_lookup, cohort_to_idx, node_perm, zero_cols)

    n_ordinal_classes = int(np.nanmax(
        pd.concat([train_y["y_T2_ordinal"], cal_y["y_T2_ordinal"], test_y["y_T2_ordinal"]])
    )) + 1

    y1_train = train_y["y_T1_significant"].to_numpy()
    counts = np.bincount(y1_train.astype(int))
    class_weights = torch.tensor(len(y1_train) / (len(counts) * counts), dtype=torch.float32)

    # SAME base SEED as B7's fit/val split, for comparability with the baseline.
    strat_key = [f"{c}_{int(y)}" for c, y in zip(train_y["cohort"], y1_train)]
    idx_all = np.arange(len(train_data))
    idx_fit, idx_val = train_test_split(idx_all, test_size=VAL_FRACTION, random_state=SEED, stratify=strat_key)

    # m=0 always for ablations -- no on-the-fly augmentation needed, so (unlike
    # B7) the fit set is constant across epochs and can be built once here.
    fit_loader = DataLoader([train_data[i] for i in idx_fit], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader([train_data[i] for i in idx_val], batch_size=BATCH_SIZE, shuffle=False)
    cal_loader = DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return dict(
        fold_id=fold_id, fold_info=fold_info, n_ordinal_classes=n_ordinal_classes, n_cohorts=n_cohorts,
        class_weights=class_weights, fit_loader=fit_loader, val_loader=val_loader,
        cal_loader=cal_loader, test_loader=test_loader,
        y_test=test_y["y_T1_significant"].to_numpy().astype(int),
    )


def run_single_ablation(fold_setup, ablation_name, cfg, seed, run_dir):
    device = get_device()
    set_seed(seed)
    model = CoDAGAT(N_FEATURES, fold_setup["n_ordinal_classes"], fold_setup["n_cohorts"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_weights = fold_setup["class_weights"].to(device)

    # Adversary OFF for every ablation (matches B7's "off" baseline exactly,
    # so each row isolates ONE additional factor against that same baseline).
    original_h4, original_h3 = b1.LOSS_WEIGHTS["h4"], b1.LOSS_WEIGHTS["h3"]
    b1.LOSS_WEIGHTS["h4"] = 0.0
    b1.LOSS_WEIGHTS["h3"] = cfg.get("h3_weight", original_h3)
    try:
        best_val_f1, best_state, bad = -1.0, None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            p = epoch / MAX_EPOCHS
            run_epoch(model, fold_setup["fit_loader"], device, optimizer, class_weights,
                      fold_setup["n_ordinal_classes"], p, train=True)
            _, val_f1 = run_epoch(model, fold_setup["val_loader"], device, optimizer, class_weights,
                                   fold_setup["n_ordinal_classes"], p, train=False)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= PATIENCE:
                break
    finally:
        b1.LOSS_WEIGHTS["h4"] = original_h4
        b1.LOSS_WEIGHTS["h3"] = original_h3

    model.load_state_dict(best_state)
    _, probs_test, preds_test, _ = extract_full(model, fold_setup["test_loader"], device)
    y_test = fold_setup["y_test"]
    p1_test = probs_test[:, 1]

    auc = float(roc_auc_score(y_test, p1_test)) if len(np.unique(y_test)) > 1 else float("nan")
    macro_f1 = float(f1_score(y_test, preds_test, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_test, preds_test))

    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "probs_test.npy", probs_test)
    np.save(run_dir / "predictions_test.npy", preds_test)
    run_metrics = {
        "fold": fold_setup["fold_id"], "ablation": ablation_name, "seed": seed,
        "auc": auc, "macro_f1": macro_f1, "balanced_acc": bal_acc, "best_val_f1": best_val_f1,
    }
    with open(run_dir / "run_metrics.json", "w") as f:
        json.dump(run_metrics, f, indent=2)
    (run_dir / "DONE.marker").touch()
    return run_metrics


# ==========================================================================
# Aggregation + comparison against the B7 "off", m=0 baseline
# ==========================================================================
def collect_ablation_results():
    rows = []
    for marker in OUT_ROOT.glob("fold*/*/seed_*/DONE.marker"):
        with open(marker.parent / "run_metrics.json") as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def load_baseline():
    df = pd.read_csv(RESULTS_B7 / "master_results.csv")
    return df[(df["mode"] == "off") & (df["m"] == 0)]


def compare_to_baseline(ablation_df, baseline_df, metric_cols=("auc", "macro_f1", "balanced_acc")):
    lines = []
    baseline_by_fold = baseline_df.groupby("fold")[list(metric_cols)].mean()

    for ablation_name in sorted(ablation_df["ablation"].unique()):
        sub = ablation_df[ablation_df["ablation"] == ablation_name]
        ablation_by_fold = sub.groupby("fold")[list(metric_cols)].mean()
        lines.append(f"\n{ablation_name} vs baseline (off, m=0):")
        for metric in metric_cols:
            joined = pd.concat([ablation_by_fold[metric], baseline_by_fold[metric]], axis=1,
                                keys=["ablation", "baseline"]).dropna()
            if len(joined) < 2:
                lines.append(f"  {metric}: SKIPPED -- fewer than 2 complete folds so far.")
                continue
            diff = joined["ablation"] - joined["baseline"]
            try:
                _, p = wilcoxon(joined["ablation"], joined["baseline"])
            except ValueError:
                p = float("nan")
            lines.append(f"  {metric}: n={len(joined)} folds, mean diff={diff.mean():+.4f} "
                        f"(baseline mean={joined['baseline'].mean():.4f}), Wilcoxon p={p:.4f} "
                        f"(descriptive/indicative only, n={len(joined)} -- not the pre-registered test)")
    return "\n".join(lines)


def compare_augmentation_sweep(metric_cols=("auc", "macro_f1", "balanced_acc")):
    lines = ["\naugmentation m-sweep (0,1,3,7) vs baseline (off, m=0):"]
    rows_by_m = {0: load_baseline()}
    for m in (1, 3, 7):
        path = REPO_ROOT / f"results_b7_m{m}" / "master_results.csv"
        if path.exists():
            df = pd.read_csv(path)
            rows_by_m[m] = df[(df["mode"] == "off") & (df["m"] == m)]
        else:
            lines.append(f"  m={m}: NOT YET RUN -- run: "
                        f"/opt/anaconda3/envs/dagat/bin/python scripts/run_full_experiment_b7.py --m {m} --modes off")

    present_ms = sorted(rows_by_m.keys())
    if len(present_ms) < 2:
        lines.append("  (need at least m=0 and one other m to compare -- see commands above)")
        return "\n".join(lines)

    for metric in metric_cols:
        by_m_fold = {m: rows_by_m[m].groupby("fold")[metric].mean() for m in present_ms}
        summary = ", ".join(f"m={m}: {by_m_fold[m].mean():.4f}" for m in present_ms)
        lines.append(f"  {metric} (mean across available folds) -- {summary}")
    return "\n".join(lines)


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT Block B9 -- ablations")
    parser.add_argument("--ablations", nargs="*", default=list(ABLATIONS.keys()), choices=list(ABLATIONS.keys()))
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        log("aggregate-only: scanning ablations_b9/ for completed runs...")
        ablation_df = collect_ablation_results()
        if not ablation_df.empty:
            ablation_df.to_csv(OUT_ROOT / "master_ablation_results.csv", index=False)
        report = compare_to_baseline(ablation_df, load_baseline()) if not ablation_df.empty else "(no completed runs yet)"
        report += "\n" + compare_augmentation_sweep()
        log(report)
        with open(OUT_ROOT / "ablation_report.txt", "w") as f:
            f.write(report + "\n")
        return

    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()) if args.folds is None \
        else [FOLDS_DIR / f for f in args.folds]

    total_runs = len(fold_dirs) * len(args.ablations) * len(args.seeds)
    log("CoDA-GAT Block B9 -- ablations")
    log(f"folds={[d.name for d in fold_dirs]}  ablations={args.ablations}  seeds={args.seeds}")
    log(f"total planned runs this invocation: {total_runs}")
    log("Resumable via DONE.marker; a failed run is logged and skipped, not fatal.")

    real_edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")

    start_time = time.time()
    run_counter, completed = 0, 0

    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        for ablation_name in args.ablations:
            cfg = ABLATIONS[ablation_name]
            log(f"=== fold {fold_id} / {ablation_name} -- one-time setup ===")
            fold_setup = setup_fold_ablation(fold_dir, cfg, real_edge_index, dev_lookup)
            cohort = fold_setup["fold_info"]["test_cohort"]

            for seed in args.seeds:
                run_counter += 1
                run_dir = OUT_ROOT / f"fold{fold_id}_{cohort}" / ablation_name / f"seed_{seed}"

                if (run_dir / "DONE.marker").exists():
                    log(f"[{run_counter}/{total_runs}] fold={fold_id} ablation={ablation_name} seed={seed} "
                        f"-- already done, skipping")
                    continue

                elapsed = time.time() - start_time
                eta_str = f"  ETA={fmt_hms(elapsed / completed * (total_runs - run_counter + 1))}" if completed else ""
                log(f"[{run_counter}/{total_runs}] fold={fold_id} ablation={ablation_name} seed={seed} "
                    f"-- starting  elapsed={fmt_hms(elapsed)}{eta_str}")

                try:
                    rm = run_single_ablation(fold_setup, ablation_name, cfg, seed, run_dir)
                    completed += 1
                    log(f"    done: AUC={rm['auc']:.4f}  macro_f1={rm['macro_f1']:.4f}  "
                        f"balanced_acc={rm['balanced_acc']:.4f}")
                except Exception:
                    log(f"    FAILED fold={fold_id} ablation={ablation_name} seed={seed} -- continuing.")
                    traceback.print_exc()

    log("all planned runs attempted. Building comparison report...")
    ablation_df = collect_ablation_results()
    if not ablation_df.empty:
        ablation_df.to_csv(OUT_ROOT / "master_ablation_results.csv", index=False)
        report = compare_to_baseline(ablation_df, load_baseline())
    else:
        report = "(no completed runs yet -- nothing to compare)"
    report += "\n" + compare_augmentation_sweep()
    for line in report.split("\n"):
        log(line)
    with open(OUT_ROOT / "ablation_report.txt", "w") as f:
        f.write(report + "\n")

    log(f"total elapsed: {fmt_hms(time.time() - start_time)}  completed this session: {completed}/{total_runs}")
    log(f"outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()
