"""
CoDA-GAT — Block B4: H2 (ordinal-stage head) keep-or-kill decision.

Runs a controlled A/B comparison across all 8 LODO folds, single seed 42,
on real data only (B2 on-the-fly augmentation OFF, to isolate the H2
effect from augmentation noise):

  Arm A (H2 ON):  loss = L_H1 + 0.5*L_H2 + 0.3*L_H3 + 0.1*L_H4  (as-is)
  Arm B (H2 OFF): loss = L_H1 +           0.3*L_H3 + 0.1*L_H4

Both arms use the exact same CoDAGAT class, same construction order, same
seed -> byte-identical initial weights and identical fit/val splits. The
ONLY difference is train_fold0_b1.LOSS_WEIGHTS["h2"] (0.5 vs 0.0), applied
by temporarily overriding that module-level constant before each arm's
training loop and calling B1's unmodified run_epoch. This is deliberately
NOT implemented by constructing a model class with the H2 head physically
removed: doing so would change the RNG draw sequence used to initialize
the H3/H4 layers (since they'd be constructed in a different position
relative to a missing H2 layer), confounding the comparison with an
unintended difference in H3/H4 initialization. Zero-weighting L_H2 gives
mathematically identical gradients to "H2 absent" for every other part of
the network, without that confound.

DECODER CORRECTION (2026-11 audit): the h2_ordinal_accuracy / h2_ordinal_mae
columns of the historical h2_decision/per_fold.csv (folds 0-4 only) were
computed with the pre-fix corn_predict_stage, which thresholded the CORN
conditional probabilities directly instead of their cumulative product. The
logits and checkpoints behind that file were never saved, so those two
columns cannot be corrected and must not be reported. This script now
inherits the fixed decoder; it still TRAINS models, so it is not part of the
analysis-only reproduction, and it refuses to overwrite the historical file.
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from scipy.stats import wilcoxon

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import train_fold0_b1 as b1
from train_fold0_b1 import (
    CoDAGAT, run_epoch, extract_outputs, corn_predict_stage,
    get_device, set_seed,
    N_NODES, N_FEATURES, N_EDGES_EXPECTED,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY, VAL_FRACTION, SEED,
    SHARED_DIR,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
OUT_DIR = REPO_ROOT / "h2_decision"

H2_WEIGHT_ON = 0.5
H2_WEIGHT_OFF = 0.0


def log(msg):
    print(f"[B4] {msg}", flush=True)


def parse_fold_index(dir_name: str) -> int:
    m = re.match(r"fold(\d+)_", dir_name)
    if not m:
        raise ValueError(f"cannot parse fold index from directory name: {dir_name}")
    return int(m.group(1))


def compute_h1_metrics(y_true, probs, preds):
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    macro_f1 = f1_score(y_true, preds, average="macro")
    bal_acc = balanced_accuracy_score(y_true, preds)
    return auc, macro_f1, bal_acc


# --------------------------------------------------------------------------
# Fold-agnostic data loading -- same construction as B1's load_split,
# parameterized by fold_dir (B1 hardcodes fold0 only).
# --------------------------------------------------------------------------
def load_edge_index_generic():
    edges = pd.read_csv(SHARED_DIR / "graph_edges.csv")
    assert len(edges) == N_EDGES_EXPECTED, f"expected {N_EDGES_EXPECTED} edges, found {len(edges)}"
    return torch.tensor(edges[["src_idx", "dst_idx"]].to_numpy().T, dtype=torch.long)


def load_fold_info(fold_dir: Path):
    import json
    with open(fold_dir / "fold_info.json") as f:
        return json.load(f)


def load_split_for_fold(fold_dir, split, edge_index, dev_lookup, cohort_to_idx):
    split_dir = fold_dir / split
    X = np.load(split_dir / "X.npy")
    assert X.shape[1:] == (N_NODES, N_FEATURES)

    y = pd.read_csv(split_dir / "y.csv", index_col=0)
    sample_ids = pd.read_csv(split_dir / "sample_ids.csv")["sample_id"].tolist()
    assert list(y.index) == sample_ids

    dev_scores = dev_lookup.loc[sample_ids, "deviation_score"].to_numpy(dtype=np.float32)
    y1 = y["y_T1_significant"].to_numpy(dtype=np.float32)
    y2_raw = y["y_T2_ordinal"].to_numpy(dtype=np.float32)  # NaN = grouped label, masked below
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


# --------------------------------------------------------------------------
# One arm, one fold
# --------------------------------------------------------------------------
def train_arm(fold_dir, edge_index, dev_lookup, h2_weight, seed):
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

    # SAME fit/val split for both arms (same seed, same stratify key) -- the
    # comparison must never accidentally differ in which rows get trained on.
    strat_key = [f"{c}_{int(y)}" for c, y in zip(train_y["cohort"], y1_train)]
    idx_all = np.arange(len(train_data))
    idx_fit, idx_val = train_test_split(idx_all, test_size=VAL_FRACTION, random_state=seed, stratify=strat_key)

    fit_loader = DataLoader([train_data[i] for i in idx_fit], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader([train_data[i] for i in idx_val], batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    set_seed(seed)  # identical construction order/RNG state for both arms
    model = CoDAGAT(N_FEATURES, n_ordinal_classes, n_cohorts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    original_h2_weight = b1.LOSS_WEIGHTS["h2"]
    b1.LOSS_WEIGHTS["h2"] = h2_weight
    try:
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
    finally:
        b1.LOSS_WEIGHTS["h2"] = original_h2_weight  # never leak the override across arms/folds

    model.load_state_dict(best_state)

    emb_test, probs_test, preds_test, _ = extract_outputs(model, test_loader, device)
    y1_test = test_y["y_T1_significant"].to_numpy().astype(int)
    auc, macro_f1, bal_acc = compute_h1_metrics(y1_test, probs_test, preds_test)

    ordinal = None
    if h2_weight > 0:
        model.eval()
        all_logits2, all_true2 = [], []
        with torch.no_grad():
            for batch in test_loader:
                mask = batch.y2_mask.view(-1)
                if not mask.any():
                    continue
                y2_true_batch = batch.y2.view(-1)[mask].clone()
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch, grl_lambd=0.0)
                mask_dev = mask.to(device)
                all_logits2.append(out["h2_logits"][mask_dev].cpu())
                all_true2.append(y2_true_batch)
        if all_logits2:
            logits2 = torch.cat(all_logits2)
            true2 = torch.cat(all_true2).long()
            pred_stage = corn_predict_stage(logits2)
            ordinal = {
                "ordinal_accuracy": float((pred_stage == true2).float().mean().item()),
                "ordinal_mae": float((pred_stage - true2).abs().float().mean().item()),
                "n_exact_label_test": int(len(true2)),
            }

    return {
        "auc": auc, "macro_f1": macro_f1, "balanced_acc": bal_acc,
        "best_val_f1": best_val_f1, "ordinal": ordinal,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT Block B4 -- H2 keep-or-kill decision")
    parser.add_argument("--folds", nargs="*", default=None, help="fold dir names (default: all 8)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="where to write results; the historical h2_decision/ is never overwritten")
    args = parser.parse_args()

    out_dir = args.out_dir
    if (out_dir / "per_fold.csv").exists():
        raise SystemExit(f"{out_dir / 'per_fold.csv'} already exists (historical record, pre-fix CORN "
                         f"decoder). Refusing to overwrite it; pass --out-dir <new directory>.")
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()) if args.folds is None \
        else [FOLDS_DIR / f for f in args.folds]

    log("CoDA-GAT Block B4 -- H2 keep-or-kill decision")
    log(f"folds: {[d.name for d in fold_dirs]}  seed={args.seed}  augmentation: OFF (real data only)")

    edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")

    per_fold_rows = []

    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        log(f"=== fold {fold_id}: {fold_dir.name} ===")

        log("  Arm A: H2 ON")
        result_on = train_arm(fold_dir, edge_index, dev_lookup, H2_WEIGHT_ON, args.seed)
        log(f"    test AUC={result_on['auc']:.4f}  macro_f1={result_on['macro_f1']:.4f}  "
            f"bal_acc={result_on['balanced_acc']:.4f}  best_val_f1={result_on['best_val_f1']:.4f}")
        if result_on["ordinal"]:
            log(f"    H2 own quality (test, exact-label n={result_on['ordinal']['n_exact_label_test']}): "
                f"accuracy={result_on['ordinal']['ordinal_accuracy']:.4f}  MAE={result_on['ordinal']['ordinal_mae']:.4f}")

        log("  Arm B: H2 OFF")
        result_off = train_arm(fold_dir, edge_index, dev_lookup, H2_WEIGHT_OFF, args.seed)
        log(f"    test AUC={result_off['auc']:.4f}  macro_f1={result_off['macro_f1']:.4f}  "
            f"bal_acc={result_off['balanced_acc']:.4f}  best_val_f1={result_off['best_val_f1']:.4f}")

        row = {
            "fold": fold_id,
            "auc_h2on": result_on["auc"], "auc_h2off": result_off["auc"], "auc_diff": result_on["auc"] - result_off["auc"],
            "macro_f1_h2on": result_on["macro_f1"], "macro_f1_h2off": result_off["macro_f1"],
            "macro_f1_diff": result_on["macro_f1"] - result_off["macro_f1"],
            "balanced_acc_h2on": result_on["balanced_acc"], "balanced_acc_h2off": result_off["balanced_acc"],
            "balanced_acc_diff": result_on["balanced_acc"] - result_off["balanced_acc"],
        }
        if result_on["ordinal"]:
            row.update({
                "h2_ordinal_accuracy": result_on["ordinal"]["ordinal_accuracy"],
                "h2_ordinal_mae": result_on["ordinal"]["ordinal_mae"],
                "h2_ordinal_n_exact_label_test": result_on["ordinal"]["n_exact_label_test"],
            })
        per_fold_rows.append(row)
        pd.DataFrame(per_fold_rows).to_csv(out_dir / "per_fold.csv", index=False)

    df = pd.DataFrame(per_fold_rows)
    log("building decision summary...")

    summary_rows = []
    for metric in ("auc", "macro_f1", "balanced_acc"):
        on_col, off_col, diff_col = f"{metric}_h2on", f"{metric}_h2off", f"{metric}_diff"
        try:
            stat, p_value = wilcoxon(df[on_col], df[off_col])
        except ValueError:
            p_value = float("nan")  # e.g. all differences identical/zero
        summary_rows.append({
            "metric": metric,
            "h2_on_mean": df[on_col].mean(), "h2_on_std": df[on_col].std(),
            "h2_off_mean": df[off_col].mean(), "h2_off_std": df[off_col].std(),
            "diff_mean": df[diff_col].mean(), "diff_std": df[diff_col].std(),
            "wilcoxon_p": p_value,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    primary = summary_df[summary_df["metric"] == "macro_f1"].iloc[0]
    diff_mean, diff_std, p_value = primary["diff_mean"], primary["diff_std"], primary["wilcoxon_p"]
    if abs(diff_mean) < 0.5 * diff_std or diff_std == 0 and abs(diff_mean) < 1e-6:
        readout = "no clear change within noise"
    elif diff_mean > 0:
        readout = "helps"
    else:
        readout = "hurts"

    log("")
    log("=" * 70)
    log("DECISION SUMMARY (primary task, H1 significant fibrosis)")
    log(summary_df.to_string(index=False))
    log("")
    log(f"Readout: keeping H2 {readout} the primary task on average "
        f"(mean macro-F1 diff = {diff_mean:+.4f} +/- {diff_std:.4f} across {len(df)} folds, "
        f"Wilcoxon p={p_value:.4f}). With only 8 folds this is indicative, not conclusive -- "
        f"treat as a decision aid, not a statistical proof.")
    log("=" * 70)

    with open(out_dir / "readout.txt", "w") as f:
        f.write(f"Keeping H2 {readout} the primary task on average "
                f"(mean macro-F1 diff = {diff_mean:+.4f} +/- {diff_std:.4f} across {len(df)} folds, "
                f"Wilcoxon p={p_value:.4f}).\n")

    log(f"saved: {out_dir / 'per_fold.csv'}, {out_dir / 'summary.csv'}, {out_dir / 'readout.txt'}")
    log("Block B4 complete.")


if __name__ == "__main__":
    main()
