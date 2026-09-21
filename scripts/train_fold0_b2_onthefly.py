"""
CoDA-GAT — Block B2 (on-the-fly variant) + fold0 proof-of-life.

Replaces the precomputed scripts/augment_b2.py (which wrote synthetic
train sets to folds/*/augmented/) with augmentation that happens live,
in memory, inside the training loop -- nothing synthetic is ever written
to disk. The four augmentation methods (A1 Gaussian noise, A2 gene
dropout, A3 library-size resampling, A4 within-class mixup) are the
SAME formulas as the old augment_b2.py; only WHEN they run has moved.

Reuses the model, loss functions, training step and output-saving from
train_fold0_b1.py by importing them directly -- they are not rebuilt.

HARD GUARDS, NO BYPASS FLAG:
  - OnTheFlyAugmenter is constructed from train/ arrays only. There is no
    code path anywhere in this file that can pass calibration/ or test/
    data into it.
  - Synthetic parent selection (which real row gets cloned/perturbed) is
    restricted to the internal "fit" pool -- never the held-out "val"
    rows carved out of train/ for early stopping, and never calibration/
    or test/.
  - Group C (columns 17-25, fold-fitted) is always carried from a real
    parent verbatim -- never recomputed, never interpolated.
  - Nothing synthetic is ever saved to folds/*/augmented/ or anywhere
    else on disk. Only final model outputs (weights, embeddings, probs
    on REAL data) are written, exactly like B1.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Reused, unmodified, from B1 -- the model, the loss functions, the
# training step, the eval/extraction step, and shared config constants.
from train_fold0_b1 import (
    CoDAGAT,
    run_epoch,
    extract_outputs,
    load_edge_index,
    load_split,
    get_device,
    set_seed,
    N_NODES,
    N_FEATURES,
    N_EDGES_EXPECTED,
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    LR,
    WEIGHT_DECAY,
    VAL_FRACTION,
    SEED,
    FOLD_DIR,
    SHARED_DIR,
)

# --------------------------------------------------------------------------
# Config -- identical augmentation constants to the old augment_b2.py
# --------------------------------------------------------------------------
EXPR_Z_IDX = 0
GROUP_C_START = 17

METHOD_WEIGHTS = {"a1_noise": 0.30, "a2_dropout": 0.25, "a3_libsize": 0.20, "a4_mixup": 0.25}
A1_SIGMA_FRAC = 0.10
A2_DROPOUT_FRAC_RANGE = (0.05, 0.15)
A4_BETA_PARAM = 0.4

M_CHOICES = [0, 1, 3, 7]


def log(msg):
    print(f"[B2-live] {msg}", flush=True)


def assert_train_only_path(path: Path):
    """Same guard as the old augment_b2.py: refuses paths touching calibration/ or test/."""
    parts = {p.lower() for p in path.parts}
    if "calibration" in parts or "test" in parts:
        raise RuntimeError(f"REFUSED by augmentation guard: path touches a non-train split -> {path}")


# --------------------------------------------------------------------------
# Shared assets
# --------------------------------------------------------------------------
def load_panel_gene_order():
    panel = pd.read_csv(SHARED_DIR / "panel_genes.csv")
    return panel["gene"].tolist()


def load_raw_panel_expression(gene_order):
    expr = pd.read_parquet(SHARED_DIR / "bulk_expression.parquet")
    return expr.loc[gene_order]


# --------------------------------------------------------------------------
# OnTheFlyAugmenter -- built once from train/ only, generates in memory
# --------------------------------------------------------------------------
class OnTheFlyAugmenter:
    """
    Holds this fold's REAL official train/ arrays plus the fitted
    statistics needed by A1-A4, and generates synthetic batches live.
    Never touches calibration/ or test/ -- those paths are never passed
    to its constructor anywhere in this file.
    """

    def __init__(self, fold_id, X_train, y_train_df, sample_ids_train,
                 raw_expr_train, h3_train):
        train_dir_marker = Path("train")
        assert train_dir_marker.name == "train"  # documents intent; see class docstring

        self.fold_id = fold_id
        self.X = X_train                      # (n_real, 72, 26) -- FULL official train/ split
        self.y = y_train_df
        self.sample_ids = sample_ids_train
        self.raw_expr = raw_expr_train        # (n_real, 72), log2, aligned to sample_ids_train
        self.h3 = h3_train
        self.n_real = X_train.shape[0]

        self.cohorts = self.y["cohort"].to_numpy()
        self.platforms = self.y["platform"].to_numpy()
        self.stages = self.y["stage"].to_numpy()

        self._fit_expr_z_affine()
        self._precompute_aggregate_stats()

        self.pool_indices = None   # set once via set_parent_pool()
        self.mixup_groups = None

    def _fit_expr_z_affine(self):
        """Identical to the old augment_b2.py: recovers z = a*r + b per gene."""
        z = self.X[:, :, EXPR_Z_IDX]
        r = self.raw_expr
        a = np.empty(N_NODES)
        b = np.empty(N_NODES)
        r2 = np.empty(N_NODES)
        for g in range(N_NODES):
            rg, zg = r[:, g], z[:, g]
            if np.std(rg) < 1e-9:
                a[g], b[g], r2[g] = 0.0, float(zg.mean()), 1.0
                continue
            slope, intercept = np.polyfit(rg, zg, deg=1)
            a[g], b[g] = slope, intercept
            pred = slope * rg + intercept
            ss_res = np.sum((zg - pred) ** 2)
            ss_tot = np.sum((zg - zg.mean()) ** 2)
            r2[g] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        self.affine_a, self.affine_b = a, b
        mean_r2 = float(np.mean(r2))
        log(f"  expression_z affine recovery: mean R^2 across 72 genes = {mean_r2:.5f}")
        if mean_r2 < 0.99:
            log("  WARNING: affine fit is imperfect for some genes -- synthetic expression_z "
                "values are an approximation.")

    def raw_to_z(self, raw_values):
        return self.affine_a * raw_values + self.affine_b

    def _precompute_aggregate_stats(self):
        """
        Aggregate statistics (medians, SDs, dispersion) are fit on the
        FULL official train/ split -- matching the old augment_b2.py and
        B1's own precedent (B1's class_weights are also fit on the full
        train/ split, not just the fit-subset). These are per-gene/
        per-cohort numbers, not per-sample identities, so this mirrors
        how Group C (columns 17-25) is itself "fitted on this fold's
        train" per fold_info.json.
        """
        self.cohort_median_z = {}
        self.cohort_dispersion = {}

        for cohort in np.unique(self.cohorts):
            mask = self.cohorts == cohort
            self.cohort_median_z[cohort] = np.median(self.X[mask, :, EXPR_Z_IDX], axis=0)

        self.fold_gene_sd_raw = np.std(self.raw_expr, axis=0, ddof=1)

        self.rnaseq_cohorts = sorted(set(self.cohorts[self.platforms == "rnaseq"]))
        for cohort in self.rnaseq_cohorts:
            mask = self.cohorts == cohort
            counts = np.power(2.0, self.raw_expr[mask])
            mean = counts.mean(axis=0)
            var = counts.var(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                disp = np.where(var > mean, (mean ** 2) / (var - mean), 1e6)
            self.cohort_dispersion[cohort] = np.clip(disp, 1e-3, 1e6)

    def set_parent_pool(self, pool_indices):
        """
        Restricts WHICH real rows can ever be cloned/perturbed/mixed into
        a synthetic sample to `pool_indices` (the "fit" subset of train/,
        excluding the internal early-stopping "val" rows). This is the
        one deliberate behavior change vs. the old disk-based augment_b2.py,
        which had no val concept to protect. Mixup pools are rebuilt here,
        scoped to the pool, so a val row can never be a mixup parent either.
        """
        self.pool_indices = np.asarray(pool_indices)
        pool_set = set(self.pool_indices.tolist())
        groups = {}
        for i in self.pool_indices.tolist():
            key = (self.cohorts[i], self.stages[i])
            groups.setdefault(key, []).append(i)
        self.mixup_groups = groups
        eligible = sum(1 for v in groups.values() if len(v) >= 2)
        log(f"  parent pool set: n_pool={len(self.pool_indices)}  "
            f"mixup-eligible (cohort,stage) groups: {eligible}/{len(groups)}")


# --------------------------------------------------------------------------
# The four augmentation methods -- SAME math as the old augment_b2.py,
# only the parameter name changed (fold -> aug) since they now read off
# an OnTheFlyAugmenter instance instead of a FoldTrainData instance.
# --------------------------------------------------------------------------
def method_a1_gaussian_noise(aug, parent_idx, rng):
    feat = aug.X[parent_idx].copy()
    r_parent = aug.raw_expr[parent_idx]
    sigma = A1_SIGMA_FRAC * aug.fold_gene_sd_raw
    r_synth = r_parent + rng.normal(0.0, sigma)
    feat[:, EXPR_Z_IDX] = aug.raw_to_z(r_synth)
    return feat, float(aug.h3[parent_idx]), {"method": "a1_noise", "parent_i": parent_idx}


def method_a2_gene_dropout(aug, parent_idx, rng):
    feat = aug.X[parent_idx].copy()
    cohort = aug.cohorts[parent_idx]
    frac = rng.uniform(*A2_DROPOUT_FRAC_RANGE)
    k = max(1, round(frac * N_NODES))
    dropped_nodes = rng.choice(N_NODES, size=k, replace=False)
    feat[dropped_nodes, EXPR_Z_IDX] = aug.cohort_median_z[cohort][dropped_nodes]
    return feat, float(aug.h3[parent_idx]), {"method": "a2_dropout", "parent_i": parent_idx}


def method_a3_libsize_resample(aug, parent_idx, rng):
    cohort = aug.cohorts[parent_idx]
    if cohort not in aug.cohort_dispersion:
        return None
    feat = aug.X[parent_idx].copy()
    r_parent = aug.raw_expr[parent_idx]
    mean_counts = np.power(2.0, r_parent)
    disp = aug.cohort_dispersion[cohort]
    p = np.clip(disp / (disp + mean_counts), 1e-6, 1.0)
    resampled_counts = rng.negative_binomial(disp, p).astype(np.float64)
    r_synth = np.log2(resampled_counts + 1.0)
    feat[:, EXPR_Z_IDX] = aug.raw_to_z(r_synth)
    return feat, float(aug.h3[parent_idx]), {"method": "a3_libsize", "parent_i": parent_idx}


def method_a4_mixup(aug, group_indices, rng):
    if len(group_indices) < 2:
        return None
    i, j = rng.choice(group_indices, size=2, replace=False)
    i, j = int(i), int(j)
    lam = float(rng.beta(A4_BETA_PARAM, A4_BETA_PARAM))
    feat = aug.X[i].copy()
    feat[:, :GROUP_C_START] = (
        lam * aug.X[i][:, :GROUP_C_START] + (1 - lam) * aug.X[j][:, :GROUP_C_START]
    )
    h3_blended = lam * aug.h3[i] + (1 - lam) * aug.h3[j]
    return feat, float(h3_blended), {
        "method": "a4_mixup", "lambda": lam, "parent_i": i, "parent_j": j,
    }


# --------------------------------------------------------------------------
# Live synthetic batch generation -- deterministic in (run_seed, fold_id, epoch)
# --------------------------------------------------------------------------
def generate_synthetic_block(aug: OnTheFlyAugmenter, m: int, run_seed: int, epoch: int):
    assert aug.pool_indices is not None, "call set_parent_pool() before generating"
    n_pool = len(aug.pool_indices)
    n_synth = m * n_pool

    method_names = list(METHOD_WEIGHTS.keys())
    probs = np.array([METHOD_WEIGHTS[k] for k in method_names])
    probs = probs / probs.sum()

    seed_seq = np.random.SeedSequence(entropy=[run_seed, aug.fold_id, epoch])
    rng = np.random.default_rng(seed_seq)

    synth_X = np.empty((n_synth, N_NODES, N_FEATURES), dtype=np.float32)
    synth_y1 = np.empty(n_synth, dtype=np.float32)
    synth_y2 = np.empty(n_synth, dtype=np.float32)
    synth_y2_mask = np.empty(n_synth, dtype=bool)
    synth_h3 = np.empty(n_synth, dtype=np.float32)
    synth_cohort = np.empty(n_synth, dtype=object)
    synth_ids = []
    method_counts = {k: 0 for k in method_names}

    for s in range(n_synth):
        method = rng.choice(method_names, p=probs)
        parent_i = int(aug.pool_indices[rng.integers(0, n_pool)])

        result = None
        if method == "a1_noise":
            result = method_a1_gaussian_noise(aug, parent_i, rng)
        elif method == "a2_dropout":
            result = method_a2_gene_dropout(aug, parent_i, rng)
        elif method == "a3_libsize":
            result = method_a3_libsize_resample(aug, parent_i, rng)
            if result is None:
                method = "a1_noise"
                result = method_a1_gaussian_noise(aug, parent_i, rng)
        elif method == "a4_mixup":
            key = (aug.cohorts[parent_i], aug.stages[parent_i])
            group = aug.mixup_groups.get(key, [])
            result = method_a4_mixup(aug, group, rng)
            if result is None:
                method = "a1_noise"
                result = method_a1_gaussian_noise(aug, parent_i, rng)

        feat, h3_val, prov = result
        method_counts[method] += 1
        label_parent = prov["parent_i"]
        yi = aug.y.iloc[label_parent]

        synth_X[s] = feat
        synth_y1[s] = yi["y_T1_significant"]
        y2_raw = yi["y_T2_ordinal"]
        synth_y2_mask[s] = not pd.isna(y2_raw)
        synth_y2[s] = y2_raw if not pd.isna(y2_raw) else -1.0
        synth_h3[s] = h3_val
        synth_cohort[s] = yi["cohort"]
        synth_ids.append(f"SYN_f{aug.fold_id}_e{epoch}_{method}_{s:05d}")

    return {
        "X": synth_X, "y1": synth_y1, "y2": synth_y2, "y2_mask": synth_y2_mask,
        "h3": synth_h3, "cohort": synth_cohort, "sample_ids": synth_ids,
        "method_counts": method_counts,
    }


# --------------------------------------------------------------------------
# Build PyG Data list from raw arrays (real+synthetic combined) -- mirrors
# the inner loop of B1's load_split() exactly, generalized to in-memory
# arrays instead of files on disk.
# --------------------------------------------------------------------------
def rows_to_data_list(X, y1, y2, y2_mask, h3, cohorts, sample_ids, cohort_to_idx, edge_index):
    data_list = []
    for i in range(X.shape[0]):
        x = torch.tensor(X[i], dtype=torch.float32)
        y1_i = torch.tensor(int(y1[i]), dtype=torch.long)
        y2_i = torch.tensor(float(y2[i]), dtype=torch.float32)
        y2_mask_i = torch.tensor(bool(y2_mask[i]), dtype=torch.bool)
        y3_i = torch.tensor(float(h3[i]), dtype=torch.float32)

        cohort_name = cohorts[i]
        if cohort_name in cohort_to_idx:
            cohort_i = torch.tensor(cohort_to_idx[cohort_name], dtype=torch.long)
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
    return data_list


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT B2 on-the-fly + fold0 proof-of-life")
    parser.add_argument("--m", type=int, default=3, choices=M_CHOICES,
                         help="augmentation multiplier: synthetic samples per real fit-row, per epoch")
    parser.add_argument("--run-seed", type=int, default=SEED,
                         help="seeds both model init (via B1's set_seed) and the augmenter's RNG schedule")
    args = parser.parse_args()

    set_seed(args.run_seed)
    device = get_device()
    log(f"device: {device}")
    log(f"augmentation multiplier m={args.m}  run_seed={args.run_seed}")
    log("GUARD: OnTheFlyAugmenter is constructed from train/ arrays only; calibration/ and "
        "test/ are never passed to it anywhere in this file. Nothing synthetic is written to disk.")

    OUT_DIR = FOLD_DIR / f"outputs_onthefly_m{args.m}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading graph edges...")
    edge_index = load_edge_index()
    log(f"edge_index shape: {tuple(edge_index.shape)} ({N_EDGES_EXPECTED} directed edges confirmed)")

    log("loading deviation targets (H3)...")
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")

    log("reading fold_info.json for cohort vocabulary...")
    with open(FOLD_DIR / "fold_info.json") as f:
        fold_info = json.load(f)
    train_cohorts = sorted(fold_info["train_cohorts"])
    cohort_to_idx = {c: i for i, c in enumerate(train_cohorts)}
    n_cohorts = len(cohort_to_idx)
    log(f"train cohorts ({n_cohorts}): {train_cohorts}")

    log("loading REAL train / calibration / test splits (via B1's load_split, unchanged)...")
    train_data, train_ids, train_y = load_split("train", edge_index, dev_lookup, cohort_to_idx)
    cal_data, cal_ids, cal_y = load_split("calibration", edge_index, dev_lookup, cohort_to_idx)
    test_data, test_ids, test_y = load_split("test", edge_index, dev_lookup, cohort_to_idx)
    log(f"train n={len(train_data)}  calibration n={len(cal_data)}  test n={len(test_data)}")

    n_ordinal_classes = int(
        np.nanmax(pd.concat([train_y["y_T2_ordinal"], cal_y["y_T2_ordinal"], test_y["y_T2_ordinal"]]))
    ) + 1
    log(f"ordinal classes inferred: {n_ordinal_classes} (F0..F{n_ordinal_classes - 1})")

    y1_train = train_y["y_T1_significant"].to_numpy()
    counts = np.bincount(y1_train.astype(int))
    weights = len(y1_train) / (len(counts) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    log(f"H1 class weights (train-derived, inverse frequency): {weights.tolist()}")

    # --- raw arrays for augmentation (train/ only) ---
    log("loading raw train/ arrays + raw expression matrix for the augmenter...")
    train_dir = FOLD_DIR / "train"
    X_train_raw = np.load(train_dir / "X.npy")
    sample_ids_train = train_ids
    h3_train = dev_lookup.loc[sample_ids_train, "deviation_score"].to_numpy(dtype=np.float32)

    gene_order = load_panel_gene_order()
    expr_panel = load_raw_panel_expression(gene_order)
    raw_expr_train = expr_panel.loc[gene_order, sample_ids_train].to_numpy(dtype=np.float64).T

    augmenter = OnTheFlyAugmenter(
        fold_id=0, X_train=X_train_raw, y_train_df=train_y, sample_ids_train=sample_ids_train,
        raw_expr_train=raw_expr_train, h3_train=h3_train,
    )
    log(f"  rnaseq cohorts eligible for A3: {augmenter.rnaseq_cohorts}")

    # --- fit/val split: SAME stratification as B1 ---
    strat_key = [f"{c}_{int(y)}" for c, y in zip(train_y["cohort"], y1_train)]
    idx_all = np.arange(len(train_data))
    idx_fit, idx_val = train_test_split(
        idx_all, test_size=VAL_FRACTION, random_state=SEED, stratify=strat_key
    )
    log(f"internal split for early stopping: fit n={len(idx_fit)}  val n={len(idx_val)} (val is REAL, never augmented)")

    augmenter.set_parent_pool(idx_fit)

    val_data = [train_data[i] for i in idx_val]
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    train_loader_full = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    cal_loader = DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = CoDAGAT(N_FEATURES, n_ordinal_classes, n_cohorts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    log("starting training with on-the-fly augmentation...")
    for epoch in range(1, MAX_EPOCHS + 1):
        p_progress = epoch / MAX_EPOCHS

        if args.m == 0:
            synth = {"X": np.empty((0, N_NODES, N_FEATURES), dtype=np.float32),
                     "y1": np.empty(0), "y2": np.empty(0), "y2_mask": np.empty(0, dtype=bool),
                     "h3": np.empty(0), "cohort": np.empty(0, dtype=object), "sample_ids": [],
                     "method_counts": {}}
        else:
            synth = generate_synthetic_block(augmenter, args.m, args.run_seed, epoch)

        fit_X = np.concatenate([X_train_raw[idx_fit], synth["X"]], axis=0)
        fit_y1 = np.concatenate([y1_train[idx_fit], synth["y1"]])
        fit_y2_raw = train_y["y_T2_ordinal"].to_numpy()[idx_fit]
        fit_y2_mask_real = ~np.isnan(fit_y2_raw)
        fit_y2_real = np.where(fit_y2_mask_real, fit_y2_raw, -1.0)
        fit_y2 = np.concatenate([fit_y2_real, synth["y2"]])
        fit_y2_mask = np.concatenate([fit_y2_mask_real, synth["y2_mask"]])
        fit_h3 = np.concatenate([h3_train[idx_fit], synth["h3"]])
        fit_cohort = np.concatenate([np.asarray(train_y["cohort"])[idx_fit], synth["cohort"]])
        fit_ids = [sample_ids_train[i] for i in idx_fit] + synth["sample_ids"]

        fit_data_epoch = rows_to_data_list(
            fit_X, fit_y1, fit_y2, fit_y2_mask, fit_h3, fit_cohort, fit_ids, cohort_to_idx, edge_index
        )
        fit_loader = DataLoader(fit_data_epoch, batch_size=BATCH_SIZE, shuffle=True)

        train_loss, train_f1 = run_epoch(
            model, fit_loader, device, optimizer, class_weights, n_ordinal_classes, p_progress, train=True,
        )
        val_loss, val_f1 = run_epoch(
            model, val_loader, device, optimizer, class_weights, n_ordinal_classes, p_progress, train=False,
        )

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        n_real_fit = len(idx_fit)
        n_synth = synth["X"].shape[0]
        pos_rate_combined = fit_y1.mean()

        if epoch == 1 or epoch % 5 == 0 or improved:
            log(
                f"epoch {epoch:3d}/{MAX_EPOCHS}  train_loss={train_loss:.4f} train_f1={train_f1:.4f}  "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}  {'*' if improved else ''}"
            )
            log(
                f"    [sanity] n_real_fit={n_real_fit}  n_synth={n_synth}  "
                f"n_combined={n_real_fit + n_synth}  pos_rate_combined={pos_rate_combined:.4f}"
                + (f"  method_mix={synth['method_counts']}" if n_synth else "")
            )

        if epochs_no_improve >= PATIENCE:
            log(f"early stopping at epoch {epoch} (patience={PATIENCE}, best_val_f1={best_val_f1:.4f})")
            break

    log(f"restoring best checkpoint (val_f1={best_val_f1:.4f})")
    model.load_state_dict(best_state)

    log("saving weights.pt...")
    torch.save(
        {"model_state_dict": best_state, "best_val_f1": best_val_f1,
         "n_ordinal_classes": n_ordinal_classes, "n_cohorts": n_cohorts,
         "cohort_to_idx": cohort_to_idx,
         "augmentation": {"mode": "on_the_fly", "m": args.m, "run_seed": args.run_seed,
                           "method_weights": METHOD_WEIGHTS}},
        OUT_DIR / "weights.pt",
    )

    log("extracting embeddings / probabilities / predictions on REAL data (never augmented)...")
    emb_train, probs_train, preds_train, _ = extract_outputs(model, train_loader_full, device)
    emb_cal, probs_cal, preds_cal, _ = extract_outputs(model, cal_loader, device)
    emb_test, probs_test, preds_test, _ = extract_outputs(model, test_loader, device)

    np.save(OUT_DIR / "embeddings_train.npy", emb_train)
    np.save(OUT_DIR / "embeddings_calibration.npy", emb_cal)
    np.save(OUT_DIR / "embeddings_test.npy", emb_test)
    np.save(OUT_DIR / "probs_calibration.npy", probs_cal)
    np.save(OUT_DIR / "probs_test.npy", probs_test)
    np.save(OUT_DIR / "predictions_test.npy", preds_test)

    y1_cal = cal_y["y_T1_significant"].to_numpy().astype(int)
    p_true_class_cal = np.where(y1_cal == 1, probs_cal, 1 - probs_cal)
    nonconformity_cal = 1.0 - p_true_class_cal
    np.save(OUT_DIR / "nonconformity_calibration.npy", nonconformity_cal)

    with open(OUT_DIR / "conformal_threshold.json", "w") as f:
        json.dump({"alpha": None}, f, indent=2)

    y1_test = test_y["y_T1_significant"].to_numpy().astype(int)
    test_f1 = f1_score(y1_test, preds_test, average="macro")
    log(f"[proof-of-life] fold0 test macro-F1 (H1, significant fibrosis): {test_f1:.4f}")
    log(f"[proof-of-life] fold0 test positive rate: {y1_test.mean():.4f} (expect ~0.32 per fold_info.json)")
    log(f"[reminder] real dataset size for the paper stays 1366 regardless of m={args.m}")

    log(f"all outputs written to {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        log(f"  {p.name}  ({p.stat().st_size} bytes)")

    log("B2 on-the-fly proof-of-life complete. folds/*/train, calibration, test, and "
        "folds/*/augmented/ were never touched by this run.")


if __name__ == "__main__":
    main()
