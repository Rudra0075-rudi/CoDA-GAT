"""
CoDA-GAT — Block B7: the full experiment (120 runs).

8 folds x 3 adversary modes {off, marginal, conditional} x 5 seeds
{42,43,44,45,46}. Reuses, unchanged, via import:
  - the model, loss functions, training step (train_fold0_b1.py)
  - the on-the-fly augmenter, its 4 methods, and its epoch-seeding
    scheme (train_fold0_b2_onthefly.py)
  - the prevalence-subsampling dose-response machinery (safety_checks_b5.py)

The ONE genuinely new piece of model code in this file is
CoDAGATConditional (a CDAN-style conditional adversary). It did not exist
before this block -- "conditional" was named in the original architecture
spec but never implemented. It is used only for the two non-primary arms
of the grid; the pre-registered primary contrast (preregistration_b6.md)
is marginal-vs-off, both of which reuse CoDAGAT completely unmodified.

Design choices worth knowing about (both documented again at point of use):
  - The train/val ("fit"/"val") split for early stopping is fixed per
    fold using the base SEED=42 (train_test_split's random_state), NOT
    the run's seed. This means all 3 modes x 5 seeds for a given fold
    train/validate on the exact same data partition -- seed-to-seed
    variance reflects stochastic training (init, batch order, dropout,
    augmentation draws) only, not "we also re-shuffled which rows are
    validation." This lets the OnTheFlyAugmenter, its parent pool, and
    the val/cal/test loaders be built ONCE per fold and reused across
    all 15 (mode, seed) runs for that fold.
  - m (augmentation multiplier) defaults to 0 for the main grid, per the
    pre-registration: the primary adversary contrast is measured on real
    data. --m is exposed so the descriptive m-sweep {0,1,3,7} can be run
    as a separate invocation later.

Resumable: each (fold, mode, seed) run's completion is marked by a
DONE.marker file written only after every artifact for that run has been
saved. Restarting the script skips any run whose marker already exists.
A single run's failure is logged and does not stop the other 119.
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
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import norm, wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import train_fold0_b1 as b1
from train_fold0_b1 import (
    CoDAGAT, run_epoch, grad_reverse, grl_lambda,
    get_device, set_seed,
    N_NODES, N_FEATURES, N_EDGES_EXPECTED,
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY, VAL_FRACTION, SEED,
    SHARED_DIR,
)
from train_fold0_b2_onthefly import (
    OnTheFlyAugmenter, generate_synthetic_block, rows_to_data_list,
    load_panel_gene_order, load_raw_panel_expression,
)
from safety_checks_b5 import synth_metric_at_prevalence, within_site_slope, METRICS

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
OUT_ROOT = REPO_ROOT / "results_b7"

MODES = ["off", "marginal", "conditional"]
SEEDS = [42, 43, 44, 45, 46]
ALPHAS = [0.05, 0.10, 0.20]
H4_WEIGHT_ON = 0.1   # matches train_fold0_b1.LOSS_WEIGHTS["h4"]
H4_WEIGHT_OFF = 0.0


def log(msg):
    print(f"[B7] {msg}", flush=True)


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
# NEW model code: the conditional adversary (did not exist before B7)
# ==========================================================================
class CoDAGATConditional(CoDAGAT):
    """
    Conditional cohort-adversary (CDAN-style multilinear conditioning,
    Long et al. 2018): the adversary sees the outer product of the shared
    embedding and the model's own H1 class-probability vector, instead of
    the embedding alone. In principle this lets the adversary erase
    cohort signal while leaving room for genuinely label-predictive
    structure that happens to correlate with cohort, unlike marginal
    (unconditional) adversarial training, which pushes to erase
    everything cohort-correlated equally.

    h1_prob is detached before conditioning: the adversary's gradient
    (pre-reversal) should shape the SHARED embedding through the
    conditioning path, not backprop into the H1 head itself through a
    second route -- keeping H1's own gradient path exactly as in the
    marginal/off arms.
    """

    def __init__(self, n_features, n_ordinal_classes, n_cohorts):
        super().__init__(n_features, n_ordinal_classes, n_cohorts)
        cond_dim = 16 * 2  # 16-dim embedding outer-producted with 2-class H1 softmax
        self.h4 = nn.Sequential(
            nn.Linear(cond_dim, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_cohorts),
        )

    def forward(self, x, edge_index, batch, grl_lambd=0.0):
        shared = self.encode(x, edge_index, batch)
        h1_logits = self.h1(shared)
        h2_logits = self.h2(shared)
        h3_pred = self.h3(shared).squeeze(-1)

        h1_prob = F.softmax(h1_logits, dim=1).detach()
        conditioning = torch.einsum("bi,bj->bij", h1_prob, shared).reshape(shared.size(0), -1)
        reversed_cond = grad_reverse(conditioning, grl_lambd)
        h4_logits = self.h4(reversed_cond)

        return {"embedding": shared, "h1_logits": h1_logits, "h2_logits": h2_logits,
                "h3_pred": h3_pred, "h4_logits": h4_logits}


def build_model(mode, n_features, n_ordinal_classes, n_cohorts):
    if mode in ("off", "marginal"):
        return CoDAGAT(n_features, n_ordinal_classes, n_cohorts)
    if mode == "conditional":
        return CoDAGATConditional(n_features, n_ordinal_classes, n_cohorts)
    raise ValueError(f"unknown adversary mode: {mode}")


# ==========================================================================
# Fold-agnostic data loading (same pattern as B3/B4/B5; B1 hardcodes fold0)
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


@torch.no_grad()
def extract_full(model, loader, device):
    """Like B1's extract_outputs, but keeps the FULL softmax vector (not
    just P(class=1)) -- APS conformal sets need every class's probability."""
    model.eval()
    embeddings, probs_list, preds, sample_ids = [], [], [], []
    for batch in loader:
        sids = batch.sample_id
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, grl_lambd=0.0)
        p = F.softmax(out["h1_logits"], dim=1)
        pred = out["h1_logits"].argmax(dim=1)
        embeddings.append(out["embedding"].cpu().numpy())
        probs_list.append(p.cpu().numpy())
        preds.append(pred.cpu().numpy())
        sample_ids.extend(sids if isinstance(sids, list) else [sids])
    return (np.concatenate(embeddings, axis=0), np.concatenate(probs_list, axis=0),
            np.concatenate(preds, axis=0), sample_ids)


def mc_dropout_probs(model, loader, device, n_passes=30):
    """30-pass MC-Dropout: model.train() keeps Dropout active (CoDAGAT has
    no BatchNorm, so this changes nothing else about the forward pass).
    Returns P(class=1) per pass, shape (n_passes, n_samples)."""
    model.train()
    all_passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            pass_probs = []
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch, grl_lambd=0.0)
                pass_probs.append(F.softmax(out["h1_logits"], dim=1)[:, 1].cpu().numpy())
            all_passes.append(np.concatenate(pass_probs))
    model.eval()
    return np.stack(all_passes, axis=0)


# ==========================================================================
# Conformal prediction: LAC + APS, both with finite-sample-corrected quantile
# ==========================================================================
def conformal_quantile(scores, alpha):
    n = len(scores)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    if k >= n:
        return float("inf")  # not enough calibration points for this alpha -- conservative include-all
    return float(np.sort(scores)[k - 1])


def lac_scores(probs_matrix, true_labels):
    n = len(true_labels)
    return 1.0 - probs_matrix[np.arange(n), true_labels]


def lac_sets(probs_matrix, q_hat):
    return [set(np.where(row >= 1 - q_hat)[0].tolist()) for row in probs_matrix]


def aps_scores(probs_matrix, true_labels):
    """Non-randomized APS (Romano, Sesia & Candès, 2020): score = cumulative
    probability mass of all classes at least as likely as the true class.
    Slightly conservative vs. the randomized variant; simpler to audit."""
    n = probs_matrix.shape[0]
    order = np.argsort(-probs_matrix, axis=1)
    sorted_probs = np.take_along_axis(probs_matrix, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    rank = np.argmax(order == true_labels[:, None], axis=1)
    return cumsum[np.arange(n), rank]


def aps_sets(probs_matrix, q_hat):
    order = np.argsort(-probs_matrix, axis=1)
    sorted_probs = np.take_along_axis(probs_matrix, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    reaches = cumsum >= q_hat
    any_reach = reaches.any(axis=1)
    k = np.where(any_reach, np.argmax(reaches, axis=1) + 1, probs_matrix.shape[1])
    return [set(order[i, :k[i]].tolist()) for i in range(probs_matrix.shape[0])]


def coverage_stats(pred_sets, true_labels):
    n = len(true_labels)
    covered = np.array([true_labels[i] in pred_sets[i] for i in range(n)])
    sizes = np.array([len(s) for s in pred_sets])
    by_class = {}
    for c in np.unique(true_labels):
        mask = true_labels == c
        by_class[int(c)] = float(covered[mask].mean()) if mask.sum() > 0 else float("nan")
    size_dist = {int(s): int((sizes == s).sum()) for s in np.unique(sizes)}
    return {
        "empirical_coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "conditional_coverage": by_class,
        "set_size_distribution": size_dist,
    }


# ==========================================================================
# DeLong AUC confidence interval (sklearn has no built-in for this)
# ==========================================================================
def _compute_midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def delong_auc_ci(y_true, y_score, alpha=0.05):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos, neg = y_score[y_true == 1], y_score[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), (float("nan"), float("nan"))  # AUC undefined with only one class present

    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz = _compute_midrank(np.concatenate([pos, neg]))
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    auc = float(v01.mean())
    var = float(np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n) if m > 1 and n > 1 else float("nan")

    if np.isnan(var) or var < 0:
        return auc, (float("nan"), float("nan"))
    se = np.sqrt(var)
    z = norm.ppf(1 - alpha / 2)
    return auc, (max(0.0, auc - z * se), min(1.0, auc + z * se))


# ==========================================================================
# Per-fold setup (done ONCE, reused across all 15 mode/seed runs)
# ==========================================================================
def setup_fold(fold_dir, edge_index, dev_lookup):
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
    class_weights = torch.tensor(len(y1_train) / (len(counts) * counts), dtype=torch.float32)

    # Fixed split (base SEED, not the run seed) -- see module docstring.
    strat_key = [f"{c}_{int(y)}" for c, y in zip(train_y["cohort"], y1_train)]
    idx_all = np.arange(len(train_data))
    idx_fit, idx_val = train_test_split(idx_all, test_size=VAL_FRACTION, random_state=SEED, stratify=strat_key)

    val_loader = DataLoader([train_data[i] for i in idx_val], batch_size=BATCH_SIZE, shuffle=False)
    cal_loader = DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    train_loader_full = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

    gene_order = load_panel_gene_order()
    expr_panel = load_raw_panel_expression(gene_order)
    X_train_raw = np.load(fold_dir / "train" / "X.npy")
    raw_expr_train = expr_panel.loc[gene_order, train_ids].to_numpy(dtype=np.float64).T
    h3_train = dev_lookup.loc[train_ids, "deviation_score"].to_numpy(dtype=np.float32)

    augmenter = OnTheFlyAugmenter(
        fold_id=parse_fold_index(fold_dir.name), X_train=X_train_raw, y_train_df=train_y,
        sample_ids_train=train_ids, raw_expr_train=raw_expr_train, h3_train=h3_train,
    )
    augmenter.set_parent_pool(idx_fit)

    return dict(
        fold_info=fold_info, cohort_to_idx=cohort_to_idx, n_cohorts=n_cohorts,
        n_ordinal_classes=n_ordinal_classes, class_weights=class_weights,
        idx_fit=idx_fit, X_train_raw=X_train_raw, y1_train=y1_train, train_y=train_y,
        h3_train=h3_train, train_ids=train_ids,
        val_loader=val_loader, cal_loader=cal_loader, test_loader=test_loader,
        train_loader_full=train_loader_full, augmenter=augmenter, edge_index=edge_index,
        y_cal=cal_y["y_T1_significant"].to_numpy().astype(int),
        y_test=test_y["y_T1_significant"].to_numpy().astype(int),
    )


# ==========================================================================
# One (fold, mode, seed) run
# ==========================================================================
def run_single(fold_setup, fold_id, mode, seed, m, run_dir, n_mc_dropout, n_prevalence_grid):
    device = get_device()
    aug = fold_setup["augmenter"]
    cohort_to_idx = fold_setup["cohort_to_idx"]
    edge_index = fold_setup["edge_index"]
    idx_fit = fold_setup["idx_fit"]

    set_seed(seed)
    model = build_model(mode, N_FEATURES, fold_setup["n_ordinal_classes"], fold_setup["n_cohorts"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_weights = fold_setup["class_weights"].to(device)

    original_h4_weight = b1.LOSS_WEIGHTS["h4"]
    b1.LOSS_WEIGHTS["h4"] = H4_WEIGHT_OFF if mode == "off" else H4_WEIGHT_ON
    try:
        best_val_f1, best_state, bad = -1.0, None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            p = epoch / MAX_EPOCHS

            if m == 0:
                synth = {"X": np.empty((0, N_NODES, N_FEATURES), dtype=np.float32),
                         "y1": np.empty(0), "y2": np.empty(0), "y2_mask": np.empty(0, dtype=bool),
                         "h3": np.empty(0), "cohort": np.empty(0, dtype=object), "sample_ids": []}
            else:
                synth = generate_synthetic_block(aug, m, seed, epoch)

            fit_X = np.concatenate([fold_setup["X_train_raw"][idx_fit], synth["X"]], axis=0)
            fit_y1 = np.concatenate([fold_setup["y1_train"][idx_fit], synth["y1"]])
            fit_y2_raw = fold_setup["train_y"]["y_T2_ordinal"].to_numpy()[idx_fit]
            fit_y2_mask_real = ~np.isnan(fit_y2_raw)
            fit_y2 = np.concatenate([np.where(fit_y2_mask_real, fit_y2_raw, -1.0), synth["y2"]])
            fit_y2_mask = np.concatenate([fit_y2_mask_real, synth["y2_mask"]])
            fit_h3 = np.concatenate([fold_setup["h3_train"][idx_fit], synth["h3"]])
            fit_cohort = np.concatenate([np.asarray(fold_setup["train_y"]["cohort"])[idx_fit], synth["cohort"]])
            fit_ids = [fold_setup["train_ids"][i] for i in idx_fit] + synth["sample_ids"]

            fit_data = rows_to_data_list(fit_X, fit_y1, fit_y2, fit_y2_mask, fit_h3, fit_cohort,
                                          fit_ids, cohort_to_idx, edge_index)
            fit_loader = DataLoader(fit_data, batch_size=BATCH_SIZE, shuffle=True)

            run_epoch(model, fit_loader, device, optimizer, class_weights,
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
        b1.LOSS_WEIGHTS["h4"] = original_h4_weight

    model.load_state_dict(best_state)

    # ---- extraction on RAW real data only ----
    emb_train, probs_train, _, _ = extract_full(model, fold_setup["train_loader_full"], device)
    emb_cal, probs_cal, preds_cal, _ = extract_full(model, fold_setup["cal_loader"], device)
    emb_test, probs_test, preds_test, _ = extract_full(model, fold_setup["test_loader"], device)
    y_cal, y_test = fold_setup["y_cal"], fold_setup["y_test"]

    # ---- discrimination metrics on test ----
    p1_test = probs_test[:, 1]
    auc, auc_ci = delong_auc_ci(y_test, p1_test)
    macro_f1 = f1_score(y_test, preds_test, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, preds_test)

    # ---- conformal: LAC + APS, fit on RAW calibration, never test ----
    lac_cal = lac_scores(probs_cal, y_cal)
    aps_cal = aps_scores(probs_cal, y_cal)

    conformal_results = {}
    thresholds = {"lac": {}, "aps": {}}
    for alpha in ALPHAS:
        q_lac = conformal_quantile(lac_cal, alpha)
        q_aps = conformal_quantile(aps_cal, alpha)
        thresholds["lac"][str(alpha)] = q_lac
        thresholds["aps"][str(alpha)] = q_aps

        sets_lac = lac_sets(probs_test, q_lac)
        sets_aps = aps_sets(probs_test, q_aps)
        stats_lac = coverage_stats(sets_lac, y_test)
        stats_aps = coverage_stats(sets_aps, y_test)
        stats_lac["signed_coverage_gap"] = stats_lac["empirical_coverage"] - (1 - alpha)
        stats_aps["signed_coverage_gap"] = stats_aps["empirical_coverage"] - (1 - alpha)

        conformal_results[str(alpha)] = {"lac": stats_lac, "aps": stats_aps}

    # ---- MC-Dropout ----
    mc_probs = mc_dropout_probs(model, fold_setup["test_loader"], device, n_passes=n_mc_dropout)

    # ---- prevalence sub-sampling dose-response (reusing B5's exact functions) ----
    pi_cal, pi_test = float(y_cal.mean()), float(y_test.mean())
    grid = np.linspace(min(pi_cal, pi_test), max(pi_cal, pi_test), n_prevalence_grid)
    rng = np.random.default_rng(seed * 7919 + fold_id * 31)
    slope, metrics_at_p = within_site_slope(y_test, p1_test, 0.5, METRICS["macro_f1"], grid, rng)
    prevalence_dose_response = {"prevalence_grid": grid.tolist(), "macro_f1_at_p": metrics_at_p}

    # ---- save artifacts ----
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "mode": mode, "seed": seed, "m": m,
                "best_val_f1": best_val_f1, "n_ordinal_classes": fold_setup["n_ordinal_classes"],
                "n_cohorts": fold_setup["n_cohorts"], "cohort_to_idx": cohort_to_idx},
               run_dir / "weights.pt")
    np.save(run_dir / "embeddings_train.npy", emb_train)
    np.save(run_dir / "embeddings_calibration.npy", emb_cal)
    np.save(run_dir / "embeddings_test.npy", emb_test)
    np.save(run_dir / "probs_calibration.npy", probs_cal)
    np.save(run_dir / "probs_test.npy", probs_test)
    np.save(run_dir / "predictions_test.npy", preds_test)
    np.save(run_dir / "nonconformity_calibration_lac.npy", lac_cal)
    np.save(run_dir / "nonconformity_calibration_aps.npy", aps_cal)
    np.save(run_dir / "mc_dropout_probs_test.npy", mc_probs)
    with open(run_dir / "conformal_threshold.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    run_metrics = {
        "fold": fold_id, "test_cohort": fold_setup["fold_info"]["test_cohort"],
        "cal_cohort": fold_setup["fold_info"]["calibration_source"],
        "mode": mode, "seed": seed, "m": m, "best_val_f1": best_val_f1,
        "auc": auc, "auc_ci_lo": auc_ci[0], "auc_ci_hi": auc_ci[1],
        "macro_f1": macro_f1, "balanced_acc": bal_acc,
        "pi_cal": pi_cal, "pi_test": pi_test,
        "prevalence_slope_macro_f1": slope,
        "prevalence_dose_response": prevalence_dose_response,
        "conformal": conformal_results,
    }
    with open(run_dir / "run_metrics.json", "w") as f:
        json.dump(run_metrics, f, indent=2)

    (run_dir / "DONE.marker").touch()
    return run_metrics


# ==========================================================================
# Master table + aggregation
# ==========================================================================
def flatten_run_metrics(rm):
    row = {k: rm[k] for k in ("fold", "test_cohort", "cal_cohort", "mode", "seed", "m",
                               "best_val_f1", "auc", "auc_ci_lo", "auc_ci_hi",
                               "macro_f1", "balanced_acc", "pi_cal", "pi_test",
                               "prevalence_slope_macro_f1")}
    for alpha_str, by_score in rm["conformal"].items():
        for score_type in ("lac", "aps"):
            s = by_score[score_type]
            prefix = f"{score_type}_a{alpha_str}"
            row[f"{prefix}_coverage"] = s["empirical_coverage"]
            row[f"{prefix}_gap"] = s["signed_coverage_gap"]
            row[f"{prefix}_mean_set_size"] = s["mean_set_size"]
    return row


def collect_all_run_metrics():
    rows = []
    for marker in OUT_ROOT.glob("fold*/mode_*/seed_*/DONE.marker"):
        run_dir = marker.parent
        with open(run_dir / "run_metrics.json") as f:
            rows.append(flatten_run_metrics(json.load(f)))
    return pd.DataFrame(rows)


def aggregate_and_report(df):
    if df.empty:
        log("no completed runs found -- nothing to aggregate.")
        return

    df.to_csv(OUT_ROOT / "master_results.csv", index=False)
    log(f"master_results.csv written ({len(df)} rows)")

    metric_cols = ["auc", "macro_f1", "balanced_acc", "prevalence_slope_macro_f1"]
    per_fold_mode = df.groupby(["fold", "mode"])[metric_cols].agg(["mean", "std"])
    per_fold_mode.columns = [f"{m}_{s}" for m, s in per_fold_mode.columns]
    per_fold_mode = per_fold_mode.reset_index()
    per_fold_mode.to_csv(OUT_ROOT / "per_fold_mode_summary.csv", index=False)

    per_mode = per_fold_mode.groupby("mode")[[f"{m}_mean" for m in metric_cols]].agg(["mean", "std"])
    per_mode.columns = [f"{m}_{s}" for m, s in per_mode.columns]
    per_mode = per_mode.reset_index()
    per_mode.to_csv(OUT_ROOT / "per_mode_summary.csv", index=False)
    log("per_fold_mode_summary.csv and per_mode_summary.csv written")

    # ---- pre-registered primary contrast: marginal vs off, macro_f1, paired across 8 folds ----
    pivot = per_fold_mode.pivot(index="fold", columns="mode", values="macro_f1_mean")
    report_lines = ["=" * 78, "B7 PRIMARY CONTRAST (pre-registered in preregistration_b6.md)", "=" * 78, ""]

    def paired_test(mode_a, mode_b, label):
        if mode_a not in pivot.columns or mode_b not in pivot.columns:
            return f"{label}: SKIPPED -- missing mode(s) in completed runs so far.\n"
        sub = pivot[[mode_a, mode_b]].dropna()
        if len(sub) < 2:
            return f"{label}: SKIPPED -- fewer than 2 complete folds for this pair so far.\n"
        diffs = sub[mode_a] - sub[mode_b]
        try:
            stat, p = wilcoxon(sub[mode_a], sub[mode_b])
        except ValueError:
            p = float("nan")
        direction = "PASS-direction (ON > OFF)" if diffs.mean() > 0 else "wrong direction (ON <= OFF)"
        verdict = "PASS" if (p < 0.05 and diffs.mean() > 0) else "NULL"
        return (f"{label}: n={len(sub)} folds, mean diff={diffs.mean():+.4f}, "
                f"Wilcoxon p={p:.4f} (n=8 floor=0.008), direction={direction} -> {verdict}\n")

    report_lines.append("PRIMARY (marginal vs off, macro_f1, paired across folds):")
    report_lines.append(paired_test("marginal", "off", "  marginal vs off"))
    report_lines.append("Reminder: n=8 is underpowered (min detectable effect ~1.15 sigma_d).")
    report_lines.append("A PASS or NULL here is reported at face value, not re-litigated by the")
    report_lines.append("descriptive contrasts below.")
    report_lines.append("")
    report_lines.append("DESCRIPTIVE ONLY (not the pre-registered primary test):")
    report_lines.append(paired_test("conditional", "off", "  conditional vs off"))
    report_lines.append(paired_test("conditional", "marginal", "  conditional vs marginal"))
    report_lines.append("=" * 78)

    report = "\n".join(report_lines)
    for line in report_lines:
        log(line)
    with open(OUT_ROOT / "primary_contrast_report.txt", "w") as f:
        f.write(report + "\n")


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT Block B7 -- full experiment (120 runs)")
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--modes", nargs="*", default=MODES, choices=MODES)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--m", type=int, default=0, choices=[0, 1, 3, 7],
                         help="augmentation multiplier; main grid uses 0 per the pre-registration")
    parser.add_argument("--n-mc-dropout", type=int, default=30)
    parser.add_argument("--n-prevalence-grid", type=int, default=7)
    parser.add_argument("--aggregate-only", action="store_true",
                         help="skip all training; just rebuild master_results.csv and the primary-contrast report from existing DONE runs")
    args = parser.parse_args()

    # BUGFIX: the run_dir path (fold/mode/seed) never included `m`, so a
    # second invocation with --m != 0 would see the --m 0 run's DONE.marker
    # already sitting at that same path and silently skip every run,
    # producing zero new data while logging "already done". Non-zero m now
    # gets its own output tree; the existing m=0 results_b7/ is untouched.
    global OUT_ROOT
    if args.m != 0:
        OUT_ROOT = REPO_ROOT / f"results_b7_m{args.m}"
        log(f"m={args.m} != 0 -- writing to a separate output tree: {OUT_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        log(f"aggregate-only mode: scanning {OUT_ROOT} for completed runs...")
        aggregate_and_report(collect_all_run_metrics())
        return

    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()) if args.folds is None \
        else [FOLDS_DIR / f for f in args.folds]

    total_runs = len(fold_dirs) * len(args.modes) * len(args.seeds)
    log("CoDA-GAT Block B7 -- full experiment")
    log(f"folds={[d.name for d in fold_dirs]}  modes={args.modes}  seeds={args.seeds}  m={args.m}")
    log(f"total planned runs this invocation: {total_runs}")
    log("Resumable: runs with an existing DONE.marker are skipped. Safe to Ctrl-C and re-run.")

    edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")

    start_time = time.time()
    run_counter = 0
    completed_this_session = 0

    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        log(f"=== fold {fold_id}: {fold_dir.name} -- one-time setup (augmenter, loaders, stats) ===")
        fold_setup = setup_fold(fold_dir, edge_index, dev_lookup)
        log(f"    test_cohort={fold_setup['fold_info']['test_cohort']}  "
            f"cal_cohort={fold_setup['fold_info']['calibration_source']}  "
            f"n_ordinal_classes={fold_setup['n_ordinal_classes']}  n_cohorts={fold_setup['n_cohorts']}")

        for mode in args.modes:
            for seed in args.seeds:
                run_counter += 1
                cohort = fold_setup["fold_info"]["test_cohort"]
                run_dir = OUT_ROOT / f"fold{fold_id}_{cohort}" / f"mode_{mode}" / f"seed_{seed}"

                if (run_dir / "DONE.marker").exists():
                    log(f"[{run_counter}/{total_runs}] fold={fold_id} mode={mode} seed={seed} "
                        f"-- already done, skipping")
                    continue

                elapsed = time.time() - start_time
                if completed_this_session > 0:
                    avg = elapsed / completed_this_session
                    eta = avg * (total_runs - run_counter + 1)
                    eta_str = f"  elapsed={fmt_hms(elapsed)}  ETA={fmt_hms(eta)}"
                else:
                    eta_str = f"  elapsed={fmt_hms(elapsed)}  ETA=(pending first completion)"
                log(f"[{run_counter}/{total_runs}] fold={fold_id} mode={mode} seed={seed} -- starting{eta_str}")

                try:
                    run_metrics = run_single(fold_setup, fold_id, mode, seed, args.m, run_dir,
                                              args.n_mc_dropout, args.n_prevalence_grid)
                    completed_this_session += 1
                    log(f"    done: AUC={run_metrics['auc']:.4f}  macro_f1={run_metrics['macro_f1']:.4f}  "
                        f"balanced_acc={run_metrics['balanced_acc']:.4f}")
                except Exception:
                    log(f"    FAILED fold={fold_id} mode={mode} seed={seed} -- see traceback below. "
                        f"Continuing to next run.")
                    traceback.print_exc()

                # rebuild the master table + report after every run so a crash never loses progress
                aggregate_and_report(collect_all_run_metrics())

    log("all planned runs attempted for this invocation.")
    log(f"total elapsed: {fmt_hms(time.time() - start_time)}")
    log(f"completed this session: {completed_this_session}/{total_runs}")
    log(f"outputs: {OUT_ROOT}")


if __name__ == "__main__":
    main()
