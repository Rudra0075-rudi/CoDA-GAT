"""
CoDA-GAT — Block B3: baselines.

Three families, all under the SAME 8-fold LODO protocol, splits, and seed
as CoDA-GAT itself:

  1. Classification baselines (spec 12.1)
       (a) 12 tabular  = 3 feature views x 4 classifiers
       (b) 5 graph controls -- the real CoDA-GAT architecture (imported,
           unmodified, from train_fold0_b1.py) but with a corrupted graph
       (c) 3 re-implemented published hybrids (GraphSAGE+SupCon,
           KnowledgeGCN+SetTransformer, Boruta+LASSO->RF/XGBoost)

  2. Coverage-estimation baselines (spec 12.2): BBSE-weighted conformal,
     an OT-based coverage-gap bound, pseudo-calibration (domain-classifier
     density-ratio weighted conformal), ATC, DoC, embedding-distance
     (adversary AUC / MMD / energy distance), and score-CDF distance at
     q-hat -- all evaluated against the main CoDA-GAT model's own
     calibration/test outputs, per fold.

Reuses CoDAGAT, run_epoch, extract_outputs, grad_reverse and all training
hyperparameters from train_fold0_b1.py unchanged. Because B1 hardcodes a
single fold, this file re-implements only the fold-agnostic data-loading
glue (load_split_for_fold) needed to run the same model across all 8
folds -- the model/loss/training-step logic itself is imported, not
rebuilt.

Results are saved incrementally per fold under baselines/, so a long run
that gets interrupted still leaves usable partial CSVs.
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
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.distance import pdist, cdist
from scipy.stats import wasserstein_distance

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from boruta import BorutaPy

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool

from train_fold0_b1 import (
    CoDAGAT,
    run_epoch,
    extract_outputs,
    grad_reverse,
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
    SHARED_DIR,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
OUT_ROOT = REPO_ROOT / "baselines"

ALPHA_CONFORMAL = 0.1  # nominal 90% coverage target for the conformal-style coverage baselines
PATHWAY_COLS = [
    "ECM_remodeling", "TGFB_signaling", "HSC_activation",
    "Inflammation", "Apoptosis_stress", "Hepatocyte_function",
]


def log(msg):
    print(f"[B3] {msg}", flush=True)


def parse_fold_index(dir_name: str) -> int:
    m = re.match(r"fold(\d+)_", dir_name)
    if not m:
        raise ValueError(f"cannot parse fold index from directory name: {dir_name}")
    return int(m.group(1))


# ==========================================================================
# PER-SAMPLE PREDICTION PERSISTENCE (added after the fact for the
# consensus-failure analysis). This section adds NOTHING to the modelling:
# every value written here was already computed and then discarded. No
# model, feature view, threshold-tuning rule, split or seed is touched.
# ==========================================================================
PER_SAMPLE_DIR = OUT_ROOT / "per_sample_predictions"


def per_sample_path(model_name, fold_id, test_cohort):
    return PER_SAMPLE_DIR / model_name / f"fold{fold_id}_{test_cohort}.csv"


def fold_test_cohort(fold_dir):
    with open(fold_dir / "fold_info.json") as f:
        return json.load(f)["test_cohort"]


def save_per_sample_predictions(model_name, fold_id, fold_dir, p_test, threshold, y_test):
    """
    Writes the already-computed test-split probabilities, the already-tuned
    threshold, and the resulting binary predictions for one model on one fold.

    sample_id order comes from folds/<fold>/test/sample_ids.csv, which is the
    same order every B3 code path produced p_test in: build_feature_views
    indexes the expression matrix by that list, and the torch loaders are all
    built shuffle=False over y.csv rows that are asserted to match it.
    """
    test_ids = pd.read_csv(fold_dir / "test" / "sample_ids.csv")["sample_id"].tolist()
    p_test = np.asarray(p_test, dtype=float)
    y_test = np.asarray(y_test, dtype=int)
    if not (len(test_ids) == len(p_test) == len(y_test)):
        raise RuntimeError(
            f"per-sample save refused for {model_name} fold {fold_id}: length mismatch -- "
            f"{len(test_ids)} sample_ids vs {len(p_test)} probabilities vs {len(y_test)} labels."
        )
    test_cohort = fold_test_cohort(fold_dir)
    out_path = per_sample_path(model_name, fold_id, test_cohort)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "sample_id": test_ids,
        "fold": fold_id,
        "test_cohort": test_cohort,
        "p_test": p_test,
        "threshold": float(threshold),
        "pred_test": (p_test >= float(threshold)).astype(int),
        "y_true": y_test,
    }).to_csv(out_path, index=False)


def group_already_persisted(model_names, fold_id, fold_dir):
    """Resumability at (group, fold) granularity: a group is skipped for a fold
    only when every model in it already has its per-sample CSV."""
    test_cohort = fold_test_cohort(fold_dir)
    return all(per_sample_path(m, fold_id, test_cohort).exists() for m in model_names)


def cached_rows_path(group, fold_id):
    return PER_SAMPLE_DIR / "_recomputed_rows" / f"{group}_fold{fold_id}.json"


def save_cached_rows(group, fold_id, rows):
    p = cached_rows_path(group, fold_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(rows, f, indent=2)


def load_cached_rows(group, fold_id):
    p = cached_rows_path(group, fold_id)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


TABULAR_MODEL_NAMES = [f"{v}__{c}" for v in ("bulk_top1000", "panel72_combined", "panel72_full")
                        for c in ("logreg", "random_forest", "xgboost", "catboost")]
BORUTA_MODEL_NAMES = ["boruta_lasso_random_forest", "boruta_lasso_xgboost"]
GNN_HYBRID_MODEL_NAMES = ["graphsage_supcon", "knowledgegcn_settransformer"]
GRAPH_CONTROL_MODEL_NAMES = ["random_graph", "shuffled_graph", "fully_connected",
                             "empty_graph", "flattened_mlp"]


def compute_h1_metrics(y_true, probs, preds):
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    macro_f1 = f1_score(y_true, preds, average="macro")
    bal_acc = balanced_accuracy_score(y_true, preds)
    return auc, macro_f1, bal_acc


def tune_threshold_macro_f1(y_cal, p_cal):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_cal, (p_cal >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ==========================================================================
# Fold-agnostic data loading (generalizes B1's fold0-only load_split)
# ==========================================================================
def load_edge_index_generic():
    edges = pd.read_csv(SHARED_DIR / "graph_edges.csv")
    assert len(edges) == N_EDGES_EXPECTED, f"expected {N_EDGES_EXPECTED} edges, found {len(edges)}"
    return torch.tensor(edges[["src_idx", "dst_idx"]].to_numpy().T, dtype=torch.long)


def load_fold_info(fold_dir: Path):
    with open(fold_dir / "fold_info.json") as f:
        return json.load(f)


def load_split_for_fold(fold_dir, split, edge_index, dev_lookup, cohort_to_idx, node_perm=None):
    """Same construction as B1's load_split, parameterized by fold_dir and
    with an optional node permutation (used by the 'shuffled graph' control)."""
    split_dir = fold_dir / split
    X = np.load(split_dir / "X.npy")
    assert X.shape[1:] == (N_NODES, N_FEATURES)
    if node_perm is not None:
        X = X[:, node_perm, :]

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

        if cohort_to_idx is not None and cohorts[i] in cohort_to_idx:
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


# ==========================================================================
# 1b. Graph controls -- reuse CoDAGAT (or a graph-free MLP) exactly,
#     varying only the edge_index / node ordering / encoder.
# ==========================================================================
def make_random_edge_index(n_nodes, n_directed_edges, rng):
    edges = set()
    while len(edges) < n_directed_edges:
        i, j = rng.integers(0, n_nodes, size=2)
        if i != j:
            edges.add((int(i), int(j)))
    arr = np.array(sorted(edges)).T
    return torch.tensor(arr, dtype=torch.long)


def make_fully_connected_edge_index(n_nodes):
    pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    return torch.tensor(np.array(pairs).T, dtype=torch.long)


def make_empty_edge_index():
    return torch.zeros((2, 0), dtype=torch.long)


class FlattenedMLPModel(nn.Module):
    """The 'no graph at all' control: node features flattened through an
    MLP instead of GAT layers, same 4-head multi-task structure as
    CoDAGAT so it plugs into B1's run_epoch/extract_outputs unchanged."""

    def __init__(self, n_nodes, n_features, n_ordinal_classes, n_cohorts):
        super().__init__()
        self.n_nodes, self.n_features = n_nodes, n_features
        in_dim = n_nodes * n_features
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(128, 32), nn.ELU(), nn.Dropout(0.3),
        )
        self.shared_dense = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.3)
        self.h1 = nn.Linear(16, 2)
        self.h2 = nn.Linear(16, n_ordinal_classes - 1)
        self.h3 = nn.Linear(16, 1)
        self.h4 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_cohorts))

    def forward(self, x, edge_index, batch, grl_lambd=0.0):
        B = int(batch.max().item()) + 1
        x_flat = x.view(B, self.n_nodes * self.n_features)
        h = self.encoder(x_flat)
        shared = F.elu(self.shared_dense(h))
        shared = self.dropout(shared)
        h1_logits = self.h1(shared)
        h2_logits = self.h2(shared)
        h3_pred = self.h3(shared).squeeze(-1)
        reversed_shared = grad_reverse(shared, grl_lambd)
        h4_logits = self.h4(reversed_shared)
        return {"embedding": shared, "h1_logits": h1_logits, "h2_logits": h2_logits,
                "h3_pred": h3_pred, "h4_logits": h4_logits}


def train_coda_gat_variant(fold_dir, edge_index, dev_lookup, seed, node_perm=None, model_factory=None):
    """Trains CoDAGAT (or a drop-in variant via model_factory) on one fold,
    real data only (no B2 augmentation), reusing B1's run_epoch/extract_outputs
    unchanged. Used for the main model AND all 5 graph controls."""
    fold_info = load_fold_info(fold_dir)
    cohort_to_idx = {c: i for i, c in enumerate(sorted(fold_info["train_cohorts"]))}
    n_cohorts = len(cohort_to_idx)

    train_data, train_ids, train_y = load_split_for_fold(fold_dir, "train", edge_index, dev_lookup, cohort_to_idx, node_perm)
    cal_data, cal_ids, cal_y = load_split_for_fold(fold_dir, "calibration", edge_index, dev_lookup, cohort_to_idx, node_perm)
    test_data, test_ids, test_y = load_split_for_fold(fold_dir, "test", edge_index, dev_lookup, cohort_to_idx, node_perm)

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
    factory = model_factory or (lambda: CoDAGAT(N_FEATURES, n_ordinal_classes, n_cohorts))
    model = factory().to(device)
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
    emb_cal, probs_cal, preds_cal, _ = extract_outputs(model, cal_loader, device)
    emb_test, probs_test, preds_test, _ = extract_outputs(model, test_loader, device)

    return dict(
        emb_cal=emb_cal, probs_cal=probs_cal, preds_cal=preds_cal,
        y1_cal=cal_y["y_T1_significant"].to_numpy().astype(int),
        emb_test=emb_test, probs_test=probs_test, preds_test=preds_test,
        y1_test=test_y["y_T1_significant"].to_numpy().astype(int),
        best_val_f1=best_val_f1, n_ordinal_classes=n_ordinal_classes, n_cohorts=n_cohorts,
    )


def run_graph_controls(fold_id, fold_dir, real_edge_index, dev_lookup, main_result, seed):
    rows = []
    auc, f1, bacc = compute_h1_metrics(main_result["y1_test"], main_result["probs_test"], main_result["preds_test"])
    rows.append({"fold": fold_id, "control": "real_graph", "auc": auc, "macro_f1": f1, "balanced_acc": bacc})

    rng = np.random.default_rng(1000 + fold_id)
    n_edges = real_edge_index.shape[1]

    variants = {
        "random_graph": dict(edge_index=make_random_edge_index(N_NODES, n_edges, rng), node_perm=None),
        "shuffled_graph": dict(edge_index=real_edge_index, node_perm=rng.permutation(N_NODES)),
        "fully_connected": dict(edge_index=make_fully_connected_edge_index(N_NODES), node_perm=None),
        "empty_graph": dict(edge_index=make_empty_edge_index(), node_perm=None),
    }
    for name, kw in variants.items():
        log(f"    graph control: {name}")
        result = train_coda_gat_variant(fold_dir, kw["edge_index"], dev_lookup, seed, node_perm=kw["node_perm"])
        auc, f1, bacc = compute_h1_metrics(result["y1_test"], result["probs_test"], result["preds_test"])
        rows.append({"fold": fold_id, "control": name, "auc": auc, "macro_f1": f1, "balanced_acc": bacc})
        # graph controls were evaluated at argmax (preds_test), i.e. threshold 0.5
        save_per_sample_predictions(name, fold_id, fold_dir, result["probs_test"], 0.5, result["y1_test"])

    log("    graph control: flattened_mlp (no graph at all)")
    factory = lambda: FlattenedMLPModel(N_NODES, N_FEATURES, main_result["n_ordinal_classes"], main_result["n_cohorts"])
    result = train_coda_gat_variant(fold_dir, real_edge_index, dev_lookup, seed, model_factory=factory)
    auc, f1, bacc = compute_h1_metrics(result["y1_test"], result["probs_test"], result["preds_test"])
    rows.append({"fold": fold_id, "control": "flattened_mlp", "auc": auc, "macro_f1": f1, "balanced_acc": bacc})
    save_per_sample_predictions("flattened_mlp", fold_id, fold_dir, result["probs_test"], 0.5, result["y1_test"])

    return rows


# ==========================================================================
# 1a. Tabular baselines: 3 feature views x 4 classifiers
# ==========================================================================
def build_feature_views(fold_dir, expr_full, healthy_ref, panel_genes_df):
    train_ids = pd.read_csv(fold_dir / "train" / "sample_ids.csv")["sample_id"].tolist()
    cal_ids = pd.read_csv(fold_dir / "calibration" / "sample_ids.csv")["sample_id"].tolist()
    test_ids = pd.read_csv(fold_dir / "test" / "sample_ids.csv")["sample_id"].tolist()
    y_train = pd.read_csv(fold_dir / "train" / "y.csv", index_col=0)
    y_cal = pd.read_csv(fold_dir / "calibration" / "y.csv", index_col=0)
    y_test = pd.read_csv(fold_dir / "test" / "y.csv", index_col=0)

    views = {}

    # --- view 1: bulk_top1000 -- top-1000 variance genes, ranked on TRAIN only ---
    train_expr_full = expr_full[train_ids]
    top_genes = train_expr_full.var(axis=1).sort_values(ascending=False).index[:1000]
    Xtr = expr_full.loc[top_genes, train_ids].to_numpy().T
    Xca = expr_full.loc[top_genes, cal_ids].to_numpy().T
    Xte = expr_full.loc[top_genes, test_ids].to_numpy().T
    scaler = StandardScaler().fit(Xtr)
    views["bulk_top1000"] = (scaler.transform(Xtr), scaler.transform(Xca), scaler.transform(Xte))

    # --- view 2: panel72_combined -- 72-gene expression (train-fit z) + per-gene
    #     deviation from the healthy reference (a fixed external panel, never
    #     part of train/cal/test in any fold, so this is not fold-dependent) ---
    panel_genes = panel_genes_df["gene"].tolist()
    raw_tr = expr_full.loc[panel_genes, train_ids].to_numpy().T
    raw_ca = expr_full.loc[panel_genes, cal_ids].to_numpy().T
    raw_te = expr_full.loc[panel_genes, test_ids].to_numpy().T
    expr_scaler = StandardScaler().fit(raw_tr)
    expr_tr, expr_ca, expr_te = expr_scaler.transform(raw_tr), expr_scaler.transform(raw_ca), expr_scaler.transform(raw_te)

    healthy_mean = healthy_ref.loc[panel_genes].mean(axis=1).to_numpy()
    healthy_std = np.clip(healthy_ref.loc[panel_genes].std(axis=1).to_numpy(), 1e-6, None)
    dev_tr = (raw_tr - healthy_mean) / healthy_std
    dev_ca = (raw_ca - healthy_mean) / healthy_std
    dev_te = (raw_te - healthy_mean) / healthy_std

    combined_tr = np.concatenate([expr_tr, dev_tr], axis=1)
    combined_ca = np.concatenate([expr_ca, dev_ca], axis=1)
    combined_te = np.concatenate([expr_te, dev_te], axis=1)
    views["panel72_combined"] = (combined_tr, combined_ca, combined_te)

    # --- view 3: panel72_full = combined + 6 pathway-mean features (train-fit z) ---
    pw_tr, pw_ca, pw_te = [], [], []
    for pw in PATHWAY_COLS:
        members = panel_genes_df[panel_genes_df["pathways"].str.contains(pw, na=False)]["gene"].tolist()
        member_idx = [panel_genes.index(g) for g in members]
        pw_tr.append(expr_tr[:, member_idx].mean(axis=1))
        pw_ca.append(expr_ca[:, member_idx].mean(axis=1))
        pw_te.append(expr_te[:, member_idx].mean(axis=1))
    pw_tr, pw_ca, pw_te = np.stack(pw_tr, 1), np.stack(pw_ca, 1), np.stack(pw_te, 1)
    views["panel72_full"] = (
        np.concatenate([combined_tr, pw_tr], axis=1),
        np.concatenate([combined_ca, pw_ca], axis=1),
        np.concatenate([combined_te, pw_te], axis=1),
    )

    labels = dict(
        y_train=y_train["y_T1_significant"].to_numpy().astype(int),
        y_cal=y_cal["y_T1_significant"].to_numpy().astype(int),
        y_test=y_test["y_T1_significant"].to_numpy().astype(int),
    )
    return views, labels


def get_classifiers(seed, scale_pos_weight):
    return {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                                  scale_pos_weight=scale_pos_weight, random_state=seed, n_jobs=-1),
        "catboost": CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                        auto_class_weights="Balanced", random_state=seed, verbose=False),
    }


def run_tabular_baselines(fold_id, views, labels, seed, fold_dir=None):
    rows = []
    n_pos = max((labels["y_train"] == 1).sum(), 1)
    n_neg = max((labels["y_train"] == 0).sum(), 1)
    scale_pos_weight = n_neg / n_pos

    for view_name, (Xtr, Xca, Xte) in views.items():
        for clf_name, clf in get_classifiers(seed, scale_pos_weight).items():
            log(f"    tabular: {view_name} / {clf_name}")
            clf.fit(Xtr, labels["y_train"])
            p_cal = clf.predict_proba(Xca)[:, 1]
            p_test = clf.predict_proba(Xte)[:, 1]
            t, _ = tune_threshold_macro_f1(labels["y_cal"], p_cal)
            pred_test = (p_test >= t).astype(int)
            auc, f1, bacc = compute_h1_metrics(labels["y_test"], p_test, pred_test)
            rows.append({"fold": fold_id, "model": f"{view_name}__{clf_name}",
                         "threshold": t, "auc": auc, "macro_f1": f1, "balanced_acc": bacc})
            if fold_dir is not None:  # persistence only -- added after the fact
                save_per_sample_predictions(f"{view_name}__{clf_name}", fold_id, fold_dir,
                                             p_test, t, labels["y_test"])
    return rows


def run_boruta_lasso_hybrid(fold_id, views, labels, seed, fold_dir=None):
    """Classical pipeline: Boruta feature selection -> LASSO sparsification
    -> RF/XGBoost final classifier, on the bulk_top1000 view, all fit on
    TRAIN only per fold."""
    Xtr, Xca, Xte = views["bulk_top1000"]
    y_train = labels["y_train"]

    rf_for_boruta = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
    selector = BorutaPy(rf_for_boruta, n_estimators="auto", random_state=seed, max_iter=50, verbose=0)
    selector.fit(Xtr, y_train)
    mask = selector.support_
    if mask.sum() == 0:
        mask = selector.support_weak_
    if mask.sum() == 0:
        mask = np.ones(Xtr.shape[1], dtype=bool)
    Xtr_b, Xca_b, Xte_b = Xtr[:, mask], Xca[:, mask], Xte[:, mask]

    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", random_state=seed)
    lasso.fit(Xtr_b, y_train)
    lasso_mask = np.abs(lasso.coef_[0]) > 1e-6
    if lasso_mask.sum() == 0:
        lasso_mask = np.ones(Xtr_b.shape[1], dtype=bool)
    Xtr_f, Xca_f, Xte_f = Xtr_b[:, lasso_mask], Xca_b[:, lasso_mask], Xte_b[:, lasso_mask]

    n_pos = max((y_train == 1).sum(), 1)
    n_neg = max((y_train == 0).sum(), 1)
    scale_pos_weight = n_neg / n_pos

    rows = []
    final_clfs = {
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                                  scale_pos_weight=scale_pos_weight, random_state=seed, n_jobs=-1),
    }
    for clf_name, clf in final_clfs.items():
        clf.fit(Xtr_f, y_train)
        p_cal = clf.predict_proba(Xca_f)[:, 1]
        p_test = clf.predict_proba(Xte_f)[:, 1]
        t, _ = tune_threshold_macro_f1(labels["y_cal"], p_cal)
        pred_test = (p_test >= t).astype(int)
        auc, f1, bacc = compute_h1_metrics(labels["y_test"], p_test, pred_test)
        rows.append({"fold": fold_id, "model": f"boruta_lasso_{clf_name}",
                     "n_features_selected": int(lasso_mask.sum()),
                     "threshold": t,
                     "auc": auc, "macro_f1": f1, "balanced_acc": bacc})
        if fold_dir is not None:  # persistence only -- added after the fact
            save_per_sample_predictions(f"boruta_lasso_{clf_name}", fold_id, fold_dir,
                                         p_test, t, labels["y_test"])
    return rows


# ==========================================================================
# 1c. Re-implemented published hybrids (GNN ones) -- H1-only, since the
#     ordinal/deviation/adversary heads are CoDA-GAT-specific, not part
#     of the published methods being compared against.
# ==========================================================================
def supcon_loss(embeddings, labels, temperature=0.1):
    """Simplified in-batch Supervised Contrastive loss (Khosla et al., 2020)."""
    embeddings = F.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.T / temperature
    labels = labels.view(-1, 1)
    mask_self = torch.eye(len(labels), device=embeddings.device)
    mask_pos = (labels == labels.T).float() - mask_self
    exp_sim = torch.exp(sim) * (1 - mask_self)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    denom = mask_pos.sum(1).clamp(min=1e-12)
    mean_log_prob_pos = (mask_pos * log_prob).sum(1) / denom
    return -mean_log_prob_pos.mean()


class GraphSAGESupConModel(nn.Module):
    """Simplified re-implementation: GraphSAGE encoder + Supervised
    Contrastive auxiliary loss on the pooled embedding, jointly trained
    with H1 cross-entropy."""

    def __init__(self, n_features):
        super().__init__()
        self.sage1 = SAGEConv(n_features, 64)
        self.sage2 = SAGEConv(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.shared_dense = nn.Linear(32, 16)
        self.h1 = nn.Linear(16, 2)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.sage1(x, edge_index)); x = self.dropout(x)
        x = F.elu(self.sage2(x, edge_index)); x = self.dropout(x)
        pooled = global_mean_pool(x, batch)
        shared = F.elu(self.shared_dense(pooled)); shared = self.dropout(shared)
        return {"embedding": shared, "h1_logits": self.h1(shared)}


class PMAPool(nn.Module):
    """Pooling by Multihead Attention -- one learned seed vector attends
    over all node embeddings (Set Transformer, Lee et al., 2019)."""

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, batch):
        B = int(batch.max().item()) + 1
        n_nodes = x.shape[0] // B  # every graph has exactly N_NODES nodes
        x = x.view(B, n_nodes, -1)
        seed = self.seed.expand(B, -1, -1)
        out, _ = self.attn(seed, x, x)
        return self.ln(out.squeeze(1))


class KnowledgeGCNSetTransformerModel(nn.Module):
    """Simplified re-implementation: GCN encoder over the same biological
    (STRING) 'knowledge' graph, pooled via Set-Transformer PMA instead of
    mean pooling."""

    def __init__(self, n_features):
        super().__init__()
        self.gcn1 = GCNConv(n_features, 64)
        self.gcn2 = GCNConv(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.pool = PMAPool(32)
        self.shared_dense = nn.Linear(32, 16)
        self.h1 = nn.Linear(16, 2)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gcn1(x, edge_index)); x = self.dropout(x)
        x = F.elu(self.gcn2(x, edge_index)); x = self.dropout(x)
        pooled = self.pool(x, batch)
        shared = F.elu(self.shared_dense(pooled)); shared = self.dropout(shared)
        return {"embedding": shared, "h1_logits": self.h1(shared)}


def load_split_binary_for_fold(fold_dir, split, edge_index, node_perm=None):
    """Like load_split_for_fold but H1-only -- used for the GNN hybrids,
    which don't carry CoDA-GAT's H2/H3/H4 heads."""
    split_dir = fold_dir / split
    X = np.load(split_dir / "X.npy")
    if node_perm is not None:
        X = X[:, node_perm, :]
    y = pd.read_csv(split_dir / "y.csv", index_col=0)
    sample_ids = pd.read_csv(split_dir / "sample_ids.csv")["sample_id"].tolist()
    y1 = y["y_T1_significant"].to_numpy(dtype=np.float32)
    data_list = []
    for i in range(X.shape[0]):
        x = torch.tensor(X[i], dtype=torch.float32)
        y1_i = torch.tensor(int(y1[i]), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index, y1=y1_i.unsqueeze(0), sample_id=sample_ids[i]))
    return data_list, sample_ids, y


def train_generic_binary_gnn(model_factory, fold_dir, edge_index, seed, aux_loss_fn=None, aux_weight=0.0):
    device = get_device()
    train_data, train_ids, train_y = load_split_binary_for_fold(fold_dir, "train", edge_index)
    cal_data, cal_ids, cal_y = load_split_binary_for_fold(fold_dir, "calibration", edge_index)
    test_data, test_ids, test_y = load_split_binary_for_fold(fold_dir, "test", edge_index)

    y1_train = train_y["y_T1_significant"].to_numpy()
    counts = np.bincount(y1_train.astype(int))
    weights = len(y1_train) / (len(counts) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    strat_key = [f"{c}_{int(y)}" for c, y in zip(train_y["cohort"], y1_train)]
    idx_all = np.arange(len(train_data))
    idx_fit, idx_val = train_test_split(idx_all, test_size=VAL_FRACTION, random_state=seed, stratify=strat_key)
    fit_loader = DataLoader([train_data[i] for i in idx_fit], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader([train_data[i] for i in idx_val], batch_size=BATCH_SIZE, shuffle=False)
    cal_loader = DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    set_seed(seed)
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in fit_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out["h1_logits"], batch.y1.view(-1), weight=class_weights)
            if aux_loss_fn is not None:
                loss = loss + aux_weight * aux_loss_fn(out, batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                val_pred.append(out["h1_logits"].argmax(1).cpu().numpy())
                val_true.append(batch.y1.view(-1).cpu().numpy())
        val_f1 = f1_score(np.concatenate(val_true), np.concatenate(val_pred), average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()

    def extract(loader):
        embs, probs, preds, y_true = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                y_true.append(batch.y1.view(-1).cpu().numpy())
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                probs.append(F.softmax(out["h1_logits"], dim=1)[:, 1].cpu().numpy())
                preds.append(out["h1_logits"].argmax(1).cpu().numpy())
                embs.append(out["embedding"].cpu().numpy())
        return np.concatenate(embs), np.concatenate(probs), np.concatenate(preds), np.concatenate(y_true)

    emb_cal, probs_cal, preds_cal, y1_cal = extract(cal_loader)
    emb_test, probs_test, preds_test, y1_test = extract(test_loader)
    return dict(probs_cal=probs_cal, preds_cal=preds_cal, y1_cal=y1_cal.astype(int),
                probs_test=probs_test, preds_test=preds_test, y1_test=y1_test.astype(int),
                best_val_f1=best_val_f1)


def run_gnn_hybrids(fold_id, fold_dir, edge_index, seed):
    rows = []

    log("    hybrid: GraphSAGE + SupCon")
    def supcon_aux(out, batch):
        return supcon_loss(out["embedding"], batch.y1.view(-1))
    result = train_generic_binary_gnn(lambda: GraphSAGESupConModel(N_FEATURES), fold_dir, edge_index, seed,
                                       aux_loss_fn=supcon_aux, aux_weight=0.5)
    t, _ = tune_threshold_macro_f1(result["y1_cal"], result["probs_cal"])
    pred_test = (result["probs_test"] >= t).astype(int)
    auc, f1, bacc = compute_h1_metrics(result["y1_test"], result["probs_test"], pred_test)
    rows.append({"fold": fold_id, "model": "graphsage_supcon", "threshold": t,
                 "auc": auc, "macro_f1": f1, "balanced_acc": bacc})
    save_per_sample_predictions("graphsage_supcon", fold_id, fold_dir,
                                 result["probs_test"], t, result["y1_test"])

    log("    hybrid: KnowledgeGCN + SetTransformer")
    result = train_generic_binary_gnn(lambda: KnowledgeGCNSetTransformerModel(N_FEATURES), fold_dir, edge_index, seed)
    t, _ = tune_threshold_macro_f1(result["y1_cal"], result["probs_cal"])
    pred_test = (result["probs_test"] >= t).astype(int)
    auc, f1, bacc = compute_h1_metrics(result["y1_test"], result["probs_test"], pred_test)
    rows.append({"fold": fold_id, "model": "knowledgegcn_settransformer", "threshold": t,
                 "auc": auc, "macro_f1": f1, "balanced_acc": bacc})
    save_per_sample_predictions("knowledgegcn_settransformer", fold_id, fold_dir,
                                 result["probs_test"], t, result["y1_test"])

    return rows


# ==========================================================================
# 2. Coverage-estimation baselines
# ==========================================================================
def compute_atc_doc(y_cal, probs_cal, preds_cal, probs_test, preds_test, y_test):
    conf_cal = np.where(preds_cal == 1, probs_cal, 1 - probs_cal)
    conf_test = np.where(preds_test == 1, probs_test, 1 - probs_test)
    acc_cal = (preds_cal == y_cal).mean()
    err_cal = 1 - acc_cal
    t = np.quantile(conf_cal, err_cal)
    atc_pred_acc = float((conf_test >= t).mean())
    doc_pred_acc = float(acc_cal - (conf_cal.mean() - conf_test.mean()))
    actual_acc = float((preds_test == y_test).mean())
    return {
        "atc_pred_acc": atc_pred_acc, "doc_pred_acc": doc_pred_acc, "actual_test_acc": actual_acc,
        "atc_abs_error": abs(atc_pred_acc - actual_acc), "doc_abs_error": abs(doc_pred_acc - actual_acc),
    }


def bbse_weighted_conformal(y_cal, preds_cal, probs_cal, preds_test, probs_test, y_test, alpha):
    """BBSE (Lipton et al.) label-shift weight estimation + weighted
    conformal calibration (Podkopaev & Ramdas)."""
    C = np.zeros((2, 2))
    for true_c in (0, 1):
        mask = y_cal == true_c
        if mask.sum() == 0:
            continue
        for pred_c in (0, 1):
            C[pred_c, true_c] = (preds_cal[mask] == pred_c).mean()
    q_pred = np.array([(preds_test == 0).mean(), (preds_test == 1).mean()])
    try:
        pi_target = np.linalg.solve(C, q_pred)
    except np.linalg.LinAlgError:
        pi_target = np.array([1 - y_cal.mean(), y_cal.mean()])
    pi_target = np.clip(pi_target, 0, None)
    pi_target = pi_target / pi_target.sum() if pi_target.sum() > 0 else np.array([0.5, 0.5])

    pi_cal = np.clip(np.array([(y_cal == 0).mean(), (y_cal == 1).mean()]), 1e-6, None)
    w = np.array([pi_target[c] / pi_cal[c] for c in y_cal])

    nonconf_cal = 1 - np.where(y_cal == 1, probs_cal, 1 - probs_cal)
    order = np.argsort(nonconf_cal)
    sorted_scores, sorted_w = nonconf_cal[order], w[order]
    cum_w = np.cumsum(sorted_w) / sorted_w.sum()
    idx = min(int(np.searchsorted(cum_w, 1 - alpha)), len(sorted_scores) - 1)
    q_hat_weighted = float(sorted_scores[idx])
    q_hat_unweighted = float(np.quantile(nonconf_cal, 1 - alpha))

    nonconf_test = 1 - np.where(y_test == 1, probs_test, 1 - probs_test)
    return {
        "bbse_pi_target_pos": float(pi_target[1]),
        "actual_test_pos_rate": float(y_test.mean()),
        "bbse_prevalence_abs_error": float(abs(pi_target[1] - y_test.mean())),
        "q_hat_unweighted": q_hat_unweighted, "q_hat_weighted": q_hat_weighted,
        "empirical_coverage_unweighted": float((nonconf_test <= q_hat_unweighted).mean()),
        "empirical_coverage_bbse_weighted": float((nonconf_test <= q_hat_weighted).mean()),
        "nominal_coverage": 1 - alpha,
    }


def pseudo_calibration(emb_cal, emb_test, y_cal, probs_cal, y_test, probs_test, alpha, seed):
    """Domain-classifier density-ratio weighted conformal -- a covariate-shift
    domain-adaptation tool in the spirit of Siahkali et al.'s pseudo-calibration
    coverage lower bound. Also yields the adversary-AUC embedding-distance
    baseline as a byproduct (same domain classifier)."""
    X_dom = np.concatenate([emb_cal, emb_test], axis=0)
    y_dom = np.concatenate([np.zeros(len(emb_cal)), np.ones(len(emb_test))])
    p_dom = np.zeros(len(y_dom))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(X_dom, y_dom):
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X_dom[tr_idx], y_dom[tr_idx])
        p_dom[te_idx] = clf.predict_proba(X_dom[te_idx])[:, 1]
    adversary_auc = float(roc_auc_score(y_dom, p_dom))

    p_target_given_cal = np.clip(p_dom[:len(emb_cal)], 1e-3, 1 - 1e-3)
    density_ratio = p_target_given_cal / (1 - p_target_given_cal)

    nonconf_cal = 1 - np.where(y_cal == 1, probs_cal, 1 - probs_cal)
    order = np.argsort(nonconf_cal)
    sorted_scores, sorted_w = nonconf_cal[order], density_ratio[order]
    cum_w = np.cumsum(sorted_w) / sorted_w.sum()
    idx = min(int(np.searchsorted(cum_w, 1 - alpha)), len(sorted_scores) - 1)
    q_hat_pseudo = float(sorted_scores[idx])

    nonconf_test = 1 - np.where(y_test == 1, probs_test, 1 - probs_test)
    return {
        "adversary_auc": adversary_auc,
        "q_hat_pseudo_calibrated": q_hat_pseudo,
        "empirical_coverage_pseudo_calibrated": float((nonconf_test <= q_hat_pseudo).mean()),
    }


def ot_coverage_gap_bound(nonconf_cal, conf_test_proxy_scores):
    """
    Documented approximation of an OT-based coverage-gap bound (spec cites
    arXiv:2507.10425, 'OT bound as predictor'). The exact constant from
    that paper was not available in this handoff, so this computes the
    1-Wasserstein distance between calibration and (label-free, predicted-
    class) test nonconformity-score distributions as a coverage-gap proxy:
    |actual - nominal coverage| ~ W1(P_cal, P_test). Treat this as a
    clearly-labeled, principled proxy -- verify against the actual paper
    before reporting it as an exact reproduction in the write-up.
    """
    w1 = float(wasserstein_distance(nonconf_cal, conf_test_proxy_scores))
    return {"ot_w1_distance": w1, "ot_predicted_coverage_gap": w1}


def score_cdf_distance_at_qhat(nonconf_cal, nonconf_test_true, q_hat):
    f_cal = float((nonconf_cal <= q_hat).mean())
    f_test = float((nonconf_test_true <= q_hat).mean())
    return {"cdf_cal_at_qhat": f_cal, "cdf_test_at_qhat": f_test, "cdf_distance_at_qhat": abs(f_cal - f_test)}


def rbf_mmd2(X, Y):
    XY = np.concatenate([X, Y], axis=0)
    d = pdist(XY, "sqeuclidean")
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    gamma = 1.0 / (2 * med) if med > 0 else 1.0

    def k(A, B):
        A2 = (A ** 2).sum(1)[:, None]
        B2 = (B ** 2).sum(1)[None, :]
        d2 = np.clip(A2 + B2 - 2 * A @ B.T, 0, None)
        return np.exp(-gamma * d2)

    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    m, n = len(X), len(Y)
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) - 2 * Kxy.mean()
    return float(mmd2)


def energy_distance(X, Y):
    dxy, dxx, dyy = cdist(X, Y).mean(), cdist(X, X).mean(), cdist(Y, Y).mean()
    return float(2 * dxy - dxx - dyy)


def run_coverage_baselines(fold_id, main_result, seed, alpha=ALPHA_CONFORMAL):
    y_cal, probs_cal, preds_cal = main_result["y1_cal"], main_result["probs_cal"], main_result["preds_cal"]
    y_test, probs_test, preds_test = main_result["y1_test"], main_result["probs_test"], main_result["preds_test"]
    emb_cal, emb_test = main_result["emb_cal"], main_result["emb_test"]

    row = {"fold": fold_id}
    row.update(compute_atc_doc(y_cal, probs_cal, preds_cal, probs_test, preds_test, y_test))
    row.update(bbse_weighted_conformal(y_cal, preds_cal, probs_cal, preds_test, probs_test, y_test, alpha))
    row.update(pseudo_calibration(emb_cal, emb_test, y_cal, probs_cal, y_test, probs_test, alpha, seed))

    nonconf_cal = 1 - np.where(y_cal == 1, probs_cal, 1 - probs_cal)
    conf_test_proxy = 1 - np.maximum(probs_test, 1 - probs_test)  # label-free proxy score
    row.update(ot_coverage_gap_bound(nonconf_cal, conf_test_proxy))

    q_hat = float(np.quantile(nonconf_cal, 1 - alpha))
    nonconf_test_true = 1 - np.where(y_test == 1, probs_test, 1 - probs_test)
    row.update(score_cdf_distance_at_qhat(nonconf_cal, nonconf_test_true, q_hat))

    row["mmd2_embeddings"] = rbf_mmd2(emb_cal, emb_test)
    row["energy_distance_embeddings"] = energy_distance(emb_cal, emb_test)
    return row


# ==========================================================================
# Summary
# ==========================================================================
def build_summary(all_tabular, all_graph, all_hybrid, all_coverage):
    def summarize(rows, group_col, metric_cols):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        agg = df.groupby(group_col)[metric_cols].agg(["mean", "std"])
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        return agg.reset_index()

    metric_cols = ["auc", "macro_f1", "balanced_acc"]
    summarize(all_tabular, "model", metric_cols).to_csv(OUT_ROOT / "summary_tabular.csv", index=False)
    summarize(all_graph, "control", metric_cols).to_csv(OUT_ROOT / "summary_graph_controls.csv", index=False)
    summarize(all_hybrid, "model", metric_cols).to_csv(OUT_ROOT / "summary_hybrids.csv", index=False)

    if all_coverage:
        cov_df = pd.DataFrame(all_coverage)
        cov_metric_cols = [c for c in cov_df.columns if c != "fold"]
        cov_summary = cov_df[cov_metric_cols].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
        cov_summary.to_csv(OUT_ROOT / "summary_coverage.csv", index=False)

    log("summaries saved: summary_tabular.csv, summary_graph_controls.csv, summary_hybrids.csv, summary_coverage.csv")


# ==========================================================================
# Main
# ==========================================================================
def check_metric_reproduction(group, recomputed_rows, original_csv, key_col, tol):
    """
    Compares metrics recomputed during a persistence re-run against the
    ORIGINAL B3 values. The persistence patch must not have changed any
    modelling, so these should match. Divergence is surfaced loudly rather
    than raised, so the run still finishes and you can inspect the diff.
    """
    if not recomputed_rows:
        return None
    if not original_csv.exists():
        log(f"  [check] no original {original_csv.name} to compare {group} against -- skipped")
        return None

    orig = pd.read_csv(original_csv)
    recomp = pd.DataFrame(recomputed_rows)
    metrics = [m for m in ("auc", "macro_f1", "balanced_acc") if m in orig.columns and m in recomp.columns]
    merged = recomp.merge(orig, on=["fold", key_col], suffixes=("_new", "_orig"), how="inner")
    if merged.empty:
        log(f"  [check] {group}: no overlapping (fold, {key_col}) rows to compare")
        return None

    for m in metrics:
        merged[f"{m}_absdiff"] = (merged[f"{m}_new"] - merged[f"{m}_orig"]).abs()
    diff_cols = [f"{m}_absdiff" for m in metrics]
    merged["max_absdiff"] = merged[diff_cols].max(axis=1)
    bad = merged[merged["max_absdiff"] > tol]

    out_path = PER_SAMPLE_DIR / f"metric_reproduction_check_{group}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    if bad.empty:
        log(f"  [check] {group}: all {len(merged)} rows reproduce the original metrics "
            f"within tol={tol} -- persistence patch changed nothing.")
    else:
        log("")
        log("  " + "!" * 72)
        log(f"  [check] WARNING -- {group}: {len(bad)}/{len(merged)} rows DIVERGE from the "
            f"original B3 metrics by more than tol={tol}.")
        log(f"  This means something other than persistence changed. Inspect {out_path.name}.")
        for _, r in bad.head(10).iterrows():
            log(f"    fold {int(r['fold'])} {r[key_col]}: max abs diff = {r['max_absdiff']:.6f}")
        log("  " + "!" * 72)
        log("")
    return merged


def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT Block B3 -- baselines")
    parser.add_argument("--folds", nargs="*", default=None, help="fold dir names (default: all 8)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-graph-controls", action="store_true")
    parser.add_argument("--skip-hybrids", action="store_true")
    parser.add_argument("--only", choices=["all", "tabular", "gnn-hybrids", "graph-controls"],
                         default="all",
                         help="'all' = the original full B3 pipeline (writes the canonical "
                              "baselines/*/all_folds.csv). Anything else is a PERSISTENCE RE-RUN: "
                              "it regenerates per-sample predictions for that group only, writes "
                              "recomputed metrics to a separate file, and does NOT touch the "
                              "existing baselines/ record.")
    parser.add_argument("--metric-tol", type=float, default=1e-6,
                         help="tolerance for the deterministic groups (tabular, boruta)")
    parser.add_argument("--metric-tol-torch", type=float, default=0.02,
                         help="tolerance for torch-trained groups (GNN hybrids, graph controls), "
                              "which are not bit-reproducible on MPS")
    args = parser.parse_args()

    persistence_rerun = args.only != "all"

    for sub in ("tabular", "graph_controls", "hybrids", "coverage"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    PER_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()) if args.folds is None \
        else [FOLDS_DIR / f for f in args.folds]

    log("CoDA-GAT Block B3 -- baselines")
    log(f"folds: {[d.name for d in fold_dirs]}  seed={args.seed}  mode={args.only}")
    if persistence_rerun:
        log("PERSISTENCE RE-RUN: regenerating per-sample predictions only. The existing")
        log("baselines/*/all_folds.csv and summary_*.csv are NOT modified; recomputed metrics")
        log("go to baselines/per_sample_predictions/ and are checked against the originals.")
    log("loading shared assets...")
    real_edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")
    panel_genes_df = pd.read_csv(SHARED_DIR / "panel_genes.csv")

    want_tabular = args.only in ("all", "tabular")
    want_gnn = args.only in ("all", "gnn-hybrids") and not args.skip_hybrids
    want_graph = args.only in ("all", "graph-controls") and not args.skip_graph_controls
    want_coverage = args.only == "all"
    # the main CoDA-GAT run is only needed by graph controls and coverage --
    # skipping it is what makes the tabular-only stage cheap
    need_main_model = want_graph or want_coverage

    expr_full = healthy_ref = None
    if want_tabular:
        expr_full = pd.read_parquet(SHARED_DIR / "bulk_expression.parquet")
        healthy_ref = pd.read_parquet(SHARED_DIR / "geo_control_reference.parquet")

    all_tabular, all_graph, all_hybrid, all_coverage = [], [], [], []

    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        log(f"=== fold {fold_id}: {fold_dir.name} ===")

        main_result = None
        if need_main_model:
            log("  training main CoDA-GAT (needed by graph controls / coverage)...")
            main_result = train_coda_gat_variant(fold_dir, real_edge_index, dev_lookup, args.seed)
            log(f"    main model best_val_f1={main_result['best_val_f1']:.4f}")

        if want_tabular:
            tab_names = TABULAR_MODEL_NAMES + BORUTA_MODEL_NAMES
            cached = load_cached_rows("tabular", fold_id)
            if group_already_persisted(tab_names, fold_id, fold_dir) and cached is not None:
                log("  [tabular] already persisted for this fold -- skipping (resumable)")
                all_tabular.extend(cached)
            else:
                log("  [tabular] building feature views + 12 tabular baselines + boruta/lasso...")
                views, labels = build_feature_views(fold_dir, expr_full, healthy_ref, panel_genes_df)
                rows = run_tabular_baselines(fold_id, views, labels, args.seed, fold_dir=fold_dir)
                rows += run_boruta_lasso_hybrid(fold_id, views, labels, args.seed, fold_dir=fold_dir)
                save_cached_rows("tabular", fold_id, rows)
                all_tabular.extend(rows)

        if want_graph:
            cached = load_cached_rows("graph_controls", fold_id)
            if group_already_persisted(GRAPH_CONTROL_MODEL_NAMES, fold_id, fold_dir) and cached is not None:
                log("  [graph controls] already persisted for this fold -- skipping (resumable)")
                all_graph.extend(cached)
            else:
                log("  [graph controls] 5 controls (incl. flattened_mlp, the only non-GAT one)...")
                rows = run_graph_controls(fold_id, fold_dir, real_edge_index, dev_lookup,
                                           main_result, args.seed)
                save_cached_rows("graph_controls", fold_id, rows)
                all_graph.extend(rows)

        if want_gnn:
            cached = load_cached_rows("gnn_hybrids", fold_id)
            if group_already_persisted(GNN_HYBRID_MODEL_NAMES, fold_id, fold_dir) and cached is not None:
                log("  [gnn hybrids] already persisted for this fold -- skipping (resumable)")
                all_hybrid.extend(cached)
            else:
                log("  [gnn hybrids] GraphSAGE+SupCon, KnowledgeGCN+SetTransformer (retraining)...")
                rows = run_gnn_hybrids(fold_id, fold_dir, real_edge_index, args.seed)
                save_cached_rows("gnn_hybrids", fold_id, rows)
                all_hybrid.extend(rows)

        if want_coverage:
            log("  [coverage] coverage-estimation baselines...")
            all_coverage.append(run_coverage_baselines(fold_id, main_result, args.seed))

        if not persistence_rerun:
            # original behaviour: keep writing the canonical incremental files
            if all_tabular:
                pd.DataFrame(all_tabular).to_csv(OUT_ROOT / "tabular" / "all_folds.csv", index=False)
            if all_graph:
                pd.DataFrame(all_graph).to_csv(OUT_ROOT / "graph_controls" / "all_folds.csv", index=False)
            if all_hybrid:
                pd.DataFrame(all_hybrid).to_csv(OUT_ROOT / "hybrids" / "all_folds.csv", index=False)
            if all_coverage:
                pd.DataFrame(all_coverage).to_csv(OUT_ROOT / "coverage" / "all_folds.csv", index=False)

    if persistence_rerun:
        log("verifying the persistence patch changed nothing...")
        tab_only = [r for r in all_tabular if r["model"] in TABULAR_MODEL_NAMES]
        bor_only = [r for r in all_tabular if r["model"] in BORUTA_MODEL_NAMES]
        if tab_only:
            pd.DataFrame(all_tabular).to_csv(PER_SAMPLE_DIR / "recomputed_metrics_tabular.csv", index=False)
            check_metric_reproduction("tabular", tab_only, OUT_ROOT / "tabular" / "all_folds.csv",
                                       "model", args.metric_tol)
        if bor_only:
            check_metric_reproduction("boruta_lasso", bor_only, OUT_ROOT / "hybrids" / "all_folds.csv",
                                       "model", args.metric_tol)
        if all_hybrid:
            pd.DataFrame(all_hybrid).to_csv(PER_SAMPLE_DIR / "recomputed_metrics_gnn_hybrids.csv", index=False)
            check_metric_reproduction("gnn_hybrids", all_hybrid, OUT_ROOT / "hybrids" / "all_folds.csv",
                                       "model", args.metric_tol_torch)
        if all_graph:
            pd.DataFrame(all_graph).to_csv(PER_SAMPLE_DIR / "recomputed_metrics_graph_controls.csv", index=False)
            check_metric_reproduction("graph_controls", all_graph,
                                       OUT_ROOT / "graph_controls" / "all_folds.csv",
                                       "control", args.metric_tol_torch)
        log(f"per-sample predictions written under {PER_SAMPLE_DIR}")
        log("Persistence re-run complete. Original baselines/ record untouched.")
    else:
        log("building summary (mean +/- std across folds)...")
        build_summary(all_tabular, all_graph, all_hybrid, all_coverage)
        log("Block B3 complete.")


if __name__ == "__main__":
    main()
