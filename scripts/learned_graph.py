"""
CoDA-GAT — Learned-graph variants: does a task-learned topology beat generic
STRING biology under missingness?

THE QUESTION (everything is framed around this): not "does it beat logistic
regression" -- the label ceiling blocks that and it is accepted. The question is
whether a topology learned from fibrosis training data recovers more information
under missing genes than generic STRING protein interactions, and whether it
beats mere random connectivity. Evaluation is on the MASKING CURVE, never on
clean-data AUC.

Known starting point (from graph_utility.py, unchanged and not revisited):
  real STRING vs empty  = +0.0243 AUC  [95% CI +0.0058, +0.0428]
  shuffled     vs empty = +0.0189
  -> so ~+0.005 of the advantage is attributable to real topology rather than
     connectivity per se. That ~+0.005 gap is what this script targets.

TWO LEARNED VARIANTS, both at STRING's exact edge budget (321 undirected / 642
directed), because density would otherwise be a confound:

  coexpr_graph -- top-|correlation| gene pairs in that fold's TRAINING SPLIT
  e2e_graph    -- two phase. Phase 1 trains a GAT over a FULLY-CONNECTED graph
                  with a learnable symmetric per-edge scalar fed to GATConv as
                  edge_attr, then keeps the top-321 pairs. Phase 2 retrains a
                  standard CoDAGAT from scratch on that fixed sparse topology,
                  so the evaluated model is ARCHITECTURALLY IDENTICAL to the
                  STRING/shuffled/random models -- only the edge list differs.

The Phase-1 topology is learned ONCE per fold at a fixed seed, not per training
seed, so that (a) the 5 seeds differ only in training stochasticity and (b) the
cross-fold topology-stability question in Step 5 is well posed.

ADDITIVE ONLY: nothing under results_b7/, ablations_b9/ or graph_utility/ is
written. All prior findings stand.
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
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

import train_fold0_b1 as b1
from train_fold0_b1 import (
    CoDAGAT, run_epoch, grad_reverse, get_device, set_seed,
    N_NODES, N_FEATURES, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY,
    VAL_FRACTION, SEED, SHARED_DIR,
)
from run_full_experiment_b7 import (
    extract_full, load_edge_index_generic, load_fold_info, parse_fold_index, fmt_hms,
)
from train_fold0_b2_onthefly import load_panel_gene_order, load_raw_panel_expression
from ablations_b9 import load_split_for_ablation
# masking protocol -- imported, NOT redefined, so the new curves are directly
# comparable to graph_utility.py's existing ones
from graph_utility import (
    apply_mask, batched_forward_probs, control_run_dir, _control_graph,
    DYNAMIC_FEATURE_SLICE, MASK_RATES, N_MASK_DRAWS, MASK_DRAW_BASE_SEED,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
RESULTS_B7 = REPO_ROOT / "results_b7"
GRAPH_UTIL_DIR = REPO_ROOT / "graph_utility"
OUT_DIR = REPO_ROOT / "learned_graph"
CKPT_DIR = OUT_DIR / "checkpoints"
EDGES_DIR = OUT_DIR / "edges"

SEEDS = [42, 43, 44, 45, 46]
TOPOLOGY_SEED = 42          # Phase-1 topology is learned once per fold at this seed
N_UNDIRECTED_STRING = 321   # the edge budget every condition must match

LEARNED_VARIANTS = ["coexpr_graph", "e2e_graph"]
CONTROL_VARIANTS = ["random_graph"]                        # no checkpoints exist anywhere yet
REUSED_VARIANTS = ["real_graph", "empty_graph", "shuffled_graph"]  # weights already on disk
ALL_CONDITIONS = REUSED_VARIANTS + CONTROL_VARIANTS + LEARNED_VARIANTS

# canonical fibrosis genes, for the hub sanity check in Step 5
FIBROSIS_CORE_GENES = {"COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1",
                        "TGFB1", "TGFBR1", "TGFBR2", "ACTA2", "TIMP1", "SPARC", "BGN", "LUM"}


def log(msg):
    print(f"[learned-graph] {msg}", flush=True)


# ==========================================================================
# Masking subsets -- formula copied VERBATIM from graph_utility.py's run_test1
# so the gene subsets at every (fold, rate, draw) are bit-identical
# ==========================================================================
def mask_subsets(fold_id, rates, n_draws):
    draws = {}
    for rate in rates:
        k = int(round(rate * N_NODES))
        draws[rate] = []
        for d in range(1 if k == 0 else n_draws):
            rng = np.random.default_rng(MASK_DRAW_BASE_SEED + fold_id * 1000 + int(rate * 100) * 10 + d)
            draws[rate].append(rng.choice(N_NODES, size=k, replace=False) if k else np.array([], dtype=int))
    return draws


# ==========================================================================
# FATAL GUARD -- learned topology must never see calibration or test
# ==========================================================================
def assert_graph_learned_from_train_only(fold_id, variant, used_sample_ids, fold_dir):
    """
    Hard assertion (not a comment). A topology that saw the held-out cohort or
    the calibration cohort would make the entire experiment circular, so any
    violation raises rather than warns.
    """
    train_ids = set(pd.read_csv(fold_dir / "train" / "sample_ids.csv")["sample_id"])
    cal_ids = set(pd.read_csv(fold_dir / "calibration" / "sample_ids.csv")["sample_id"])
    test_ids = set(pd.read_csv(fold_dir / "test" / "sample_ids.csv")["sample_id"])
    used = set(used_sample_ids)

    leaked_test = used & test_ids
    if leaked_test:
        raise RuntimeError(
            f"FATAL [fold {fold_id}, {variant}]: the learned topology used {len(leaked_test)} "
            f"HELD-OUT TEST sample(s), e.g. {sorted(leaked_test)[:5]}. The experiment would be "
            f"circular. Refusing to continue."
        )
    leaked_cal = used & cal_ids
    if leaked_cal:
        raise RuntimeError(
            f"FATAL [fold {fold_id}, {variant}]: the learned topology used {len(leaked_cal)} "
            f"CALIBRATION sample(s). Refusing to continue."
        )
    not_train = used - train_ids
    if not_train:
        raise RuntimeError(
            f"FATAL [fold {fold_id}, {variant}]: {len(not_train)} sample(s) used by the topology "
            f"are not in this fold's training split at all. Refusing to continue."
        )
    test_cohort = load_fold_info(fold_dir)["test_cohort"]
    train_cohorts = set(pd.read_csv(fold_dir / "train" / "y.csv", index_col=0)["cohort"])
    if test_cohort in train_cohorts:
        raise RuntimeError(
            f"FATAL [fold {fold_id}]: test cohort '{test_cohort}' appears in the training split. "
            f"LODO is violated. Refusing to continue."
        )
    log(f"    GUARD PASSED [{variant}]: topology built from {len(used)} training samples only; "
        f"calibration ({len(cal_ids)}) and test ({len(test_ids)}) untouched.")


# ==========================================================================
# Edge-list helpers -- every condition gets exactly STRING's budget
# ==========================================================================
def undirected_pairs_to_edge_index(pairs):
    src, dst = [], []
    for i, j in pairs:
        src += [i, j]
        dst += [j, i]
    return torch.tensor([src, dst], dtype=torch.long)


def edge_index_to_pair_set(edge_index):
    return {(min(int(s), int(d)), max(int(s), int(d)))
            for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()) if int(s) != int(d)}


def save_learned_edges(variant, fold_id, pairs, scores, gene_order, string_pairs):
    EDGES_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"rank": r + 1, "src_idx": i, "dst_idx": j,
             "src": gene_order[i], "dst": gene_order[j],
             "score": float(s), "in_string": (min(i, j), max(i, j)) in string_pairs}
            for r, ((i, j), s) in enumerate(zip(pairs, scores))]
    pd.DataFrame(rows).to_csv(EDGES_DIR / f"learned_edges_fold{fold_id}_{variant}.csv", index=False)


# ==========================================================================
# Variant 1 -- co-expression topology (training split only)
# ==========================================================================
def build_coexpression_topology(fold_dir, gene_order, expr_panel, n_undirected):
    fold_id = parse_fold_index(fold_dir.name)
    train_ids = pd.read_csv(fold_dir / "train" / "sample_ids.csv")["sample_id"].tolist()
    assert_graph_learned_from_train_only(fold_id, "coexpr_graph", train_ids, fold_dir)

    # (n_train x 72) log2 expression, training samples only
    X = expr_panel.loc[gene_order, train_ids].to_numpy(dtype=np.float64).T
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    scored = []
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            scored.append(((i, j), abs(corr[i, j])))
    scored.sort(key=lambda t: -t[1])
    top = scored[:n_undirected]
    return [p for p, _ in top], [s for _, s in top]


# ==========================================================================
# Variant 2 -- end-to-end learned adjacency (Phase 1), then sparsify
# ==========================================================================
def fully_connected_with_pair_ids(n_nodes):
    pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    pair_id = {p: k for k, p in enumerate(pairs)}
    src, dst, pid = [], [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            pid.append(pair_id[(min(i, j), max(i, j))])
    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(pid, dtype=torch.long), pairs)


class LearnableAdjacencyGAT(nn.Module):
    """
    Phase-1 topology learner: the CoDA-GAT encoder over a FULLY-CONNECTED graph,
    with one learnable scalar per UNORDERED gene pair fed to GATConv as edge_attr
    (so it is symmetric by construction and genuinely participates in attention).
    After training, the top-k pairs by learned weight become the topology.

    This class is only ever used to LEARN the edge list. The model actually
    evaluated in Phase 2 is a stock CoDAGAT, so no architectural difference
    contaminates the masking comparison.
    """

    def __init__(self, n_features, n_ordinal_classes, n_cohorts, pair_id_per_edge, n_pairs):
        super().__init__()
        self.register_buffer("pair_id_per_edge", pair_id_per_edge)
        self.edge_logit = nn.Parameter(torch.zeros(n_pairs))
        self.gat1 = GATConv(n_features, 32, heads=4, concat=True, dropout=0.3,
                             add_self_loops=True, edge_dim=1)
        self.gat2 = GATConv(32 * 4, 32, heads=1, concat=False, dropout=0.3,
                             add_self_loops=True, edge_dim=1)
        self.shared_dense = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.3)
        self.h1 = nn.Linear(16, 2)
        self.h2 = nn.Linear(16, n_ordinal_classes - 1)
        self.h3 = nn.Linear(16, 1)
        self.h4 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(32, n_cohorts))

    def _edge_attr(self, edge_index):
        n_per_graph = self.pair_id_per_edge.numel()
        total = edge_index.shape[1]
        if total % n_per_graph != 0:
            raise RuntimeError(
                f"batched edge count {total} is not a multiple of the per-graph fully-connected "
                f"edge count {n_per_graph}; cannot map learned weights onto edges."
            )
        w = self.edge_logit[self.pair_id_per_edge]
        return w.repeat(total // n_per_graph).unsqueeze(-1)

    def forward(self, x, edge_index, batch, grl_lambd=0.0):
        ea = self._edge_attr(edge_index)
        h = self.dropout(F.elu(self.gat1(x, edge_index, edge_attr=ea)))
        h = self.dropout(F.elu(self.gat2(h, edge_index, edge_attr=ea)))
        shared = self.dropout(F.elu(self.shared_dense(global_mean_pool(h, batch))))
        return {"embedding": shared,
                "h1_logits": self.h1(shared), "h2_logits": self.h2(shared),
                "h3_pred": self.h3(shared).squeeze(-1),
                "h4_logits": self.h4(grad_reverse(shared, grl_lambd))}


def learn_e2e_topology(fold_dir, dev_lookup, n_undirected, epochs):
    fold_id = parse_fold_index(fold_dir.name)
    train_ids = pd.read_csv(fold_dir / "train" / "sample_ids.csv")["sample_id"].tolist()
    assert_graph_learned_from_train_only(fold_id, "e2e_graph", train_ids, fold_dir)

    fc_ei, pair_ids, pairs = fully_connected_with_pair_ids(N_NODES)
    setup = setup_fold_custom_graph(fold_dir, fc_ei, dev_lookup)
    device = get_device()

    set_seed(TOPOLOGY_SEED)
    model = LearnableAdjacencyGAT(N_FEATURES, setup["n_ordinal_classes"], setup["n_cohorts"],
                                   pair_ids, len(pairs)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_weights = setup["class_weights"].to(device)

    original_h4 = b1.LOSS_WEIGHTS["h4"]
    b1.LOSS_WEIGHTS["h4"] = 0.0  # adversary off, as in every other condition
    try:
        # One code path only: the SAME multi-task objective and loop every other
        # condition trains under (run_epoch, adversary off). No sparsity penalty
        # is applied -- the top-k cut is the only sparsification, so the edge
        # budget rather than a tuned regulariser determines the topology.
        best_val, best_logits, bad = -1.0, None, 0
        for epoch in range(1, epochs + 1):
            p = epoch / epochs
            run_epoch(model, setup["fit_loader"], device, optimizer, class_weights,
                      setup["n_ordinal_classes"], p, train=True)
            _, val_f1 = run_epoch(model, setup["val_loader"], device, optimizer, class_weights,
                                   setup["n_ordinal_classes"], p, train=False)
            if val_f1 > best_val:
                best_val = val_f1
                best_logits = model.edge_logit.detach().cpu().numpy().copy()
                bad = 0
            else:
                bad += 1
            if bad >= PATIENCE:
                break
    finally:
        b1.LOSS_WEIGHTS["h4"] = original_h4

    log(f"    phase-1 topology learned (best val macro-F1={best_val:.4f})")
    order = np.argsort(-best_logits)[:n_undirected]
    return [pairs[k] for k in order], [float(best_logits[k]) for k in order]


# ==========================================================================
# Training on an arbitrary topology (Phase 2 / controls)
# ==========================================================================
def setup_fold_custom_graph(fold_dir, edge_index, dev_lookup):
    """Mirrors ablations_b9.setup_fold_ablation, but takes an explicit
    edge_index -- b9's version can only express its own named configs."""
    fold_info = load_fold_info(fold_dir)
    cohort_to_idx = {c: i for i, c in enumerate(sorted(fold_info["train_cohorts"]))}

    train_data, train_ids, train_y = load_split_for_ablation(
        fold_dir, "train", edge_index, dev_lookup, cohort_to_idx)
    cal_data, cal_ids, cal_y = load_split_for_ablation(
        fold_dir, "calibration", edge_index, dev_lookup, cohort_to_idx)
    test_data, test_ids, test_y = load_split_for_ablation(
        fold_dir, "test", edge_index, dev_lookup, cohort_to_idx)

    n_ordinal_classes = int(np.nanmax(
        pd.concat([train_y["y_T2_ordinal"], cal_y["y_T2_ordinal"], test_y["y_T2_ordinal"]]))) + 1
    y1_train = train_y["y_T1_significant"].to_numpy()
    counts = np.bincount(y1_train.astype(int))
    class_weights = torch.tensor(len(y1_train) / (len(counts) * counts), dtype=torch.float32)

    strat_key = [f"{c}_{int(y)}" for c, y in zip(train_y["cohort"], y1_train)]
    idx_fit, idx_val = train_test_split(np.arange(len(train_data)), test_size=VAL_FRACTION,
                                         random_state=SEED, stratify=strat_key)
    return dict(
        fold_info=fold_info, n_ordinal_classes=n_ordinal_classes, n_cohorts=len(cohort_to_idx),
        class_weights=class_weights,
        fit_loader=DataLoader([train_data[i] for i in idx_fit], batch_size=BATCH_SIZE, shuffle=True),
        val_loader=DataLoader([train_data[i] for i in idx_val], batch_size=BATCH_SIZE, shuffle=False),
        cal_loader=DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False),
        test_loader=DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False),
        y_test=test_y["y_T1_significant"].to_numpy().astype(int),
    )


def variant_run_dir(variant, fold_id, test_cohort, seed):
    return CKPT_DIR / f"fold{fold_id}_{test_cohort}" / variant / f"seed_{seed}"


def train_on_graph(variant, fold_dir, setup, seed, run_dir):
    """Identical hyperparameters and loop to B7/B9 (adversary off). Only the
    graph differs between conditions."""
    device = get_device()
    set_seed(seed)
    model = CoDAGAT(N_FEATURES, setup["n_ordinal_classes"], setup["n_cohorts"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_weights = setup["class_weights"].to(device)

    original_h4 = b1.LOSS_WEIGHTS["h4"]
    b1.LOSS_WEIGHTS["h4"] = 0.0
    try:
        best_val, best_state, bad = -1.0, None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            p = epoch / MAX_EPOCHS
            run_epoch(model, setup["fit_loader"], device, optimizer, class_weights,
                      setup["n_ordinal_classes"], p, train=True)
            _, val_f1 = run_epoch(model, setup["val_loader"], device, optimizer, class_weights,
                                   setup["n_ordinal_classes"], p, train=False)
            if val_f1 > best_val:
                best_val = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= PATIENCE:
                break
    finally:
        b1.LOSS_WEIGHTS["h4"] = original_h4

    model.load_state_dict(best_state)
    _, probs_test, preds_test, _ = extract_full(model, setup["test_loader"], device)
    y_test = setup["y_test"]

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "variant": variant, "seed": seed,
                "best_val_f1": best_val, "n_ordinal_classes": setup["n_ordinal_classes"],
                "n_cohorts": setup["n_cohorts"]}, run_dir / "weights.pt")
    np.save(run_dir / "probs_test.npy", probs_test)
    np.save(run_dir / "predictions_test.npy", preds_test)
    metrics = {
        "variant": variant, "fold": parse_fold_index(fold_dir.name), "seed": seed,
        "best_val_f1": best_val,
        "auc": float(roc_auc_score(y_test, probs_test[:, 1])) if len(np.unique(y_test)) > 1 else float("nan"),
        "macro_f1": float(f1_score(y_test, preds_test, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_test, preds_test)),
    }
    with open(run_dir / "run_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    (run_dir / "DONE.marker").touch()
    return metrics


# ==========================================================================
# Masking evaluation -- the protocol imported from graph_utility, unchanged
# ==========================================================================
def condition_checkpoint(condition, fold_id, test_cohort, seed):
    """Where each condition's weights live. real_graph comes from B7;
    empty/shuffled from graph_utility's retrain; everything else from here."""
    if condition == "real_graph":
        return RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / "mode_off" / f"seed_{seed}" / "weights.pt"
    if condition in ("empty_graph", "shuffled_graph"):
        return control_run_dir(condition, fold_id, test_cohort, seed) / "weights.pt"
    return variant_run_dir(condition, fold_id, test_cohort, seed) / "weights.pt"


def condition_graph(condition, fold_id, real_edge_index, learned_pairs):
    """(edge_index, node_perm) for a condition. Learned variants use their
    saved per-fold edge list; controls reuse B9/graph_utility's constructors."""
    if condition == "real_graph":
        return real_edge_index, None
    if condition in ("empty_graph", "shuffled_graph", "random_graph"):
        return _control_graph(condition, real_edge_index, fold_id)
    return undirected_pairs_to_edge_index(learned_pairs[(condition, fold_id)]), None


def audit_edge_budgets(fold_ids, conditions, real_edge_index, learned_pairs):
    """
    Measures each condition's actual edge counts rather than assuming they match.

    This matters: STRING, coexpr and e2e are symmetric (321 undirected pairs
    mirrored to 642 directed). B9's random_graph control, which this analysis
    reuses unchanged, samples 642 unique DIRECTED edges without enforcing
    symmetry -- so it holds the directed count constant but can span up to 642
    distinct undirected pairs. The directed budget is therefore matched across
    all conditions; the undirected pair count is not, for that one control.
    Reported explicitly so no comparison rests on an unchecked density claim.
    """
    rows = []
    for fold_id in fold_ids:
        for cond in conditions:
            if cond in ("coexpr_graph", "e2e_graph"):
                if (cond, fold_id) not in learned_pairs:
                    continue
                ei = undirected_pairs_to_edge_index(learned_pairs[(cond, fold_id)])
            elif cond == "real_graph":
                ei = real_edge_index
            else:
                ei, _ = _control_graph(cond, real_edge_index, fold_id)
            pair_set = edge_index_to_pair_set(ei)
            n_dir = int(ei.shape[1])
            rows.append({
                "fold": fold_id, "condition": cond,
                "n_directed_edges": n_dir,
                "n_undirected_pairs": len(pair_set),
                "symmetric": bool(n_dir == 2 * len(pair_set)),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUT_DIR / "edge_budget_audit.csv", index=False)
    return df


def run_masking(fold_dirs, conditions, real_edge_index, learned_pairs, args):
    log("masking sweep (inference only; protocol imported from graph_utility.py)")
    rows_path = OUT_DIR / "masking_curves_all_conditions.csv"
    rows = []
    done = set()
    if rows_path.exists() and not args.recompute_masking:
        rows = pd.read_csv(rows_path).to_dict("records")
        done = {(r["model"], r["fold"], r["seed"]) for r in rows}
        log(f"  resuming: {len(done)} (condition, fold, seed) combinations already done")

    device = get_device()
    t0 = time.time()
    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        test_cohort = load_fold_info(fold_dir)["test_cohort"]
        X_test_raw = np.load(fold_dir / "test" / "X.npy")
        y_test = pd.read_csv(fold_dir / "test" / "y.csv", index_col=0)["y_T1_significant"].to_numpy().astype(int)
        train_node_means = np.load(fold_dir / "train" / "X.npy").mean(axis=0)
        draws = mask_subsets(fold_id, args.mask_rates, args.n_draws)

        for condition in conditions:
            for seed in args.seeds:
                if (condition, fold_id, seed) in done:
                    continue
                ckpt_path = condition_checkpoint(condition, fold_id, test_cohort, seed)
                if not ckpt_path.exists():
                    log(f"    MISSING {ckpt_path.relative_to(REPO_ROOT)} -- skipping "
                        f"{condition} fold{fold_id} seed{seed}")
                    continue
                ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
                model = CoDAGAT(N_FEATURES, ckpt["n_ordinal_classes"], ckpt["n_cohorts"]).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                edge_index, node_perm = condition_graph(condition, fold_id, real_edge_index, learned_pairs)

                for rate, subsets in draws.items():
                    for d, masked_nodes in enumerate(subsets):
                        Xm = apply_mask(X_test_raw, masked_nodes, train_node_means)
                        if node_perm is not None:
                            Xm = Xm[:, node_perm, :]
                        p = batched_forward_probs(model, Xm, edge_index, device)
                        pred = (p >= 0.5).astype(int)
                        rows.append({
                            "model": condition, "fold": fold_id, "test_cohort": test_cohort,
                            "seed": seed, "mask_rate": rate, "draw": d,
                            "n_masked_genes": int(len(masked_nodes)),
                            "auc": float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else float("nan"),
                            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                            "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
                        })
                done.add((condition, fold_id, seed))
                pd.DataFrame(rows).to_csv(rows_path, index=False)
                log(f"    {condition:15s} fold{fold_id} seed{seed} done  elapsed={fmt_hms(time.time()-t0)}")
    return pd.DataFrame(rows)


def verify_protocol_unchanged(df, tol):
    """The recomputed curves for the three conditions graph_utility already
    measured must match its saved values. If they do, the masking protocol is
    provably identical and the new conditions are directly comparable."""
    old_path = GRAPH_UTIL_DIR / "masking_curves.csv"
    if not old_path.exists():
        log("  [verify] graph_utility/masking_curves.csv not found -- protocol check skipped")
        return
    old = pd.read_csv(old_path)
    keys = ["model", "fold", "seed", "mask_rate", "draw"]
    merged = df.merge(old, on=keys, suffixes=("_new", "_old"), how="inner")
    if merged.empty:
        log("  [verify] no overlapping rows with graph_utility -- protocol check skipped")
        return
    merged["auc_absdiff"] = (merged["auc_new"] - merged["auc_old"]).abs()
    merged.to_csv(OUT_DIR / "protocol_verification.csv", index=False)
    bad = merged[merged["auc_absdiff"] > tol]
    if bad.empty:
        log(f"  [verify] all {len(merged)} overlapping rows reproduce graph_utility's AUCs within "
            f"tol={tol}. Masking protocol is unchanged; curves are comparable.")
    else:
        log("  " + "!" * 70)
        log(f"  [verify] WARNING: {len(bad)}/{len(merged)} rows differ from graph_utility by >{tol}.")
        log("  The masking protocol may have drifted -- new curves are NOT comparable to the old.")
        log("  Inspect learned_graph/protocol_verification.csv before trusting any verdict.")
        log("  " + "!" * 70)


# ==========================================================================
# Comparison + verdict
# ==========================================================================
def compare_conditions(df, args):
    if df.empty:
        return "NO RESULTS"

    per_fold = df.groupby(["model", "fold", "mask_rate"])[
        ["auc", "macro_f1", "balanced_acc"]].mean().reset_index()
    pooled = per_fold.groupby(["model", "mask_rate"])[["auc", "macro_f1", "balanced_acc"]].agg(["mean", "std"])
    pooled.columns = [f"{a}_{b}" for a, b in pooled.columns]
    pooled.reset_index().to_csv(OUT_DIR / "masking_curves_pooled.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for cond in ALL_CONDITIONS:
        sub = pooled.reset_index()
        sub = sub[sub["model"] == cond].sort_values("mask_rate")
        if sub.empty:
            continue
        ax.errorbar(sub["mask_rate"] * 100, sub["auc_mean"], yerr=sub["auc_std"],
                    marker="o", capsize=3, label=cond)
    ax.set_xlabel("Genes masked (%)")
    ax.set_ylabel("AUC (mean over folds, seeds, draws)")
    ax.set_title("Learned vs STRING vs connectivity controls under missing genes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "masking_curves_all_conditions.png", dpi=150)
    plt.close(fig)

    def paired(a, b, metric="auc", masked_only=True):
        pa = per_fold[(per_fold["model"] == a)]
        pb = per_fold[(per_fold["model"] == b)]
        if masked_only:
            pa, pb = pa[pa["mask_rate"] > 0], pb[pb["mask_rate"] > 0]
        j = pa.merge(pb, on=["fold", "mask_rate"], suffixes=("_a", "_b"))
        if len(j) < 2:
            return None
        delta = j[f"{metric}_a"] - j[f"{metric}_b"]
        try:
            _, p = wilcoxon(j[f"{metric}_a"], j[f"{metric}_b"])
        except ValueError:
            p = float("nan")
        lo, hi = np.percentile(delta, [2.5, 97.5])
        return {"comparison": f"{a} vs {b}", "metric": metric, "n_cells": len(j),
                "mean_delta": float(delta.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                "cells_favouring_a": int((delta > 0).sum()), "wilcoxon_p": p}

    comps = []
    for learned in LEARNED_VARIANTS:
        for ref in ("empty_graph", "shuffled_graph", "random_graph", "real_graph"):
            c = paired(learned, ref)
            if c:
                comps.append(c)
    for ref in ("empty_graph", "shuffled_graph", "random_graph"):
        c = paired("real_graph", ref)
        if c:
            comps.append(c)
    comp_df = pd.DataFrame(comps)
    if not comp_df.empty:
        comp_df.to_csv(OUT_DIR / "paired_comparisons.csv", index=False)

    # clean-data sanity check
    clean = per_fold[per_fold["mask_rate"] == 0.0].groupby("model")["auc"].mean()

    lines = ["PAIRED COMPARISONS (masked rates only; positive delta favours the first condition):"]
    for c in comps:
        lines.append(f"  {c['comparison']:38s} delta={c['mean_delta']:+.4f} "
                     f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]  "
                     f"{c['cells_favouring_a']}/{c['n_cells']} cells  p={c['wilcoxon_p']:.4g}")
    lines.append("")
    lines.append("CLEAN-DATA (0% masking) AUC -- sanity check that the protocol is unchanged:")
    for m, v in clean.items():
        lines.append(f"  {m:18s} {v:.4f}")
    lines.append("")

    def get(a, b):
        return next((c for c in comps if c["comparison"] == f"{a} vs {b}"), None)

    verdicts = []
    for learned in LEARNED_VARIANTS:
        vs_string = get(learned, "real_graph")
        vs_shuffled = get(learned, "shuffled_graph")
        if not (vs_string and vs_shuffled):
            verdicts.append(f"{learned}: INCONCLUSIVE -- missing comparisons.")
            continue
        beats_string = vs_string["mean_delta"] > 0 and vs_string["ci_lo"] > 0
        beats_shuffled = vs_shuffled["mean_delta"] > 0 and vs_shuffled["ci_lo"] > 0
        if beats_string:
            v = (f"LEARNED > STRING ({learned}) -- a task-learned topology recovers more "
                 f"information under missingness than generic protein interactions "
                 f"(delta={vs_string['mean_delta']:+.4f}, CI [{vs_string['ci_lo']:+.4f}, "
                 f"{vs_string['ci_hi']:+.4f}]).")
        elif beats_shuffled:
            v = (f"LEARNED ~ STRING ({learned}) -- the specific edges matter little; the benefit is "
                 f"largely connectivity plus mild topology. vs STRING delta="
                 f"{vs_string['mean_delta']:+.4f} CI [{vs_string['ci_lo']:+.4f}, {vs_string['ci_hi']:+.4f}]; "
                 f"vs shuffled delta={vs_shuffled['mean_delta']:+.4f}.")
        else:
            v = (f"LEARNED ~ SHUFFLED ({learned}) -- only connectivity matters, not which "
                 f"connections. vs shuffled delta={vs_shuffled['mean_delta']:+.4f} "
                 f"CI [{vs_shuffled['ci_lo']:+.4f}, {vs_shuffled['ci_hi']:+.4f}].")
        verdicts.append(v)

    return "\n".join(lines + verdicts)


# ==========================================================================
# Step 5 -- topology inspection
# ==========================================================================
def topology_inspection(learned_pairs, string_pairs, gene_order, fold_ids):
    rows, stab_rows = [], []
    for variant in LEARNED_VARIANTS:
        per_fold_sets = {f: set(map(lambda p: (min(p), max(p)), learned_pairs[(variant, f)]))
                         for f in fold_ids if (variant, f) in learned_pairs}
        if not per_fold_sets:
            continue
        for f, s in per_fold_sets.items():
            inter = len(s & string_pairs)
            deg = {}
            for i, j in s:
                deg[gene_order[i]] = deg.get(gene_order[i], 0) + 1
                deg[gene_order[j]] = deg.get(gene_order[j], 0) + 1
            hubs = sorted(deg.items(), key=lambda t: -t[1])[:10]
            rows.append({
                "variant": variant, "fold": f, "n_edges": len(s),
                "overlap_with_string": inter,
                "jaccard_with_string": inter / len(s | string_pairs),
                "top_hubs": "; ".join(f"{g}({d})" for g, d in hubs),
                "n_fibrosis_core_in_top10_hubs": sum(1 for g, _ in hubs if g in FIBROSIS_CORE_GENES),
            })
        keys = sorted(per_fold_sets)
        jaccs = [len(per_fold_sets[a] & per_fold_sets[b]) / len(per_fold_sets[a] | per_fold_sets[b])
                 for i, a in enumerate(keys) for b in keys[i + 1:]]
        stab_rows.append({"variant": variant, "mean_cross_fold_edge_jaccard": float(np.mean(jaccs)),
                           "min": float(np.min(jaccs)), "max": float(np.max(jaccs)),
                           "n_fold_pairs": len(jaccs)})

    topo_df = pd.DataFrame(rows)
    stab_df = pd.DataFrame(stab_rows)
    if not topo_df.empty:
        topo_df.to_csv(OUT_DIR / "topology_comparison.csv", index=False)
    if not stab_df.empty:
        stab_df.to_csv(OUT_DIR / "topology_cross_fold_stability.csv", index=False)

    lines = ["TOPOLOGY INSPECTION:"]
    for variant in LEARNED_VARIANTS:
        sub = topo_df[topo_df["variant"] == variant]
        st = stab_df[stab_df["variant"] == variant]
        if sub.empty:
            continue
        lines.append(f"  {variant}: mean Jaccard with STRING = {sub['jaccard_with_string'].mean():.3f} "
                     f"({sub['overlap_with_string'].mean():.1f}/{int(sub['n_edges'].mean())} edges shared)")
        lines.append(f"    canonical fibrosis genes among top-10 hubs (mean over folds): "
                     f"{sub['n_fibrosis_core_in_top10_hubs'].mean():.1f}/10")
        if not st.empty:
            mj = st["mean_cross_fold_edge_jaccard"].iloc[0]
            lines.append(f"    cross-fold edge-set Jaccard = {mj:.3f} -- "
                         + ("topology is reasonably stable across folds."
                            if mj > 0.5 else
                            "LOW: each fold learns a substantially different graph, which is what "
                            "fitting noise looks like. Treat any advantage with corresponding caution."))
    return "\n".join(lines)


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT learned-graph variants")
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--mask-rates", nargs="*", type=float, default=MASK_RATES)
    parser.add_argument("--n-draws", type=int, default=N_MASK_DRAWS)
    parser.add_argument("--e2e-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--protocol-tol", type=float, default=1e-4,
                         help="tolerance when checking recomputed curves against graph_utility's. "
                              "Loose enough for MPS inference jitter, tight enough that any real "
                              "protocol drift (which would shift AUC by ~0.01+) still trips it.")
    parser.add_argument("--stage", choices=["all", "topology", "train", "mask", "report"], default="all")
    parser.add_argument("--recompute-masking", action="store_true")
    args = parser.parse_args()

    for d in (OUT_DIR, CKPT_DIR, EDGES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log("Learned-graph analysis: does task-learned topology beat generic STRING under missingness?")
    log("Evaluation is on the MASKING CURVE, not clean-data AUC. Nothing prior is modified.")

    real_edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")
    gene_order = load_panel_gene_order()
    string_pairs = edge_index_to_pair_set(real_edge_index)
    log(f"STRING edge budget: {len(string_pairs)} undirected / {real_edge_index.shape[1]} directed "
        f"-- every condition matches this exactly")

    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()) if args.folds is None \
        else [FOLDS_DIR / f for f in args.folds]
    fold_ids = [parse_fold_index(d.name) for d in fold_dirs]

    # ---- STAGE 1: build the learned topologies (train split only) ----
    learned_pairs, learned_scores = {}, {}
    if args.stage in ("all", "topology", "train", "mask", "report"):
        expr_panel = None
        for fold_dir in fold_dirs:
            fold_id = parse_fold_index(fold_dir.name)
            for variant in LEARNED_VARIANTS:
                path = EDGES_DIR / f"learned_edges_fold{fold_id}_{variant}.csv"
                if path.exists():
                    e = pd.read_csv(path)
                    learned_pairs[(variant, fold_id)] = list(zip(e["src_idx"], e["dst_idx"]))
                    learned_scores[(variant, fold_id)] = e["score"].tolist()
                    continue
                if args.stage not in ("all", "topology"):
                    log(f"  {variant} fold{fold_id}: edge list missing -- run --stage topology first")
                    continue
                log(f"=== fold {fold_id}: learning {variant} topology ===")
                if variant == "coexpr_graph":
                    if expr_panel is None:
                        expr_panel = load_raw_panel_expression(gene_order)
                    pairs, scores = build_coexpression_topology(
                        fold_dir, gene_order, expr_panel, N_UNDIRECTED_STRING)
                else:
                    pairs, scores = learn_e2e_topology(
                        fold_dir, dev_lookup, N_UNDIRECTED_STRING, args.e2e_epochs)
                save_learned_edges(variant, fold_id, pairs, scores, gene_order, string_pairs)
                learned_pairs[(variant, fold_id)] = pairs
                learned_scores[(variant, fold_id)] = scores
                log(f"    saved {len(pairs)} undirected edges; "
                    f"{sum(1 for p in pairs if (min(p), max(p)) in string_pairs)} overlap STRING")

    # ---- STAGE 2: train the learned variants + the random control ----
    to_train = LEARNED_VARIANTS + CONTROL_VARIANTS
    if args.stage in ("all", "train"):
        total = len(fold_dirs) * len(to_train) * len(args.seeds)
        i, done_n, t0 = 0, 0, time.time()
        for fold_dir in fold_dirs:
            fold_id = parse_fold_index(fold_dir.name)
            test_cohort = load_fold_info(fold_dir)["test_cohort"]
            for variant in to_train:
                pending = [s for s in args.seeds
                           if not (variant_run_dir(variant, fold_id, test_cohort, s) / "DONE.marker").exists()]
                i += len(args.seeds) - len(pending)
                if not pending:
                    continue
                if variant in CONTROL_VARIANTS:
                    edge_index, _ = _control_graph(variant, real_edge_index, fold_id)
                else:
                    if (variant, fold_id) not in learned_pairs:
                        log(f"  skipping {variant} fold{fold_id}: no learned edge list")
                        i += len(pending)
                        continue
                    edge_index = undirected_pairs_to_edge_index(learned_pairs[(variant, fold_id)])
                setup = setup_fold_custom_graph(fold_dir, edge_index, dev_lookup)
                for seed in pending:
                    i += 1
                    eta = f" ETA={fmt_hms((time.time()-t0)/max(done_n,1)*(total-i))}" if done_n else ""
                    log(f"  [{i}/{total}] training {variant} fold{fold_id} seed{seed}{eta}")
                    try:
                        m = train_on_graph(variant, fold_dir,
                                            setup, seed,
                                            variant_run_dir(variant, fold_id, test_cohort, seed))
                        done_n += 1
                        log(f"    done: AUC={m['auc']:.4f} macro_f1={m['macro_f1']:.4f}")
                    except Exception:
                        log(f"    FAILED {variant} fold{fold_id} seed{seed} -- continuing")
                        traceback.print_exc()

    # ---- STAGE 3: masking sweep over all five conditions ----
    df = pd.DataFrame()
    if args.stage in ("all", "mask"):
        df = run_masking(fold_dirs, ALL_CONDITIONS, real_edge_index, learned_pairs, args)
        verify_protocol_unchanged(df, args.protocol_tol)
    elif args.stage == "report":
        p = OUT_DIR / "masking_curves_all_conditions.csv"
        if p.exists():
            df = pd.read_csv(p)

    # ---- STAGE 4: report ----
    if args.stage in ("all", "mask", "report") and not df.empty:
        comparison_text = compare_conditions(df, args)
        topo_text = topology_inspection(learned_pairs, string_pairs, gene_order, fold_ids)
        budget_df = audit_edge_budgets(fold_ids, ALL_CONDITIONS, real_edge_index, learned_pairs)
        budget_lines = ["EDGE-BUDGET AUDIT (measured, not assumed):"]
        if not budget_df.empty:
            for cond, sub in budget_df.groupby("condition"):
                budget_lines.append(
                    f"  {cond:16s} directed={sub['n_directed_edges'].mean():.0f}  "
                    f"undirected pairs={sub['n_undirected_pairs'].mean():.0f}  "
                    f"symmetric={'yes' if sub['symmetric'].all() else 'NO'}")
            asym = sorted(budget_df[~budget_df["symmetric"]]["condition"].unique())
            if asym:
                budget_lines.append(
                    f"  NOTE: {', '.join(asym)} is asymmetric by construction (B9's control, reused "
                    f"unchanged). The DIRECTED edge budget is matched across all conditions; its "
                    f"undirected pair count is not. Any comparison involving it carries that caveat.")

        lines = ["=" * 78, "Learned-Graph Analysis", "=" * 78, "",
                 "QUESTION: under missingness, does a topology learned from fibrosis training",
                 "data outperform generic STRING protein interactions, and does it beat random",
                 "connectivity? Clean-data AUC is not the criterion -- the label ceiling is",
                 "accepted and the accuracy finding stands.",
                 "",
                 *budget_lines,
                 "",
                 "Masking protocol imported unchanged from graph_utility.py (Group A features",
                 "replaced by the fold's training-split per-node mean; masked nodes kept in",
                 "message passing; identical rates, draw count and draw seeds).",
                 "",
                 "POWER: paired cross-fold comparisons are DESCRIPTIVE and underpowered at n=8",
                 "folds; CIs are percentile intervals over (fold x masking-rate) cells and should",
                 "be read as indicative, not confirmatory.",
                 "", comparison_text, "", topo_text, "", "=" * 78]
        summary = "\n".join(lines)
        with open(OUT_DIR / "summary.txt", "w") as f:
            f.write(summary + "\n")
        log("")
        for line in lines:
            log(line)

    log(f"outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
