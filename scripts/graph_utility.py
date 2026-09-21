"""
CoDA-GAT — Graph utility analysis: does the STRING graph contribute on axes
OTHER than accuracy?

The accuracy finding stands as reported (shuffled/random/empty tie or slightly
beat the real graph). Nothing here modifies, re-tunes or overwrites any B7/B9
result -- all output goes to graph_utility/.

  TEST 1  robustness to missing genes   (inference-only masking, the primary test)
  TEST 2  edge-level interpretability   (what only a graph can produce)
  TEST 3  embedding quality + calibration

UNAVOIDABLE RETRAINING, STATED UP FRONT: B9 persisted only probs_test.npy and
predictions_test.npy for its ablations -- no weights.pt, no embeddings, no
calibration probabilities. Every test here needs more than saved test
predictions (Test 1 needs forward passes on modified inputs, Test 2 needs
attention, Test 3 needs embeddings and calibration probs). So the two controls
(empty_graph, shuffled_graph) are retrained ONCE here, mirroring B9's protocol
exactly, into graph_utility/control_checkpoints/. ablations_b9/ is never
written to. The retrained controls are then checked against B9's published
metrics -- if they reproduce, the accuracy finding is demonstrably unchanged.

NOT DONE (deliberately): evaluating the real-graph model with an empty
edge_index at inference. That is a graph-TRAINED model deprived of its graph,
which is a different claim from B9's graph-free control, and substituting it
would confound the comparison.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr, wilcoxon, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import silhouette_score, f1_score, balanced_accuracy_score, roc_auc_score

import train_fold0_b1 as b1
from train_fold0_b1 import (
    CoDAGAT, run_epoch, get_device, set_seed,
    N_NODES, N_FEATURES, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY, SEED, SHARED_DIR,
)
from run_full_experiment_b7 import (
    extract_full, load_edge_index_generic, load_fold_info, parse_fold_index, fmt_hms,
    lac_scores, aps_scores, conformal_quantile, lac_sets, aps_sets, coverage_stats, ALPHAS,
)
from ablations_b9 import setup_fold_ablation, load_split_for_ablation, ABLATIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
RESULTS_B7 = REPO_ROOT / "results_b7"
ABLATIONS_B9 = REPO_ROOT / "ablations_b9"
OUT_DIR = REPO_ROOT / "graph_utility"
CTRL_DIR = OUT_DIR / "control_checkpoints"

SEEDS = [42, 43, 44, 45, 46]
MASK_RATES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
N_MASK_DRAWS = 20
MASK_DRAW_BASE_SEED = 20260919

# Group A (feature idx 0..7) are the DYNAMIC, sample-varying, measurement-derived
# features. Groups B (8..16, graph/GTEx statics) and C (17..25, fold-fitted
# per-gene statistics) are properties of the gene and the panel, not of the
# patient's measurement -- they remain known even if a gene was not assayed.
# "Gene unmeasured" therefore == Group A replaced by that fold's TRAINING-split
# per-(node, feature) mean.
DYNAMIC_FEATURE_SLICE = (0, 8)

# Models: "real" from B7 mode_off; the two controls retrained here.
MODELS = ["real_graph", "empty_graph", "shuffled_graph"]


def log(msg):
    print(f"[graph-util] {msg}", flush=True)


# ==========================================================================
# PART A -- retrain the two controls, persisting the artifacts B9 omitted
# ==========================================================================
def control_run_dir(control, fold_id, test_cohort, seed):
    return CTRL_DIR / f"fold{fold_id}_{test_cohort}" / control / f"seed_{seed}"


def retrain_control(control, fold_dir, fold_setup, seed, run_dir, real_edge_index, dev_lookup):
    """
    Mirrors ablations_b9.run_single_ablation's training loop exactly (same
    hyperparameters, same adversary-off / h3 handling, same early stopping),
    but additionally saves weights, embeddings and calibration probabilities.
    Writes only under graph_utility/.
    """
    device = get_device()
    cfg = ABLATIONS[control]

    set_seed(seed)
    model = CoDAGAT(N_FEATURES, fold_setup["n_ordinal_classes"], fold_setup["n_cohorts"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    class_weights = fold_setup["class_weights"].to(device)

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

    # a full-train loader (b9's setup does not build one) for Test 3
    from torch_geometric.loader import DataLoader
    cohort_to_idx = {c: i for i, c in enumerate(sorted(fold_setup["fold_info"]["train_cohorts"]))}
    edge_index, node_perm = _control_graph(control, real_edge_index, parse_fold_index(fold_dir.name))
    train_data, train_ids, train_y = load_split_for_ablation(
        fold_dir, "train", edge_index, dev_lookup, cohort_to_idx, node_perm, cfg.get("zero_cols"))
    train_loader_full = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

    emb_train, _, _, _ = extract_full(model, train_loader_full, device)
    emb_cal, probs_cal, preds_cal, _ = extract_full(model, fold_setup["cal_loader"], device)
    emb_test, probs_test, preds_test, _ = extract_full(model, fold_setup["test_loader"], device)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "control": control, "seed": seed,
                "best_val_f1": best_val_f1,
                "n_ordinal_classes": fold_setup["n_ordinal_classes"],
                "n_cohorts": fold_setup["n_cohorts"]}, run_dir / "weights.pt")
    np.save(run_dir / "embeddings_train.npy", emb_train)
    np.save(run_dir / "embeddings_calibration.npy", emb_cal)
    np.save(run_dir / "embeddings_test.npy", emb_test)
    np.save(run_dir / "probs_calibration.npy", probs_cal)
    np.save(run_dir / "probs_test.npy", probs_test)
    np.save(run_dir / "predictions_test.npy", preds_test)

    y_test = fold_setup["y_test"]
    metrics = {
        "control": control, "fold": parse_fold_index(fold_dir.name), "seed": seed,
        "best_val_f1": best_val_f1,
        "auc": float(roc_auc_score(y_test, probs_test[:, 1])) if len(np.unique(y_test)) > 1 else float("nan"),
        "macro_f1": float(f1_score(y_test, preds_test, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_test, preds_test)),
    }
    with open(run_dir / "run_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    (run_dir / "DONE.marker").touch()
    return metrics


def _control_graph(control, real_edge_index, fold_id):
    """The same edge_index / node permutation B9 used for this control."""
    from ablations_b9 import build_edge_index_and_perm
    return build_edge_index_and_perm(ABLATIONS[control], real_edge_index, fold_id)


def verify_controls_match_b9(retrained_rows, tol):
    """The retrain must reproduce B9's published control metrics. If it does,
    the accuracy finding is demonstrably unchanged by this analysis."""
    b9_path = ABLATIONS_B9 / "master_ablation_results.csv"
    if not b9_path.exists() or not retrained_rows:
        log("  [verify] no B9 master results to compare against -- skipped")
        return
    orig = pd.read_csv(b9_path)
    new = pd.DataFrame(retrained_rows).rename(columns={"control": "ablation"})
    merged = new.merge(orig, on=["ablation", "fold", "seed"], suffixes=("_new", "_b9"), how="inner")
    if merged.empty:
        log("  [verify] no overlapping (ablation, fold, seed) rows -- skipped")
        return
    for m in ("auc", "macro_f1", "balanced_acc"):
        merged[f"{m}_absdiff"] = (merged[f"{m}_new"] - merged[f"{m}_b9"]).abs()
    merged["max_absdiff"] = merged[[f"{m}_absdiff" for m in ("auc", "macro_f1", "balanced_acc")]].max(axis=1)
    merged.to_csv(OUT_DIR / "control_retrain_verification.csv", index=False)
    bad = merged[merged["max_absdiff"] > tol]
    if bad.empty:
        log(f"  [verify] all {len(merged)} retrained control runs reproduce B9 within tol={tol}. "
            f"The published accuracy finding is unchanged.")
    else:
        log("  " + "!" * 70)
        log(f"  [verify] WARNING: {len(bad)}/{len(merged)} retrained runs diverge from B9 by >{tol}.")
        log("  MPS is not bit-deterministic so small drift is expected; large drift is not.")
        log("  Inspect graph_utility/control_retrain_verification.csv before trusting Tests 1-3.")
        log("  " + "!" * 70)


# ==========================================================================
# TEST 1 -- robustness to missing genes (inference only)
# ==========================================================================
def batched_forward_probs(model, X, edge_index, device, chunk=64):
    """Direct batched forward without PyG Data objects (much faster for the
    14k forward passes Test 1 needs). Every graph has exactly N_NODES nodes,
    so the batched edge_index is the per-sample edge_index tiled with offsets."""
    model.eval()
    out_probs = []
    n = X.shape[0]
    ei = edge_index.to(device)
    with torch.no_grad():
        for start in range(0, n, chunk):
            xb = X[start:start + chunk]
            b = xb.shape[0]
            x = torch.tensor(xb.reshape(-1, X.shape[2]), dtype=torch.float32, device=device)
            if ei.shape[1] > 0:
                big_ei = torch.cat([ei + i * N_NODES for i in range(b)], dim=1)
            else:
                big_ei = torch.zeros((2, 0), dtype=torch.long, device=device)
            batch_vec = torch.arange(b, device=device).repeat_interleave(N_NODES)
            out = model(x, big_ei, batch_vec, grl_lambd=0.0)
            out_probs.append(torch.softmax(out["h1_logits"], dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out_probs)


def apply_mask(X, masked_nodes, train_node_means):
    """Masking convention (identical for every model): the masked gene's
    DYNAMIC features (Group A, idx 0..7) are replaced by that fold's
    training-split per-(node, feature) mean. Static (B) and fold-fitted (C)
    features are left intact -- they are properties of the gene, not of the
    patient's measurement. The node is KEPT in message passing; no edges are
    removed. Removing edges would change topology per draw and confound
    'the graph recovers missing information' with 'the graph got sparser'."""
    lo, hi = DYNAMIC_FEATURE_SLICE
    Xm = X.copy()
    Xm[:, masked_nodes, lo:hi] = train_node_means[masked_nodes, lo:hi]
    return Xm


def run_test1(fold_dirs, real_edge_index, dev_lookup, args):
    log("TEST 1: robustness to missing genes (inference only, no retraining)")
    rows_path = OUT_DIR / "masking_curves.csv"
    done = set()
    existing = []
    if rows_path.exists() and not args.recompute_test1:
        existing = pd.read_csv(rows_path).to_dict("records")
        done = {(r["model"], r["fold"], r["seed"]) for r in existing}
        log(f"  resuming: {len(done)} (model, fold, seed) combinations already done")

    device = get_device()
    rows = list(existing)
    t0 = time.time()

    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        fold_info = load_fold_info(fold_dir)
        test_cohort = fold_info["test_cohort"]

        X_test_raw = np.load(fold_dir / "test" / "X.npy")
        y_test = pd.read_csv(fold_dir / "test" / "y.csv", index_col=0)["y_T1_significant"].to_numpy().astype(int)
        train_node_means = np.load(fold_dir / "train" / "X.npy").mean(axis=0)  # (72, 26)

        # gene subsets are drawn ONCE per (fold, rate, draw) and reused by every
        # model, so the comparison is paired at the draw level
        draws = {}
        for rate in args.mask_rates:
            k = int(round(rate * N_NODES))
            draws[rate] = []
            for d in range(1 if k == 0 else args.n_draws):
                rng = np.random.default_rng(MASK_DRAW_BASE_SEED + fold_id * 1000 + int(rate * 100) * 10 + d)
                draws[rate].append(rng.choice(N_NODES, size=k, replace=False) if k else np.array([], dtype=int))

        for model_name in MODELS:
            for seed in args.seeds:
                if (model_name, fold_id, seed) in done:
                    continue
                ckpt_path = (RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / "mode_off" / f"seed_{seed}" / "weights.pt"
                              if model_name == "real_graph"
                              else control_run_dir(model_name, fold_id, test_cohort, seed) / "weights.pt")
                if not ckpt_path.exists():
                    log(f"    MISSING {ckpt_path.relative_to(REPO_ROOT)} -- skipping "
                        f"{model_name} fold{fold_id} seed{seed}")
                    continue

                ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
                model = CoDAGAT(N_FEATURES, ckpt["n_ordinal_classes"], ckpt["n_cohorts"]).to(device)
                model.load_state_dict(ckpt["model_state_dict"])

                if model_name == "real_graph":
                    edge_index, node_perm = real_edge_index, None
                else:
                    edge_index, node_perm = _control_graph(model_name, real_edge_index, fold_id)

                for rate, subsets in draws.items():
                    for d, masked_nodes in enumerate(subsets):
                        # mask in ORIGINAL gene space first, then apply the model's
                        # node permutation -- so the same GENES are masked for every model
                        Xm = apply_mask(X_test_raw, masked_nodes, train_node_means)
                        if node_perm is not None:
                            Xm = Xm[:, node_perm, :]
                        p = batched_forward_probs(model, Xm, edge_index, device)
                        pred = (p >= 0.5).astype(int)
                        rows.append({
                            "model": model_name, "fold": fold_id, "test_cohort": test_cohort,
                            "seed": seed, "mask_rate": rate, "draw": d,
                            "n_masked_genes": int(len(masked_nodes)),
                            "auc": float(roc_auc_score(y_test, p)) if len(np.unique(y_test)) > 1 else float("nan"),
                            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                            "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
                        })
                done.add((model_name, fold_id, seed))
                pd.DataFrame(rows).to_csv(rows_path, index=False)
                log(f"    {model_name:15s} fold{fold_id} seed{seed} done  "
                    f"({len(args.mask_rates)} rates)  elapsed={fmt_hms(time.time() - t0)}")

    return pd.DataFrame(rows)


def analyse_test1(df, args):
    """Degradation curves + the paired real-vs-empty flatness comparison."""
    if df.empty:
        return None, "TEST 1 (robustness): NOT RUN -- no masking results."

    # average over draws and seeds within (model, fold, rate)
    per_fold = df.groupby(["model", "fold", "mask_rate"])[["auc", "macro_f1", "balanced_acc"]].mean().reset_index()
    pooled = per_fold.groupby(["model", "mask_rate"])[["auc", "macro_f1", "balanced_acc"]].agg(["mean", "std"])
    pooled.columns = [f"{a}_{b}" for a, b in pooled.columns]
    pooled = pooled.reset_index()
    pooled.to_csv(OUT_DIR / "masking_curves_pooled.csv", index=False)

    # degradation = metric(0%) - metric(rate), per model per fold
    base = per_fold[per_fold["mask_rate"] == 0.0].set_index(["model", "fold"])
    per_fold = per_fold.join(base[["auc"]].rename(columns={"auc": "auc_at_0"}), on=["model", "fold"])
    per_fold["auc_drop"] = per_fold["auc_at_0"] - per_fold["auc"]
    per_fold.to_csv(OUT_DIR / "masking_degradation_per_fold.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_name in MODELS:
        sub = pooled[pooled["model"] == model_name].sort_values("mask_rate")
        if sub.empty:
            continue
        ax.errorbar(sub["mask_rate"] * 100, sub["auc_mean"], yerr=sub["auc_std"],
                    marker="o", capsize=3, label=model_name)
    ax.set_xlabel("Genes masked (%)")
    ax.set_ylabel("AUC (mean over folds, seeds, draws)")
    ax.set_title("Test 1: degradation under missing genes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "masking_curves.png", dpi=150)
    plt.close(fig)

    # paired real vs empty on the DROP, across (fold, rate) cells with rate>0
    real = per_fold[(per_fold["model"] == "real_graph") & (per_fold["mask_rate"] > 0)]
    empty = per_fold[(per_fold["model"] == "empty_graph") & (per_fold["mask_rate"] > 0)]
    joined = real.merge(empty, on=["fold", "mask_rate"], suffixes=("_real", "_empty"))
    if len(joined) < 2:
        return pooled, "TEST 1 (robustness): INCONCLUSIVE -- not enough paired cells."

    # positive delta = real degrades LESS than empty = graph helps
    joined["flatness_advantage"] = joined["auc_drop_empty"] - joined["auc_drop_real"]
    joined.to_csv(OUT_DIR / "masking_paired_real_vs_empty.csv", index=False)
    adv = joined["flatness_advantage"]
    try:
        _, p = wilcoxon(joined["auc_drop_real"], joined["auc_drop_empty"])
    except ValueError:
        p = float("nan")
    lo, hi = np.percentile(adv, [2.5, 97.5])
    cohen_dz = adv.mean() / adv.std() if adv.std() > 0 else float("nan")

    helps = adv.mean() > 0 and p < 0.05
    verdict = (
        f"TEST 1 (robustness): the graph {'DOES' if helps else 'DOES NOT'} degrade more gracefully "
        f"than the graph-free control.\n"
        f"  mean AUC-drop advantage (empty minus real, positive = graph more robust) = "
        f"{adv.mean():+.4f} [95% pct {lo:+.4f}, {hi:+.4f}], Cohen's dz={cohen_dz:.2f}, "
        f"Wilcoxon p={p:.4g} over {len(joined)} (fold x rate) cells.\n"
        f"  Cells favouring the real graph: {int((adv > 0).sum())}/{len(adv)}."
    )
    return pooled, verdict


# ==========================================================================
# TEST 2 -- edge-level interpretability
# ==========================================================================
def extract_edge_and_node_attention(model, X, edge_index, device, n_real_edges):
    """Mean GAT-layer-2 attention per real STRING edge, and per node (as
    source), averaged over samples. GATConv appends self-loops to the edge
    index it reports, so the first n_real_edges entries are the real edges."""
    model.eval()
    edge_sum = np.zeros(n_real_edges)
    node_sum = np.zeros(N_NODES)
    n = 0
    ei = edge_index.to(device)
    with torch.no_grad():
        for i in range(X.shape[0]):
            x = torch.tensor(X[i], dtype=torch.float32, device=device)
            batch_vec = torch.zeros(N_NODES, dtype=torch.long, device=device)
            _ = model(x, ei, batch_vec, grl_lambd=0.0)
            attn_ei, alpha = model.last_attn
            if n == 0:
                # GATConv(add_self_loops=True) concatenates the self-loops AFTER the
                # supplied edges, so alpha[:n_real_edges] are the real STRING edges.
                # Verified once rather than assumed, since it depends on PyG internals.
                if attn_ei.shape[1] != n_real_edges + N_NODES or \
                        not torch.equal(attn_ei[:, :n_real_edges].cpu(), edge_index.cpu()):
                    raise RuntimeError(
                        f"attention edge ordering is not (real edges, then self-loops): got "
                        f"{attn_ei.shape[1]} entries for {n_real_edges} real edges + {N_NODES} "
                        f"nodes. Edge-level attention cannot be attributed reliably -- aborting "
                        f"rather than mislabelling edges."
                    )
            a = alpha.squeeze(-1).cpu().numpy()
            src = attn_ei[0].cpu().numpy()
            edge_sum += a[:n_real_edges]
            np.add.at(node_sum, src, a)
            n += 1
    return edge_sum / max(n, 1), node_sum / max(n, 1)


def run_test2(fold_dirs, real_edge_index, args):
    log("TEST 2: edge-level interpretability (real graph vs shuffled control)")
    edges = pd.read_csv(SHARED_DIR / "graph_edges.csv")
    panel = pd.read_csv(SHARED_DIR / "panel_genes.csv")
    gene_pathways = {r["gene"]: set(str(r["pathways"]).split(";")) for _, r in panel.iterrows()}
    n_real_edges = len(edges)
    device = get_device()

    per_fold_edge = {m: {} for m in ("real_graph", "shuffled_graph")}
    per_fold_node = {m: {} for m in ("real_graph", "shuffled_graph")}

    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        test_cohort = load_fold_info(fold_dir)["test_cohort"]
        X_test = np.load(fold_dir / "test" / "X.npy")

        for model_name in ("real_graph", "shuffled_graph"):
            e_seeds, n_seeds = [], []
            for seed in args.seeds:
                ckpt_path = (RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / "mode_off" / f"seed_{seed}" / "weights.pt"
                              if model_name == "real_graph"
                              else control_run_dir(model_name, fold_id, test_cohort, seed) / "weights.pt")
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
                model = CoDAGAT(N_FEATURES, ckpt["n_ordinal_classes"], ckpt["n_cohorts"]).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                if model_name == "real_graph":
                    ei, perm = real_edge_index, None
                else:
                    ei, perm = _control_graph(model_name, real_edge_index, fold_id)
                Xin = X_test[:, perm, :] if perm is not None else X_test
                e, nd = extract_edge_and_node_attention(model, Xin, ei, device, n_real_edges)
                e_seeds.append(e)
                n_seeds.append(nd)
            if e_seeds:
                per_fold_edge[model_name][fold_id] = np.mean(e_seeds, axis=0)
                per_fold_node[model_name][fold_id] = np.mean(n_seeds, axis=0)
        log(f"    fold {fold_id} edge attention extracted")

    if not per_fold_edge["real_graph"]:
        return None, "TEST 2 (edge interpretability): NOT RUN -- no real-graph attention extracted."

    # cross-fold stability of the edge ranking, real vs shuffled
    stability = {}
    for model_name, folds in per_fold_edge.items():
        if len(folds) < 2:
            stability[model_name] = float("nan")
            continue
        keys = sorted(folds)
        rhos = [spearmanr(folds[a], folds[b]).statistic
                for i, a in enumerate(keys) for b in keys[i + 1:]]
        stability[model_name] = float(np.mean(rhos))

    consensus = np.mean(list(per_fold_edge["real_graph"].values()), axis=0)
    node_consensus = np.mean(list(per_fold_node["real_graph"].values()), axis=0)

    edge_df = edges.copy()
    edge_df["mean_attention"] = consensus
    edge_df["endpoint_mean_node_attention"] = [
        (node_consensus[s] + node_consensus[d]) / 2
        for s, d in zip(edge_df["src_idx"], edge_df["dst_idx"])]
    edge_df["same_pathway"] = [
        bool(gene_pathways.get(s, set()) & gene_pathways.get(d, set()))
        for s, d in zip(edge_df["src"], edge_df["dst"])]
    edge_df = edge_df.sort_values("mean_attention", ascending=False)
    edge_df.to_csv(OUT_DIR / "edge_attention.csv", index=False)

    # the honest caveat: is edge attention just a re-description of node importance?
    r_p = pearsonr(edge_df["mean_attention"], edge_df["endpoint_mean_node_attention"])
    r_s = spearmanr(edge_df["mean_attention"], edge_df["endpoint_mean_node_attention"])

    # pathway co-membership enrichment among top-K edges
    k = args.top_k_edges
    top = edge_df.head(k)
    a = int(top["same_pathway"].sum())
    b = k - a
    c = int(edge_df["same_pathway"].sum()) - a
    d = (len(edge_df) - k) - c
    or_pw, p_pw = fisher_exact([[a, b], [c, d]], alternative="greater")

    more_stable = stability.get("real_graph", np.nan) > stability.get("shuffled_graph", np.nan)
    if abs(r_p.statistic) > 0.8:
        endpoint_note = ("HIGH: the edge signal is largely a re-description of node importance "
                         "and must be reported as such, not as independent edge-level insight.")
    else:
        endpoint_note = ("not so high as to make the edge signal purely a restatement of node "
                         "importance.")
    top5 = ", ".join(f"{r['src']}-{r['dst']}" for _, r in edge_df.head(5).iterrows())
    verdict = (
        f"TEST 2 (edge interpretability): edge attention "
        f"{'IS' if more_stable else 'IS NOT'} more stable across folds than the shuffled control.\n"
        f"  mean pairwise Spearman of edge ranking: real={stability.get('real_graph', float('nan')):.3f}, "
        f"shuffled={stability.get('shuffled_graph', float('nan')):.3f}\n"
        f"  endpoint-correlation (edge attention vs mean of its two endpoint genes' attention): "
        f"Pearson r={r_p.statistic:.3f}, Spearman rho={r_s.statistic:.3f}\n"
        f"    -> {endpoint_note}\n"
        f"  top-{k} edges connecting same-pathway genes: {a}/{k} vs "
        f"{int(edge_df['same_pathway'].sum())}/{len(edge_df)} background "
        f"(Fisher one-sided OR={or_pw:.2f}, p={p_pw:.4g})\n"
        f"  Top 5 edges: {top5}"
    )
    pd.DataFrame([{"model": m, "mean_pairwise_spearman": v} for m, v in stability.items()]).to_csv(
        OUT_DIR / "edge_attention_stability.csv", index=False)
    return edge_df, verdict


# ==========================================================================
# TEST 3 -- embedding quality and calibration
# ==========================================================================
def run_test3(fold_dirs, args):
    log("TEST 3: embedding quality + calibration (real graph vs empty-graph control)")
    emb_rows, cal_rows = [], []

    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        test_cohort = load_fold_info(fold_dir)["test_cohort"]
        train_y = pd.read_csv(fold_dir / "train" / "y.csv", index_col=0)
        cal_y = pd.read_csv(fold_dir / "calibration" / "y.csv", index_col=0)
        test_y = pd.read_csv(fold_dir / "test" / "y.csv", index_col=0)
        y_cal = cal_y["y_T1_significant"].to_numpy().astype(int)
        y_test = test_y["y_T1_significant"].to_numpy().astype(int)

        for model_name in ("real_graph", "empty_graph"):
            for seed in args.seeds:
                d = (RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / "mode_off" / f"seed_{seed}"
                     if model_name == "real_graph"
                     else control_run_dir(model_name, fold_id, test_cohort, seed))
                if not (d / "embeddings_train.npy").exists():
                    continue

                emb_train = np.load(d / "embeddings_train.npy")
                emb_cal = np.load(d / "embeddings_calibration.npy")
                emb_test = np.load(d / "embeddings_test.npy")
                probs_cal = np.load(d / "probs_calibration.npy")
                probs_test = np.load(d / "probs_test.npy")

                # cohort separability -- needs multiple cohorts, so it uses TRAIN
                # embeddings (IN-SAMPLE; flagged as such). Lower = more invariant.
                cohorts = train_y["cohort"].to_numpy()
                # no multi_class= : that parameter was removed in sklearn 1.7+
                clf = LogisticRegression(max_iter=2000)
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
                try:
                    cohort_acc = float(cross_val_score(clf, emb_train, cohorts, cv=skf).mean())
                except Exception:
                    cohort_acc = float("nan")
                try:
                    cohort_sil = float(silhouette_score(emb_train, cohorts))
                except Exception:
                    cohort_sil = float("nan")

                # biological structure, OUT-OF-SAMPLE (calibration + test)
                emb_oos = np.vstack([emb_cal, emb_test])
                stage_oos = np.concatenate([cal_y["stage"].to_numpy(), test_y["stage"].to_numpy()])
                etio_oos = np.concatenate([cal_y["etiology"].to_numpy(), test_y["etiology"].to_numpy()])

                def safe_sil(X, labels):
                    try:
                        return float(silhouette_score(X, labels)) if len(set(labels)) > 1 else float("nan")
                    except Exception:
                        return float("nan")

                emb_rows.append({
                    "model": model_name, "fold": fold_id, "seed": seed,
                    "cohort_pred_acc_train_insample": cohort_acc,
                    "cohort_silhouette_train_insample": cohort_sil,
                    "stage_silhouette_oos": safe_sil(emb_oos, stage_oos),
                    "etiology_silhouette_oos": safe_sil(emb_oos, etio_oos),
                })

                # calibration: the SAME conformal layer B7 used, imported unchanged
                lac_cal = lac_scores(probs_cal, y_cal)
                aps_cal = aps_scores(probs_cal, y_cal)
                for alpha in ALPHAS:
                    for score_name, cal_scores, set_fn in (("lac", lac_cal, lac_sets),
                                                            ("aps", aps_cal, aps_sets)):
                        q = conformal_quantile(cal_scores, alpha)
                        stats = coverage_stats(set_fn(probs_test, q), y_test)
                        cal_rows.append({
                            "model": model_name, "fold": fold_id, "seed": seed,
                            "alpha": alpha, "score": score_name, "q_hat": q,
                            "empirical_coverage": stats["empirical_coverage"],
                            "signed_coverage_gap": stats["empirical_coverage"] - (1 - alpha),
                            "mean_set_size": stats["mean_set_size"],
                        })
        log(f"    fold {fold_id} embeddings + calibration done")

    emb_df, cal_df = pd.DataFrame(emb_rows), pd.DataFrame(cal_rows)
    if emb_df.empty:
        return emb_df, cal_df, "TEST 3 (embeddings/calibration): NOT RUN -- control artifacts missing."
    emb_df.to_csv(OUT_DIR / "embedding_analysis.csv", index=False)
    cal_df.to_csv(OUT_DIR / "calibration_comparison.csv", index=False)

    def paired(df, col, lower_is_better=False):
        a = df[df["model"] == "real_graph"].groupby("fold")[col].mean()
        b = df[df["model"] == "empty_graph"].groupby("fold")[col].mean()
        j = pd.concat([a, b], axis=1, keys=["real", "empty"]).dropna()
        if len(j) < 2:
            return None
        delta = j["real"] - j["empty"]
        try:
            _, p = wilcoxon(j["real"], j["empty"])
        except ValueError:
            p = float("nan")
        favours_real = (delta.mean() < 0) if lower_is_better else (delta.mean() > 0)
        return {"metric": col, "mean_real": float(j["real"].mean()), "mean_empty": float(j["empty"].mean()),
                "mean_delta": float(delta.mean()), "wilcoxon_p": p, "n_folds": len(j),
                "favours_real": bool(favours_real)}

    comps = [c for c in (paired(emb_df, "cohort_pred_acc_train_insample", lower_is_better=True),
                          paired(emb_df, "cohort_silhouette_train_insample", lower_is_better=True),
                          paired(emb_df, "stage_silhouette_oos"),
                          paired(emb_df, "etiology_silhouette_oos")) if c]
    for alpha in ALPHAS:
        for score_name in ("lac", "aps"):
            sub = cal_df[(cal_df["alpha"] == alpha) & (cal_df["score"] == score_name)].copy()
            sub["abs_gap"] = sub["signed_coverage_gap"].abs()
            c = paired(sub, "abs_gap", lower_is_better=True)
            if c:
                c["metric"] = f"{score_name}_a{alpha}_abs_coverage_gap"
                comps.append(c)
            c2 = paired(sub, "mean_set_size", lower_is_better=True)
            if c2:
                c2["metric"] = f"{score_name}_a{alpha}_mean_set_size"
                comps.append(c2)

    comp_df = pd.DataFrame(comps)
    comp_df.to_csv(OUT_DIR / "embedding_calibration_comparison.csv", index=False)

    invariance = [c for c in comps if c["metric"].startswith("cohort")]
    calib = [c for c in comps if "coverage_gap" in c["metric"]]
    inv_favours = sum(c["favours_real"] for c in invariance)
    cal_favours = sum(c["favours_real"] for c in calib)
    verdict = (
        f"TEST 3 (embeddings/calibration): the graph "
        f"{'DOES' if inv_favours == len(invariance) and invariance else 'DOES NOT'} produce more "
        f"cohort-invariant embeddings; "
        f"{'DOES' if calib and cal_favours > len(calib) / 2 else 'DOES NOT'} give better-calibrated "
        f"uncertainty.\n"
        f"  cohort-invariance metrics favouring the real graph: {inv_favours}/{len(invariance)} "
        f"(lower cohort separability = more invariant; measured on IN-SAMPLE train embeddings)\n"
        f"  conformal |coverage gap| comparisons favouring the real graph: {cal_favours}/{len(calib)} "
        f"across alpha in {ALPHAS} x (LAC, APS)\n"
        + "\n".join(f"    {c['metric']}: real={c['mean_real']:.4f} empty={c['mean_empty']:.4f} "
                    f"delta={c['mean_delta']:+.4f} p={c['wilcoxon_p']:.4g}" for c in comps)
    )
    return emb_df, cal_df, verdict


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT graph utility analysis")
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--mask-rates", nargs="*", type=float, default=MASK_RATES)
    parser.add_argument("--n-draws", type=int, default=N_MASK_DRAWS)
    parser.add_argument("--top-k-edges", type=int, default=30)
    parser.add_argument("--control-tol", type=float, default=0.03,
                         help="tolerance when checking retrained controls against B9's metrics")
    parser.add_argument("--stage", choices=["all", "controls", "test1", "test2", "test3"], default="all")
    parser.add_argument("--recompute-test1", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CTRL_DIR.mkdir(parents=True, exist_ok=True)

    log("Graph utility analysis -- does the STRING graph contribute beyond accuracy?")
    log("The accuracy finding is NOT revisited. Nothing under results_b7/ or ablations_b9/ is written.")

    real_edge_index = load_edge_index_generic()
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")
    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()) if args.folds is None \
        else [FOLDS_DIR / f for f in args.folds]

    # ---- PART A: retrain controls (the one unavoidable retraining) ----
    if args.stage in ("all", "controls"):
        log("PART A: retraining the two controls to persist weights/embeddings/cal-probs")
        log("  (B9 saved only test predictions, so none of Tests 1-3 can use its runs directly)")
        retrained = []
        total = len(fold_dirs) * 2 * len(args.seeds)
        i, t0 = 0, time.time()
        for fold_dir in fold_dirs:
            fold_id = parse_fold_index(fold_dir.name)
            test_cohort = load_fold_info(fold_dir)["test_cohort"]
            for control in ("empty_graph", "shuffled_graph"):
                pending = [s for s in args.seeds
                           if not (control_run_dir(control, fold_id, test_cohort, s) / "DONE.marker").exists()]
                i += len(args.seeds) - len(pending)
                if pending:
                    fold_setup = setup_fold_ablation(fold_dir, ABLATIONS[control], real_edge_index, dev_lookup)
                for seed in pending:
                    i += 1
                    rd = control_run_dir(control, fold_id, test_cohort, seed)
                    eta = f" ETA={fmt_hms((time.time()-t0)/max(len(retrained),1)*(total-i))}" if retrained else ""
                    log(f"  [{i}/{total}] retraining {control} fold{fold_id} seed{seed}{eta}")
                    try:
                        retrained.append(retrain_control(control, fold_dir, fold_setup, seed, rd,
                                                          real_edge_index, dev_lookup))
                    except Exception:
                        log(f"    FAILED {control} fold{fold_id} seed{seed} -- continuing")
                        traceback.print_exc()

        # collect every control run on disk (this session's and any earlier session's)
        all_control_metrics = []
        for mp in CTRL_DIR.glob("fold*/*/seed_*/run_metrics.json"):
            with open(mp) as f:
                all_control_metrics.append(json.load(f))
        log(f"  {len(all_control_metrics)} control runs available on disk")
        verify_controls_match_b9(all_control_metrics, args.control_tol)

    verdicts = []
    if args.stage in ("all", "test1"):
        df1 = run_test1(fold_dirs, real_edge_index, dev_lookup, args)
        _, v1 = analyse_test1(df1, args)
        verdicts.append(v1)
        log(v1)
    if args.stage in ("all", "test2"):
        _, v2 = run_test2(fold_dirs, real_edge_index, args)
        verdicts.append(v2)
        log(v2)
    if args.stage in ("all", "test3"):
        _, _, v3 = run_test3(fold_dirs, args)
        verdicts.append(v3)
        log(v3)

    if verdicts:
        any_contribution = any(("DOES degrade" in v) or ("IS more stable" in v) or
                                (" DOES produce" in v) or (" DOES give" in v) for v in verdicts)
        lines = ["=" * 78, "Graph Utility Analysis", "=" * 78, "",
                 "The accuracy finding is unchanged and not revisited: shuffled/random/empty",
                 "graphs tie or slightly beat the real graph on AUC. This analysis asks only",
                 "whether the graph contributes on OTHER axes, each against a matched control.",
                 "",
                 "Power: cross-fold statements (n=8) are DESCRIPTIVE and underpowered.",
                 "Sample-level statements (n~1,064 held-out samples) are adequately powered.",
                 "Masking results are averaged over >=20 random gene subsets per rate, with the",
                 "same subsets reused across models so comparisons are paired at the draw level.",
                 ""]
        lines += [v + "\n" for v in verdicts]
        lines += ["OVERALL: " + ("the graph contributes on at least one axis tested (see above)."
                                  if any_contribution else
                                  "the graph contributes NOTHING on any axis tested -- robustness, "
                                  "edge interpretability and embedding/calibration quality all fail "
                                  "to separate it from its matched controls. Reported at face value."),
                  "", "=" * 78]
        summary = "\n".join(lines)
        with open(OUT_DIR / "summary.txt", "w") as f:
            f.write(summary + "\n")
        log("")
        for line in lines:
            log(line)

    log(f"outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
