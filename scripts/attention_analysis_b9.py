"""
CoDA-GAT — Block B9: attention analysis (interpretability contribution).

Loads the already-trained B7 "off"-mode checkpoints (8 folds x 5 seeds --
no retraining needed, this is pure analysis) and asks two questions:

  1. Cross-fold stability: does the model attend to the same genes
     regardless of which cohort is held out, or is attention just
     fitting to whatever's idiosyncratic about each training pool?
  2. Biological plausibility: are the top-attended genes concentrated in
     the pathways classically implicated in fibrosis progression (HSC
     activation, ECM remodeling, TGF-beta signaling), more than chance
     would predict from the 72-gene panel's own pathway composition?

Attention is read from CoDAGAT's own `self.last_attn`, already populated
by GAT layer 2's `return_attention_weights=True` inside `encode()` (see
train_fold0_b1.py) -- no new model code, just a new reading of an
attribute the model already exposes on every forward pass.
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

from scipy.stats import spearmanr, fisher_exact
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train_fold0_b1 import CoDAGAT, get_device, N_NODES, N_FEATURES, SHARED_DIR
from run_full_experiment_b7 import load_edge_index_generic, load_fold_info

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
RESULTS_B7 = REPO_ROOT / "results_b7"
OUT_DIR = REPO_ROOT / "attention_analysis_b9"

MODE = "off"
SEEDS = [42, 43, 44, 45, 46]
TOP_K = 15  # ~20% of the 72-gene panel
FIBROSIS_PATHWAYS = {"HSC_activation", "ECM_remodeling", "TGFB_signaling"}


def log(msg):
    print(f"[B9-attn] {msg}", flush=True)


def parse_fold_index(dir_name: str) -> int:
    m = re.match(r"fold(\d+)_", dir_name)
    return int(m.group(1))


def load_test_loader_for_fold(fold_dir, edge_index):
    """Batch size 1 -- keeps attn_edge_index in the un-offset 0..71 range
    per sample, so no batching-offset arithmetic is needed to read it."""
    X = np.load(fold_dir / "test" / "X.npy")
    sample_ids = pd.read_csv(fold_dir / "test" / "sample_ids.csv")["sample_id"].tolist()
    data_list = [Data(x=torch.tensor(X[i], dtype=torch.float32), edge_index=edge_index, sample_id=sample_ids[i])
                 for i in range(X.shape[0])]
    return DataLoader(data_list, batch_size=1, shuffle=False)


def extract_gene_attention(model, loader, device):
    """For each test sample, sums GAT-layer-2 attention over edges where
    each gene is the SOURCE (its total outgoing influence on neighbors'
    updated representations), then averages over all test samples.
    Returns a (72,) importance vector."""
    model.eval()
    totals = np.zeros(N_NODES)
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _ = model(batch.x, batch.edge_index, batch.batch, grl_lambd=0.0)
            attn_edge_index, attn_alpha = model.last_attn
            src = attn_edge_index[0].cpu().numpy()
            alpha = attn_alpha.squeeze(-1).cpu().numpy()  # layer 2 has 1 head
            per_sample = np.zeros(N_NODES)
            np.add.at(per_sample, src, alpha)
            totals += per_sample
            n_samples += 1
    return totals / max(n_samples, 1)


def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT B9 -- attention stability + biology analysis")
    parser.add_argument("--mode", default=MODE)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading shared assets (edges, panel genes)...")
    edge_index = load_edge_index_generic()
    panel = pd.read_csv(SHARED_DIR / "panel_genes.csv")
    gene_order = panel["gene"].tolist()
    device = get_device()

    fold_dirs = sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir())
    fold_profiles = {}

    for fold_dir in fold_dirs:
        fold_id = parse_fold_index(fold_dir.name)
        fold_info = load_fold_info(fold_dir)
        test_cohort = fold_info["test_cohort"]

        log(f"=== fold {fold_id}: {test_cohort} ===")
        test_loader = load_test_loader_for_fold(fold_dir, edge_index)

        seed_profiles = []
        for seed in args.seeds:
            run_dir = RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / f"mode_{args.mode}" / f"seed_{seed}"
            ckpt = torch.load(run_dir / "weights.pt", weights_only=False, map_location=device)
            model = CoDAGAT(N_FEATURES, ckpt["n_ordinal_classes"], ckpt["n_cohorts"]).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            profile = extract_gene_attention(model, test_loader, device)
            seed_profiles.append(profile)
        log(f"    extracted attention from {len(args.seeds)} seed checkpoints "
            f"over {len(test_loader)} test samples each")

        fold_profiles[fold_id] = np.mean(seed_profiles, axis=0)

    profile_df = pd.DataFrame(fold_profiles, index=gene_order).T
    profile_df.index.name = "fold"
    profile_df.to_csv(OUT_DIR / "per_fold_gene_attention.csv")
    log(f"per-fold gene attention profiles saved ({profile_df.shape[0]} folds x {profile_df.shape[1]} genes)")

    # ---- cross-fold stability ----
    n_folds = len(profile_df)
    corr_matrix = np.zeros((n_folds, n_folds))
    for i in range(n_folds):
        for j in range(n_folds):
            corr_matrix[i, j], _ = spearmanr(profile_df.iloc[i], profile_df.iloc[j])
    pairwise = corr_matrix[np.triu_indices(n_folds, k=1)]
    mean_pairwise_corr = float(np.mean(pairwise))
    pd.DataFrame(corr_matrix, index=profile_df.index, columns=profile_df.index).to_csv(
        OUT_DIR / "cross_fold_attention_correlation.csv")
    log(f"cross-fold attention stability: mean pairwise Spearman rho = {mean_pairwise_corr:.3f} "
        f"(across {len(pairwise)} fold pairs)")

    # ---- consensus ranking + pathway enrichment ----
    consensus = profile_df.mean(axis=0).sort_values(ascending=False)
    consensus.to_csv(OUT_DIR / "consensus_gene_ranking.csv", header=["mean_attention"])
    top_genes = consensus.index[:args.top_k].tolist()
    log(f"top {args.top_k} genes by consensus attention: {top_genes}")

    gene_pathway_map = {row["gene"]: set(row["pathways"].split(";")) for _, row in panel.iterrows()}
    all_pathways = set()
    for pw_set in gene_pathway_map.values():
        all_pathways.update(pw_set)

    n_total, n_top = len(panel), len(top_genes)
    enrichment_rows = []
    for pw in sorted(all_pathways):
        n_pw_total = sum(1 for g in gene_order if pw in gene_pathway_map[g])
        n_pw_top = sum(1 for g in top_genes if pw in gene_pathway_map[g])
        table = [[n_pw_top, n_top - n_pw_top], [n_pw_total - n_pw_top, (n_total - n_top) - (n_pw_total - n_pw_top)]]
        _, p_value = fisher_exact(table, alternative="greater")
        enrichment_rows.append({
            "pathway": pw, "n_in_panel": n_pw_total, "n_in_top_k": n_pw_top,
            "fraction_of_panel": n_pw_total / n_total, "fraction_of_top_k": n_pw_top / n_top,
            "fisher_p_enriched": p_value, "fibrosis_classical": pw in FIBROSIS_PATHWAYS,
        })
    enrichment_df = pd.DataFrame(enrichment_rows).sort_values("fisher_p_enriched")
    enrichment_df.to_csv(OUT_DIR / "pathway_enrichment.csv", index=False)

    # ---- summary ----
    lines = ["=" * 78, "CoDA-GAT Block B9 -- Attention Analysis", "=" * 78, ""]
    lines.append(f"Model: mode='{args.mode}', {len(args.seeds)} seeds averaged per fold, {n_folds} folds.")
    lines.append("")
    lines.append(f"CROSS-FOLD STABILITY: mean pairwise Spearman correlation of gene-attention "
                 f"rankings across the {n_folds} held-out-cohort folds = {mean_pairwise_corr:.3f}.")
    if mean_pairwise_corr > 0.5:
        lines.append("Interpreted as reasonably stable: the model attends to a broadly similar "
                     "gene set regardless of which cohort is held out.")
    elif mean_pairwise_corr > 0.2:
        lines.append("Interpreted as weakly/moderately stable -- some consistency, but attention "
                     "shifts meaningfully depending on which cohort is held out.")
    else:
        lines.append("Interpreted as UNSTABLE: attention is largely cohort-specific, not a "
                     "consistent biological signature. Report this plainly -- it is a real finding, "
                     "not a bug to fix before writing up.")
    lines.append("")
    lines.append(f"TOP {args.top_k} GENES BY CONSENSUS ATTENTION (mean across folds and seeds):")
    lines.append(", ".join(top_genes))
    lines.append("")
    lines.append("PATHWAY ENRICHMENT among top genes (one-sided Fisher's exact vs. panel background):")
    for _, row in enrichment_df.iterrows():
        star = "  <-- classical fibrosis pathway" if row["fibrosis_classical"] else ""
        lines.append(f"  {row['pathway']}: {row['n_in_top_k']}/{n_top} of top genes vs "
                     f"{row['n_in_panel']}/{n_total} of panel  (p={row['fisher_p_enriched']:.3f}){star}")
    lines.append("")
    lines.append(f"n={n_folds} folds: the stability estimate above is descriptive, not a hypothesis")
    lines.append("test -- no formal inferential claim is made about it.")
    lines.append("=" * 78)

    summary = "\n".join(lines)
    for line in lines:
        log(line)
    with open(OUT_DIR / "b9_attention_summary.txt", "w") as f:
        f.write(summary + "\n")

    log(f"outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
