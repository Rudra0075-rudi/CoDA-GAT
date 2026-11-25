"""Positive control for T2: can the network tests detect a graph effect when one exists?

The real F2+ label is replaced by a planted label that, by construction, depends on each
gene's STRING neighbourhood:

    z_i  = independent standard-normal value per sample and gene; it replaces feature
           column 0 (expression_z) in the model input, all other features are kept,
    m_i  = mean of z over gene i and its STRING neighbours,
    s    = sum over genes of |m_i|,
    y*   = 1 if s > median of s over the training split of the fold, else 0,

with a share NOISE of labels flipped at random (fixed seed per fold and split). The
model, optimiser, early stopping, threshold rule and folds are those of
scripts/ablations_b9.py, unchanged. Three arms are trained: STRING, disconnected (no
edges) and feature-permuted STRING (same edges, gene features moved to other nodes).

If the T2 tests are able to detect a real network contribution, STRING should beat both
controls here, with an interval that excludes zero. Without edges, or with the features
moved to the wrong genes, a model sees only part of this signal: in a simulation with
logistic regression on per-gene terms, the AUROC reachable with the true neighbourhoods
was 0.98, against 0.67 with no neighbourhoods and 0.59 with permuted ones.

Usage (from the repository root, Zenodo data restored; see run_positive_control.sh):
    python paper_analyses/additional_analyses/positive_control.py . OUT FOLDS SEEDS ARMS [NOISE]
    FOLDS  comma-separated fold folder names, e.g. fold0_GSE130970,fold1_GSE135251
    SEEDS  comma-separated seeds, e.g. 42,43,44,45,46
    ARMS   comma-separated, from string,empty,shuffled
    NOISE  share of planted labels flipped at random (default 0.1)
Each run writes OUT/<fold>/<arm>/seed_<s>/ (probs_test.npy, predictions_test.npy,
run_metrics.json, DONE.marker); finished runs are skipped. Summarise with
positive_control_summary.py.
"""
import os
import sys
import time
from pathlib import Path

W = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
folds = sys.argv[3].split(',')
seeds = [int(s) for s in sys.argv[4].split(',')]
arms = sys.argv[5].split(',')
NOISE = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1

sys.path.insert(0, str(W / 'scripts'))
os.chdir(W)
import numpy as np
import pandas as pd
import torch
import ablations_b9 as ab
from ablations_b9 import setup_fold_ablation, run_single_ablation, load_edge_index_generic, SHARED_DIR

torch.set_num_threads(1)

edges = pd.read_csv(SHARED_DIR / 'graph_edges.csv')
und = edges[edges.src_idx < edges.dst_idx]
ADJ = np.eye(ab.N_NODES)
ADJ[und.src_idx, und.dst_idx] = 1
ADJ[und.dst_idx, und.src_idx] = 1                           # 321 undirected STRING edges + self
MEAN = ADJ / ADJ.sum(axis=1, keepdims=True)                 # row i: mean over gene i and neighbours
THRESH = {}                                                 # fold dir -> training median of s


def planted_signal(fold_dir, split, n):
    """Independent standard-normal value per sample and gene (fixed seed per fold and split)."""
    rng = np.random.default_rng([1000 + ab.parse_fold_index(fold_dir.name),
                                 ['train', 'calibration', 'test'].index(split)])
    return rng.standard_normal((n, ab.N_NODES)).astype(np.float32)


orig_loader = ab.load_split_for_ablation


def patched_loader(fold_dir, split, edge_index, dev_lookup, cohort_to_idx, node_perm=None, zero_cols=None):
    data_list, sample_ids, y = orig_loader(fold_dir, split, edge_index, dev_lookup, cohort_to_idx,
                                           node_perm, zero_cols)
    z = planted_signal(fold_dir, split, len(data_list))
    s = np.abs(z @ MEAN.T).sum(axis=1)                      # signal carried by STRING neighbourhoods
    if split == 'train':
        THRESH[str(fold_dir)] = float(np.median(s))
    y_star = (s > THRESH[str(fold_dir)]).astype(int)
    rng = np.random.default_rng([ab.parse_fold_index(fold_dir.name),
                                 ['train', 'calibration', 'test'].index(split)])
    flip = rng.random(len(y_star)) < NOISE
    y_star = np.where(flip, 1 - y_star, y_star)
    y = y.copy()
    y['y_T1_significant'] = y_star
    zp = z if node_perm is None else z[:, node_perm]           # permuted arm: moved with the features
    for d, zi, v in zip(data_list, zp, y_star):
        d.x = d.x.clone()
        d.x[:, 0] = torch.from_numpy(zi)                      # replaces expression_z in the model input
        d.y1 = torch.tensor([int(v)], dtype=torch.long)
    return data_list, sample_ids, y


ab.load_split_for_ablation = patched_loader
CONDS = {'string': {}, 'empty': {'graph': 'empty'}, 'shuffled': {'graph': 'shuffled'}}

real = load_edge_index_generic()
dev = pd.read_parquet(SHARED_DIR / 'deviation_targets.parquet')
for f in folds:
    fd = W / 'folds' / f
    for c in arms:
        setup = setup_fold_ablation(fd, CONDS[c], real, dev)
        for s in seeds:
            rd = OUT / f / c / f'seed_{s}'
            if (rd / 'DONE.marker').exists():
                continue
            t = time.time()
            m = run_single_ablation(setup, 'planted_' + c, CONDS[c], s, rd)
            print(f, c, s, round(m['auc'], 4), round(m['macro_f1'], 4), f'{time.time() - t:.0f}s', flush=True)
