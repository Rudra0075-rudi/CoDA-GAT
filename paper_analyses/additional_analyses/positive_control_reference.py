"""Reference AUROCs for the positive control of T2 (no graph model involved).

The planted label of positive_control.py is reproduced on simulated samples:
z is an independent standard-normal value per sample and gene, m_i is the mean of z
over gene i and its STRING neighbours, s = sum_i |m_i|, y = 1 if s exceeds the
training median of s, and 10% of labels are flipped at random in every split, as in
positive_control.py. Logistic regression is then fitted on per-gene terms built from
  - the true neighbourhoods:      |m_i|,
  - no neighbourhoods:            |z_i|,
  - permuted neighbourhoods:      |m_i| computed after moving each gene's value to
                                  another gene (a random permutation per replicate),
and scored on an independent test set against both the noisy labels (the labels the
graph models were scored against) and the noise-free labels. The oracle row scores s
itself. Ten replicates of 1,000 training and 1,000 test samples, seeds 0-9.

Usage (from the repository root, Zenodo data restored):
    python paper_analyses/additional_analyses/positive_control_reference.py shared/graph_edges.csv
Output: positive_control_reference.txt
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N_NODES, NOISE, N_TRAIN, N_TEST, N_REP = 72, 0.10, 1000, 1000, 10

edges = pd.read_csv(sys.argv[1])
und = edges[edges.src_idx < edges.dst_idx]
adj = np.eye(N_NODES)
adj[und.src_idx, und.dst_idx] = 1
adj[und.dst_idx, und.src_idx] = 1
mean_op = adj / adj.sum(axis=1, keepdims=True)

rows = []
for rep in range(N_REP):
    rng = np.random.default_rng(rep)
    perm = rng.permutation(N_NODES)
    z_tr, z_te = rng.standard_normal((N_TRAIN, N_NODES)), rng.standard_normal((N_TEST, N_NODES))
    s_tr, s_te = np.abs(z_tr @ mean_op.T).sum(1), np.abs(z_te @ mean_op.T).sum(1)
    thr = np.median(s_tr)

    def labels(s):
        clean = (s > thr).astype(int)
        flip = rng.random(len(clean)) < NOISE
        return clean, np.where(flip, 1 - clean, clean)

    _, ytr_noisy = labels(s_tr)
    yte_clean, yte_noisy = labels(s_te)
    feats = {
        "true neighbourhoods": lambda z: np.abs(z @ mean_op.T),
        "no neighbourhoods": lambda z: np.abs(z),
        "permuted neighbourhoods": lambda z: np.abs(z[:, perm] @ mean_op.T),
    }
    for name, f in feats.items():
        p = LogisticRegression(max_iter=5000).fit(f(z_tr), ytr_noisy).predict_proba(f(z_te))[:, 1]
        rows.append((name, rep, roc_auc_score(yte_noisy, p), roc_auc_score(yte_clean, p)))
    rows.append(("oracle score s", rep, roc_auc_score(yte_noisy, s_te), roc_auc_score(yte_clean, s_te)))

df = pd.DataFrame(rows, columns=["inputs", "replicate", "auroc_noisy_labels", "auroc_clean_labels"])
summary = df.groupby("inputs", sort=False)[["auroc_noisy_labels", "auroc_clean_labels"]].agg(["mean", "std"])
print(f"Positive-control reference AUROCs: {N_REP} replicates, {N_TRAIN} training and "
      f"{N_TEST} test samples, {int(NOISE * 100)}% label noise in every split")
print(summary.round(3).to_string())
