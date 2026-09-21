"""
CoDA-GAT — Block B1 proof-of-life.
Trains ONE fold (fold0_GSE130970) end-to-end to the locked architecture
contract and writes checkpoints/embeddings/predictions for the conformal
layer to consume later (B7). This script does NOT run the 120-run
experiment (B7) — it is the B1 gate only.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SEED = 42
FOLD_DIR = Path("folds/fold0_GSE130970")
SHARED_DIR = Path("shared")
OUT_DIR = FOLD_DIR / "outputs"

BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.15

N_NODES = 72
N_FEATURES = 26
N_EDGES_EXPECTED = 642

LOSS_WEIGHTS = {"h1": 1.0, "h2": 0.5, "h3": 0.3, "h4": 0.1}


def log(msg):
    print(f"[B1] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_edge_index():
    edges = pd.read_csv(SHARED_DIR / "graph_edges.csv")
    assert len(edges) == N_EDGES_EXPECTED, (
        f"expected {N_EDGES_EXPECTED} directed edges, found {len(edges)}"
    )
    assert edges["src_idx"].max() < N_NODES and edges["dst_idx"].max() < N_NODES
    edge_index = torch.tensor(
        edges[["src_idx", "dst_idx"]].to_numpy().T, dtype=torch.long
    )
    return edge_index


def load_split(split, edge_index, dev_lookup, cohort_to_idx=None):
    """Load one split (train/calibration/test) into a list of PyG Data objects."""
    split_dir = FOLD_DIR / split
    X = np.load(split_dir / "X.npy")
    assert X.shape[1:] == (N_NODES, N_FEATURES), f"{split} X shape {X.shape} unexpected"

    y = pd.read_csv(split_dir / "y.csv", index_col=0)
    sample_ids = pd.read_csv(split_dir / "sample_ids.csv")["sample_id"].tolist()
    assert list(y.index) == sample_ids, f"{split}: y.csv order does not match sample_ids.csv"

    dev_scores = dev_lookup.loc[sample_ids, "deviation_score"].to_numpy(dtype=np.float32)

    y1 = y["y_T1_significant"].to_numpy(dtype=np.float32)
    y2_raw = y["y_T2_ordinal"].to_numpy(dtype=np.float32)  # NaN where grouped label
    y3 = dev_scores
    cohorts = y["cohort"].tolist()

    data_list = []
    for i in range(X.shape[0]):
        x = torch.tensor(X[i], dtype=torch.float32)
        y1_i = torch.tensor(int(y1[i]), dtype=torch.long)
        y2_valid = not np.isnan(y2_raw[i])
        y2_i = torch.tensor(y2_raw[i] if y2_valid else -1.0, dtype=torch.float32)
        y2_mask_i = torch.tensor(y2_valid, dtype=torch.bool)
        y3_i = torch.tensor(y3[i], dtype=torch.float32)

        if cohort_to_idx is not None and cohorts[i] in cohort_to_idx:
            cohort_i = torch.tensor(cohort_to_idx[cohorts[i]], dtype=torch.long)
            cohort_valid = torch.tensor(True)
        else:
            cohort_i = torch.tensor(-1, dtype=torch.long)
            cohort_valid = torch.tensor(False)

        data_list.append(
            Data(
                x=x,
                edge_index=edge_index,
                y1=y1_i.unsqueeze(0),
                y2=y2_i.unsqueeze(0),
                y2_mask=y2_mask_i.unsqueeze(0),
                y3=y3_i.unsqueeze(0),
                cohort=cohort_i.unsqueeze(0),
                cohort_valid=cohort_valid.unsqueeze(0),
                sample_id=sample_ids[i],
            )
        )
    return data_list, sample_ids, y


# --------------------------------------------------------------------------
# Gradient Reversal Layer
# --------------------------------------------------------------------------
class GradReverseFn(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradReverseFn.apply(x, lambd)


def grl_lambda(p):
    """Ganin schedule: p in [0, 1] -> lambda in [0, 1]."""
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


# --------------------------------------------------------------------------
# CORN ordinal loss (Shi, Cao & Raschka, 2022)
# --------------------------------------------------------------------------
def corn_loss(logits, targets, num_classes):
    """
    logits: (N, num_classes-1)
    targets: (N,) long, ranks in [0, num_classes-1]
    Rank-conditional binary decomposition; samples that have not survived
    to threshold k are excluded from that task's loss.
    """
    total_loss = 0.0
    num_examples = 0
    for task_idx in range(num_classes - 1):
        mask = targets > (task_idx - 1)
        if mask.sum() == 0:
            continue
        task_targets = (targets[mask] > task_idx).float()
        task_logits = logits[mask, task_idx]
        num_examples += mask.sum().item()
        total_loss = total_loss + F.binary_cross_entropy_with_logits(
            task_logits, task_targets, reduction="sum"
        )
    if num_examples == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return total_loss / num_examples


def corn_predict_stage(logits):
    """Convert CORN logits (N, K-1) to predicted ordinal rank (N,)."""
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class CoDAGAT(nn.Module):
    def __init__(self, in_features, n_ordinal_classes, n_cohorts):
        super().__init__()
        self.gat1 = GATConv(in_features, 32, heads=4, concat=True, dropout=0.3, add_self_loops=True)
        self.gat2 = GATConv(32 * 4, 32, heads=1, concat=False, dropout=0.3, add_self_loops=True)
        self.shared_dense = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.3)

        self.h1 = nn.Linear(16, 2)
        self.h2 = nn.Linear(16, n_ordinal_classes - 1)
        self.h3 = nn.Linear(16, 1)
        self.h4 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_cohorts),
        )

        self.last_attn = None  # (edge_index, alpha) from GAT layer 2

    def encode(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        x, (attn_edge_index, attn_alpha) = self.gat2(
            x, edge_index, return_attention_weights=True
        )
        self.last_attn = (attn_edge_index.detach(), attn_alpha.detach())
        x = F.elu(x)
        x = self.dropout(x)

        pooled = global_mean_pool(x, batch)  # (B, 32)
        shared = F.elu(self.shared_dense(pooled))
        shared = self.dropout(shared)  # (B, 16)
        return shared

    def forward(self, x, edge_index, batch, grl_lambd=0.0):
        shared = self.encode(x, edge_index, batch)
        h1_logits = self.h1(shared)
        h2_logits = self.h2(shared)
        h3_pred = self.h3(shared).squeeze(-1)
        reversed_shared = grad_reverse(shared, grl_lambd)
        h4_logits = self.h4(reversed_shared)
        return {
            "embedding": shared,
            "h1_logits": h1_logits,
            "h2_logits": h2_logits,
            "h3_pred": h3_pred,
            "h4_logits": h4_logits,
        }


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer, class_weights, n_ordinal_classes,
              p_progress, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    all_y1, all_pred1 = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            lambd = grl_lambda(p_progress) if train else 0.0

            out = model(batch.x, batch.edge_index, batch.batch, grl_lambd=lambd)

            y1 = batch.y1.view(-1)
            loss_h1 = F.cross_entropy(out["h1_logits"], y1, weight=class_weights)

            y2_mask = batch.y2_mask.view(-1)
            if y2_mask.any():
                y2 = batch.y2.view(-1)[y2_mask].long()
                h2_logits_masked = out["h2_logits"][y2_mask]
                loss_h2 = corn_loss(h2_logits_masked, y2, n_ordinal_classes)
            else:
                loss_h2 = torch.tensor(0.0, device=device)

            y3 = batch.y3.view(-1)
            loss_h3 = F.mse_loss(out["h3_pred"], y3)

            cohort_valid = batch.cohort_valid.view(-1)
            if train and cohort_valid.any():
                cohort = batch.cohort.view(-1)[cohort_valid]
                h4_logits_masked = out["h4_logits"][cohort_valid]
                loss_h4 = F.cross_entropy(h4_logits_masked, cohort)
            else:
                loss_h4 = torch.tensor(0.0, device=device)

            loss = (
                LOSS_WEIGHTS["h1"] * loss_h1
                + LOSS_WEIGHTS["h2"] * loss_h2
                + LOSS_WEIGHTS["h3"] * loss_h3
                + LOSS_WEIGHTS["h4"] * loss_h4
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            all_y1.append(y1.detach().cpu().numpy())
            all_pred1.append(out["h1_logits"].detach().argmax(dim=1).cpu().numpy())

    all_y1 = np.concatenate(all_y1)
    all_pred1 = np.concatenate(all_pred1)
    macro_f1 = f1_score(all_y1, all_pred1, average="macro")
    return total_loss / max(n_batches, 1), macro_f1


@torch.no_grad()
def extract_outputs(model, loader, device):
    model.eval()
    embeddings, probs, preds, sample_ids = [], [], [], []
    for batch in loader:
        sids = batch.sample_id
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, grl_lambd=0.0)
        p = F.softmax(out["h1_logits"], dim=1)[:, 1]
        pred = out["h1_logits"].argmax(dim=1)
        embeddings.append(out["embedding"].cpu().numpy())
        probs.append(p.cpu().numpy())
        preds.append(pred.cpu().numpy())
        sample_ids.extend(sids if isinstance(sids, list) else [sids])
    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(probs, axis=0),
        np.concatenate(preds, axis=0),
        sample_ids,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    set_seed(SEED)
    device = get_device()
    log(f"device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading graph edges...")
    edge_index = load_edge_index()
    log(f"edge_index shape: {tuple(edge_index.shape)} (642 directed edges confirmed)")

    log("loading deviation targets (H3)...")
    dev_lookup = pd.read_parquet(SHARED_DIR / "deviation_targets.parquet")

    log("reading fold_info.json for cohort vocabulary...")
    with open(FOLD_DIR / "fold_info.json") as f:
        fold_info = json.load(f)
    train_cohorts = sorted(fold_info["train_cohorts"])
    cohort_to_idx = {c: i for i, c in enumerate(train_cohorts)}
    n_cohorts = len(cohort_to_idx)
    log(f"train cohorts ({n_cohorts}): {train_cohorts}")

    log("loading train / calibration / test splits...")
    train_data, train_ids, train_y = load_split("train", edge_index, dev_lookup, cohort_to_idx)
    cal_data, cal_ids, cal_y = load_split("calibration", edge_index, dev_lookup, cohort_to_idx)
    test_data, test_ids, test_y = load_split("test", edge_index, dev_lookup, cohort_to_idx)
    log(f"train n={len(train_data)}  calibration n={len(cal_data)}  test n={len(test_data)}")

    n_ordinal_classes = int(
        np.nanmax(
            pd.concat([train_y["y_T2_ordinal"], cal_y["y_T2_ordinal"], test_y["y_T2_ordinal"]])
        )
    ) + 1
    log(f"ordinal classes inferred: {n_ordinal_classes} (F0..F{n_ordinal_classes - 1})")

    # --- class weights for H1 (inverse frequency, train split only) ---
    y1_train = train_y["y_T1_significant"].to_numpy()
    counts = np.bincount(y1_train.astype(int))
    weights = len(y1_train) / (len(counts) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    log(f"H1 class weights (train-derived, inverse frequency): {weights.tolist()}")

    # --- validation split: 15% of train, stratified by cohort x class ---
    strat_key = [f"{c}_{int(y)}" for c, y in zip(train_y["cohort"], y1_train)]
    idx_all = np.arange(len(train_data))
    idx_fit, idx_val = train_test_split(
        idx_all, test_size=VAL_FRACTION, random_state=SEED, stratify=strat_key
    )
    fit_data = [train_data[i] for i in idx_fit]
    val_data = [train_data[i] for i in idx_val]
    log(f"internal split for early stopping: fit n={len(fit_data)}  val n={len(val_data)}")

    fit_loader = DataLoader(fit_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    train_loader_full = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    cal_loader = DataLoader(cal_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = CoDAGAT(N_FEATURES, n_ordinal_classes, n_cohorts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    log("starting training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        p_progress = epoch / MAX_EPOCHS
        train_loss, train_f1 = run_epoch(
            model, fit_loader, device, optimizer, class_weights,
            n_ordinal_classes, p_progress, train=True,
        )
        val_loss, val_f1 = run_epoch(
            model, val_loader, device, optimizer, class_weights,
            n_ordinal_classes, p_progress, train=False,
        )

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            log(
                f"epoch {epoch:3d}/{MAX_EPOCHS}  "
                f"train_loss={train_loss:.4f} train_f1={train_f1:.4f}  "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}  "
                f"{'*' if improved else ''}"
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
         "cohort_to_idx": cohort_to_idx},
        OUT_DIR / "weights.pt",
    )

    log("extracting embeddings / probabilities / predictions...")
    emb_train, probs_train, preds_train, _ = extract_outputs(model, train_loader_full, device)
    emb_cal, probs_cal, preds_cal, _ = extract_outputs(model, cal_loader, device)
    emb_test, probs_test, preds_test, _ = extract_outputs(model, test_loader, device)

    np.save(OUT_DIR / "embeddings_train.npy", emb_train)
    np.save(OUT_DIR / "embeddings_calibration.npy", emb_cal)
    np.save(OUT_DIR / "embeddings_test.npy", emb_test)
    np.save(OUT_DIR / "probs_calibration.npy", probs_cal)
    np.save(OUT_DIR / "probs_test.npy", probs_test)
    np.save(OUT_DIR / "predictions_test.npy", preds_test)

    log(f"embeddings_train {emb_train.shape}  embeddings_calibration {emb_cal.shape}  embeddings_test {emb_test.shape}")

    # nonconformity score on calibration: 1 - p(true class)
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

    log("all B1 gate artifacts written to " + str(OUT_DIR))
    for p in sorted(OUT_DIR.iterdir()):
        log(f"  {p.name}  ({p.stat().st_size} bytes)")

    log("B1 proof-of-life complete.")


if __name__ == "__main__":
    main()
