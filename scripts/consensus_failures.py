"""
CoDA-GAT — Model-agreement / consensus-failure analysis.

Asks whether architecturally different models fail on the same patients, as a
second line of evidence for the performance ceiling that requires no kappa
inversion. The discriminating test is not the agreement itself (models sharing
input features agree for trivial reasons) but WHERE the consensus failures land:
concentrated at the F1/F2 boundary -> consistent with label error; scattered
across F0/F3/F4 -> more consistent with shared-input correlation or genuinely
hard cases.

WHAT IS ACTUALLY AVAILABLE (discovered, not assumed -- see the startup report):
Only models whose per-sample test predictions were saved can enter this
analysis. As of writing that is 9 CoDA-GAT variants (B7's 3 adversary modes +
B9's 6 ablations). The 12 tabular baselines, the published hybrids, and B3's
flattened_mlp / fully_connected controls saved AGGREGATE METRICS ONLY -- their
per-sample predictions were computed in memory by baselines_b3.py and
discarded. That means the linear, tree-ensemble and hybrid families are absent.

Consequently the verdict is GATED ON ARCHITECTURAL DIVERSITY: with fewer than
two distinct architecture families present, this script will NOT report the
convergent-failure argument as supported, however significant the adjacency
test is, because agreement among variants of one architecture sharing one
feature tensor cannot distinguish label error from shared-input correlation.
The adjacency result is still computed and reported as a descriptive finding.

No retraining, no threshold re-tuning. Saved predictions only.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import matthews_corrcoef

from test_a_adjacency import enrichment_test, parse_fold_index, ADJACENT_STAGES

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
SHARED_DIR = REPO_ROOT / "shared"
RESULTS_B7 = REPO_ROOT / "results_b7"
ABLATIONS_B9 = REPO_ROOT / "ablations_b9"
BASELINES_DIR = REPO_ROOT / "baselines"
TEST_A_FLAGS = REPO_ROOT / "test_a_adjacency" / "per_sample_flags.csv"
OUT_DIR = REPO_ROOT / "consensus_failures"

SEEDS = [42, 43, 44, 45, 46]
DECISION_THRESHOLD = 0.5  # how every available model was originally evaluated (argmax on 2-class softmax)
CONSENSUS_LEVELS = [0.80, 0.90, 1.00]
PRIMARY_CONSENSUS_LEVEL = 0.80
N_PERMUTATIONS = 1000

# variant_group = what differs between these models; architecture_family = the
# thing the diversity gate cares about. All currently-available models are 'gat'.
#
# B3 baselines persist per-sample predictions as CSV (sample_id + p_test +
# the model's OWN tuned threshold), not as the 5-seed .npy layout B7/B9 use.
# They were run at a single seed and evaluated at a calibration-tuned
# threshold, so they cannot be expressed in the .npy format without
# fabricating seed copies and discarding the threshold. The "baselines_csv"
# source below reads them natively and applies each model's saved threshold.
#
BASELINES_PER_SAMPLE = BASELINES_DIR / "per_sample_predictions"

MODEL_REGISTRY = {
    # B7 adversary modes
    "codagat_off":              {"source": "b7", "key": "off",                      "variant_group": "codagat_core",    "architecture_family": "gat"},
    "codagat_marginal":         {"source": "b7", "key": "marginal",                 "variant_group": "codagat_core",    "architecture_family": "gat"},
    "codagat_conditional":      {"source": "b7", "key": "conditional",              "variant_group": "codagat_core",    "architecture_family": "gat"},
    # B9 graph corruptions
    "shuffled_graph":           {"source": "b9", "key": "shuffled_graph",           "variant_group": "graph_corrupted", "architecture_family": "gat"},
    "random_graph":             {"source": "b9", "key": "random_graph",             "variant_group": "graph_corrupted", "architecture_family": "gat"},
    "empty_graph":              {"source": "b9", "key": "empty_graph",              "variant_group": "graph_corrupted", "architecture_family": "gat"},
    # B9 feature ablations
    "no_h3_deviation":          {"source": "b9", "key": "no_h3_deviation",          "variant_group": "feature_ablated", "architecture_family": "gat"},
    "no_pathway_features":      {"source": "b9", "key": "no_pathway_features",      "variant_group": "feature_ablated", "architecture_family": "gat"},
    "no_fold_fitted_features":  {"source": "b9", "key": "no_fold_fitted_features",  "variant_group": "feature_ablated", "architecture_family": "gat"},
}

# --- B3 baselines (added once baselines_b3.py was patched to persist them) ---
for _view in ("bulk_top1000", "panel72_combined", "panel72_full"):
    MODEL_REGISTRY[f"{_view}__logreg"] = {
        "source": "baselines_csv", "key": f"{_view}__logreg",
        "variant_group": f"tabular_{_view}", "architecture_family": "linear"}
    for _clf in ("random_forest", "xgboost", "catboost"):
        MODEL_REGISTRY[f"{_view}__{_clf}"] = {
            "source": "baselines_csv", "key": f"{_view}__{_clf}",
            "variant_group": f"tabular_{_view}", "architecture_family": "tree_ensemble"}

for _clf in ("random_forest", "xgboost"):
    MODEL_REGISTRY[f"boruta_lasso_{_clf}"] = {
        "source": "baselines_csv", "key": f"boruta_lasso_{_clf}",
        "variant_group": "boruta_lasso", "architecture_family": "tree_ensemble"}

MODEL_REGISTRY["graphsage_supcon"] = {
    "source": "baselines_csv", "key": "graphsage_supcon",
    "variant_group": "published_hybrid", "architecture_family": "gnn_sage"}
MODEL_REGISTRY["knowledgegcn_settransformer"] = {
    "source": "baselines_csv", "key": "knowledgegcn_settransformer",
    "variant_group": "published_hybrid", "architecture_family": "gnn_gcn_settransformer"}

# flattened_mlp is the only genuinely graph-free model in the whole project --
# it is the single most valuable addition for the architectural-diversity gate.
MODEL_REGISTRY["flattened_mlp"] = {
    "source": "baselines_csv", "key": "flattened_mlp",
    "variant_group": "non_graph_control", "architecture_family": "mlp"}
# fully_connected is still a GAT, just over a complete graph
MODEL_REGISTRY["fully_connected"] = {
    "source": "baselines_csv", "key": "fully_connected",
    "variant_group": "graph_corrupted", "architecture_family": "gat"}

def log(msg):
    print(f"[consensus] {msg}", flush=True)


# ==========================================================================
# Discovery + loading (saved predictions only)
# ==========================================================================
def fold_test_index():
    """sample_id + fold for every held-out test sample, in the exact row order
    the saved probs_test.npy arrays use (test loaders were built shuffle=False)."""
    rows = []
    for fold_dir in sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir()):
        fold_id = parse_fold_index(fold_dir.name)
        with open(fold_dir / "fold_info.json") as f:
            test_cohort = json.load(f)["test_cohort"]
        ids = pd.read_csv(fold_dir / "test" / "sample_ids.csv")["sample_id"].tolist()
        rows.append(pd.DataFrame({"fold": fold_id, "test_cohort": test_cohort, "sample_id": ids}))
    return pd.concat(rows, ignore_index=True)


def run_dir_for(spec, fold_id, test_cohort, seed):
    if spec["source"] == "b7":
        return RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / f"mode_{spec['key']}" / f"seed_{seed}"
    if spec["source"] == "b9":
        return ABLATIONS_B9 / f"fold{fold_id}_{test_cohort}" / spec["key"] / f"seed_{seed}"
    raise ValueError(spec["source"])


def load_baselines_csv_probs(name, spec, index_df):
    """
    Reads B3's persisted per-sample CSVs and aligns them to index_df's sample
    order. Returns (p_test, per-sample threshold).

    These models were run at a SINGLE seed (B3's default, 42) and evaluated at
    a calibration-tuned threshold. Both facts are preserved here, not changed:
    the threshold comes from the file, so each model's error set matches how it
    was originally evaluated.
    """
    frames = []
    for (fold_id, test_cohort), grp in index_df.groupby(["fold", "test_cohort"], sort=True):
        path = BASELINES_PER_SAMPLE / spec["key"] / f"fold{fold_id}_{test_cohort}.csv"
        if not path.exists():
            log(f"    {name}: MISSING {path.relative_to(REPO_ROOT)} -- excluding this model")
            return None
        df = pd.read_csv(path)
        absent = set(grp["sample_id"]) - set(df["sample_id"])
        if absent:
            log(f"    {name}: {len(absent)} sample_id(s) absent at fold {fold_id} -- excluding")
            return None
        frames.append(df.set_index("sample_id").loc[grp["sample_id"]].reset_index())
    out = pd.concat(frames, ignore_index=True)
    return out["p_test"].to_numpy(dtype=float), out["threshold"].to_numpy(dtype=float)


def load_model_probs(name, spec, index_df, seeds):
    """Returns (P(F2+) per test sample, per-sample decision threshold), or None.
    B7/B9 models are seed-averaged .npy and were evaluated at argmax (0.5);
    B3 baselines are single-seed CSV carrying their own tuned threshold."""
    if spec["source"] == "baselines_csv":
        return load_baselines_csv_probs(name, spec, index_df)

    per_fold = []
    for (fold_id, test_cohort), grp in index_df.groupby(["fold", "test_cohort"], sort=True):
        seed_probs = []
        for seed in seeds:
            run_dir = run_dir_for(spec, fold_id, test_cohort, seed)
            path = run_dir / "probs_test.npy"
            if not path.exists():
                log(f"    {name}: MISSING {path.relative_to(REPO_ROOT)} -- excluding this model")
                return None
            probs = np.load(path)
            if probs.shape[0] != len(grp):
                log(f"    {name}: shape mismatch at fold {fold_id} "
                    f"({probs.shape[0]} vs {len(grp)} samples) -- excluding this model")
                return None
            seed_probs.append(probs[:, 1])
        per_fold.append(np.mean(seed_probs, axis=0))
    p = np.concatenate(per_fold)
    return p, np.full(len(p), DECISION_THRESHOLD)


# ==========================================================================
# Step 3: agreement, against chance
# ==========================================================================
def expected_jaccard(p1, p2):
    """Analytic E[Jaccard] for two independent error sets with rates p1, p2."""
    denom = p1 + p2 - p1 * p2
    return float(p1 * p2 / denom) if denom > 0 else float("nan")


def jaccard(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union > 0 else float("nan")


def permute_within_fold(err, fold_ids, rng):
    """Shuffles a model's error vector WITHIN each fold, preserving its per-fold
    error rate. A global shuffle would destroy the fold/cohort difficulty
    structure and inflate apparent excess agreement."""
    out = err.copy()
    for f in np.unique(fold_ids):
        mask = fold_ids == f
        out[mask] = rng.permutation(err[mask])
    return out


def agreement_analysis(err_matrix, model_names, fold_ids, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    rates = {m: float(err_matrix[m].mean()) for m in model_names}

    # observed
    rows = []
    for a, b in itertools.combinations(model_names, 2):
        ea, eb = err_matrix[a].to_numpy(), err_matrix[b].to_numpy()
        obs_j = jaccard(ea, eb)
        exp_j = expected_jaccard(rates[a], rates[b])
        mcc = matthews_corrcoef(ea, eb) if (ea.std() > 0 and eb.std() > 0) else float("nan")
        rows.append({
            "model_a": a, "model_b": b,
            "error_rate_a": rates[a], "error_rate_b": rates[b],
            "n_both_wrong": int(np.logical_and(ea, eb).sum()),
            "jaccard_observed": obs_j, "jaccard_expected_chance": exp_j,
            "jaccard_excess": obs_j - exp_j, "mcc": mcc,
        })
    pair_df = pd.DataFrame(rows)

    # within-fold permutation null
    log(f"    running {n_perm} within-fold permutations for the chance null...")
    perm_j = {(a, b): np.empty(n_perm) for a, b in itertools.combinations(model_names, 2)}
    err_np = {m: err_matrix[m].to_numpy() for m in model_names}
    for i in range(n_perm):
        shuffled = {m: permute_within_fold(err_np[m], fold_ids, rng) for m in model_names}
        for a, b in perm_j:
            perm_j[(a, b)][i] = jaccard(shuffled[a], shuffled[b])

    perm_mean, perm_sd, perm_p = [], [], []
    for _, r in pair_df.iterrows():
        dist = perm_j[(r["model_a"], r["model_b"])]
        perm_mean.append(float(np.nanmean(dist)))
        sd = float(np.nanstd(dist))
        perm_sd.append(sd)
        perm_p.append(float((np.nansum(dist >= r["jaccard_observed"]) + 1) / (len(dist) + 1)))
    pair_df["jaccard_perm_null_mean"] = perm_mean
    pair_df["jaccard_perm_null_sd"] = perm_sd
    pair_df["jaccard_perm_z"] = (pair_df["jaccard_observed"] - pair_df["jaccard_perm_null_mean"]) / \
                                 pair_df["jaccard_perm_null_sd"].replace(0, np.nan)
    pair_df["jaccard_perm_p_onesided"] = perm_p
    return pair_df, rates


def group_summary(pair_df, registry, level_key):
    """Within- vs between-group agreement at the requested grouping level."""
    def grp(m):
        return registry[m][level_key]

    rows = []
    pair_df = pair_df.copy()
    pair_df["group_a"] = pair_df["model_a"].map(grp)
    pair_df["group_b"] = pair_df["model_b"].map(grp)
    pair_df["relation"] = np.where(pair_df["group_a"] == pair_df["group_b"], "within", "between")

    for relation, sub in pair_df.groupby("relation"):
        rows.append({
            "level": level_key, "relation": relation, "n_pairs": len(sub),
            "mean_jaccard_observed": float(sub["jaccard_observed"].mean()),
            "mean_jaccard_expected_chance": float(sub["jaccard_expected_chance"].mean()),
            "mean_jaccard_excess": float(sub["jaccard_excess"].mean()),
            "mean_mcc": float(sub["mcc"].mean()),
            "mean_perm_z": float(sub["jaccard_perm_z"].mean()),
        })

    # unordered group-pair key, built explicitly -- a DataFrame.min(axis=1) over
    # string columns is not reliably supported across pandas versions
    pair_df["_g1"] = [min(a, b) for a, b in zip(pair_df["group_a"], pair_df["group_b"])]
    pair_df["_g2"] = [max(a, b) for a, b in zip(pair_df["group_a"], pair_df["group_b"])]
    for (ga, gb), sub in pair_df.groupby(["_g1", "_g2"]):
        rows.append({
            "level": f"{level_key}_pairwise", "relation": f"{ga} vs {gb}", "n_pairs": len(sub),
            "mean_jaccard_observed": float(sub["jaccard_observed"].mean()),
            "mean_jaccard_expected_chance": float(sub["jaccard_expected_chance"].mean()),
            "mean_jaccard_excess": float(sub["jaccard_excess"].mean()),
            "mean_mcc": float(sub["mcc"].mean()),
            "mean_perm_z": float(sub["jaccard_perm_z"].mean()),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT consensus-failure analysis")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--consensus-level", type=float, default=PRIMARY_CONSENSUS_LEVEL,
                         help="fraction of models that must be wrong to call a sample a consensus failure")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("Model-agreement / consensus-failure analysis (saved predictions only, no retraining)")

    # ---- Step 1: discovery ----
    index_df = fold_test_index()
    log(f"held-out test samples across all folds: {len(index_df)}")

    labels_df = pd.read_parquet(SHARED_DIR / "labels.parquet")
    index_df["y_true"] = labels_df.loc[index_df["sample_id"], "y_T1_significant"].to_numpy().astype(int)
    index_df["stage"] = labels_df.loc[index_df["sample_id"], "stage"].to_numpy()

    log("discovering models with SAVED per-sample predictions...")
    probs, thresholds, used_specs, missing = {}, {}, {}, {}
    for name, spec in MODEL_REGISTRY.items():
        loaded = load_model_probs(name, spec, index_df, args.seeds)
        if loaded is not None:
            probs[name], thresholds[name] = loaded
            used_specs[name] = spec
            log(f"    {name:30s} OK  ({spec['variant_group']}, family={spec['architecture_family']})")
        else:
            missing.setdefault(spec["architecture_family"], []).append(name)
    if len(probs) < 2:
        raise RuntimeError(f"only {len(probs)} model(s) had usable saved per-sample predictions; "
                            f"cannot compute agreement.")

    n_missing = sum(len(v) for v in missing.values())
    log("")
    if n_missing:
        log(f"MODELS WITHOUT saved per-sample predictions ({n_missing}) -- excluded, not fabricated:")
        for fam, models in sorted(missing.items()):
            log(f"    {fam:24s} ({len(models)}): {', '.join(models)}")
    else:
        log("every registered model had saved per-sample predictions.")
    families_present = sorted({s["architecture_family"] for s in used_specs.values()})
    families_absent = sorted(missing.keys())
    log(f"architecture families present ({len(families_present)}): {families_present}")
    if families_absent:
        log(f"architecture families absent:  {families_absent}")
    log("")

    # ---- Step 2: error matrix ----
    err_matrix = pd.DataFrame({"sample_id": index_df["sample_id"], "fold": index_df["fold"],
                                "stage": index_df["stage"], "y_true": index_df["y_true"]})
    for name, p in probs.items():
        # each model's OWN threshold: 0.5 (argmax) for the B7/B9 GAT models,
        # the saved calibration-tuned value for the B3 baselines -- so every
        # error set matches how that model was originally evaluated
        pred = (p >= thresholds[name]).astype(int)
        err_matrix[name] = (pred != index_df["y_true"].to_numpy()).astype(bool)

    model_names = list(probs.keys())
    log(f"Step 2: error matrix = {len(err_matrix)} samples x {len(model_names)} models")
    for name in model_names:
        log(f"    {name:26s} error rate = {err_matrix[name].mean():.1%}")
    err_matrix.to_csv(OUT_DIR / "error_matrix.csv", index=False)

    # ---- Step 3: agreement vs chance ----
    log("Step 3: pairwise error agreement, observed vs chance...")
    pair_df, rates = agreement_analysis(err_matrix, model_names,
                                         err_matrix["fold"].to_numpy(), args.n_permutations)
    pair_df.to_csv(OUT_DIR / "pairwise_agreement.csv", index=False)
    log(f"    mean observed Jaccard = {pair_df['jaccard_observed'].mean():.3f}  "
        f"vs chance {pair_df['jaccard_expected_chance'].mean():.3f}  "
        f"(excess {pair_df['jaccard_excess'].mean():+.3f}, mean MCC {pair_df['mcc'].mean():.3f})")

    fam_df = pd.concat([group_summary(pair_df, used_specs, "variant_group"),
                        group_summary(pair_df, used_specs, "architecture_family")],
                       ignore_index=True)
    fam_df.to_csv(OUT_DIR / "family_agreement_summary.csv", index=False)
    for _, r in fam_df[fam_df["level"] == "variant_group"].iterrows():
        log(f"    variant_group {r['relation']:8s}: observed {r['mean_jaccard_observed']:.3f} "
            f"vs chance {r['mean_jaccard_expected_chance']:.3f} "
            f"(excess {r['mean_jaccard_excess']:+.3f}, z={r['mean_perm_z']:.1f}, n={int(r['n_pairs'])} pairs)")

    # ---- Step 4: consensus failures ----
    n_models = len(model_names)
    err_matrix["n_models_wrong"] = err_matrix[model_names].sum(axis=1)
    dist = err_matrix["n_models_wrong"].value_counts().sort_index()
    log("Step 4: distribution of how many models get each sample wrong:")
    for k, v in dist.items():
        log(f"    {int(k):2d}/{n_models} models wrong: {v:4d} samples ({v/len(err_matrix):.1%})")

    consensus_sets = {}
    for lvl in sorted(set(CONSENSUS_LEVELS + [args.consensus_level])):
        need = int(np.ceil(lvl * n_models))
        mask = (err_matrix["n_models_wrong"] >= need).to_numpy()
        consensus_sets[lvl] = mask
        log(f"    consensus failures at >={lvl:.0%} ({need}/{n_models} models wrong): "
            f"{int(mask.sum())} samples ({mask.mean():.1%})")

    primary_mask = consensus_sets[args.consensus_level]
    err_matrix.loc[primary_mask, ["sample_id", "fold", "stage", "y_true", "n_models_wrong"]] \
        .to_csv(OUT_DIR / "consensus_failure_samples.csv", index=False)

    # ---- Step 5: the discriminating adjacency test (Test A's function, unchanged) ----
    log("Step 5: adjacency test on the consensus-failure set (Test A's enrichment_test, unchanged)...")
    adj_lines = ["Adjacency test on consensus failures", "=" * 60, ""]
    adj_results = {}
    for lvl in sorted(consensus_sets):
        mask = consensus_sets[lvl]
        if mask.sum() == 0:
            adj_lines.append(f">={lvl:.0%}: no consensus failures at this level -- test not defined.")
            continue
        table_5, (chi2, p_chi2), table_2, (odds_ratio, p_fisher) = enrichment_test(err_matrix, mask)
        adj_results[lvl] = {"n": int(mask.sum()), "odds_ratio": odds_ratio, "fisher_p": p_fisher,
                             "chi2": chi2, "chi2_p": p_chi2}
        adj_lines += [
            f">= {lvl:.0%} of models wrong  (n={int(mask.sum())} consensus failures)",
            "  stage distribution (consensus failure vs rest):",
            "    " + table_5.to_string().replace("\n", "\n    "),
            f"  2x5 chi-square: chi2={chi2:.3f}, p={p_chi2:.4g}",
            f"  2x2 Fisher (adjacent {sorted(ADJACENT_STAGES)} vs distant, one-sided): "
            f"OR={odds_ratio:.3f}, p={p_fisher:.4g}",
            "",
        ]
    with open(OUT_DIR / "adjacency_test.txt", "w") as f:
        f.write("\n".join(adj_lines) + "\n")
    for line in adj_lines:
        log("    " + line)

    # ---- Step 6: overlap with the Test A detector ----
    log("Step 6: overlap with the existing CoDA-GAT flags...")
    ov_lines = ["Overlap: consensus failures vs Test A detector flags", "=" * 60, ""]
    if TEST_A_FLAGS.exists():
        flags = pd.read_csv(TEST_A_FLAGS)[["sample_id", "flagged"]]
        merged = err_matrix[["sample_id"]].assign(consensus=primary_mask).merge(
            flags, on="sample_id", how="left")
        # robust to the left-merge introducing NaN and to object dtype
        merged["flagged"] = (merged["flagged"] == True)  # noqa: E712
        both = int((merged["consensus"] & merged["flagged"]).sum())
        only_cons = int((merged["consensus"] & ~merged["flagged"]).sum())
        only_flag = int((~merged["consensus"] & merged["flagged"]).sum())
        neither = int((~merged["consensus"] & ~merged["flagged"]).sum())
        union = both + only_cons + only_flag
        jac = both / union if union else float("nan")
        ov_lines += [
            f"consensus-failure level: >={args.consensus_level:.0%} of models wrong",
            f"  both consensus AND flagged      : {both}",
            f"  consensus only (not flagged)    : {only_cons}",
            f"  flagged only (not consensus)    : {only_flag}",
            f"  neither                         : {neither}",
            f"  Jaccard(consensus, flagged)     : {jac:.3f}",
            "",
            ("The two methods largely identify the SAME patients."
             if jac >= 0.5 else
             "The two methods largely identify DIFFERENT patients -- reported at face value. "
             "Low overlap is informative either way: it means consensus failure and the "
             "uncertainty-gated detector are picking up partly distinct phenomena, and neither "
             "result validates the other."),
        ]
    else:
        ov_lines.append(f"{TEST_A_FLAGS} not found -- overlap not computed.")
    with open(OUT_DIR / "overlap_with_flags.txt", "w") as f:
        f.write("\n".join(ov_lines) + "\n")
    for line in ov_lines:
        log("    " + line)

    # ---- Step 7: verdict, gated on architectural diversity ----
    diverse = len(families_present) >= 2
    primary_adj = adj_results.get(args.consensus_level)

    lines = ["=" * 78, "Consensus-Failure Analysis (kappa-free line of evidence)", "=" * 78, ""]
    lines += [
        "CONFOUND, STATED UP FRONT: models can agree on their errors simply because they",
        "share input features. Agreement magnitude alone therefore cannot support a",
        "label-error reading. The adjacency test (Step 5) is what discriminates: consensus",
        "failures concentrated at F1/F2 point at the labels; failures scattered across",
        "F0/F3/F4 point at shared-input correlation or genuinely hard cases.",
        "",
        f"MODELS INCLUDED ({len(model_names)}): {', '.join(model_names)}",
        f"architecture families present: {families_present}",
        "",
    ]
    if n_missing:
        lines.append(f"MODELS EXCLUDED ({n_missing}) -- no per-sample predictions found on disk. "
                      "Run the B3 persistence re-run to add them:")
        for fam, models in sorted(missing.items()):
            lines.append(f"  {fam} ({len(models)}): {', '.join(models)}")
    else:
        lines.append("MODELS EXCLUDED: none -- every registered model had per-sample predictions.")
    lines += [
        "",
        "Sample-level tests below use n = %d held-out samples and are adequately powered."
        % len(err_matrix),
        "Any fold-level statement (n=8 cohorts) is descriptive and underpowered.",
        "",
        "AGREEMENT (observed vs chance; raw agreement alone is not interpretable):",
        f"  mean Jaccard observed {pair_df['jaccard_observed'].mean():.3f} vs chance "
        f"{pair_df['jaccard_expected_chance'].mean():.3f} "
        f"(excess {pair_df['jaccard_excess'].mean():+.3f}); mean MCC {pair_df['mcc'].mean():.3f}",
        "  chance null = within-fold permutation (preserves each model's per-fold error rate,",
        "  so cohort difficulty differences are not counted as agreement).",
        "",
    ]
    for _, r in fam_df[fam_df["level"] == "variant_group"].iterrows():
        lines.append(f"  variant_group {r['relation']}: observed {r['mean_jaccard_observed']:.3f}, "
                     f"excess {r['mean_jaccard_excess']:+.3f}, z={r['mean_perm_z']:.1f} "
                     f"({int(r['n_pairs'])} pairs)")
    lines.append("")

    if primary_adj:
        lines += [
            f"ADJACENCY TEST at >={args.consensus_level:.0%} consensus "
            f"(n={primary_adj['n']} consensus failures):",
            f"  Fisher one-sided OR={primary_adj['odds_ratio']:.3f}, p={primary_adj['fisher_p']:.4g}",
            "",
        ]

    if not diverse:
        lines += [
            "CONVERGENT FAILURE: NOT ESTABLISHED -- insufficient architectural diversity.",
            "",
            f"Every included model is the same architecture family ({families_present[0]}): the same",
            "GAT encoder over the same 26-feature node tensor, differing only in adversary mode,",
            "graph corruption, or feature subset. Agreement among them is the MAXIMAL shared-input",
            "case -- exactly the alternative explanation this analysis exists to rule out. The",
            "adjacency result above is reported as a DESCRIPTIVE finding and must not be presented",
            "as a kappa-free confirmation of the ceiling.",
            "",
            "TO MAKE THIS CLAIM TESTABLE: baselines_b3.py has been patched to persist per-sample",
            "predictions. Run the persistence re-run (see that script's --only flag) to populate",
            "baselines/per_sample_predictions/, then re-run this analysis. The tabular stage alone",
            "adds the linear and tree_ensemble families and makes between-family agreement -- the",
            "informative quantity -- computable.",
        ]
    elif primary_adj and primary_adj["fisher_p"] < 0.05 and primary_adj["odds_ratio"] > 1:
        lines += [
            f"CONVERGENT FAILURE: SUPPORTED -- consensus failures enrich at F1/F2 "
            f"(OR={primary_adj['odds_ratio']:.2f}, p={primary_adj['fisher_p']:.3g}), consistent with",
            "label error independently of the kappa inversion.",
        ]
    else:
        lines += [
            "CONVERGENT FAILURE: NOT SUPPORTED -- consensus failures do not concentrate at the",
            "decision boundary; convergent failure is more consistent with shared-input",
            "correlation or genuinely hard cases.",
        ]

    lines += ["", "=" * 78]
    summary = "\n".join(lines)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(summary + "\n")
    log("")
    for line in lines:
        log(line)
    log(f"outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
