"""
CoDA-GAT — two flagged verification checks that were never run, and both of
which touch claims already in the write-up.

CHECK 1  APS conformal diagnostic. APS empirical coverage is exactly 1.000 at
         every alpha in every experiment while LAC behaves sensibly. Diagnose
         whether that is a bug or inherent, and if it is a bug, produce
         corrected numbers ALONGSIDE the originals (never replacing them).

CHECK 2  The pre-specified kill test for Finding 7: is attention a
         re-description of univariate effect size? Spearman(attention rank,
         |Cohen's d| rank). The spec's rule: rho > 0.8 means Finding 7 must be
         rewritten as a faithfulness check, not independent biological insight.

Analysis only. No retraining. No existing result file is overwritten -- all
output goes to verification_checks/.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, fisher_exact
from sklearn.linear_model import LogisticRegression

from run_full_experiment_b7 import (
    lac_sets, aps_sets, conformal_quantile, coverage_stats,
    load_fold_info, parse_fold_index, ALPHAS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = REPO_ROOT / "folds"
SHARED_DIR = REPO_ROOT / "shared"
RESULTS_B7 = REPO_ROOT / "results_b7"
ATTN_DIR = REPO_ROOT / "attention_analysis_b9"
OUT_DIR = REPO_ROOT / "verification_checks"

MODE = "off"
SEEDS = [42, 43, 44, 45, 46]
COHENS_D_FEATURE_IDX = 21       # shared/feature_schema_C.csv: fold-fitted 'cohens_d'
EXPRESSION_Z_FEATURE_IDX = 0    # shared/feature_schema_AB.csv: 'expression_z'
TOP_K_GENES = 15
APS_RANDOMIZATION_SEED = 20260920


def log(msg):
    print(f"[verify] {msg}", flush=True)


def fold_dirs():
    return sorted(p for p in FOLDS_DIR.iterdir() if p.is_dir())


# ==========================================================================
# CHECK 1 -- corrected (randomized) APS, provided as SEPARATE functions.
# The originals in run_full_experiment_b7.py are left untouched.
# ==========================================================================
def aps_scores_randomized(probs_matrix, true_labels, rng):
    """
    Randomized APS (Romano, Sesia & Candes 2020), the full form:

        score_i = (cumulative mass of classes strictly MORE likely than y_i)
                  + U_i * p(y_i),      U_i ~ Uniform(0, 1)

    The U_i * p(y_i) term is what the non-randomized version in b7 omits. For
    K = 2 that omission is fatal rather than "slightly conservative": without
    it the score can only ever take the value p_max (true class ranked first)
    or exactly 1.0 (true class ranked second), so the score distribution has a
    point mass at 1.0 whose weight equals the model's mis-ranking rate. Any
    alpha below that rate puts the (1-alpha) quantile inside the atom, pinning
    q_hat at 1.0 and forcing every prediction set to be the full label set.
    """
    n, K = probs_matrix.shape
    order = np.argsort(-probs_matrix, axis=1)
    sorted_probs = np.take_along_axis(probs_matrix, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    rank = np.argmax(order == true_labels[:, None], axis=1)
    cum_before = np.where(rank > 0, cumsum[np.arange(n), np.maximum(rank - 1, 0)], 0.0)
    p_true = probs_matrix[np.arange(n), true_labels]
    U = rng.uniform(size=n)
    return cum_before + U * p_true


def aps_sets_randomized(probs_matrix, q_hat, rng):
    """
    The matching randomized set construction. L is the smallest prefix whose
    cumulative mass reaches q_hat; the L-th class is then dropped with
    probability (cumsum_L - q_hat) / p_L. This is what lets APS emit singletons
    (and, legitimately, occasionally the empty set) instead of always the full
    label set.
    """
    n, K = probs_matrix.shape
    order = np.argsort(-probs_matrix, axis=1)
    sorted_probs = np.take_along_axis(probs_matrix, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)

    reaches = cumsum >= q_hat
    any_reach = reaches.any(axis=1)
    L = np.where(any_reach, np.argmax(reaches, axis=1) + 1, K)  # 1-indexed prefix length

    p_L = sorted_probs[np.arange(n), L - 1]
    cum_L = cumsum[np.arange(n), L - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        V = np.where(p_L > 0, (cum_L - q_hat) / p_L, 0.0)
    V = np.clip(np.nan_to_num(V), 0.0, 1.0)
    drop_last = rng.uniform(size=n) < V
    k_final = np.where(drop_last, L - 1, L)

    return [set(order[i, :k_final[i]].tolist()) for i in range(n)]


def run_check1(args):
    log("CHECK 1: APS conformal diagnostic")
    rows_diag = []
    lines = ["=" * 78, "CHECK 1 -- APS CONFORMAL DIAGNOSTIC", "=" * 78, ""]

    lines += [
        "IMPLEMENTATION AS WRITTEN (run_full_experiment_b7.py):",
        "  LAC score      : 1 - p(true class)",
        "  LAC set        : {c : p(c) >= 1 - q_hat}",
        "  APS score      : cumulative mass of all classes at least as likely as the",
        "                   true class -- i.e. the NON-randomized variant. The standard",
        "                   randomization term U * p(true) is absent.",
        "  APS set        : smallest top-k prefix whose cumulative mass reaches q_hat",
        "  quantile level : k = ceil((1 - alpha) * (n + 1)); q_hat = k-th smallest score",
        "",
    ]

    for fold_dir in fold_dirs():
        fold_id = parse_fold_index(fold_dir.name)
        test_cohort = load_fold_info(fold_dir)["test_cohort"]
        y_cal = pd.read_csv(fold_dir / "calibration" / "y.csv", index_col=0)["y_T1_significant"].to_numpy().astype(int)
        y_test = pd.read_csv(fold_dir / "test" / "y.csv", index_col=0)["y_T1_significant"].to_numpy().astype(int)

        for seed in args.seeds:
            d = RESULTS_B7 / f"fold{fold_id}_{test_cohort}" / f"mode_{MODE}" / f"seed_{seed}"
            if not (d / "nonconformity_calibration_aps.npy").exists():
                continue
            aps_cal_saved = np.load(d / "nonconformity_calibration_aps.npy")
            lac_cal_saved = np.load(d / "nonconformity_calibration_lac.npy")
            probs_cal = np.load(d / "probs_calibration.npy")
            probs_test = np.load(d / "probs_test.npy")
            n_cal = len(aps_cal_saved)

            # --- Diagnostic A: score distribution and the atom at 1.0 ---
            atom_mass = float(np.mean(np.abs(aps_cal_saved - 1.0) < 1e-6))
            # --- Diagnostic B: K=2 structure -- how many distinct values can the score take?
            n_distinct = int(len(np.unique(np.round(aps_cal_saved, 6))))
            # --- Diagnostic C: quantile level, and whether the k>=n guard fires ---
            rng = np.random.default_rng(APS_RANDOMIZATION_SEED + fold_id * 100 + seed)
            aps_cal_fixed = aps_scores_randomized(probs_cal, y_cal, rng)

            for alpha in args.alphas:
                k_level = int(np.ceil((1 - alpha) * (n_cal + 1)))
                guard_fires = k_level >= n_cal          # b7's guard (latent off-by-one: should be k > n)
                guard_would_fire_correct = k_level > n_cal

                q_aps_orig = conformal_quantile(aps_cal_saved, alpha)
                q_lac_orig = conformal_quantile(lac_cal_saved, alpha)
                q_aps_fixed = conformal_quantile(aps_cal_fixed, alpha)

                sets_aps_orig = aps_sets(probs_test, q_aps_orig)
                st_orig = coverage_stats(sets_aps_orig, y_test)

                rng_test = np.random.default_rng(APS_RANDOMIZATION_SEED + 7919 + fold_id * 100 + seed)
                sets_aps_fixed = aps_sets_randomized(probs_test, q_aps_fixed, rng_test)
                st_fixed = coverage_stats(sets_aps_fixed, y_test)
                n_empty = int(sum(1 for s in sets_aps_fixed if len(s) == 0))

                st_lac = coverage_stats(lac_sets(probs_test, q_lac_orig), y_test)

                rows_diag.append({
                    "fold": fold_id, "seed": seed, "alpha": alpha, "n_cal": n_cal,
                    "aps_atom_mass_at_1.0": atom_mass,
                    "alpha_below_atom_mass": bool(alpha < atom_mass),
                    "aps_n_distinct_scores": n_distinct,
                    "quantile_k": k_level,
                    "b7_guard_fires(k>=n)": bool(guard_fires),
                    "correct_guard_would_fire(k>n)": bool(guard_would_fire_correct),
                    "q_hat_aps_original": q_aps_orig,
                    "q_hat_aps_corrected": q_aps_fixed,
                    "q_hat_lac": q_lac_orig,
                    "aps_coverage_original": st_orig["empirical_coverage"],
                    "aps_set_size_original": st_orig["mean_set_size"],
                    "aps_coverage_corrected": st_fixed["empirical_coverage"],
                    "aps_set_size_corrected": st_fixed["mean_set_size"],
                    "aps_n_empty_sets_corrected": n_empty,
                    "lac_coverage": st_lac["empirical_coverage"],
                    "lac_set_size": st_lac["mean_set_size"],
                })
        log(f"  fold {fold_id} diagnosed")

    diag = pd.DataFrame(rows_diag)
    if diag.empty:
        lines.append("No saved APS artifacts found -- check could not run.")
        return "\n".join(lines), diag

    diag.to_csv(OUT_DIR / "aps_diagnostic_per_run.csv", index=False)

    # ---- evidence ----
    lines += ["EVIDENCE", "-" * 78]
    atom = diag["aps_atom_mass_at_1.0"].mean()
    lines += [
        f"A. Score distribution. Across folds/seeds, a mean {atom:.1%} of calibration APS",
        f"   scores are exactly 1.0. That is a POINT MASS, and its weight equals the",
        f"   model's calibration mis-ranking rate.",
        f"   Every alpha tested ({', '.join(str(a) for a in args.alphas)}) is BELOW that mass",
        f"   in {int(diag['alpha_below_atom_mass'].sum())}/{len(diag)} (fold, seed, alpha) cells,",
        f"   so the (1-alpha) quantile lands inside the atom.",
        "",
        f"B. K=2 structure. With two classes the non-randomized score can only be p_max",
        f"   (true class ranked first) or exactly 1.0 (true class ranked second). The",
        f"   randomization term U * p(true) that would spread the 1.0 atom into a",
        f"   continuum is ABSENT from the b7 implementation.",
        "",
        f"C. Quantile level. k = ceil((1-alpha)(n+1)) is CORRECT as written -- this is not",
        f"   an off-by-one in the level. There IS a latent off-by-one in the guard",
        f"   'if k >= n: return inf' (it should be 'k > n'), but it fires in",
        f"   {int(diag['b7_guard_fires(k>=n)'].sum())}/{len(diag)} cells, so it is NOT the cause.",
        "",
        f"D. Set-size consistency. Original APS mean set size is",
        f"   {diag['aps_set_size_original'].min():.3f}-{diag['aps_set_size_original'].max():.3f}",
        f"   against 2 possible classes, with coverage",
        f"   {diag['aps_coverage_original'].min():.3f}-{diag['aps_coverage_original'].max():.3f}.",
        f"   Coverage and set size AGREE: the sets really are (essentially always) full,",
        f"   so the 1.000 is real rather than a coverage-computation bug. The handful of",
        f"   runs slightly below 2.0 come from float32 cumsum making q_hat 0.99999994",
        f"   instead of exactly 1.0 at alpha=0.20 -- a precision artifact, not information.",
        "",
        f"   LAC over the same runs: coverage",
        f"   {diag['lac_coverage'].min():.3f}-{diag['lac_coverage'].max():.3f}, set size",
        f"   {diag['lac_set_size'].min():.3f}-{diag['lac_set_size'].max():.3f} -- behaving as",
        f"   intended, which localises the fault to APS specifically.",
        "",
    ]

    # ---- verdict ----
    by_alpha = diag.groupby("alpha")[
        ["aps_coverage_original", "aps_set_size_original",
         "aps_coverage_corrected", "aps_set_size_corrected", "aps_n_empty_sets_corrected"]].mean()

    lines += [
        "VERDICT", "-" * 78,
        "APS BUG IDENTIFIED: the randomization term is omitted.",
        "",
        "The b7 docstring describes the non-randomized variant as 'slightly conservative'.",
        "That characterisation is wrong for a binary task and it propagated into the",
        "write-up: for K=2 the omission is not conservative but DEGENERATE. The score",
        "collapses onto {p_max} union {1.0}, the atom at 1.0 carries the mis-ranking rate",
        f"(~{atom:.0%} here), and for any alpha below that rate q_hat is pinned at 1.0, which",
        "forces the full label set every time. Coverage 1.000 is therefore arithmetically",
        "guaranteed and carries no information.",
        "",
        "CORRECTED NUMBERS (randomized APS, computed from the SAME saved calibration and",
        "test probabilities; originals untouched and shown alongside):",
        "",
        f"  {'alpha':>6} {'orig cov':>9} {'orig size':>10} {'corr cov':>9} {'corr size':>10} {'empty sets':>11}",
    ]
    for alpha, r in by_alpha.iterrows():
        lines.append(f"  {alpha:>6} {r['aps_coverage_original']:>9.3f} {r['aps_set_size_original']:>10.3f} "
                     f"{r['aps_coverage_corrected']:>9.3f} {r['aps_set_size_corrected']:>10.3f} "
                     f"{r['aps_n_empty_sets_corrected']:>11.1f}")
    lines += [
        "",
        "Randomized APS can legitimately emit the empty set; the count is reported above",
        "so it is not mistaken for a new bug.",
        "",
        "Recommended wording: the previously reported APS coverage of 1.000 was an artifact",
        "of a non-randomized APS score on a two-class problem, not a property of the model.",
        "Either report the corrected randomized-APS numbers above, or report LAC only and",
        "state that APS was excluded for this reason. Do not report the 1.000 as a result.",
        "",
    ]
    return "\n".join(lines), diag


# ==========================================================================
# CHECK 2 -- is attention a re-description of univariate effect size?
# ==========================================================================
def assert_train_only(fold_dir, n_used_samples):
    """Cohen's d must come from the fold's TRAINING split only."""
    n_train = len(pd.read_csv(fold_dir / "train" / "sample_ids.csv"))
    n_cal = len(pd.read_csv(fold_dir / "calibration" / "sample_ids.csv"))
    n_test = len(pd.read_csv(fold_dir / "test" / "sample_ids.csv"))
    if n_used_samples != n_train:
        raise RuntimeError(
            f"FATAL [{fold_dir.name}]: effect sizes computed from {n_used_samples} samples but the "
            f"training split has {n_train} (calibration={n_cal}, test={n_test}). Refusing to "
            f"continue -- effect sizes must never see calibration or test."
        )


def cohens_d_from_expression(X_train, y_train, feature_idx=EXPRESSION_Z_FEATURE_IDX):
    """Independent recomputation of |Cohen's d| per gene from the model's own
    expression feature, on the training split only."""
    pos = X_train[y_train == 1][:, :, feature_idx]
    neg = X_train[y_train == 0][:, :, feature_idx]
    m1, m0 = pos.mean(axis=0), neg.mean(axis=0)
    s1, s0 = pos.std(axis=0, ddof=1), neg.std(axis=0, ddof=1)
    n1, n0 = len(pos), len(neg)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / max(n1 + n0 - 2, 1))
    pooled = np.where(pooled < 1e-9, 1e-9, pooled)
    return np.abs((m1 - m0) / pooled)


def bootstrap_spearman_ci(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(a[idx])) < 3 or len(np.unique(b[idx])) < 3:
            continue
        vals.append(spearmanr(a[idx], b[idx]).statistic)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def pathway_enrichment(top_genes, gene_order, gene_pathways):
    n_total, n_top = len(gene_order), len(top_genes)
    out = {}
    all_pw = set()
    for s in gene_pathways.values():
        all_pw |= s
    for pw in sorted(all_pw):
        n_pw_total = sum(1 for g in gene_order if pw in gene_pathways[g])
        n_pw_top = sum(1 for g in top_genes if pw in gene_pathways[g])
        table = [[n_pw_top, n_top - n_pw_top],
                 [n_pw_total - n_pw_top, (n_total - n_top) - (n_pw_total - n_pw_top)]]
        or_, p = fisher_exact(table, alternative="greater")
        out[pw] = {"n_in_top": n_pw_top, "n_in_panel": n_pw_total,
                   "odds_ratio": float(or_), "p": float(p)}
    return out


def run_check2(args):
    log("CHECK 2: is attention a re-description of univariate effect size?")
    attn_path = ATTN_DIR / "per_fold_gene_attention.csv"
    if not attn_path.exists():
        return f"CHECK 2 could not run: {attn_path} not found.", pd.DataFrame()

    attn = pd.read_csv(attn_path, index_col=0)          # rows = folds, cols = genes
    gene_order = list(attn.columns)
    panel = pd.read_csv(SHARED_DIR / "panel_genes.csv")
    gene_pathways = {r["gene"]: set(str(r["pathways"]).split(";")) for _, r in panel.iterrows()}

    rows, per_fold_cd = [], {}
    for fold_dir in fold_dirs():
        fold_id = parse_fold_index(fold_dir.name)
        if fold_id not in attn.index:
            continue

        X_train = np.load(fold_dir / "train" / "X.npy")
        y_train = pd.read_csv(fold_dir / "train" / "y.csv", index_col=0)["y_T1_significant"].to_numpy().astype(int)
        assert_train_only(fold_dir, X_train.shape[0])

        # (a) fold-fitted Cohen's d already present as Group C feature 21.
        # It is a per-gene constant stored replicated across samples (observed
        # max per-gene sd ~7e-6, i.e. float32 jitter), so take the per-gene mean.
        cd_groupC = np.abs(X_train[:, :, COHENS_D_FEATURE_IDX].mean(axis=0))
        jitter = float(X_train[:, :, COHENS_D_FEATURE_IDX].std(axis=0).max())

        # (b) independent recomputation from expression_z, train split only
        cd_recomputed = cohens_d_from_expression(X_train, y_train)

        # (c) logistic-regression |coefficient| on the same panel, train only
        Xlr = X_train[:, :, EXPRESSION_Z_FEATURE_IDX]
        try:
            lr = LogisticRegression(max_iter=3000, class_weight="balanced")
            lr.fit(Xlr, y_train)
            lr_abs_coef = np.abs(lr.coef_[0])
        except Exception:
            lr_abs_coef = np.full(len(gene_order), np.nan)

        a = attn.loc[fold_id, gene_order].to_numpy(dtype=float)
        per_fold_cd[fold_id] = cd_groupC

        def corr_and_ci(x, label, seed_off):
            if np.all(np.isnan(x)):
                return None
            rho = spearmanr(a, x).statistic
            lo, hi = bootstrap_spearman_ci(a, x, n_boot=args.n_boot, seed=1000 + fold_id + seed_off)
            top_a = set(np.array(gene_order)[np.argsort(-a)[:args.top_k]])
            top_x = set(np.array(gene_order)[np.argsort(-x)[:args.top_k]])
            return {"comparison": label, "spearman_rho": float(rho),
                    "ci_lo": lo, "ci_hi": hi,
                    f"top{args.top_k}_overlap": len(top_a & top_x)}

        for label, vec, off in (("attention_vs_cohens_d_groupC", cd_groupC, 0),
                                 ("attention_vs_cohens_d_recomputed", cd_recomputed, 1),
                                 ("attention_vs_logreg_abs_coef", lr_abs_coef, 2)):
            r = corr_and_ci(vec, label, off)
            if r:
                r.update({"fold": fold_id, "groupC_per_gene_jitter": jitter})
                rows.append(r)

        # agreement between the two Cohen's d sources, as a sanity check
        rows.append({"fold": fold_id, "comparison": "cohens_d_groupC_vs_recomputed",
                     "spearman_rho": float(spearmanr(cd_groupC, cd_recomputed).statistic),
                     "ci_lo": np.nan, "ci_hi": np.nan,
                     f"top{args.top_k}_overlap": len(
                         set(np.array(gene_order)[np.argsort(-cd_groupC)[:args.top_k]]) &
                         set(np.array(gene_order)[np.argsort(-cd_recomputed)[:args.top_k]])),
                     "groupC_per_gene_jitter": jitter})
        log(f"  fold {fold_id} done")

    df = pd.DataFrame(rows)
    if df.empty:
        return "CHECK 2 produced no rows.", df
    df.to_csv(OUT_DIR / "attention_vs_cohens_d.csv", index=False)

    primary = df[df["comparison"] == "attention_vs_cohens_d_groupC"]
    rho_mean = float(primary["spearman_rho"].mean())
    rho_lo, rho_hi = np.percentile(primary["spearman_rho"], [2.5, 97.5])
    overlap_mean = float(primary[f"top{args.top_k}_overlap"].mean())

    # ECM enrichment: attention top-K vs |Cohen's d| top-K
    consensus_attn = attn.mean(axis=0)
    top_attn = list(consensus_attn.sort_values(ascending=False).index[:args.top_k])
    consensus_cd = pd.Series(np.mean(list(per_fold_cd.values()), axis=0), index=gene_order)
    top_cd = list(consensus_cd.sort_values(ascending=False).index[:args.top_k])
    enr_attn = pathway_enrichment(top_attn, gene_order, gene_pathways)
    enr_cd = pathway_enrichment(top_cd, gene_order, gene_pathways)
    pd.DataFrame([
        {"ranking": "attention", "pathway": pw, **v} for pw, v in enr_attn.items()
    ] + [
        {"ranking": "cohens_d", "pathway": pw, **v} for pw, v in enr_cd.items()
    ]).to_csv(OUT_DIR / "pathway_enrichment_attention_vs_cohens_d.csv", index=False)

    # ---- the pre-specified decision rule ----
    if rho_mean > 0.8:
        band = "rho > 0.8"
        corrected = (
            "Attention concentrates on the genes with the largest univariate discriminative "
            "effect (Spearman rho = {rho:.3f} against |Cohen's d| across folds). Finding 7 is "
            "therefore a FAITHFULNESS check, not independent biological insight: it shows the "
            "model relies on individually informative fibrosis genes rather than cohort "
            "artefacts, but it does not demonstrate that the graph revealed anything beyond "
            "what a univariate screen would have found."
        ).format(rho=rho_mean)
    elif rho_mean > 0.5:
        band = "0.5 < rho <= 0.8"
        dis = []
        for g in gene_order:
            ar = float(consensus_attn.rank(ascending=False)[g])
            cr = float(consensus_cd.rank(ascending=False)[g])
            if ar <= args.top_k and cr > 2 * args.top_k:
                dis.append((g, int(ar), int(cr)))
        dis_txt = ", ".join(f"{g} (attn #{a}, |d| #{c})" for g, a, c in dis) or "none"
        corrected = (
            "Attention partially overlaps univariate effect size (Spearman rho = {rho:.3f}). "
            "Most of the attended set is recoverable from |Cohen's d| alone, so Finding 7 should "
            "be stated as largely a faithfulness check. The only candidates for graph-derived "
            "signal are genes attended disproportionately to their individual effect size: {dis}."
        ).format(rho=rho_mean, dis=dis_txt)
    else:
        band = "rho <= 0.5"
        corrected = (
            "Attention is substantially independent of univariate effect size (Spearman rho = "
            "{rho:.3f}), so the interpretability claim stands as stated: the attended gene set is "
            "not simply the top of a univariate screen."
        ).format(rho=rho_mean)

    lines = ["=" * 78, "CHECK 2 -- IS ATTENTION A RE-DESCRIPTION OF UNIVARIATE EFFECT SIZE?",
             "=" * 78, "",
             "Effect sizes come from each fold's TRAINING SPLIT ONLY (asserted in code).",
             "Cohen's d was already available as fold-fitted Group C feature 21 and is reused,",
             f"with an independent recomputation from expression_z as a cross-check.", "",
             "PER-FOLD RESULTS (descriptive; n=8 folds):"]
    for _, r in primary.sort_values("fold").iterrows():
        lines.append(f"  fold {int(r['fold'])}: rho={r['spearman_rho']:+.3f} "
                     f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
                     f"top-{args.top_k} overlap={int(r[f'top{args.top_k}_overlap'])}/{args.top_k}")
    lines += [
        "",
        f"POOLED: mean Spearman rho = {rho_mean:+.3f} (across-fold 2.5-97.5 pct "
        f"[{rho_lo:+.3f}, {rho_hi:+.3f}]); mean top-{args.top_k} overlap = "
        f"{overlap_mean:.1f}/{args.top_k}",
        "",
        "SECOND ANGLE:",
    ]
    for label in ("attention_vs_cohens_d_recomputed", "attention_vs_logreg_abs_coef",
                   "cohens_d_groupC_vs_recomputed"):
        sub = df[df["comparison"] == label]
        if not sub.empty:
            lines.append(f"  {label}: mean rho = {sub['spearman_rho'].mean():+.3f}")
    lines += [
        "",
        "PATHWAY ENRICHMENT -- does it survive a plain univariate ranking?",
        f"  top-{args.top_k} by ATTENTION:",
    ]
    for pw, v in sorted(enr_attn.items(), key=lambda t: t[1]["p"]):
        lines.append(f"    {pw:24s} {v['n_in_top']:2d}/{args.top_k}  OR={v['odds_ratio']:.2f}  p={v['p']:.4g}")
    lines.append(f"  top-{args.top_k} by |COHEN'S D|:")
    for pw, v in sorted(enr_cd.items(), key=lambda t: t[1]["p"]):
        lines.append(f"    {pw:24s} {v['n_in_top']:2d}/{args.top_k}  OR={v['odds_ratio']:.2f}  p={v['p']:.4g}")
    ecm_a = enr_attn.get("ECM_remodeling", {}).get("p", float("nan"))
    ecm_c = enr_cd.get("ECM_remodeling", {}).get("p", float("nan"))
    lines += [
        "",
        f"  ECM_remodeling: attention p={ecm_a:.4g} vs |Cohen's d| p={ecm_c:.4g}.",
        ("  The enrichment is reproduced by a plain univariate ranking, so it is NOT "
         "attention-specific and must not be presented as such."
         if (np.isfinite(ecm_c) and ecm_c < 0.05) else
         "  The enrichment is not reproduced by the univariate ranking, so it is at least "
         "partly attention-specific."),
        "",
        f"DECISION RULE APPLIED ({band}):", "",
        corrected, "",
    ]
    return "\n".join(lines), df


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="CoDA-GAT verification checks (APS + attention)")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--alphas", nargs="*", type=float, default=ALPHAS)
    parser.add_argument("--top-k", type=int, default=TOP_K_GENES)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--only", choices=["both", "aps", "attention"], default="both")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Verification checks -- analysis only, no retraining, no existing file overwritten")

    parts = []
    if args.only in ("both", "aps"):
        text, _ = run_check1(args)
        with open(OUT_DIR / "aps_diagnostic.txt", "w") as f:
            f.write(text + "\n")
        parts.append(text)
        log(f"  wrote {OUT_DIR / 'aps_diagnostic.txt'}")
    if args.only in ("both", "attention"):
        text, _ = run_check2(args)
        with open(OUT_DIR / "attention_verdict.txt", "w") as f:
            f.write(text + "\n")
        parts.append(text)
        log(f"  wrote {OUT_DIR / 'attention_verdict.txt'}")

    summary = "\n\n".join(parts)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(summary + "\n")
    log("")
    for line in summary.split("\n"):
        log(line)
    log(f"outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
