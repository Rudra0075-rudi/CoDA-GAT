"""
Issue 2: shared-error analysis on consistent decisions.

1. Runs the corrected scripts/consensus_failures.py into outputs/consensus_failures/.
2. Rebuilds the HISTORICAL error matrix independently (default-precision CSV
   parse, p_test >= threshold) and checks it reproduces
   consensus_failures/error_matrix.csv exactly -- so the comparison below is
   between two known decision rules, not between two unknown runs.
3. Lists every changed decision and every record whose consensus membership
   changes, and checks that pairwise rows not involving a changed model are
   reproduced exactly (the permutation RNG stream does not depend on values).
"""

import sys

import numpy as np
import pandas as pd

import guard

REPO, OUT = guard.REPO, guard.OUT / "consensus_failures"
HIST = REPO / "consensus_failures"


def run_corrected():
    import consensus_failures as cf
    argv = sys.argv
    sys.argv = ["consensus_failures.py", "--out-dir", str(OUT)]
    try:
        cf.main()
    finally:
        sys.argv = argv
    return cf


def historical_error_matrix(cf):
    """The pre-fix rule, rebuilt here only to reproduce and diff the old outputs."""
    index_df = cf.fold_test_index()
    labels = pd.read_parquet(REPO / "shared" / "labels.parquet")
    y = labels.loc[index_df["sample_id"], "y_T1_significant"].to_numpy().astype(int)
    em = pd.DataFrame({"sample_id": index_df["sample_id"], "fold": index_df["fold"],
                       "stage": labels.loc[index_df["sample_id"], "stage"].to_numpy(), "y_true": y})
    for name, spec in cf.MODEL_REGISTRY.items():
        if spec["source"] == "baselines_csv":
            frames = []
            for (fold_id, cohort), grp in index_df.groupby(["fold", "test_cohort"], sort=True):
                path = cf.BASELINES_PER_SAMPLE / spec["key"] / f"fold{fold_id}_{cohort}.csv"
                if not path.exists():
                    break
                frames.append(pd.read_csv(path).set_index("sample_id").loc[grp["sample_id"]].reset_index())
            else:
                d = pd.concat(frames, ignore_index=True)
                em[name] = ((d["p_test"] >= d["threshold"]).astype(int).to_numpy() != y)
        else:
            loaded = cf.load_model_probs(name, spec, index_df, cf.SEEDS)
            if loaded is not None:
                em[name] = (loaded[2] != y)
    return em


def main():
    cf = run_corrected()
    models = [c for c in pd.read_csv(HIST / "error_matrix.csv", nrows=0).columns
              if c not in {"sample_id", "fold", "stage", "y_true"}]
    hist_saved = pd.read_csv(HIST / "error_matrix.csv")
    hist_rebuilt = historical_error_matrix(cf)
    corr = pd.read_csv(OUT / "error_matrix.csv")
    labels = pd.read_parquet(REPO / "shared" / "labels.parquet")

    lines = []

    def say(x=""):
        lines.append(x)
        print(f"[consensus-diff] {x}", flush=True)

    reproduced = bool((hist_rebuilt[models].to_numpy() == hist_saved[models].to_numpy()).all()
                      and (hist_rebuilt["sample_id"] == hist_saved["sample_id"]).all())
    say(f"historical error matrix rebuilt independently (default CSV parse): "
        f"{'reproduces consensus_failures/error_matrix.csv exactly' if reproduced else 'DOES NOT reproduce'}")
    assert (corr["sample_id"] == hist_saved["sample_id"]).all()
    assert list(c for c in corr.columns if c in models) == models

    # changed decisions
    diff = hist_saved[models].to_numpy() != corr[models].to_numpy()
    rows = []
    for i, j in zip(*np.nonzero(diff)):
        sid = corr.loc[i, "sample_id"]
        rows.append({"sample_id": sid, "fold": int(corr.loc[i, "fold"]),
                     "cohort": labels.loc[sid, "cohort"], "stage_stored": labels.loc[sid, "stage"],
                     "resolution": labels.loc[sid, "resolution"], "y_true": int(corr.loc[i, "y_true"]),
                     "model": models[j], "error_historical": bool(hist_saved.loc[i, models[j]]),
                     "error_corrected": bool(corr.loc[i, models[j]])})
    changed = pd.DataFrame(rows)
    changed.to_csv(OUT / "changed_decisions.csv", index=False)
    say(f"changed model decisions: {len(changed)}")
    for _, r in changed.iterrows():
        say(f"  {r['sample_id']} ({r['cohort']}, {r['resolution']}, y={r['y_true']}) {r['model']}: "
            f"error {r['error_historical']} -> {r['error_corrected']}")

    n = len(models)
    k_hist, k_corr = hist_saved[models].sum(axis=1), corr[models].sum(axis=1)
    count_rows, member_rows = [], []
    for lvl in (0.80, 0.90, 1.00):
        need = int(np.ceil(lvl * n))
        h, c = (k_hist >= need).to_numpy(), (k_corr >= need).to_numpy()
        count_rows.append({"level": lvl, "models_wrong_at_least": need, "n_models": n,
                           "historical": int(h.sum()), "corrected": int(c.sum()),
                           "entered": int((c & ~h).sum()), "left": int((h & ~c).sum())})
        for i in np.nonzero(h != c)[0]:
            sid = corr.loc[i, "sample_id"]
            member_rows.append({"level": lvl, "sample_id": sid, "cohort": labels.loc[sid, "cohort"],
                                "resolution": labels.loc[sid, "resolution"], "stage_stored": labels.loc[sid, "stage"],
                                "y_true": int(corr.loc[i, "y_true"]), "n_wrong_historical": int(k_hist[i]),
                                "n_wrong_corrected": int(k_corr[i]),
                                "change": "entered" if c[i] else "left"})
    counts = pd.DataFrame(count_rows)
    counts.to_csv(OUT / "consensus_counts_historical_vs_corrected.csv", index=False)
    pd.DataFrame(member_rows).to_csv(OUT / "consensus_membership_changes.csv", index=False)
    say("")
    say("consensus-failure counts (historical -> corrected):")
    for r in count_rows:
        say(f"  >= {r['models_wrong_at_least']}/{n} wrong: {r['historical']} -> {r['corrected']} "
            f"(+{r['entered']} entered, -{r['left']} left)")
    for r in member_rows:
        say(f"    level {r['level']:.0%}: {r['sample_id']} ({r['cohort']}, {r['resolution']}) "
            f"{r['n_wrong_historical']} -> {r['n_wrong_corrected']} models wrong: {r['change']}")

    # pairwise rows untouched by the changed model must be reproduced exactly
    changed_models = set(changed["model"]) if len(changed) else set()
    ph, pc = pd.read_csv(HIST / "pairwise_agreement.csv"), pd.read_csv(OUT / "pairwise_agreement.csv")
    key = ["model_a", "model_b"]
    m = ph.merge(pc, on=key, suffixes=("_h", "_c"))
    untouched = ~(m["model_a"].isin(changed_models) | m["model_b"].isin(changed_models))
    cols = ["jaccard_observed", "jaccard_expected_chance", "jaccard_perm_null_mean", "mcc"]
    max_dev = max(float((m.loc[untouched, f"{c}_h"] - m.loc[untouched, f"{c}_c"]).abs().max()) for c in cols)
    say("")
    say(f"pairwise rows not involving {sorted(changed_models)}: {int(untouched.sum())}/{len(m)}, "
        f"max |historical - corrected| over {cols} = {max_dev:.3g}")
    touched = m.loc[~untouched]
    say(f"pairs involving the changed model(s): {len(touched)}; max |change in observed Jaccard| = "
        f"{(touched['jaccard_observed_h'] - touched['jaccard_observed_c']).abs().max():.4f}")

    (OUT / "historical_vs_corrected.txt").write_text("\n".join(lines) + "\n")
    return {"reproduced_historical": reproduced, "n_changed_decisions": len(changed),
            "counts": count_rows, "untouched_pairs_max_dev": max_dev}

