"""
Issue 3: grouped stage labels.

Recomputes every F1/F2 adjacency enrichment for which per-patient flags were
SAVED, on the exact-stage subset, from existing outputs only:
  * Test A detector flags          test_a_adjacency/per_sample_flags.csv
  * flagging v2 variants (12)      flagging_v2/per_patient_flags.csv
  * flagging v3 confident learning flagging_v3/per_patient_{real,empty}_graph.csv
  * consensus failures (3 levels)  outputs/consensus_failures/error_matrix.csv (corrected)

For each flag set it reports
  exact_only         primary: exact-stage records only (800 of 1064)
  as_stored          the historical coding (grouped F0-F1 counted as F0, F3-F4 as F3)
  grouped_F0F1_as_F1 the opposite extreme for the unknown records
The two codings of the grouped F0-F1 records are extremes of an unknown
assignment, not confidence bounds. Grouped F3-F4 records are distant from the
F1/F2 boundary under any assignment.

Also: exact-stage subset composition per cohort, and planted-error
eligibility (flagging_v4 plants only at recorded exact F1 with y=0 and
exact F2 with y=1).
"""

import numpy as np
import pandas as pd

import guard
from stage_labels import enrichment_test, ADJACENT_STAGES, ALL_STAGES

REPO, OUT = guard.REPO, guard.OUT / "stage_enrichment"


# ---- verbatim from scripts/flagging_v2.py (pure numpy; that module imports torch) ----
N_BOOT = 2000
BOOT_SEED = 20260922
HALDANE = 0.5


def or_counts(flags, adjacent):
    a = int((flags & adjacent).sum())         # flagged, F1/F2
    b = int((flags & ~adjacent).sum())        # flagged, F0/F3/F4
    c = int((~flags & adjacent).sum())
    d = int((~flags & ~adjacent).sum())
    return a, b, c, d


def log_or_haldane(flags, adjacent):
    a, b, c, d = or_counts(flags, adjacent)
    a, b, c, d = a + HALDANE, b + HALDANE, c + HALDANE, d + HALDANE
    return float(np.log((a * d) / (b * c)))


def boot_indices(folds, n_boot, seed):
    rng = np.random.default_rng(seed)
    by_fold = [np.where(folds == f)[0] for f in np.unique(folds)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(ix, size=ix.size, replace=True) for ix in by_fold])


def paired_log_or_ci(flags_a, flags_b, adjacent, folds, n_boot=N_BOOT, seed=BOOT_SEED):
    diffs = []
    for idx in boot_indices(folds, n_boot, seed):
        a = adjacent[idx]
        diffs.append(log_or_haldane(flags_a[idx], a) - log_or_haldane(flags_b[idx], a))
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return np.nan, np.nan, np.nan
    return (float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
# -------------------------------------------------------------------------------------


def graph_comparison(lab):
    """flagging_v2's real/shuffled-vs-empty paired contrast at matched counts.
    as_stored must reproduce flagging_v2/graph_comparison.csv; exact_only keeps
    the ORIGINAL flags (ranked over all 1064 records) and restricts to exact
    records, so the arms no longer have equal flag counts within the subset."""
    v2 = pd.read_csv(REPO / "flagging_v2" / "per_patient_flags.csv")
    v2["resolution"] = lab.loc[v2["sample_id"], "resolution"].to_numpy()
    hist = pd.read_csv(REPO / "flagging_v2" / "graph_comparison.csv").set_index("variant")
    as_bool = lambda c: v2[c].astype(str).str.lower().eq("true").to_numpy()  # noqa: E731
    rows = []
    for n_target in (52, 192):
        fe = as_bool(f"V4 empty_graph @{n_target}")
        for cond in ("real_graph", "shuffled_graph"):
            fa = as_bool(f"V4 {cond} @{n_target}")
            for coding in ("as_stored", "exact_only"):
                keep = (v2["resolution"] == "exact").to_numpy() if coding == "exact_only" else np.ones(len(v2), bool)
                adj = v2.loc[keep, "stage"].isin(ADJACENT_STAGES).to_numpy()
                mean, lo, hi = paired_log_or_ci(fa[keep], fe[keep], adj, v2.loc[keep, "fold"].to_numpy())
                verdict = ("BETTER than no graph" if lo > 0 else "WORSE than no graph" if hi < 0
                           else "NO DIFFERENCE DETECTED")
                h = hist.loc[f"V4 {cond} @{n_target}"]
                rows.append({"condition": cond, "target_n": n_target, "coding": coding,
                             "n_records": int(keep.sum()), "n_flagged_arm": int(fa[keep].sum()),
                             "n_flagged_empty": int(fe[keep].sum()),
                             "paired_diff_mean": mean, "paired_diff_lo": lo, "paired_diff_hi": hi,
                             "verdict": verdict, "historical_verdict": h["verdict"],
                             "reproduces_historical": bool(coding != "as_stored" or (
                                 np.isclose(mean, h["paired_diff_mean"]) and np.isclose(lo, h["paired_diff_lo"])
                                 and np.isclose(hi, h["paired_diff_hi"])))})
    return pd.DataFrame(rows)


def labels():
    return pd.read_parquet(REPO / "shared" / "labels.parquet")


def two_stage_share(df, flags):
    sub = df.loc[flags & (df["resolution"] == "exact").to_numpy(), "stage"]
    return float(sub.isin(["F0", "F3"]).mean()) if len(sub) else np.nan


COHORT = pd.read_parquet(REPO / "shared" / "labels.parquet")["cohort"]


def score(df, flags, name, source):
    flags = np.asarray(flags, dtype=bool)
    rows = []
    codings = {
        "exact_only": (df, True),
        "as_stored": (df, False),
        "grouped_F0F1_as_F1": (df.assign(stage=np.where(df["resolution"] == "grouped_F0_F1", "F1", df["stage"])), False),
        # descriptive sensitivity only: GSE213621's exact records are all F2, so every one
        # of its exact-stage flags is "adjacent" by construction
        "exact_excluding_GSE213621": (df.assign(resolution=np.where(
            df["sample_id"].map(COHORT).eq("GSE213621"), "excluded_cohort", df["resolution"])), True),
    }
    for coding, (d, exact_only) in codings.items():
        t5, (chi2, p_chi2), t2, (orr, p) = enrichment_test(d, flags, exact_only=exact_only)
        keep = (d["resolution"] == "exact").to_numpy() if exact_only else np.ones(len(d), bool)
        row = {"source": source, "flag_set": name, "coding": coding,
               "n_records": int(keep.sum()), "n_flagged": int(flags[keep].sum()),
               "n_flagged_all_records": int(flags.sum()),
               "n_flag_adjacent": int(t2[0, 0]), "n_flag_distant": int(t2[0, 1]),
               "n_unflag_adjacent": int(t2[1, 0]), "n_unflag_distant": int(t2[1, 1]),
               "odds_ratio": orr, "fisher_p_onesided": p, "chi2": chi2, "chi2_p": p_chi2}
        row.update({f"flag_{s}": int(t5.loc["flagged", s]) for s in ALL_STAGES})
        if coding == "exact_only":
            row["two_stage_share_exact"] = two_stage_share(df, flags)
            for res in ("grouped_F0_F1", "grouped_F3_F4"):
                m = (df["resolution"] == res).to_numpy()
                row[f"flagged_{res}"] = int((flags & m).sum())
        rows.append(row)
    return rows


def flag_sets(lab):
    """(source, name, frame with sample_id/stage/resolution/fold, flags) for every saved flag set."""
    def attach(d):
        d = d.copy()
        d["stage"] = lab.loc[d["sample_id"], "stage"].to_numpy()
        d["resolution"] = lab.loc[d["sample_id"], "resolution"].to_numpy()
        return d

    def as_bool(s):
        return s.astype(str).str.strip().str.lower().isin(["true", "1"]).to_numpy()

    out = []
    ta = attach(pd.read_csv(REPO / "test_a_adjacency" / "per_sample_flags.csv"))
    out.append(("test_a_adjacency", "Test A flags (primary)", ta, as_bool(ta["flagged"])))

    v2 = attach(pd.read_csv(REPO / "flagging_v2" / "per_patient_flags.csv"))
    meta = {"fold", "test_cohort", "sample_id", "stage", "resolution", "y_true", "p_f2plus",
            "uncertainty", "disagreement"}
    for col in [c for c in v2.columns if c not in meta]:
        out.append(("flagging_v2", col, v2, as_bool(v2[col])))

    for cond in ("real_graph", "empty_graph"):
        v3 = attach(pd.read_csv(REPO / "flagging_v3" / f"per_patient_{cond}.csv"))
        out.append(("flagging_v3", f"CL pruned [{cond}]", v3, as_bool(v3["cl_pruned"])))

    em = pd.read_csv(guard.OUT / "consensus_failures" / "error_matrix.csv")
    models = [c for c in em.columns if c not in {"sample_id", "fold", "stage", "resolution", "y_true", "n_models_wrong"}]
    k = em[models].apply(lambda s: as_bool(s)).sum(axis=1).to_numpy()
    for need in (19, 21, 23):
        out.append(("consensus_failures (corrected decisions)", f">= {need}/{len(models)} models wrong",
                    em, k >= need))
    return out


def check_against_historical(table):
    """The as_stored coding must reproduce the historical summaries exactly."""
    checks = []
    v2 = pd.read_csv(REPO / "flagging_v2" / "variant_summary.csv").set_index("variant")
    v3 = pd.read_csv(REPO / "flagging_v3" / "variant_summary.csv").set_index("variant")
    hist = {("flagging_v2", v): (r["or_f1f2"], r["or_p_onesided"], r["n_flagged"]) for v, r in v2.iterrows()}
    hist.update({("flagging_v3", v.replace("CL pruned", "CL pruned")): (r["or_f1f2"], r["or_p_onesided"], r["n_flagged"])
                 for v, r in v3.iterrows() if v.startswith("CL pruned")})
    ta = pd.read_csv(REPO / "test_a_adjacency" / "adjacency_2x2_table.csv", index_col=0)
    for (src, name), (orr, p, n) in hist.items():
        r = table[(table["source"] == src) & (table["flag_set"] == name) & (table["coding"] == "as_stored")]
        if r.empty:
            checks.append({"source": src, "flag_set": name, "match": False, "note": "not recomputed"})
            continue
        r = r.iloc[0]
        checks.append({"source": src, "flag_set": name, "historical_or": orr, "recomputed_or": r["odds_ratio"],
                       "historical_p": p, "recomputed_p": r["fisher_p_onesided"],
                       "match": bool(np.isclose(orr, r["odds_ratio"], rtol=1e-9, equal_nan=True)
                                     and np.isclose(p, r["fisher_p_onesided"], rtol=1e-9)
                                     and int(n) == int(r["n_flagged"]))})
    r = table[(table["source"] == "test_a_adjacency") & (table["coding"] == "as_stored")].iloc[0]
    checks.append({"source": "test_a_adjacency", "flag_set": "Test A flags (primary)",
                   "match": bool(int(ta.loc["flagged", "adjacent_F1F2"]) == r["n_flag_adjacent"]
                                 and int(ta.loc["flagged", "distant_F0F3F4"]) == r["n_flag_distant"]
                                 and int(ta.loc["unflagged", "adjacent_F1F2"]) == r["n_unflag_adjacent"])})
    return pd.DataFrame(checks)


def composition(lab):
    t = lab[lab["tier"] == "T1"]
    rows = []
    for cohort, g in t.groupby("cohort"):
        ex = g[g["resolution"] == "exact"]
        y_all, y_ex = g["y_T1_significant"].mean(), ex["y_T1_significant"].mean() if len(ex) else np.nan
        row = {"cohort": cohort, "n_all": len(g), "n_exact": len(ex), "n_grouped_F0_F1": int((g["resolution"] == "grouped_F0_F1").sum()),
               "n_grouped_F3_F4": int((g["resolution"] == "grouped_F3_F4").sum()),
               "share_of_all_1064": len(g) / len(t), "share_of_exact_800": len(ex) / int((t["resolution"] == "exact").sum()),
               "f2plus_rate_all": y_all, "f2plus_rate_exact": y_ex}
        row.update({f"exact_{s}": int((ex["stage"] == s).sum()) for s in ALL_STAGES})
        rows.append(row)
    return pd.DataFrame(rows)


def planted_eligibility(lab):
    """flagging_v4.plant_labels pools: recorded stage F1 with y=0 (flip up) and
    recorded stage F2 with y=1 (flip down), over the 1064 test records, read from
    each fold's test/y.csv exactly as flagging_v4.split_labels does."""
    frames = []
    for fold_dir in sorted(p for p in (REPO / "folds").iterdir() if p.is_dir()):
        y = pd.read_csv(fold_dir / "test" / "y.csv", index_col=0)
        frames.append(y.assign(fold=fold_dir.name))
    y = pd.concat(frames)
    assert (y["stage"] == lab.loc[y.index, "stage"]).all() and (y["resolution"] == lab.loc[y.index, "resolution"]).all()
    up = (y["stage"] == "F1") & (y["y_T1_significant"] == 0)
    down = (y["stage"] == "F2") & (y["y_T1_significant"] == 1)
    rows = []
    for res, g in y.groupby("resolution"):
        rows.append({"resolution": res, "n_test": len(g), "eligible_up_F1": int(up[g.index].sum()),
                     "eligible_down_F2": int(down[g.index].sum()), "eligible_total": int((up | down)[g.index].sum())})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"resolution": "ALL", "n_test": len(y), "eligible_up_F1": int(up.sum()),
                       "eligible_down_F2": int(down.sum()), "eligible_total": int((up | down).sum())}
    per_cohort = y.assign(eligible=(up | down)).groupby("cohort")["eligible"].sum().astype(int)
    return df, per_cohort


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lab = labels()
    rows = []
    for src, name, df, flags in flag_sets(lab):
        rows += score(df, flags, name, src)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "adjacency_enrichment_by_coding.csv", index=False)
    checks = check_against_historical(table)
    checks.to_csv(OUT / "historical_reproduction_check.csv", index=False)
    comp = composition(lab)
    comp.to_csv(OUT / "exact_stage_subset_composition.csv", index=False)
    elig, elig_cohort = planted_eligibility(lab)
    elig.to_csv(OUT / "planted_error_eligibility.csv", index=False)
    elig_cohort.to_csv(OUT / "planted_error_eligibility_per_cohort.csv")

    gc = graph_comparison(lab)
    gc.to_csv(OUT / "graph_comparison_by_coding.csv", index=False)

    lines = ["Exact-stage adjacency enrichment (F1/F2 vs F0/F3/F4, one-sided Fisher)", "=" * 78, "",
             f"exact-stage records: {int((lab.loc[lab['tier'] == 'T1', 'resolution'] == 'exact').sum())} of "
             f"{int((lab['tier'] == 'T1').sum())} held-out test records. Excluded from stage tests only:",
             "  137 grouped F0-F1 (stored F0; F1/F2-boundary membership unknown)",
             "  127 grouped F3-F4 (stored F3; outside the boundary, not known exact F3)",
             "Cohort representation changes: GSE31803 contributes 0 exact records, GSE213621 only",
             "its 107 exact F2 records (see exact_stage_subset_composition.csv). The exact subset",
             "is therefore NOT representative of all 1,064 records.", "",
             f"{'source':28s} {'flag set':40s} {'exact OR (p)':>20s} {'as-stored OR (p)':>20s} {'F0-F1->F1 OR':>12s}"
             f" {'exact w/o GSE213621 OR (p)':>26s}"]
    for (src, name), g in table.groupby(["source", "flag_set"], sort=False):
        g = g.set_index("coding")
        e, h, u = g.loc["exact_only"], g.loc["as_stored"], g.loc["grouped_F0F1_as_F1"]
        x = g.loc["exact_excluding_GSE213621"]
        lines.append(f"{src[:28]:28s} {name[:40]:40s} "
                     f"{e['odds_ratio']:6.2f} ({e['fisher_p_onesided']:.2g}) n={int(e['n_flagged']):3d}"
                     f"  {h['odds_ratio']:6.2f} ({h['fisher_p_onesided']:.2g})  {u['odds_ratio']:6.2f}"
                     f"  {x['odds_ratio']:6.2f} ({x['fisher_p_onesided']:.2g}) n={int(x['n_flagged']):3d}")
    lines += ["", f"as-stored coding reproduces historical summaries: {int(checks['match'].sum())}/{len(checks)}",
              "", "flagging_v2 paired contrast vs empty graph (dlogOR, 95% bootstrap CI). exact_only keeps the",
              "original matched-count flags, so arms are NOT count-matched within the exact subset:",
              gc[["condition", "target_n", "coding", "n_flagged_arm", "n_flagged_empty", "paired_diff_mean",
                  "paired_diff_lo", "paired_diff_hi", "verdict"]].to_string(index=False),
              f"as_stored reproduces flagging_v2/graph_comparison.csv: {bool(gc['reproduces_historical'].all())}",
              "", "Planted-error eligibility (flagging_v4 pools):", elig.to_string(index=False), "",
              "per cohort:", elig_cohort.to_string()]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return {"checks_pass": bool(checks["match"].all() and gc["reproduces_historical"].all()),
            "eligibility": elig, "graph_comparison": gc}
