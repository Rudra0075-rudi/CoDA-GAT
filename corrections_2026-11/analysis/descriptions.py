"""
Issue 5: experiment descriptions, checked against code and saved files (no
design change). Every graph is reconstructed with numpy from the same RNG
calls as scripts/ablations_b9.build_edge_index_and_perm, which cannot be
imported here (it imports torch):

    rng = np.random.default_rng(2000 + fold_id)
    shuffled: rng.permutation(72) applied to X[:, perm, :]   (edges unchanged)
    random:   draw (i, j) = rng.integers(0, 72, size=2), keep i != j, until
              642 unique ORDERED pairs                       (no symmetrisation)

The reconstruction is validated against learned_graph/edge_budget_audit.csv,
which was produced from the torch originals.
"""

import json

import numpy as np
import pandas as pd

import guard

REPO, OUT = guard.REPO, guard.OUT / "descriptions"
N_NODES, N_DIRECTED = 72, 642


def pair_stats(directed):
    s = set(directed)
    und = {(min(a, b), max(a, b)) for a, b in s if a != b}
    return {"n_directed": len(s), "n_unordered_pairs": len(und),
            "n_reverse_present": sum((b, a) in s for a, b in s),
            "self_loops": sum(a == b for a, b in s), "symmetric": all((b, a) in s for a, b in s)}, und


def random_graph(fold_id):
    rng = np.random.default_rng(2000 + fold_id)
    edges = set()
    while len(edges) < N_DIRECTED:
        i, j = rng.integers(0, N_NODES, size=2)
        if i != j:
            edges.add((int(i), int(j)))
    return edges


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e = pd.read_csv(REPO / "shared" / "graph_edges.csv")
    string = set(zip(e["src_idx"], e["dst_idx"]))
    s_stats, s_pairs = pair_stats(string)
    audit = pd.read_csv(REPO / "learned_graph" / "edge_budget_audit.csv")

    rows = [{"graph": "STRING (shared/graph_edges.csv)", "fold": "all", **s_stats,
             "unordered_pairs_shared_with_STRING": len(s_pairs)}]
    for f in range(8):
        st, und = pair_stats(random_graph(f))
        a = audit[(audit["fold"] == f) & (audit["condition"] == "random_graph")].iloc[0]
        rows.append({"graph": "random_graph (B9)", "fold": f, **st,
                     "unordered_pairs_shared_with_STRING": len(und & s_pairs),
                     "matches_edge_budget_audit": bool(a["n_directed_edges"] == st["n_directed"]
                                                       and a["n_undirected_pairs"] == st["n_unordered_pairs"])})
        perm = np.random.default_rng(2000 + f).permutation(N_NODES)
        # node i carries gene perm[i]'s features over STRING edge (i, j): the gene pair
        # whose features meet along that edge is (perm[i], perm[j])
        meet = {(min(perm[a], perm[b]), max(perm[a], perm[b])) for a, b in s_pairs}
        rows.append({"graph": "shuffled_graph (B9)", "fold": f, **s_stats,
                     "unordered_pairs_shared_with_STRING": len(meet & s_pairs),
                     "nodes_keeping_own_features": int((perm == np.arange(N_NODES)).sum()),
                     "matches_edge_budget_audit": True})
    for path in sorted((REPO / "learned_graph" / "edges").glob("learned_edges_fold*_*.csv")):
        d = pd.read_csv(path)
        fold = int(path.stem.split("fold")[1].split("_")[0])
        variant = path.stem.split(f"fold{fold}_")[1]
        pairs = list(zip(d["src_idx"], d["dst_idx"]))
        mirrored = set(pairs) | {(b, a) for a, b in pairs}   # learned_graph.undirected_pairs_to_edge_index
        st, und = pair_stats(mirrored)
        a = audit[(audit["fold"] == fold) & (audit["condition"] == variant)].iloc[0]
        rows.append({"graph": f"{variant} (learned; file stores each pair once)", "fold": fold,
                     "rows_in_file": len(d), **st, "unordered_pairs_shared_with_STRING": len(und & s_pairs),
                     "matches_edge_budget_audit": bool(a["n_directed_edges"] == st["n_directed"]
                                                       and a["n_undirected_pairs"] == st["n_unordered_pairs"])})
    graphs = pd.DataFrame(rows)
    graphs.to_csv(OUT / "graph_conditions.csv", index=False)

    # augmentation arms: completed, so "not yet run" text is stale
    aug = []
    for m, root in ((1, "results_b7_m1"), (3, "results_b7_m3"), (7, "results_b7_m7")):
        r = REPO / root
        aug.append({"m": m, "directory": root, "exists": r.exists(),
                    "done_markers": len(list(r.glob("fold*/mode_off/seed_*/DONE.marker"))) if r.exists() else 0,
                    "master_rows": len(pd.read_csv(r / "master_results.csv")) if (r / "master_results.csv").exists() else 0})
    aug = pd.DataFrame(aug)
    aug.to_csv(OUT / "augmentation_arms.csv", index=False)

    # head-3 / directional_ps
    dev = pd.read_parquet(REPO / "shared" / "deviation_targets.parquet")
    handoff = json.loads((REPO / "shared" / "handoff.json").read_text())
    h3 = {"training_target_column": "deviation_score (ablations_b9.py, baselines_b3.py, run_full_experiment_b7.py)",
          "deviation_score_equals_directional": bool(np.allclose(dev["deviation_score"], dev["directional"], equal_nan=True)),
          "deviation_score_equals_directional_ps": bool(np.allclose(dev["deviation_score"], dev["directional_ps"], equal_nan=True)),
          "handoff_head3_arms": list(handoff["head3"]["arms"]),
          "handoff_head3_decision": handoff["head3"]["decision"],
          "files_reporting_a_directional_ps_model": sorted(
              str(p.relative_to(REPO)) for p in REPO.rglob("*")
              if p.is_file() and "directional_ps" in p.name)}
    (OUT / "head3_directional_ps.json").write_text(json.dumps(h3, indent=2))

    lines = ["Experiment descriptions vs code and files", "=" * 78, "",
             graphs.groupby("graph")[["n_directed", "n_unordered_pairs", "n_reverse_present", "self_loops",
                                      "unordered_pairs_shared_with_STRING"]].agg(["min", "max"]).to_string(), "",
             f"random/learned reconstructions match learned_graph/edge_budget_audit.csv: "
             f"{bool(graphs['matches_edge_budget_audit'].fillna(True).all())}", "",
             aug.to_string(index=False), "", json.dumps(h3, indent=2)]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return {"graphs": graphs, "aug": aug, "h3": h3,
            "audit_match": bool(graphs["matches_edge_budget_audit"].fillna(True).all())}
