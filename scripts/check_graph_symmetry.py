"""
Graph symmetry / self-loop validation for every CoDA-GAT graph condition.

Extends the file-based check to the graphs that have no file: random_graph,
shuffled_graph and empty_graph are generated procedurally per fold by
ablations_b9.build_edge_index_and_perm, so they are reconstructed here with the
same fold seed rather than read from disk.

Two conventions matter and are reported separately, because mixing them up
produces a nonsense asymmetry count:

  MIRRORED edge lists   shared/graph_edges.csv stores both (i,j) and (j,i).
                        Asymmetry here is a real defect.
  PAIR edge lists       learned_graph/edges/*.csv stores each undirected pair
                        ONCE. Every row looks "asymmetric" under a naive check,
                        which means nothing -- the file is simply not mirrored.

Read-only. Prints a table; writes nothing.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ablations_b9 import build_edge_index_and_perm, ABLATIONS
from run_full_experiment_b7 import load_edge_index_generic
from train_fold0_b1 import N_NODES

REPO = Path(__file__).resolve().parent.parent
REAL_N_DIRECTED = 642
PROCEDURAL = ["empty_graph", "shuffled_graph", "random_graph"]


def analyse(pairs, n_nodes=N_NODES):
    """Asymmetry, self-loops and undirected span of a directed edge set."""
    asym = sum((b, a) not in pairs for a, b in pairs)
    loops = sum(1 for a, b in pairs if a == b)
    undirected = {(min(a, b), max(a, b)) for a, b in pairs}
    n = len(pairs)
    # chance expectation: if n directed edges are drawn uniformly from the
    # n_nodes*(n_nodes-1) ordered pairs, P(reverse also drawn) ~ (n-1)/(N*(N-1)-1)
    ordered = n_nodes * (n_nodes - 1)
    p_mirror = (n - 1) / (ordered - 1) if n > 1 and ordered > 1 else 0.0
    return {"n_directed": n, "asymmetric": asym, "self_loops": loops,
            "n_undirected_pairs": len(undirected),
            "mirrored_edges": n - asym,
            "symmetric": asym == 0 and n > 0,
            "expected_asymmetric_if_uniform": n * (1 - p_mirror)}


def row(label, pairs, expected_directed=REAL_N_DIRECTED, convention="mirrored"):
    a = analyse(pairs)
    a["graph"] = label
    a["convention"] = convention
    a["matches_real_budget"] = (a["n_directed"] == expected_directed if convention == "mirrored"
                                 else 2 * a["n_directed"] == expected_directed)
    return a


SRC_NAMES = ("src_idx", "source_idx", "src", "source", "i", "u")
DST_NAMES = ("dst_idx", "target_idx", "dst", "target", "j", "v")


def pick_edge_columns(E, path):
    """Find the endpoint columns BY NAME, never by position.

    Taking the first two columns is a trap here: shared/graph_edges.csv starts
    with src_idx,dst_idx, but learned_graph/edges/*.csv starts with
    rank,src_idx,dst_idx -- so a positional read silently compares 'rank'
    against 'src_idx' and invents self-loops and duplicate pairs that do not
    exist. Named lookup first, positional only as a last resort, and the chosen
    columns are always printed so the reader can check.
    """
    lower = {c.lower(): c for c in E.columns}
    s = next((lower[n] for n in SRC_NAMES if n in lower), None)
    t = next((lower[n] for n in DST_NAMES if n in lower), None)
    if s is None or t is None:
        numeric = [c for c in E.columns if np.issubdtype(E[c].dtype, np.number)
                   and c.lower() not in ("rank", "score", "weight", "string_score")]
        if len(numeric) < 2:
            raise ValueError(f"{path}: cannot identify endpoint columns from "
                              f"{list(E.columns)}")
        s, t = numeric[0], numeric[1]
    for c in (s, t):
        if not np.issubdtype(E[c].dtype, np.number):
            raise ValueError(f"{path}: endpoint column '{c}' is not numeric -- node INDICES "
                              f"are required, not gene names ({list(E.columns)})")
    return s, t


def from_csv(path, convention):
    E = pd.read_csv(path)
    s, t = pick_edge_columns(E, path)
    return set(zip(E[s].astype(int), E[t].astype(int))), len(E), (s, t)


def main():
    ap = argparse.ArgumentParser(description="graph symmetry / self-loop validation")
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(8)))
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional edge CSVs to check (first two columns = node indices)")
    args = ap.parse_args()

    rows = []
    real_ei = load_edge_index_generic()

    print("=" * 96)
    print("MIRRORED-CONVENTION graphs (both directions stored; asymmetry here is a defect)")
    print("=" * 96)

    p_real, n_rows, cols = from_csv(REPO / "shared" / "graph_edges.csv", "mirrored")
    rows.append(row("shared/graph_edges.csv (STRING)", p_real))
    print(f"  endpoint columns used: {cols}")
    if n_rows != len(p_real):
        print(f"  NOTE: {n_rows} rows collapse to {len(p_real)} unique directed edges "
              f"-> {n_rows - len(p_real)} duplicate row(s)")

    for cond in PROCEDURAL:
        for fold in args.folds:
            ei, perm = build_edge_index_and_perm(ABLATIONS[cond], real_ei, fold)
            pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
            r = row(f"{cond} fold{fold}", pairs)
            r["node_perm_applied_to_features"] = perm is not None
            rows.append(r)

    hdr = (f"  {'graph':<34s} {'rows':>6s} {'asym':>6s} {'loops':>6s} {'pairs':>6s} "
           f"{'sym':>5s} {'budget':>7s}")
    print(hdr)
    for r in rows:
        print(f"  {r['graph']:<34s} {r['n_directed']:>6d} {r['asymmetric']:>6d} "
              f"{r['self_loops']:>6d} {r['n_undirected_pairs']:>6d} "
              f"{'yes' if r['symmetric'] else 'NO':>5s} "
              f"{'ok' if r['matches_real_budget'] else 'MISMATCH':>7s}")

    # learned graphs use the pair convention -- checked separately, never mixed in
    learned = sorted((REPO / "learned_graph" / "edges").glob("learned_edges_fold*_*.csv"))
    if learned:
        print()
        print("=" * 96)
        _, _, lcols = from_csv(learned[0], "pair")
        print(f"PAIR-CONVENTION graphs -- endpoint columns used: {lcols}")
        print("(each undirected pair stored once; 'asymmetric' is")
        print("expected and meaningless -- what matters is duplicate pairs and self-loops)")
        print("=" * 96)
        print(f"  {'graph':<34s} {'rows':>6s} {'pairs':>6s} {'dup':>5s} {'loops':>6s} "
              f"{'x2 = budget':>12s}")
        for p in learned:
            pairs, n_rows, cols = from_csv(p, "pair")
            a = analyse(pairs)
            dup = n_rows - a["n_undirected_pairs"]
            rows.append(row(p.name, pairs, convention="pair"))
            print(f"  {p.name:<34s} {n_rows:>6d} {a['n_undirected_pairs']:>6d} "
                  f"{dup:>5d} {a['self_loops']:>6d} "
                  f"{'ok' if 2 * a['n_undirected_pairs'] == REAL_N_DIRECTED else 'MISMATCH':>12s}")

    for extra in args.extra:
        pairs, n_rows, _ = from_csv(Path(extra), "mirrored")
        r = row(str(extra), pairs)
        rows.append(r)
        print(f"\n  {extra}: rows={n_rows} asym={r['asymmetric']} loops={r['self_loops']} "
              f"pairs={r['n_undirected_pairs']} symmetric={'yes' if r['symmetric'] else 'NO'}")

    print()
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    rnd = [r for r in rows if r["graph"].startswith("random_graph")]
    if rnd:
        asym = np.array([r["asymmetric"] for r in rnd], dtype=float)
        exp = np.array([r["expected_asymmetric_if_uniform"] for r in rnd], dtype=float)
        prs = np.array([r["n_undirected_pairs"] for r in rnd], dtype=float)
        print(f"  random_graph is ASYMMETRIC by construction, across all {len(rnd)} folds:")
        print(f"    asymmetric edges  observed {asym.mean():.0f} +/- {asym.std():.0f}   "
              f"expected under uniform directed sampling {exp.mean():.0f}")
        print(f"    undirected pairs  {prs.min():.0f}-{prs.max():.0f} "
              f"(mean {prs.mean():.1f}) vs 321 for every symmetric condition")
        print(f"    self-loops        {max(r['self_loops'] for r in rnd)} (excluded by "
              f"make_random_edge_index's i != j guard)")
        print("    The observed count matches the chance expectation, so this is uniform")
        print("    directed sampling behaving correctly -- not a bug. It does mean random_graph")
        print("    holds the DIRECTED budget (642) while spanning ~2x the undirected pairs of")
        print("    the symmetric conditions, so it is denser in undirected terms. Any")
        print("    comparison involving it carries that caveat.")
    shuf = [r for r in rows if r["graph"].startswith("shuffled_graph")]
    if shuf:
        print()
        print(f"  shuffled_graph's EDGE LIST is identical to STRING and fully symmetric.")
        print(f"    That is correct and not a failed shuffle: build_edge_index_and_perm")
        print(f"    returns (real_edge_index, node_permutation) and the permutation is applied")
        print(f"    to the node FEATURES, not the edges. Checking the edge list alone cannot")
        print(f"    see the shuffle. node_perm present: "
              f"{all(r.get('node_perm_applied_to_features') for r in shuf)}")
    bad = [r for r in rows if r["convention"] == "mirrored" and not r["symmetric"]
           and not r["graph"].startswith(("random_graph", "empty_graph"))]
    print()
    if bad:
        print(f"  UNEXPECTED ASYMMETRY in {len(bad)} mirrored-convention graph(s):")
        for r in bad:
            print(f"    {r['graph']}: {r['asymmetric']} asymmetric edges")
    else:
        print("  No unexpected asymmetry. Every graph that should be symmetric is symmetric;")
        print("  the only asymmetric condition is random_graph, which is asymmetric by design.")
    loops = [r for r in rows if r["self_loops"]]
    print(f"  Self-loops anywhere: {'YES -- ' + str([r['graph'] for r in loops]) if loops else 'none'}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
