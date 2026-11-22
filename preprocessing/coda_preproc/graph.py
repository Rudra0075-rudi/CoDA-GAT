"""Stage 7: 72-gene panel and STRING subgraph (notebook Cell 29, Part B).

Behaviour kept exactly, including one quirk that must be disclosed:
74 pathway genes are available in the common gene space, and the panel is
the first 72 of them in alphabetical order. VIM and XBP1 are therefore
excluded by sort order, not by any biological or statistical criterion.
The top-up branch (high-variance genes) never runs because 74 >= 72.
"""
import networkx as nx
import pandas as pd

from .config import ECM_UP, HEP_DOWN, N_PANEL_GENES, PATHWAYS, STRING_THRESHOLD


def select_panel(bulk, n=N_PANEL_GENES):
    avail = set(bulk.index)
    panel = sorted({g for gs in PATHWAYS.values() for g in gs if g in avail})
    if len(panel) < n:                       # label-free top-up (unused for this data)
        pool = [g for g in ECM_UP + HEP_DOWN if g in avail and g not in panel]
        add = list(bulk.loc[pool].var(axis=1).sort_values(ascending=False).index[:n - len(panel)])
        panel = sorted(panel + add)
    dropped = panel[n:]
    return panel[:n], dropped


def panel_table(panel):
    return pd.DataFrame({"gene": panel, "pathways": [
        ";".join(p for p, gs in PATHWAYS.items() if g in gs) or "extra" for g in panel]})


def string_subgraph(links_gz, info_gz, panel, threshold=STRING_THRESHOLD):
    sinfo = pd.read_csv(info_gz, sep="\t", compression="gzip")
    pid = next(c for c in sinfo.columns if "protein_external_id" in c or c == "#string_protein_id")
    name = next(c for c in sinfo.columns if "preferred_name" in c)
    prot2sym = dict(zip(sinfo[pid].astype(str), sinfo[name].astype(str)))
    keep = {p for p, s in prot2sym.items() if s in set(panel)}

    parts = []
    for ch in pd.read_csv(links_gz, sep=" ", compression="gzip", chunksize=2_000_000):
        ch = ch[(ch.combined_score >= threshold) & ch.protein1.isin(keep) & ch.protein2.isin(keep)]
        if len(ch):
            parts.append(ch)
    E = pd.concat(parts, ignore_index=True)
    E["g1"], E["g2"] = E.protein1.map(prot2sym), E.protein2.map(prot2sym)
    E = E[E.g1 != E.g2]
    E["key"] = [tuple(sorted(t)) for t in zip(E.g1, E.g2)]
    und = E.groupby("key").combined_score.max().reset_index()
    und["g1"], und["g2"] = und.key.str[0], und.key.str[1]

    gi = {g: i for i, g in enumerate(panel)}
    rows = []
    for a, b, s in zip(und.g1, und.g2, und.combined_score):
        for u, v in ((a, b), (b, a)):        # both directions for PyG
            rows.append(dict(src_idx=gi[u], dst_idx=gi[v], src=u, dst=v,
                             string_score=int(s), weight=1.0))
    edges = pd.DataFrame(rows)

    G = nx.Graph()
    G.add_nodes_from(panel)
    G.add_edges_from(zip(und.g1, und.g2))
    dc, bc, cc, cl = (nx.degree_centrality(G), nx.betweenness_centrality(G),
                      nx.closeness_centrality(G), nx.clustering(G))
    topo = pd.DataFrame({
        "gene": panel,
        "degree_centrality": [dc[g] for g in panel],
        "betweenness_centrality": [bc[g] for g in panel],
        "closeness_centrality": [cc[g] for g in panel],
        "clustering_coefficient": [cl[g] for g in panel],
        "pathway_membership_count": [sum(g in gs for gs in PATHWAYS.values()) / len(PATHWAYS)
                                     for g in panel],
    }).set_index("gene")
    return edges, topo
