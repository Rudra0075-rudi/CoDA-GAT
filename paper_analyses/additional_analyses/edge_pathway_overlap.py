"""Share of STRING panel edges that join two genes of a shared pathway, against all gene pairs.
Usage (from the repository root, Zenodo data restored):
    python paper_analyses/additional_analyses/edge_pathway_overlap.py shared
Output: edge_pathway_overlap.txt
Reported: 209 of 321 edges (65.1%) against 17.8% of all 2,556 gene pairs."""
import sys, itertools
import pandas as pd
p = sys.argv[1].rstrip('/') + '/'
e = pd.read_csv(p + 'graph_edges.csv'); g = pd.read_csv(p + 'panel_genes.csv')
pw = {r.gene: set(str(r.pathways).split(';')) for r in g.itertuples()}
u = e[e.src_idx < e.dst_idx]
k = sum(len(pw[a] & pw[b]) > 0 for a, b in zip(u.src, u.dst))
pairs = list(itertools.combinations(pw, 2))
allp = sum(len(pw[a] & pw[b]) > 0 for a, b in pairs)
print(f'edges in shared pathway: {k}/{len(u)} = {k/len(u):.3f}')
print(f'all gene pairs in shared pathway: {allp}/{len(pairs)} = {allp/len(pairs):.3f}')
