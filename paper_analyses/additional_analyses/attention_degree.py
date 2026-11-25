"""Summed and per-edge layer-2 attention against STRING degree (attention paragraph of T2).

Per-edge attention = summed outgoing layer-2 attention divided by out-edges, self-loop included.
Usage (from the repository root, Zenodo data restored):
    python paper_analyses/additional_analyses/attention_degree.py \
        attention_analysis_b9/consensus_gene_ranking.csv shared/graph_topology.csv \
        shared/panel_genes.csv shared/graph_edges.csv
Output: attention_degree.txt
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, fisher_exact
a=pd.read_csv(sys.argv[1],index_col=0); t=pd.read_csv(sys.argv[2],index_col=0); p=pd.read_csv(sys.argv[3])
e=pd.read_csv(sys.argv[4]); print(e.columns.tolist(), len(e))
j=a.join(t)
deg=np.rint(j.degree_centrality*71).astype(int); j['deg']=deg
j['per_edge']=j.mean_attention/(deg+1)
pw={r.gene:set(r.pathways.split(';')) for r in p.itertuples()}
for col in ['mean_attention','per_edge']:
    top=j[col].nlargest(15).index
    k=sum('ECM_remodeling' in pw[g] for g in top); n=sum('ECM_remodeling' in pw[g] for g in j.index)
    print(col, list(top), 'ECM %d/15 vs %d/72'%(k,n), 'p=%.3f'%fisher_exact([[k,15-k],[n-k,72-15-(n-k)]],alternative='greater')[1],
          'rho_deg=%.3f p=%.2g'%spearmanr(j[col],j.deg))
nz=j[j.deg>0]; print('connected genes only rho(per_edge,deg)=%.3f p=%.2g n=%d'%(*spearmanr(nz.per_edge,nz.deg),len(nz)))
print('sum degree', deg.sum())
top_att=set(j.mean_attention.nlargest(15).index); top_deg=set(j.deg.nlargest(15).index)
print('top 15 by summed attention that are among the 15 highest-degree genes: %d' % len(top_att & top_deg))
nz=j[j.deg>0]
top=nz.per_edge.nlargest(15).index
k=sum('ECM_remodeling' in pw[g] for g in top); n=sum('ECM_remodeling' in pw[g] for g in nz.index)
print('connected only per-edge top15',list(top),'ECM %d/15 vs %d/%d p=%.3f'%(k,n,len(nz),fisher_exact([[k,15-k],[n-k,len(nz)-15-(n-k)]],alternative='greater')[1]))
print(nz.loc[list(top),'deg'].tolist())
# ECM genes mean degree vs others
ecm=[('ECM_remodeling' in pw[g]) for g in j.index]
print('mean deg ECM %.2f other %.2f'%(j.deg[ecm].mean(), j.deg[[not e for e in ecm]].mean()))
from scipy.stats import mannwhitneyu
print(mannwhitneyu(j.deg[ecm], j.deg[[not e for e in ecm]],alternative='greater'))
