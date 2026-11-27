"""Observed layer-2 attention against the uniform-attention null (attention paragraph of T2).

Under uniform attention each receiving gene l spreads its attention equally over its
neighbours and itself, 1/(deg(l)+1) each, so gene j has an expected summed attention of
1/(deg(j)+1) (self-loop) plus the sum of 1/(deg(l)+1) over the genes l that receive from j.
The ratio of observed to expected attention is what the model learned beyond the graph.
Usage (from the repository root, Zenodo data restored):
    python paper_analyses/additional_analyses/attention_uniform_null.py \
        attention_analysis_b9/consensus_gene_ranking.csv shared/graph_edges.csv shared/panel_genes.csv
Output: attention_uniform_null.txt
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, fisher_exact, mannwhitneyu

att = pd.read_csv(sys.argv[1], index_col=0).mean_attention
edges = pd.read_csv(sys.argv[2])
panel = pd.read_csv(sys.argv[3])

deg = pd.Series(0, index=att.index)
for s in edges.src:
    deg[s] += 1
expected = pd.Series(0.0, index=att.index)
for g in att.index:
    expected[g] += 1 / (deg[g] + 1)
for r in edges.itertuples():
    expected[r.src] += 1 / (deg[r.dst] + 1)
ratio = att / expected

print("genes %d, directed edge entries %d" % (len(att), len(edges)))
print("Spearman rho(observed, uniform null) = %.3f, Pearson r = %.3f" % (spearmanr(att, expected)[0], pearsonr(att, expected)[0]))
print("Spearman rho(observed, degree) = %.3f" % spearmanr(att, deg)[0])
print("ratio observed/expected: range %.2f-%.2f" % (ratio.min(), ratio.max()))
nz = deg > 0
print("connected genes: Spearman rho(ratio, degree) = %.3f (n=%d)" % (spearmanr(ratio[nz], deg[nz])[0], nz.sum()))

pw = {r.gene: set(str(r.pathways).split(';')) for r in panel.itertuples()}
ecm = pd.Series({g: 'ECM_remodeling' in pw[g] for g in att.index})
top = ratio.sort_values(ascending=False).index[:15]
k, n = int(ecm[top].sum()), int(ecm.sum())
p = fisher_exact([[k, 15 - k], [n - k, len(att) - 15 - (n - k)]], alternative='greater')[1]
print("15 most up-weighted genes:", list(top))
print("ECM among them: %d of 15, against %d of %d in the panel; one-sided Fisher p = %.2g" % (k, n, len(att), p))
mw = mannwhitneyu(ratio[ecm], ratio[~ecm], alternative='greater').pvalue
print("median ratio ECM %.2f, other genes %.2f; one-sided Mann-Whitney p = %.2g" % (ratio[ecm].median(), ratio[~ecm].median(), mw))
