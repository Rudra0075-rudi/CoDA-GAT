"""Paired contrasts for the context-free experiment (table of node features with and without network context).

Usage (from the repository root):
    python paper_analyses/additional_analyses/ctx_summary.py \
        paper_analyses/additional_analyses/ctxfree_runs \
        paper_analyses/additional_analyses/ctxfree_runs.csv
Output: ctxfree_runs.csv (one row per run) and, from stdout, ctxfree_summary.txt
"""
import json,glob,sys,numpy as np,pandas as pd
from scipy.stats import wilcoxon,t
rows=[json.load(open(f)) for f in glob.glob(sys.argv[1]+'/*/*/seed_*/run_metrics.json')]
d=pd.DataFrame(rows)
print(d.groupby('ablation').size())
for met in ['auc','macro_f1','balanced_acc']:
    g=d.groupby(['fold','ablation'])[met].mean().unstack()
    print('\n',met); print(g.round(4).to_string()); print('means',g.mean().round(4).to_dict())
    def ci(x):
        x=x.dropna(); h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)); 
        try: p=wilcoxon(x).pvalue
        except Exception: p=float('nan')
        return f'{x.mean():+.4f} [{x.mean()-h:+.4f}, {x.mean()+h:+.4f}] n={len(x)} pos={int((x>0).sum())} p={p:.3f}'
    print(' STRING-empty full   ',ci(g.string_full-g.empty_full))
    print(' STRING-empty ctxfree',ci(g.string_ctxfree-g.empty_ctxfree))
    print(' interaction         ',ci((g.string_ctxfree-g.empty_ctxfree)-(g.string_full-g.empty_full)))
    print(' ctx loss STRING     ',ci(g.string_full-g.string_ctxfree))
    print(' ctx loss empty      ',ci(g.empty_full-g.empty_ctxfree))
d.to_csv(sys.argv[2],index=False)
