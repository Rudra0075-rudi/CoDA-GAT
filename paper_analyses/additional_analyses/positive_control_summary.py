"""Paired contrasts for the positive control (T2 run on a planted, STRING-dependent label).

Usage (from the repository root):
    python paper_analyses/additional_analyses/positive_control_summary.py \
        paper_analyses/additional_analyses/positive_control_runs \
        paper_analyses/additional_analyses/positive_control_runs.csv
Output: positive_control_runs.csv (one row per run) and, from stdout, positive_control_summary.txt
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

rows = [json.load(open(f)) for f in glob.glob(sys.argv[1] + '/*/*/seed_*/run_metrics.json')]
d = pd.DataFrame(rows)
d['graph'] = d.ablation.str.replace('planted_', '', regex=False)
print(d.groupby('graph').size().to_string())


def ci(x):
    x = x.dropna()
    h = t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    p = wilcoxon(x).pvalue
    return f'{x.mean():+.4f} [{x.mean() - h:+.4f}, {x.mean() + h:+.4f}] n={len(x)} STRING higher in {int((x > 0).sum())} p={p:.3f}'


for met in ['auc', 'macro_f1', 'balanced_acc']:
    g = d.groupby(['fold', 'graph'])[met].mean().unstack()      # five-seed mean per fold
    print('\n', met)
    print(g.round(4).to_string())
    print(' means', g.mean().round(4).to_dict())
    print(' STRING - disconnected     ', ci(g['string'] - g['empty']))
    print(' STRING - feature-permuted ', ci(g['string'] - g['shuffled']))
s = d[d.graph == 'string']
print('\nSTRING runs with test AUROC above 0.7 (planted signal learned): %d of %d' % ((s.auc > 0.7).sum(), len(s)))
for gname in ['empty', 'shuffled']:
    print('%s runs with test AUROC above 0.7: %d of 40' % (gname, (d[d.graph == gname].auc > 0.7).sum()))
d.drop(columns='graph').sort_values(['fold', 'ablation', 'seed']).to_csv(sys.argv[2], index=False)
