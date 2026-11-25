"""Cohort-composition sensitivity for the exact-stage boundary-enrichment tests.
Usage (from the repository root): python paper_analyses/exact_stage_cohort_sensitivity.py .
Output: exact_stage_cohort_sensitivity.txt
Inputs (deposited): corrections_2026-11/outputs/consensus_failures/error_matrix.csv,
test_a_adjacency/per_sample_flags.csv."""
import sys, numpy as np, pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable
R = sys.argv[1] if len(sys.argv) > 1 else '.'
em = pd.read_csv(f'{R}/corrections_2026-11/outputs/consensus_failures/error_matrix.csv')
models = [c for c in em.columns if c not in ('sample_id','fold','stage','resolution','y_true')]
em['n_wrong'] = em[models].astype(bool).sum(axis=1)
fl = pd.read_csv(f'{R}/test_a_adjacency/per_sample_flags.csv')[['sample_id','test_cohort','flagged']]
d = em.merge(fl, on='sample_id')
d = d[d.resolution == 'exact'].copy()
d['boundary'] = d.stage.isin(['F1','F2'])
d['consensus'] = d.n_wrong >= 19
def report(name, col, sub):
    t = pd.crosstab(sub[col], sub.boundary).reindex(index=[True,False], columns=[True,False]).fillna(0).astype(int).values
    o, p = fisher_exact(t, alternative='greater')
    tabs = []
    for c, g in sub.groupby('test_cohort'):
        tt = pd.crosstab(g[col], g.boundary).reindex(index=[True,False], columns=[True,False]).fillna(0).values
        if (tt.sum(0) > 0).all() and (tt.sum(1) > 0).all(): tabs.append(tt + 0.0)
    st = StratifiedTable(tabs)
    print(f'{name}: n={len(sub)} table={t.tolist()} pooled OR={o:.2f} one-sided p={p:.2g} | '
          f'MH OR over {len(tabs)} informative cohorts={st.oddsratio_pooled:.2f} '
          f'95% CI {st.oddsratio_pooled_confint()[0]:.2f}-{st.oddsratio_pooled_confint()[1]:.2f} '
          f'CMH p={st.test_null_odds().pvalue:.2g}')
for col, nm in [('consensus','Consensus failures (>=19 of 23 wrong)'), ('flagged','Test A flags (52-flag point)')]:
    report(nm, col, d)
    report(nm + ', excluding GSE213621', col, d[d.test_cohort != 'GSE213621'])
