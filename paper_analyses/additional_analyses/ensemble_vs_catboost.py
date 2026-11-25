"""CoDA-GAT (five-seed ensemble and per-seed mean) against single-seed CatBoost, paired over cohorts (T1).

Usage (from the repository root, Zenodo data restored):
    python paper_analyses/additional_analyses/ensemble_vs_catboost.py results_b7 folds \
        corrections_2026-11/outputs/aggregation/per_cohort_baselines.csv
Output: ensemble_vs_catboost.txt
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, t
R=Path(sys.argv[1]); F=Path(sys.argv[2]); B=pd.read_csv(sys.argv[3])
ens={}; per={}
for fd in sorted(F.iterdir()):
    if not fd.name.startswith('fold'): continue
    yt=pd.read_csv(fd/'test'/'y.csv',index_col=0)['y_T1_significant'].astype(int).to_numpy()
    P=[np.load(s/'probs_test.npy')[:,1] for s in sorted((R/fd.name/'mode_off').glob('seed_*'))]
    coh=fd.name.split('_',1)[1]
    ens[coh]=roc_auc_score(yt,np.mean(P,0)); per[coh]=np.mean([roc_auc_score(yt,p) for p in P])
e=pd.Series(ens); p=pd.Series(per)
print('ensemble mean %.4f sd %.4f | per-seed mean %.4f'%(e.mean(),e.std(ddof=1),p.mean()))
print(B.columns.tolist())
for arm in ['panel72_full__catboost','panel72_combined__catboost']:
    c=B[B.arm==arm].set_index('test_cohort')['auc'].reindex(e.index)
    for name,g in [('ensemble',e),('per-seed',p)]:
        d=c-g; ci=t.ppf(.975,7)*d.std(ddof=1)/np.sqrt(8)
        print(arm,name,'cat %.4f'%c.mean(),'diff %.4f [%.4f, %.4f]'%(d.mean(),d.mean()-ci,d.mean()+ci),'higher %d/8'%(d>0).sum(),'p=%.3f'%wilcoxon(c,g).pvalue)
