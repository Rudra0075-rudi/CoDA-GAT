"""LAC coverage at 95, 90 and 80% nominal, recomputed from the saved probabilities and split by true class (T4).

Usage (from the repository root, Zenodo data restored):
    python paper_analyses/additional_analyses/conformal_classcond.py results_b7 folds \
        paper_analyses/additional_analyses/conformal_classcond.csv
Output: conformal_classcond.csv (one row per fold, seed and level) and conformal_classcond.txt
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
R=Path(sys.argv[1]); F=Path(sys.argv[2])
rows=[]
for fd in sorted(F.iterdir()):
    if not fd.name.startswith('fold'): continue
    yc=pd.read_csv(fd/'calibration'/'y.csv',index_col=0)['y_T1_significant'].astype(int).to_numpy()
    yt=pd.read_csv(fd/'test'/'y.csv',index_col=0)['y_T1_significant'].astype(int).to_numpy()
    for sd in sorted((R/fd.name/'mode_off').glob('seed_*')):
        pc=np.load(sd/'probs_calibration.npy'); pt=np.load(sd/'probs_test.npy')
        sc=1-pc[np.arange(len(yc)),yc]; n=len(sc)
        for a in (0.05,0.1,0.2):
            k=int(np.ceil((1-a)*(n+1))); q=np.sort(sc)[k-1] if k<=n else np.inf
            S=(pt>=1-q)  # set membership
            cov=S[np.arange(len(yt)),yt]
            rows.append(dict(fold=fd.name,seed=sd.name,alpha=a,cov=cov.mean(),cov_neg=cov[yt==0].mean(),cov_pos=cov[yt==1].mean(),
                 size=S.sum(1).mean(),pi_cal=yc.mean(),pi_test=yt.mean(),empty=(S.sum(1)==0).mean()))
d=pd.DataFrame(rows)
g=d.groupby(['alpha','fold']).mean(numeric_only=True)
pd.set_option('display.width',200)
print(g.round(3).to_string())
for a in (0.05,0.1,0.2):
    x=g.loc[a]; print('alpha',a,'mean cov',round(x['cov'].mean(),4),'mean size',round(x['size'].mean(),3),
      'neg cov',round(x.cov_neg.mean(),3),'pos cov',round(x.cov_pos.mean(),3))
    dp=x.pi_test-x.pi_cal; gap=x['cov']-(1-a)
    print('  spearman(cov gap, pi_test-pi_cal)=%.3f p=%.3f'%spearmanr(gap,dp), ' spearman(|gap|,|dp|)=%.3f p=%.3f'%spearmanr(gap.abs(),dp.abs()))
    print('  spearman(cov_pos-cov_neg, dp)=%.3f p=%.3f'%spearmanr(x.cov_pos-x.cov_neg,dp))
d.to_csv(sys.argv[3],index=False)
