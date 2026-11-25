"""Class-conditional (Mondrian) LAC sets from the same saved probabilities, compared with LAC (T4).

Usage (from the repository root, Zenodo data restored; run conformal_classcond.py first):
    python paper_analyses/additional_analyses/conformal_mondrian.py results_b7 folds \
        paper_analyses/additional_analyses/conformal_classcond.csv \
        paper_analyses/additional_analyses/conformal_mondrian.csv
Output: conformal_mondrian.csv and conformal_mondrian.txt
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
R=Path(sys.argv[1]); F=Path(sys.argv[2]); rows=[]
def q_of(s,a):
    n=len(s); k=int(np.ceil((1-a)*(n+1))); return np.sort(s)[k-1] if k<=n else np.inf
for fd in sorted(F.iterdir()):
    if not fd.name.startswith('fold'): continue
    yc=pd.read_csv(fd/'calibration'/'y.csv',index_col=0)['y_T1_significant'].astype(int).to_numpy()
    yt=pd.read_csv(fd/'test'/'y.csv',index_col=0)['y_T1_significant'].astype(int).to_numpy()
    for sd in sorted((R/fd.name/'mode_off').glob('seed_*')):
        pc=np.load(sd/'probs_calibration.npy'); pt=np.load(sd/'probs_test.npy')
        sc=1-pc[np.arange(len(yc)),yc]
        for a in (0.05,0.1,0.2):
            S=np.zeros_like(pt,dtype=bool)
            for c in (0,1):
                qc=q_of(sc[yc==c],a); S[:,c]=(1-pt[:,c])<=qc
            cov=S[np.arange(len(yt)),yt]
            rows.append(dict(fold=fd.name,seed=sd.name,alpha=a,cov=cov.mean(),cov_neg=cov[yt==0].mean(),cov_pos=cov[yt==1].mean(),size=S.sum(1).mean()))
d=pd.DataFrame(rows); g=d.groupby(['alpha','fold']).mean(numeric_only=True)
print(g.round(3).to_string())
m=pd.read_csv(sys.argv[3]); gm=m.groupby(['alpha','fold']).mean(numeric_only=True)
for a in (0.05,0.1,0.2):
    x=g.loc[a]; y=gm.loc[a]
    print(f'alpha {a}: Mondrian mean cov {x["cov"].mean():.4f} (min {x["cov"].min():.3f}) size {x["size"].mean():.3f} | LAC mean cov {y["cov"].mean():.4f} (min {y["cov"].min():.3f}) size {y["size"].mean():.3f}')
    print('   cohorts below nominal-0.02: Mondrian',(x['cov']<1-a-0.02).sum(),' LAC',(y['cov']<1-a-0.02).sum(), '| below nominal: M',(x['cov']<1-a).sum(),'L',(y['cov']<1-a).sum())
    print('   worst-class cov mean: M %.3f L %.3f'%(np.minimum(x.cov_neg,x.cov_pos).mean(), np.minimum(y.cov_neg,y.cov_pos).mean()))
    print('   mean abs dev from nominal: M %.4f L %.4f'%((x['cov']-(1-a)).abs().mean(),(y['cov']-(1-a)).abs().mean()))
d.to_csv(sys.argv[4],index=False)
