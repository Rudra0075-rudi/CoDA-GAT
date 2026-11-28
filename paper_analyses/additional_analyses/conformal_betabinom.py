"""Coverage of the class-conditional (Mondrian) sets at 90% nominal against the coverage expected
under exchangeability (T4).

If the calibration and test scores of a class are exchangeable, the number X of the m test
samples of that class covered by a split-conformal set calibrated on n samples follows
BetaBinomial(m, k, n + 1 - k), with k = ceil((1 - alpha)(n + 1)). The script gives, for each
cohort and class, the probability under this distribution that coverage falls below 85%, the
expected and observed number of cohort-class cells below 85% in each seed, and the lower-tail
probability of the observed F2+ coverage in GSE84044 in each seed.

Usage (from the repository root, Zenodo data restored; run conformal_mondrian.py first):
    python paper_analyses/additional_analyses/conformal_betabinom.py folds \
        paper_analyses/additional_analyses/conformal_mondrian.csv
Output: conformal_betabinom.txt
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import betabinom

ALPHA, LEVEL_PCT = 0.10, 85
F = Path(sys.argv[1])
mon = pd.read_csv(sys.argv[2])
mon = mon[np.isclose(mon.alpha, ALPHA)]

cells, runs = [], []
for fd in sorted(F.iterdir()):
    if not fd.name.startswith('fold'):
        continue
    yc = pd.read_csv(fd / 'calibration' / 'y.csv', index_col=0)['y_T1_significant'].astype(int)
    yt = pd.read_csv(fd / 'test' / 'y.csv', index_col=0)['y_T1_significant'].astype(int)
    cal = json.load(open(fd / 'fold_info.json'))['calibration_source'].split(':')[-1]
    for c, col, name in ((0, 'cov_neg', 'F0-F1'), (1, 'cov_pos', 'F2+')):
        n, m = int((yc == c).sum()), int((yt == c).sum())
        k = int(np.ceil((1 - ALPHA) * (n + 1)))
        dist = betabinom(m, k, n + 1 - k) if k <= n else None   # k > n: the class is always in the set
        below = (LEVEL_PCT * m - 1) // 100                        # largest count with coverage < 85%
        cells.append(dict(test_cohort=fd.name.split('_', 1)[1], calibration_cohort=cal, cls=name, n_cal=n, m_test=m, k=k,
                          p_below=float(dist.cdf(below)) if dist else 0.0))
        for _, r in mon[mon.fold == fd.name].iterrows():
            x = int(round(r[col] * m))
            runs.append(dict(test_cohort=fd.name.split('_', 1)[1], cls=name, seed=r.seed, m_test=m, covered=x,
                             coverage=x / m, below=100 * x < LEVEL_PCT * m,
                             p_tail=float(dist.cdf(x)) if dist else 1.0))
cells, runs = pd.DataFrame(cells), pd.DataFrame(runs)

print(f'Mondrian sets at {int(100 * (1 - ALPHA))}% nominal: coverage against BetaBinomial(m, k, n + 1 - k), '
      f'k = ceil((1 - alpha)(n + 1))')
print()
print(f'Probability under exchangeability that coverage falls below {LEVEL_PCT}%, by test cohort and class')
print(cells.to_string(index=False, formatters={'p_below': '{:.4f}'.format}))
print()
print(f'cohort-class cells below {LEVEL_PCT}% per seed: expected {cells.p_below.sum():.2f} of {len(cells)}; observed',
      ', '.join(f'{s} {v}' for s, v in runs.groupby('seed').below.sum().items()))
print()
g = runs[(runs.test_cohort == 'GSE84044') & (runs.cls == 'F2+')]
c = cells[(cells.test_cohort == 'GSE84044') & (cells.cls == 'F2+')].iloc[0]
print(f'GSE84044, F2+: {c.n_cal} calibration samples ({c.calibration_cohort}) and {c.m_test} test samples, k = {c.k}')
print(g[['seed', 'covered', 'coverage', 'p_tail']].to_string(
    index=False, formatters={'coverage': '{:.3f}'.format, 'p_tail': '{:.4f}'.format}))
print(f'lower-tail probability P(X <= observed): {g.p_tail.min():.4f} to {g.p_tail.max():.4f}')
