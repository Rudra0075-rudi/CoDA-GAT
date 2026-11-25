"""Context-free node features (T2): pathway activity, pathway membership and the four STRING
topology measures (feature columns 2-12) set to zero, STRING versus disconnected encoder.

Reuses the training code of scripts/ablations_b9.py unchanged and writes only to OUT.
Usage (from the repository root, Zenodo data restored; see run_ctx_free.sh):
    python paper_analyses/additional_analyses/ctx_free.py . OUT FOLDS SEEDS ARMS
    FOLDS  comma-separated fold folder names, e.g. fold0_GSE130970,fold1_GSE135251
    SEEDS  comma-separated seeds, e.g. 42,43,44,45,46
    ARMS   comma-separated, from string_full,empty_full,string_ctxfree,empty_ctxfree
Each run writes OUT/<fold>/<arm>/seed_<s>/ (probs_test.npy, predictions_test.npy,
run_metrics.json, DONE.marker); finished runs are skipped.
"""
import sys, json, time
from pathlib import Path
W = Path(sys.argv[1]); OUT = Path(sys.argv[2]); folds = sys.argv[3].split(','); seeds=[int(s) for s in sys.argv[4].split(',')]
conds = sys.argv[5].split(',')
sys.path.insert(0, str(W/'scripts'))
import os; os.chdir(W)
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
import ablations_b9 as ab
from ablations_b9 import setup_fold_ablation, run_single_ablation, load_edge_index_generic, SHARED_DIR

# patch loader to accept a list of column ranges
orig = ab.load_split_for_ablation
def patched(fold_dir, split, edge_index, dev_lookup, cohort_to_idx, node_perm=None, zero_cols=None):
    if zero_cols is None or isinstance(zero_cols, tuple):
        return orig(fold_dir, split, edge_index, dev_lookup, cohort_to_idx, node_perm, zero_cols)
    # list of ranges: zero sequentially by wrapping np.load
    real_load = np.load
    def fake_load(p, *a, **k):
        X = real_load(p, *a, **k)
        if str(p).endswith('X.npy'):
            X = X.copy()
            for s,e in zero_cols: X[:,:,s:e]=0.0
        return X
    ab.np.load = fake_load
    try:
        return orig(fold_dir, split, edge_index, dev_lookup, cohort_to_idx, node_perm, None)
    finally:
        ab.np.load = real_load
ab.load_split_for_ablation = patched
CONDS = {
 'string_full': {},
 'empty_full': {'graph':'empty'},
 'string_ctxfree': {'zero_cols':[(2,13)]},
 'empty_ctxfree': {'graph':'empty','zero_cols':[(2,13)]},
}
real = load_edge_index_generic(); dev = pd.read_parquet(SHARED_DIR/'deviation_targets.parquet')
for f in folds:
    fd = W/'folds'/f
    for c in conds:
        setup = setup_fold_ablation(fd, CONDS[c], real, dev)
        for s in seeds:
            rd = OUT/f/c/f'seed_{s}'
            if (rd/'DONE.marker').exists(): continue
            t=time.time(); m = run_single_ablation(setup, c, CONDS[c], s, rd)
            print(f, c, s, round(m['auc'],4), round(m['macro_f1'],4), f'{time.time()-t:.0f}s', flush=True)
