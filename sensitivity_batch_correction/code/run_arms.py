"""Tabular baselines under five preprocessing arms. Feature construction, classifiers,
hyperparameters, threshold rule and metrics are taken verbatim (via ast) from
scripts/baselines_b3.py so that only the input matrices differ.
usage: run_arms.py DATA REPO OUT ARM SEED"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["OMP_NUM_THREADS"] = "1"
import sys, ast, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
D, REPO, OUT, ARM, SEED = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
sys.path.insert(0, str(Path(__file__).parent))
import combat_fw as cf

src = (REPO / "scripts/baselines_b3.py").read_text()
tree = ast.parse(src)
want = {"compute_h1_metrics", "tune_threshold_macro_f1", "build_feature_views", "get_classifiers"}
ns = {}
exec("import numpy as np, pandas as pd\n"
     "from sklearn.preprocessing import StandardScaler\n"
     "from sklearn.linear_model import LogisticRegression\n"
     "from sklearn.ensemble import RandomForestClassifier\n"
     "from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score\n"
     "from xgboost import XGBClassifier\nfrom catboost import CatBoostClassifier\n", ns)
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PATHWAY_COLS" for t in node.targets):
        exec(ast.get_source_segment(src, node), ns)
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(ast.get_source_segment(src, node), ns)
assert want <= ns.keys()

MICRO = {"GSE31803", "GSE48452", "GSE84044", "GSE89632"}
dep = pd.read_parquet(D / "shared/bulk_expression.parquet")
dep_h = pd.read_parquet(D / "shared/healthy_reference.parquet")
raw = pd.read_parquet(D / "shared/bulk_expression_nocombat.parquet")
raw_h = pd.read_parquet(D / "recovered_controls_raw.parquet")
lab = pd.read_parquet(D / "shared/labels.parquet")
ccoh = pd.read_csv(D / "control_cohorts.csv", index_col=0)["cohort"]
panel = pd.read_csv(D / "shared/panel_genes.csv")
coh = pd.concat([lab.cohort, ccoh])
allraw = pd.concat([raw, raw_h], axis=1)


def combat_all(exclude=None):
    """ComBat within each platform. exclude=None: every cohort in the fit (transductive).
    exclude=g: cohort g left out of the fit and adjusted afterwards as a new batch from its
    own (unlabelled) samples; g's control samples are dropped."""
    parts = []
    for micro in (False, True):
        cols = [c for c in allraw.columns if (coh[c] in MICRO) == micro and coh[c] != exclude]
        adj, par = cf.fit(allraw[cols], coh[cols].to_numpy())
        parts.append(adj)
        if exclude is not None and ((exclude in MICRO) == micro):
            tcols = list(lab.index[lab.cohort == exclude])
            parts.append(cf.adjust_new_batch(raw[tcols], par))
    return pd.concat(parts, axis=1)


fold_dirs = sorted(p for p in (D / "folds").iterdir() if p.is_dir())
refit_all = combat_all() if ARM == "transductive_refit" else None
rows = []
for fd in fold_dirs:
    info = json.load(open(fd / "fold_info.json"))
    test = info["test_cohort"]
    train_ctrl = [c for c in ccoh.index if ccoh[c] != test]
    if ARM == "deposited":
        expr, href = dep, dep_h
    elif ARM == "deposited_trainctrl":
        expr, href = dep, dep_h[train_ctrl]
    elif ARM == "transductive_refit":
        expr, href = refit_all[dep.columns], refit_all[dep_h.columns]
    elif ARM == "inductive":
        m = combat_all(exclude=test)
        expr = m[[c for c in dep.columns if c in m.columns]]
        href = m[train_ctrl]
    elif ARM == "no_combat":
        expr, href = raw, raw_h[train_ctrl]
    else:
        raise SystemExit(ARM)
    views, labels = ns["build_feature_views"](fd, expr, href, panel)
    n_pos = max((labels["y_train"] == 1).sum(), 1); n_neg = max((labels["y_train"] == 0).sum(), 1)
    for vname, (Xtr, Xca, Xte) in views.items():
        for cname, clf in ns["get_classifiers"](SEED, n_neg / n_pos).items():
            clf.fit(Xtr, labels["y_train"])
            p_cal = clf.predict_proba(Xca)[:, 1]; p_te = clf.predict_proba(Xte)[:, 1]
            t, _ = ns["tune_threshold_macro_f1"](labels["y_cal"], p_cal)
            auc, f1, bacc = ns["compute_h1_metrics"](labels["y_test"], p_te, (p_te >= t).astype(int))
            rows.append(dict(arm=ARM, seed=SEED, fold=info["fold"], test_cohort=test, model=f"{vname}__{cname}",
                             n_test=len(p_te), n_train_controls=href.shape[1], threshold=t, auc=auc, macro_f1=f1, balanced_acc=bacc))
    print(ARM, SEED, test, flush=True)
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / f"{ARM}_seed{SEED}.csv", index=False)
