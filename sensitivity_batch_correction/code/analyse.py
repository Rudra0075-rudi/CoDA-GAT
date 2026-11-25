import sys, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
OUT, REPO, DEST = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/*_seed*.csv")])
arms = ["deposited", "transductive_refit", "deposited_trainctrl", "inductive", "no_combat"]
df = df[df.arm.isin(arms)]
seeds = sorted(df.seed.unique()); print("seeds", seeds, df.groupby("arm").seed.nunique().to_dict())
m = pd.read_csv(f"{REPO}/results_b7/master_results.csv")
gat = m[m["mode"] == "off"].groupby("test_cohort").auc.mean()

# per cohort, per model, per arm: mean over seeds
pc = df.groupby(["arm", "model", "fold", "test_cohort"]).agg(auc=("auc", "mean"), auc_seed_sd=("auc", "std"),
     macro_f1=("macro_f1", "mean"), balanced_acc=("balanced_acc", "mean"), n_test=("n_test", "first"),
     n_train_controls=("n_train_controls", "first"), n_seeds=("seed", "nunique")).reset_index()
pc["gat_auc"] = pc.test_cohort.map(gat)
base = pc[pc.arm == "deposited"].set_index(["model", "test_cohort"]).auc
pc["delta_vs_deposited"] = pc.auc - pc.set_index(["model", "test_cohort"]).index.map(base)
pc.sort_values(["model", "arm", "fold"]).to_csv(f"{DEST}/batch_correction_sensitivity_per_cohort.csv", index=False)

def wil(x):
    x = np.asarray(x)
    return float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0

rows = []
for (arm, model), g in pc.groupby(["arm", "model"]):
    g = g.sort_values("fold")
    s = df[(df.arm == arm) & (df.model == model)].groupby("seed").auc.mean()
    d = g.delta_vs_deposited.to_numpy(); dg = (g.auc - g.gat_auc).to_numpy()
    rows.append(dict(arm=arm, model=model, mean_auc=g.auc.mean(), between_cohort_sd=g.auc.std(),
        seed42_mean_auc=float(s.get(42, np.nan)), seed_min=s.min(), seed_max=s.max(), seed_sd=s.std(),
        delta_vs_deposited=d.mean(), cohorts_lower_than_deposited=int((d < 0).sum()), wilcoxon_p_vs_deposited=wil(d) if arm != "deposited" else np.nan,
        diff_vs_codagat=dg.mean(), cohorts_above_codagat=int((dg > 0).sum()), wilcoxon_p_vs_codagat=wil(dg),
        mean_macro_f1=g.macro_f1.mean(), mean_balanced_acc=g.balanced_acc.mean()))
summ = pd.DataFrame(rows)
summ["arm"] = pd.Categorical(summ.arm, arms); summ = summ.sort_values(["model", "arm"])
summ.to_csv(f"{DEST}/batch_correction_sensitivity_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(summ.round(4).to_string(index=False))
print("\nCatBoost panel+deviation+pathway, per cohort:")
w = pc[pc.model == "panel72_full__catboost"].pivot(index="test_cohort", columns="arm", values="auc")[arms]
w["codagat"] = gat; print(w.round(4)); print(w.mean().round(4))
print("\nMean change over 12 models vs deposited:")
print(summ.groupby("arm", observed=True).delta_vs_deposited.agg(["mean", "min", "max"]).round(4))
print("best model per arm:"); print(summ.loc[summ.groupby("arm", observed=True).mean_auc.idxmax(), ["arm", "model", "mean_auc", "diff_vs_codagat", "cohorts_above_codagat", "wilcoxon_p_vs_codagat"]].round(4))
