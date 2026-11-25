"""1) recover uncorrected values of the 148 controls by inverting the per-cohort,
per-gene affine ComBat map (estimated from the 1,366 samples that have both matrices);
2) check our ComBat against inmoose on identical input;
3) check how close an all-cohort refit (without the unlabelled samples that are not in
the deposit) comes to the deposited ComBat matrix."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2])
import combat_fw as cf
D = sys.argv[1]
b = pd.read_parquet(f"{D}/shared/bulk_expression.parquet")
n = pd.read_parquet(f"{D}/shared/bulk_expression_nocombat.parquet")
h = pd.read_parquet(f"{D}/shared/healthy_reference.parquet")
lab = pd.read_parquet(f"{D}/shared/labels.parquet")
num = lambda s: int(s[3:])
rng = lab.assign(g=[num(s) for s in lab.index]).groupby("cohort").g.agg(["min", "max"])
ctrl_cohort = pd.Series({c: [k for k, r in rng.iterrows() if r["min"] - 300 <= num(c) <= r["max"] + 300][0] for c in h.columns})
print(ctrl_cohort.value_counts().to_dict())

raw_ctrl = pd.DataFrame(index=h.index, columns=h.columns, dtype=float)
worst = 0
for g, ids in lab.groupby("cohort").groups.items():
    R, C = n[ids].to_numpy(), b[ids].to_numpy()
    rm, cm = R.mean(1, keepdims=True), C.mean(1, keepdims=True)
    slope = ((R - rm) * (C - cm)).sum(1) / ((R - rm) ** 2).sum(1)
    icpt = cm[:, 0] - slope * rm[:, 0]
    resid = np.abs(C - (slope[:, None] * R + icpt[:, None])).max()
    worst = max(worst, resid)
    cc = ctrl_cohort.index[ctrl_cohort == g]
    if len(cc):
        raw_ctrl[cc] = (h[cc].to_numpy() - icpt[:, None]) / slope[:, None]
print("max residual of affine fit (should be ~float noise):", worst)
assert not raw_ctrl.isna().any().any()
raw_ctrl.to_parquet(f"{D}/recovered_controls_raw.parquet")
ctrl_cohort.rename("cohort").to_csv(f"{D}/control_cohorts.csv")

# 2) implementation check vs inmoose on the RNA-seq block (supervised + controls)
from inmoose.pycombat import pycombat_norm
allraw = pd.concat([n, raw_ctrl], axis=1)
coh = pd.concat([lab.cohort, ctrl_cohort])
plat = coh.map(lambda g: "microarray" if g in ("GSE31803", "GSE48452", "GSE84044", "GSE89632") else "rnaseq")
for p in ["rnaseq", "microarray"]:
    cols = plat.index[plat == p]
    ours, _ = cf.fit(allraw[cols], coh[cols].to_numpy())
    ref = pycombat_norm(allraw[cols], coh[cols].to_numpy())
    print(p, "ours vs inmoose max abs diff:", float(np.abs(ours.to_numpy() - np.asarray(ref)).max()))
    dep = pd.concat([b, h], axis=1)[cols]
    d = np.abs(ours.to_numpy() - dep.to_numpy())
    print(p, "all-cohort refit vs deposited ComBat: max", float(d.max()), "mean", float(d.mean()),
          "| SD of deposited values", float(dep.to_numpy().std()))
