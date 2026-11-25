"""Parametric ComBat (no covariates), written to match inmoose.pycombat_norm 0.9.1
line for line (population variances, ddof=0; same priors; same iteration rule),
split into a fit step on training batches and an apply step for a new batch.

Without covariates, every batch's gamma*/delta* depend only on that batch's own
standardised data; batches interact only through the per-gene grand mean and
pooled variance. 'fit' estimates those two from training batches; 'adjust_new_batch'
standardises a new batch with them and estimates its own gamma*/delta* by the same
empirical Bayes rule, using only that batch's (unlabelled) samples.
"""
import numpy as np
import pandas as pd


def _prior(delta_hat):
    m = np.mean(delta_hat); s2 = np.var(delta_hat)
    return (2 * s2 + m * m) / s2, (m * s2 + m ** 3) / s2


def _it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4, max_it=int(1e6)):
    n = sdat.shape[1]
    t2_n = t2 * n
    t2_n_g_hat = t2_n * g_hat
    g_old, d_old = g_hat.copy(), d_hat.copy()
    change, it = 1.0, 0
    while change > conv and it < max_it:
        g_new = (t2_n_g_hat + d_old * g_bar) / (t2_n + d_old)
        sum2 = np.sum((sdat - g_new[:, None]) ** 2, axis=1)
        d_new = (0.5 * sum2 + b) / (0.5 * n + a - 1)
        change = max(np.max(np.abs(g_new - g_old) / g_old), np.max(np.abs(d_new - d_old) / d_old))
        g_old, d_old = g_new, d_new
        it += 1
    return g_new, d_new


def _eb_adjust_block(s_block):
    """s_block: genes x samples of ONE batch, already standardised."""
    g_hat = s_block.mean(axis=1)
    d_hat = s_block.var(axis=1)            # ddof=0, as inmoose
    g_bar, t2 = np.mean(g_hat), np.var(g_hat)
    a, b = _prior(d_hat)
    g_star, d_star = _it_sol(s_block, g_hat, d_hat, g_bar, t2, a, b)
    return (s_block - g_star[:, None]) / np.sqrt(d_star)[:, None]


def fit(X: pd.DataFrame, batch):
    """X genes x samples (raw), batch labels per column. Returns adjusted X and params."""
    batch = np.asarray(batch)
    D = X.to_numpy(dtype=float)
    labs = list(pd.unique(batch))
    n = D.shape[1]
    means = np.stack([D[:, batch == b].mean(axis=1) for b in labs], 1)   # per-batch OLS = batch mean
    sizes = np.array([(batch == b).sum() for b in labs])
    grand_mean = means @ (sizes / n)
    fitted = np.empty_like(D)
    for j, b in enumerate(labs):
        fitted[:, batch == b] = means[:, [j]]
    var_pooled = ((D - fitted) ** 2).mean(axis=1)
    out = np.empty_like(D)
    for b in labs:
        m = batch == b
        s = (D[:, m] - grand_mean[:, None]) / np.sqrt(var_pooled)[:, None]
        out[:, m] = _eb_adjust_block(s) * np.sqrt(var_pooled)[:, None] + grand_mean[:, None]
    return pd.DataFrame(out, index=X.index, columns=X.columns), dict(grand_mean=grand_mean, var_pooled=var_pooled)


def adjust_new_batch(Xnew: pd.DataFrame, params):
    gm, vp = params["grand_mean"], params["var_pooled"]
    s = (Xnew.to_numpy(dtype=float) - gm[:, None]) / np.sqrt(vp)[:, None]
    out = _eb_adjust_block(s) * np.sqrt(vp)[:, None] + gm[:, None]
    return pd.DataFrame(out, index=Xnew.index, columns=Xnew.columns)
