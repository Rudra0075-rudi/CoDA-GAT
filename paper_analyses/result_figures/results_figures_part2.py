# Part 2 of results_figures.py (executed in its namespace): figures p11-p26.

COND = ["real_graph", "shuffled_graph", "random_graph", "e2e_graph", "empty_graph", "coexpr_graph"]
COND_LABEL = {"real_graph": "STRING (CoDA-GAT)", "shuffled_graph": "Feature-permuted STRING",
              "random_graph": "Random directed graph", "e2e_graph": "Learned: end-to-end",
              "coexpr_graph": "Learned: co-expression", "empty_graph": "Disconnected (no edges)"}
COND_COLOR = {"real_graph": S.BLUE, "shuffled_graph": S.TEAL, "random_graph": S.VIOLET,
              "e2e_graph": S.GOLD, "coexpr_graph": S.RED, "empty_graph": S.GREY}
P13_NAME = {"STRING (CoDA-GAT)": "real_graph", "feature-permuted STRING": "shuffled_graph",
            "random directed": "random_graph", "learned: co-expression": "coexpr_graph",
            "learned: end-to-end": "e2e_graph"}
METRIC_NICE = {"auc": "AUROC", "macro_f1": "Macro-F1", "balanced_acc": "Balanced accuracy"}


def masking_raw():
    d = pd.read_csv(GUTIL / "masking_curves.csv")
    lp = LEARNED / "masking_curves_all_conditions.csv"
    d = pd.concat([d, pd.read_csv(lp)], ignore_index=True).drop_duplicates(
        subset=["model", "fold", "seed", "mask_rate", "draw"])
    return d


def levels(d, metric):
    return d.groupby(["model", "fold", "mask_rate"])[metric].mean().unstack("model")


def drop_contrast(d, metric, a, b="empty_graph"):
    """per (fold, rate>0): loss of b minus loss of a, each from its own unmasked level."""
    lv = levels(d, metric)
    out = []
    for (fold, rate), r in lv.iterrows():
        if rate == 0:
            continue
        base = lv.loc[(fold, 0.0)]
        out.append({"fold": fold, "mask_rate": rate,
                    "delta": (base[b] - r[b]) - (base[a] - r[a])})
    return pd.DataFrame(out)


@fig(11)
def p11():
    ref = pd.read_csv(FIGIN / "fig_p11_data.csv")
    d = masking_raw()
    rows = []
    for metric in ("auc", "macro_f1", "balanced_acc"):
        dc = drop_contrast(d, metric, "real_graph")
        for rate, g in dc.groupby("mask_rate"):
            m, lo, hi = t_ci(g["delta"])
            rows.append({"metric": metric, "mask_rate": rate, "mean": m, "lo": lo, "hi": hi, "n": len(g),
                         "p_wilcoxon": stats.wilcoxon(g["delta"]).pvalue})
    t = pd.DataFrame(rows)
    mg = t.merge(ref, on=["metric", "mask_rate"], suffixes=("", "_ref"))
    err = max((mg["mean"] - mg["mean_ref"]).abs().max(), (mg["lo"] - mg["lo_ref"]).abs().max(), (mg["hi"] - mg["hi_ref"]).abs().max())
    print(f"   p11 check: {len(mg)} cells, max |recomputed - figure_data| = {err:.2e}")
    assert err < 1e-9
    f, axes = plt.subplots(1, 3, figsize=(S.TEXTW, 2.35), sharey=True)
    for ax, metric in zip(axes, ("auc", "macro_f1", "balanced_acc")):
        g = t[t["metric"] == metric].sort_values("mask_rate")
        x = g["mask_rate"].values * 100
        ax.axhline(0, color=S.INK, lw=0.7, zorder=1)
        excl = (g["hi"] < 0) | (g["lo"] > 0)
        ax.vlines(x, g["lo"], g["hi"], color=S.BLUE, lw=1.0, zorder=2)
        ax.plot(x[excl.values], g["mean"].values[excl.values], "o", ms=4, color=S.BLUE, zorder=3)
        ax.plot(x[~excl.values], g["mean"].values[~excl.values], "o", ms=4, mfc="white", mec=S.BLUE, mew=1.0, zorder=3)
        ax.set_title(METRIC_NICE[metric], loc="left")
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_xlim(4, 56)
    axes[0].set_ylabel("Contrast (disconnected loss\nminus STRING loss)")
    axes[0].text(0.02, 0.03, "STRING degrades more", transform=axes[0].transAxes, fontsize=6.2, color=S.GREY, va="bottom")
    axes[0].text(0.02, 0.97, "STRING degrades less", transform=axes[0].transAxes, fontsize=6.2, color=S.GREY, va="top")
    axes[2].legend(handles=[Line2D([], [], marker="o", ls="", color=S.BLUE, label="95% interval excludes 0"),
                            Line2D([], [], marker="o", ls="", mfc="white", mec=S.BLUE, label="includes 0")],
                   loc="upper left", bbox_to_anchor=(0.0, 1.0))
    f.supxlabel("Genes masked (%)", fontsize=7.5, y=0.0)
    f.subplots_adjust(left=0.13, right=0.99, bottom=0.18, top=0.9, wspace=0.12)
    return f, t


@fig(12)
def p12():
    d = masking_raw()
    lv = levels(d, "auc")
    rows = []
    for c in COND:
        s = lv[c].groupby(level="mask_rate").agg(["mean", "std", "count"])
        s["se"] = s["std"] / np.sqrt(s["count"])
        for rate, r in s.iterrows():
            rows.append({"condition": c, "mask_rate": rate, "mean": r["mean"], "lo": r["mean"] - 1.96 * r["se"],
                         "hi": r["mean"] + 1.96 * r["se"], "n_cohorts": int(r["count"])})
    t = pd.DataFrame(rows)
    print("   p12 unmasked:", {COND_LABEL[c]: round(t[(t.condition == c) & (t.mask_rate == 0)]["mean"].iloc[0], 3) for c in COND})
    f, axes = plt.subplots(2, 3, figsize=(S.TEXTW, 3.35), sharex=True, sharey=True)
    ref = t[t["condition"] == "real_graph"]
    for ax, c in zip(axes.flat, COND):
        g = t[t["condition"] == c]
        x = g["mask_rate"] * 100
        if c != "real_graph":
            ax.plot(ref["mask_rate"] * 100, ref["mean"], color=S.BLUE, lw=0.9, ls=(0, (3, 1.5)), zorder=2)
        ax.fill_between(x, g["lo"], g["hi"], color=COND_COLOR[c], alpha=0.16, lw=0, zorder=1)
        ax.plot(x, g["mean"], "-o", color=COND_COLOR[c], ms=2.6, lw=1.2, zorder=3)
        ax.set_title(COND_LABEL[c], loc="left", color=COND_COLOR[c] if c != "empty_graph" else S.GREY_D)
        ax.text(0.97, 0.95, f"{g['mean'].iloc[0]:.3f} → {g['mean'].iloc[-1]:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.2, color=S.INK)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
    axes[0, 0].legend(handles=[Line2D([], [], color=S.BLUE, lw=1.2, label="Condition: mean"),
                               Patch(color=S.BLUE, alpha=0.16, label="±1.96 SE"),
                               Line2D([], [], color=S.BLUE, lw=0.9, ls=(0, (3, 1.5)), label="STRING (reference)")],
                      loc="lower left", fontsize=6.0, handlelength=1.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC")
    f.supxlabel("Genes masked (%)", fontsize=7.5, y=0.0)
    f.subplots_adjust(left=0.1, right=0.99, bottom=0.11, top=0.94, wspace=0.08, hspace=0.3)
    return f, t


@fig(13)
def p13():
    ref = pd.read_csv(FIGIN / "fig_p13_data.csv")
    d = masking_raw()
    rows = []
    for c in COND:
        if c == "empty_graph":
            continue
        for metric in ("auc", "macro_f1"):
            dc = drop_contrast(d, metric, c)
            per_fold = dc.groupby("fold")["delta"].mean()
            m, lo, hi = t_ci(per_fold)
            rows.append({"condition": c, "panel": "masking", "metric": metric, "mean": m, "lo": lo, "hi": hi})
        lv0 = levels(d, "auc").xs(0.0, level="mask_rate")
        m, lo, hi = t_ci(lv0[c] - lv0["empty_graph"])
        rows.append({"condition": c, "panel": "clean", "metric": "auc", "mean": m, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    r2 = ref.assign(condition=ref["cond"].map(P13_NAME), metric=ref["metric"].map({"AUROC": "auc", "macro-F1": "macro_f1"}))
    mg = t.merge(r2, on=["condition", "panel", "metric"], suffixes=("", "_ref"))
    err = max((mg["mean"] - mg["mean_ref"]).abs().max(), (mg["lo"] - mg["lo_ref"]).abs().max())
    print(f"   p13 check: {len(mg)} of {len(ref)} cells matched, max |recomputed - figure_data| = {err:.2e}")
    print(mg.assign(e=(mg["mean"]-mg["mean_ref"]).abs()).sort_values("e").tail(2)[["condition","panel","metric","e"]].to_string())
    assert len(mg) == len(ref) and err < 1e-6

    order = [c for c in COND if c != "empty_graph"]
    y = np.arange(len(order))
    f, axes = plt.subplots(1, 2, figsize=(S.TEXTW, 2.25), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for k, (metric, mk, col, off) in enumerate((("auc", "o", S.BLUE, -0.14), ("macro_f1", "s", S.GREEN, 0.14))):
        g = t[(t.panel == "masking") & (t.metric == metric)].set_index("condition").reindex(order)
        ax.hlines(y + off, g["lo"], g["hi"], color=col, lw=1.0)
        ax.plot(g["mean"], y + off, mk, ms=3.8, color=col, label=METRIC_NICE[metric])
    ax.axvline(0, color=S.INK, lw=0.7)
    ax.set_title("Under masking (10–50%)", loc="left", pad=12)
    ax.set_xlabel("Advantage over disconnected")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.1, borderaxespad=0.0)
    ax.set_yticks(y); ax.set_yticklabels([COND_LABEL[c] for c in order])
    ax.invert_yaxis()
    only_bottom(ax)
    ax = axes[1]
    g = t[t.panel == "clean"].set_index("condition").reindex(order)
    ax.hlines(y, g["lo"], g["hi"], color=S.BLUE, lw=1.0)
    ax.plot(g["mean"], y, "o", ms=3.8, color=S.BLUE)
    ax.axvline(0, color=S.INK, lw=0.7)
    ax.set_title("Unmasked data", loc="left", pad=12)
    ax.set_xlabel("AUROC difference from disconnected")
    only_bottom(ax)
    for a in axes:
        lim = max(abs(np.array(a.get_xlim())))
        a.set_xlim(-lim, lim)
    f.subplots_adjust(left=0.24, right=0.99, bottom=0.2, top=0.9, wspace=0.08)
    return f, t


def string_pairs():
    e = pd.read_csv(SHARED / "graph_edges.csv")
    return {frozenset((a, b)) for a, b in zip(e["src"], e["dst"]) if a != b}


def learned_edges():
    out = {}
    for p in sorted((LEARNED / "edges").glob("learned_edges_fold*_*.csv")):
        fold = int(p.name.split("_fold")[1].split("_")[0])
        var = p.name.split(f"fold{fold}_")[1].replace(".csv", "")
        e = pd.read_csv(p)
        out.setdefault(var, {})[fold] = {frozenset((a, b)) for a, b in zip(e["src"], e["dst"]) if a != b}
    return out


def jac(a, b):
    return len(a & b) / len(a | b)


@fig(14)
def p14():
    cl = fold_cohort()
    st = string_pairs()
    ed = learned_edges()
    n_genes = len(pd.read_csv(SHARED / "panel_genes.csv"))
    n_pairs = n_genes * (n_genes - 1) // 2
    sizes = [len(s) for v in ed.values() for s in v.values()] + [len(st)]
    k = float(np.mean(sizes))
    exp_inter = k * k / n_pairs
    chance = exp_inter / (2 * k - exp_inter)
    rows = []
    for v in ("e2e_graph", "coexpr_graph"):
        for fold, s in sorted(ed[v].items()):
            rows.append({"variant": v, "kind": "vs_STRING", "fold": fold, "cohort": cl[fold], "n_edges": len(s),
                         "shared_with_string": len(s & st), "jaccard": jac(s, st)})
        for a, b in itertools.combinations(sorted(ed[v]), 2):
            rows.append({"variant": v, "kind": "cross_fold", "fold": f"{a}-{b}", "jaccard": jac(ed[v][a], ed[v][b])})
    t = pd.DataFrame(rows)
    for v in ("e2e_graph", "coexpr_graph"):
        a = t[(t.variant == v) & (t.kind == "vs_STRING")]
        b = t[(t.variant == v) & (t.kind == "cross_fold")]
        print(f"   p14 {v}: edges {a.n_edges.mean():.1f}, shared {a.shared_with_string.mean():.1f}, J {a.jaccard.mean():.3f}, cross-fold J {b.jaccard.mean():.3f}")
    print(f"   p14 STRING edges {len(st)}, chance J {chance:.3f}, expected shared {exp_inter:.1f}")
    t.attrs["chance"] = chance

    VAR = {"e2e_graph": ("Learned: end-to-end", S.GOLD), "coexpr_graph": ("Learned: co-expression", S.RED)}
    f, axes = plt.subplots(1, 2, figsize=(S.TEXTW, 2.45), gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axes[0]
    for i, v in enumerate(("coexpr_graph", "e2e_graph")):
        a = t[(t.variant == v) & (t.kind == "vs_STRING")]
        rng = np.random.default_rng(i)
        ax.plot(a["jaccard"], i + rng.uniform(-0.12, 0.12, len(a)), "o", ms=3.4, color=VAR[v][1], alpha=0.85, mec="white", mew=0.3)
        ax.plot([a["jaccard"].mean()] * 2, [i - 0.26, i + 0.26], color=S.INK, lw=1.3)
        ax.text(a["jaccard"].mean(), i - 0.34, f"{a['jaccard'].mean():.3f}", ha="center", va="bottom", fontsize=6.3)
    ax.axvline(chance, color=S.GREY, lw=0.8, ls=(0, (3, 2)))
    ax.text(chance, 1.62, f"chance {chance:.3f}", fontsize=6.2, color=S.GREY, ha="center", va="bottom")
    ax.set_yticks([0, 1]); ax.set_yticklabels([VAR["coexpr_graph"][0], VAR["e2e_graph"][0]])
    ax.set_ylim(1.75, -0.55)
    ax.set_xlim(0, 0.2)
    ax.set_xlabel("Jaccard with STRING (points: folds)")
    ax.set_title("Overlap with STRING", loc="left")
    only_bottom(ax)
    ax = axes[1]
    for i, v in enumerate(("coexpr_graph", "e2e_graph")):
        b = t[(t.variant == v) & (t.kind == "cross_fold")]
        rng = np.random.default_rng(10 + i)
        ax.plot(b["jaccard"], i + rng.uniform(-0.15, 0.15, len(b)), "o", ms=2.6, color=VAR[v][1], alpha=0.75, mec="white", mew=0.25)
        ax.plot([b["jaccard"].mean()] * 2, [i - 0.26, i + 0.26], color=S.INK, lw=1.3)
        ax.text(b["jaccard"].mean(), i - 0.34, f"{b['jaccard'].mean():.2f}", ha="center", va="bottom", fontsize=6.3)
    ax.axvline(1.0, color=S.BLUE, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.985, 1.62, "STRING: 1 (fixed)", fontsize=6.2, color=S.BLUE, ha="right", va="bottom")
    ax.set_yticks([0, 1]); ax.set_yticklabels([])
    ax.set_ylim(1.75, -0.55)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Jaccard between folds (points: fold pairs)")
    ax.set_title("Stability across folds", loc="left")
    only_bottom(ax)
    f.subplots_adjust(left=0.2, right=0.98, bottom=0.18, top=0.9, wspace=0.16)
    t["chance_jaccard"] = chance
    return f, t


@fig(15)
def p15():
    a = pd.read_csv(ATTN / "per_fold_gene_attention.csv", index_col=0)
    pg = pd.read_csv(SHARED / "panel_genes.csv")
    ecm = {g for g, p in zip(pg["gene"], pg["pathways"]) if "ECM" in str(p)}
    mean, sd = a.mean(axis=0), a.std(axis=0)
    top = mean.sort_values(ascending=False).head(15)
    n_ecm = sum(g in ecm for g in top.index)
    print(f"   p15 check: {len(a)} folds, top-15 ECM genes = {n_ecm}, ECM in panel = {len(ecm)} of {len(pg)}; top7 {list(top.index[:7])}")
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.7))
    y = np.arange(len(top))
    cols = [S.BLUE if g in ecm else S.GREY for g in top.index]
    for yi, g, c in zip(y, top.index, cols):
        ax.hlines(yi, top[g] - sd[g], top[g] + sd[g], color=c, lw=0.9, alpha=0.6)
        ax.hlines(yi, 0, top[g], color=c, lw=0.6, alpha=0.25)
        ax.plot(top[g], yi, "o", ms=4.4, color=c)
    ax.set_yticks(y); ax.set_yticklabels(top.index)
    for t_, g in zip(ax.get_yticklabels(), top.index):
        t_.set_color(S.BLUE if g in ecm else S.INK)
    ax.invert_yaxis()
    ax.set_xlim(0, None)
    ax.set_xlabel("Summed layer-2 attention, mean over held-out rounds (line: ± SD across rounds)")
    only_bottom(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.BLUE, label=f"ECM remodelling gene ({n_ecm} of 15)"),
                       Line2D([], [], marker="o", ls="", color=S.GREY, label="Other pathway")], loc="lower right")
    d = pd.DataFrame({"gene": top.index, "mean": top.values, "sd": sd[top.index].values, "ecm": [g in ecm for g in top.index]})
    return f, d


@fig(16)
def p16():
    cl = fold_cohort()
    c = pd.read_csv(ATTN / "cross_fold_attention_correlation.csv", index_col=0)
    names = [cl[int(i)] for i in c.columns]
    M = c.values.astype(float)
    off = M[np.triu_indices_from(M, 1)]
    print(f"   p16 check: mean off-diagonal Spearman = {off.mean():.4f} (min {off.min():.3f}, max {off.max():.3f})")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#FFFFFF", S.BLUE_XL, S.BLUE_L, S.BLUE2, S.BLUE])
    vmin, vmax = 0.85, 1.0
    f, ax = plt.subplots(figsize=(135 / 25.4, 3.0))
    f._fixed_canvas = True
    f.subplots_adjust(left=0.24, right=0.70, top=0.98, bottom=0.2)
    n = len(names)
    for i in range(n):
        for j in range(i):
            v = M[i, j]
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap((v - vmin) / (vmax - vmin)), edgecolor="white", lw=0.8))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color="white" if v > 0.955 else S.INK)
    ax.set_xlim(-0.5, n - 1.5); ax.set_ylim(n - 0.5, 0.5)
    ax.set_xticks(range(n - 1)); ax.set_xticklabels(names[:-1], rotation=35, ha="right")
    ax.set_yticks(range(1, n)); ax.set_yticklabels(names[1:])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.text(n - 1.6, 1.0, f"Mean ρ = {off.mean():.3f}\n{len(off)} pairs of rounds", ha="right", va="top", fontsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cax = f.add_axes([0.73, 0.45, 0.015, 0.45])
    cb = f.colorbar(sm, cax=cax)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6, length=2); cb.set_label("Spearman ρ", fontsize=6.5)
    d = pd.DataFrame(M, index=names, columns=names).reset_index().rename(columns={"index": "cohort"})
    return f, d


MODEL_NICE = {
    "codagat_off": "CoDA-GAT, adversary off", "codagat_marginal": "CoDA-GAT, marginal", "codagat_conditional": "CoDA-GAT, class-cond.",
    "empty_graph": "Disconnected graph", "random_graph": "Random graph", "shuffled_graph": "Feature-permuted STRING",
    "no_pathway_features": "No pathway features", "no_fold_fitted_features": "No fold-fitted features", "no_h3_deviation": "No deviation head",
}
MODEL_NICE.update({k: v for k, v in NICE.items()})
FAMILY_COL = {"gat": S.BLUE, "linear": S.GREEN, "tree": S.GOLD}
FAMILY_NICE = {"gat": "Graph attention", "linear": "Linear", "tree": "Tree ensemble"}


def family_of(m):
    if m.startswith(("codagat", "empty", "random_graph", "shuffled", "no_")):
        return "gat"
    if m.endswith("logreg"):
        return "linear"
    return "tree"


@fig(17)
def p17():
    M = pd.read_csv(CORR / "fig17_error_agreement_heatmap.csv", index_col=0)
    assert list(M.index) == list(M.columns)
    fams = [family_of(m) for m in M.index]
    print(f"   p17: {len(M)} models; families", pd.Series(fams).value_counts().to_dict())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#FFFFFF", S.BLUE_XL, S.BLUE_L, S.BLUE2, S.BLUE])
    n = len(M)
    f, ax = plt.subplots(figsize=(135 / 25.4, 3.75))
    f._fixed_canvas = True
    f.subplots_adjust(left=0.30, right=0.97, top=0.995, bottom=0.035)
    V = M.values.astype(float)
    for i in range(n):
        for j in range(i):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(V[i, j]), edgecolor="white", lw=0.3))
    b = [i for i in range(1, n) if fams[i] != fams[i - 1]]
    for k in b:
        ax.plot([-0.5, k - 0.5], [k - 0.5, k - 0.5], color=S.INK, lw=0.6)
        ax.plot([k - 0.5, k - 0.5], [k - 0.5, n - 0.5], color=S.INK, lw=0.6)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{MODEL_NICE.get(m, m)}  {i+1:>2d}" for i, m in enumerate(M.index)], fontsize=5.9)
    ax.set_xticks(range(n)); ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=5.6)
    for t_, fm in zip(ax.get_yticklabels(), fams):
        t_.set_color(FAMILY_COL[fm])
    for t_, fm in zip(ax.get_xticklabels(), fams):
        t_.set_color(FAMILY_COL[fm])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0, pad=1.5)
    cax = ax.inset_axes([0.62, 0.62, 0.32, 0.025])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = f.colorbar(sm, cax=cax, orientation="horizontal"); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=2); cb.set_label("Jaccard overlap of error sets", fontsize=6.5)
    ax.legend(handles=[Patch(color=FAMILY_COL[k], label=FAMILY_NICE[k]) for k in ("gat", "linear", "tree")],
              loc="upper right", bbox_to_anchor=(0.98, 0.98), title="Model family", title_fontsize=6.5, alignment="left")
    return f, M.reset_index().rename(columns={"index": "model"})


@fig(18)
def p18():
    d = pd.read_csv(CORR / "fig18_family_agreement_vs_chance.csv")
    d["lab"] = d["label"].str.replace("\n", " ")
    rel = []
    for l in d["lab"]:
        if l.startswith("all between"):
            rel.append("All between-family pairs" + (" (pair-weighted)" if "model-pair" in l else " (family-weighted)"))
        else:
            a, _, rest = l.partition(" vs ")
            b, _, n = rest.partition(" (")
            nice = {"gat": "Graph", "linear": "Linear", "tree": "Tree"}
            rel.append(f"{nice[a]} – {nice[b]} ({n}")
    d["relation"] = rel
    d["within"] = [("–" in r) and (r.split(" – ")[0] == r.split(" – ")[1].split(" (")[0]) for r in rel]
    order = [1, 0] + [i for i in range(2, len(d)) if not d["within"][i]] + [i for i in range(2, len(d)) if d["within"][i]]
    order = [0, 1] + [i for i in range(2, len(d)) if not d["within"][i]] + [i for i in range(2, len(d)) if d["within"][i]]
    d = d.iloc[order].reset_index(drop=True)
    ypos = []
    yy = 0
    for i in range(len(d)):
        if i in (2, 2 + sum(~d["within"][2:])):
            yy += 0.6
        ypos.append(yy); yy += 1
    ypos = np.array(ypos)
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.75))
    for yi, (_, r) in zip(ypos, d.iterrows()):
        ax.hlines(yi, r["mean_jaccard_perm_null_mean"], r["mean_jaccard_observed"], color=S.GREY_XL, lw=2.4, zorder=1)
        ax.plot(r["mean_jaccard_expected_chance"], yi, "|", ms=7, mew=1.3, color=S.GREY, zorder=2)
        ax.plot(r["mean_jaccard_perm_null_mean"], yi, "o", ms=3.6, mfc="white", mec=S.GREY_D, mew=0.9, zorder=3)
        ax.plot(r["mean_jaccard_observed"], yi, "o", ms=4.6, color=S.RED, zorder=4)
        ax.text(r["mean_jaccard_observed"] + 0.015, yi, f"{r['mean_jaccard_observed']:.3f}", va="center", fontsize=6.3)
    ax.set_yticks(ypos); ax.set_yticklabels(d["relation"])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Mean Jaccard overlap of error sets")
    only_bottom(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.RED, ms=4.6, label="Observed"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=S.GREY_D, label="Within-cohort permutation null"),
                       Line2D([], [], marker="|", ls="", color=S.GREY, ms=7, mew=1.3, label="Independent errors")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    return f, d.drop(columns=["label"])


def stage_panels(csv, grp_a, lab_a, lab_b):
    d = pd.read_csv(csv, index_col=0)
    stg = ["F0", "F1", "F2", "F3", "F4"]
    ex = d[[f"exact_{s}" for s in stg]]
    ex.columns = stg
    share = ex.div(ex.sum(axis=1), axis=0) * 100
    ga, gb = d.index[0], d.index[1]
    grouped_tot = d[["n_grouped_F0_F1", "n_grouped_F3_F4"]].sum()
    gshare = d.loc[ga, ["n_grouped_F0_F1", "n_grouped_F3_F4"]] / grouped_tot * 100
    f, axes = plt.subplots(1, 2, figsize=(S.TEXTW, 2.5), gridspec_kw={"width_ratios": [2.6, 1]})
    ax = axes[0]
    x = np.arange(5)
    ax.axvspan(0.5, 2.5, color=S.GREY_XL, lw=0, zorder=0)
    ax.text(1.5, 1.0, "F1/F2 boundary", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.2, color=S.GREY)
    w = 0.36
    na, nb = int(ex.loc[ga].sum()), int(ex.loc[gb].sum())
    ba = ax.bar(x - w / 2, share.loc[ga], w, color=S.RED, label=f"{lab_a} (n = {na})", zorder=2)
    bb = ax.bar(x + w / 2, share.loc[gb], w, color=S.GREY_L, label=f"{lab_b} (n = {nb})", zorder=2)
    for bars, cnt in ((ba, ex.loc[ga]), (bb, ex.loc[gb])):
        for b_, c_ in zip(bars, cnt):
            ax.annotate(f"{b_.get_height():.0f}", (b_.get_x() + b_.get_width() / 2, b_.get_height()), xytext=(0, 1.5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=5.9)
    ax.set_xticks(x); ax.set_xticklabels(stg)
    ax.set_xlabel("Recorded stage (exact-stage records)")
    ax.set_ylabel("Share of group (%)")
    ax.set_ylim(0, max(share.values.max() * 1.18, 10))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax = axes[1]
    xs = np.arange(2)
    bars = ax.bar(xs, gshare.values, 0.55, color=S.RED, zorder=2)
    for b_, num, den in zip(bars, d.loc[ga, ["n_grouped_F0_F1", "n_grouped_F3_F4"]], grouped_tot):
        ax.annotate(f"{num}/{den}", (b_.get_x() + b_.get_width() / 2, b_.get_height()), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(xs); ax.set_xticklabels(["F0–F1", "F3–F4"])
    ax.set_xlabel("Grouped records")
    ax.set_ylabel(f"{lab_a} (% of group)")
    ax.set_ylim(0, gshare.max() * 1.25)
    ax.set_xlim(-0.6, 1.6)
    f.subplots_adjust(left=0.09, right=0.99, bottom=0.19, top=0.9, wspace=0.35)
    out = share.copy(); out["n_exact"] = ex.sum(axis=1)
    out["grouped_F0_F1"] = d["n_grouped_F0_F1"]; out["grouped_F3_F4"] = d["n_grouped_F3_F4"]
    return f, out.reset_index().rename(columns={"index": "group"}), ex


@fig(19)
def p19():
    f, d, ex = stage_panels(CORR / "fig19_consensus_failure_stages.csv", "consensus failure", "Consensus failures", "Other records")
    print(f"   p19 check: consensus exact {int(ex.iloc[0].sum())}, F1+F2 {int(ex.iloc[0][['F1','F2']].sum())}; others {int(ex.iloc[1].sum())}, F1+F2 {int(ex.iloc[1][['F1','F2']].sum())}")
    return f, d


@fig(20)
def p20():
    e = pd.read_csv(CORR / "fig20_error_rate_all_models.csv", index_col=0)["error_rate"].sort_values()
    fams = [family_of(m) for m in e.index]
    f, ax = plt.subplots(figsize=(S.TEXTW, 3.6))
    y = np.arange(len(e))
    for yi, (m, v), fm in zip(y, e.items(), fams):
        ax.hlines(yi, 0, v * 100, color=FAMILY_COL[fm], lw=0.6, alpha=0.25)
        ax.plot(v * 100, yi, "o", ms=4, color=FAMILY_COL[fm])
        ax.text(v * 100 + 0.25, yi, f"{v*100:.1f}%", va="center", fontsize=5.9)
    ax.set_yticks(y); ax.set_yticklabels([MODEL_NICE.get(m, m) for m in e.index], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 34)
    ax.set_xlabel(f"Error rate on the {1064:,} held-out samples (%)")
    only_bottom(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FAMILY_COL[k], label=FAMILY_NICE[k]) for k in ("gat", "linear", "tree")],
              loc="lower left", bbox_to_anchor=(0.0, 0.0))
    print(f"   p20 check: {len(e)} models, range {e.min()*100:.1f}-{e.max()*100:.1f}%")
    return f, pd.DataFrame({"model": e.index, "label": [MODEL_NICE.get(m, m) for m in e.index], "family": fams, "error_rate": e.values})


@fig(21)
def p21():
    f, d, ex = stage_panels(CORR / "fig21_testA_flagged_stage_distribution.csv", "flagged", "Flagged", "Not flagged")
    print(f"   p21 check: flagged exact {int(ex.iloc[0].sum())}, F1+F2 {int(ex.iloc[0][['F1','F2']].sum())}; unflagged {int(ex.iloc[1].sum())}, F1+F2 {int(ex.iloc[1][['F1','F2']].sum())}")
    return f, d


@fig(22)
def p22():
    d = pd.read_csv(TB / "threshold_sweep_extended.csv").sort_values("pooled_flagging_rate")
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.5))
    ax.axhline(1.0, color=S.GREY, lw=0.7, ls=(0, (1, 1.5)))
    ax.text(27.5, 1.03, "no enrichment (OR = 1)", fontsize=6.2, color=S.GREY, ha="right", va="bottom")
    ax.plot(d["pooled_flagging_rate"] * 100, d["odds_ratio"], "-o", color=S.GREY_D, ms=3.4, lw=1.0)
    pred = float((d["pooled_flagging_rate"] + d["abs_diff_from_predicted_rate"]).round(4).mode().iloc[0])
    print(f"   p22 predicted label-error rate recovered from the sweep file: {pred*100:.1f}%")
    ax.axvline(pred * 100, color=S.GREY, lw=0.7, ls=(0, (3, 2)))
    ax.text(pred * 100 + 0.3, 0.15, f"predicted label-error rate {pred*100:.1f}%", fontsize=6.2, color=S.GREY, va="bottom")
    for _, r in d.iterrows():
        if int(r["uncertainty_percentile"]) in (15, 25, 30):
            continue
        ax.annotate(f"p{int(r['uncertainty_percentile'])}", (r["pooled_flagging_rate"] * 100, r["odds_ratio"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=5.6, color=S.GREY)
    ax.set_xlabel("Samples flagged (%)")
    ax.set_ylabel("Boundary enrichment\n(odds ratio)")
    ax.set_ylim(-0.1, max(d["odds_ratio"]) * 1.15)
    ax.set_xlim(-0.5, 28)
    ax.text(0.99, 0.97, "Historical analysis (grouped stages coded as exact)", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.4, color=S.RED, style="italic")
    return f, d


@fig(23)
def p23():
    cl = fold_cohort()
    ent = json.loads((TC / "per_fold_deltas.json").read_text())

    def pick(comp):
        m = [e for e in ent if e["comparison"] == comp and e["metric"] == "macro_f1" and e["operating_point"] == "p70"]
        assert len(m) == 1
        return m[0]["per_fold_deltas"]
    cle, rnd = pick("cleaned vs baseline"), pick("random_draw0 vs baseline")
    folds = sorted(int(k) for k in cle)
    d = pd.DataFrame({"fold": folds, "cohort": [cl[k] for k in folds],
                      "cleaned": [cle[str(k)] for k in folds], "random": [rnd[str(k)] for k in folds]})
    print(f"   p23 check (p70, primary): cleaned mean {d.cleaned.mean():+.4f}, random mean {d.random.mean():+.4f}, cleaned-random Wilcoxon p = {stats.wilcoxon(d.cleaned - d.random).pvalue:.4f}")
    rows = d.copy()
    rows.loc[len(rows)] = [np.nan, "Mean", d.cleaned.mean(), d.random.mean()]
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.75))
    y = np.r_[np.arange(len(folds)), len(folds) + 0.6]
    for yi, (_, r) in zip(y, rows.iterrows()):
        ax.hlines(yi, min(r.cleaned, r.random), max(r.cleaned, r.random), color=S.GREY_L, lw=0.9, zorder=1)
        big = r["cohort"] == "Mean"
        ax.plot(r.random, yi, "o", ms=5 if big else 4, mfc="white", mec=S.GREY_D, mew=1.0, zorder=2)
        ax.plot(r.cleaned, yi, "o", ms=5 if big else 4, color=S.BLUE, zorder=3)
    ax.axvline(0, color=S.INK, lw=0.7)
    ax.axhline(len(folds) - 0.2, color=S.GREY_XL, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(rows["cohort"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Change in held-out macro-F1 versus no removal")
    only_bottom(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.BLUE, label="Flagged samples removed"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=S.GREY_D, label="Equal-size random removal")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return f, rows


@fig(24)
def p24():
    a = pd.read_csv(B9 / "master_ablation_results.csv")
    b = pd.read_csv(B7 / "master_results.csv")
    base = b[(b["mode"] == "off") & (b["m"] == 0)].groupby("fold")[["auc", "macro_f1"]].mean()
    order = ["empty_graph", "shuffled_graph", "random_graph", "no_h3_deviation", "no_fold_fitted_features", "no_pathway_features"]
    lab = {"empty_graph": "Disconnected graph", "shuffled_graph": "Feature-permuted STRING", "random_graph": "Random directed graph",
           "no_h3_deviation": "No deviation head", "no_fold_fitted_features": "No fold-fitted features", "no_pathway_features": "No pathway features"}
    rows = []
    for ab in order:
        per = a[a["ablation"] == ab].groupby("fold")[["auc", "macro_f1"]].mean()
        j = per.add_suffix("_abl").join(base.add_suffix("_base"))
        assert len(j.dropna()) == 8
        for metric in ("auc", "macro_f1"):
            dd = j[f"{metric}_abl"] - j[f"{metric}_base"]
            m, lo, hi = t_ci(dd)
            rows.append({"ablation": ab, "label": lab[ab], "metric": metric, "mean": m, "lo": lo, "hi": hi,
                         "sd": dd.std(ddof=1), "p_wilcoxon": stats.wilcoxon(dd).pvalue})
    t = pd.DataFrame(rows)
    print("   p24:", {(r.ablation, r.metric): (round(r["mean"], 3), round(r.p_wilcoxon, 3)) for _, r in t.iterrows()})
    f, axes = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), sharey=True)
    ypos = np.array([0, 1, 2, 3.5, 4.5, 5.5])
    for ax, metric in zip(axes, ("auc", "macro_f1")):
        g = t[t.metric == metric].set_index("ablation").reindex(order)
        cols = [S.BLUE2] * 3 + [S.RED] * 3
        for yi, (_, r), c in zip(ypos, g.iterrows(), cols):
            ax.hlines(yi, r["lo"], r["hi"], color=c, lw=1.0)
            ax.plot(r["mean"], yi, "o", ms=4, color=c)
            ax.text(1.0, yi, f"{r['mean']:+.3f}", transform=ax.get_yaxis_transform(), fontsize=6.2, va="center", ha="right", color=c)
        ax.axvline(0, color=S.INK, lw=0.7)
        lim = max(abs(g["lo"]).max(), abs(g["hi"]).max()) * 1.1
        ax.set_xlim(-lim, lim * 1.45)
        ax.set_xlabel(f"Change vs full model (95% interval)")
        ax.set_title(METRIC_NICE[metric], loc="left")
        only_bottom(ax)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels([lab[o] for o in order])
    axes[0].invert_yaxis()
    f.legend(handles=[Line2D([], [], marker="o", color=S.BLUE2, ms=4, lw=1.0, label="Network altered"),
                      Line2D([], [], marker="o", color=S.RED, ms=4, lw=1.0, label="Node features or head removed")],
             loc="lower center", bbox_to_anchor=(0.6, 0.0), ncol=2)
    f.subplots_adjust(left=0.24, right=0.99, bottom=0.27, top=0.92, wspace=0.12)
    return f, t


@fig(25)
def p25():
    pl = pd.read_csv(B5 / "placebo_summary.csv")
    co = pd.read_csv(B5 / "consistency_detail.csv")
    cl = fold_cohort()
    n_pass = int(pl["pass"].astype(str).str.lower().eq("true").sum())
    n_agree = int(co["agree"].astype(str).str.lower().eq("true").sum())
    print(f"   p25 check: placebo {n_pass}/{len(pl)} include 0; consistency {n_agree}/{len(co)} agree")
    f, axes = plt.subplots(1, 2, figsize=(S.TEXTW, 3.1), sharey=False)
    ax = axes[0]
    groups = [("macro_f1", "prevalence_component"), ("macro_f1", "residual_component"),
              ("accuracy", "prevalence_component"), ("accuracy", "residual_component")]
    cohorts = sorted(pl["cohort"].unique(), key=lambda c: [k for k, v in cl.items() if v == c][0])
    yy = 0
    yt, yl = [], []
    for gi, (met, comp) in enumerate(groups):
        g = pl[(pl.metric == met) & (pl.component == comp)].set_index("cohort").reindex(cohorts)
        col = S.RED if comp.startswith("prev") else S.GREY_D
        for k, (c, r) in enumerate(g.iterrows()):
            yv = yy + k * 0.22
            ax.hlines(yv, r["ci_lo"], r["ci_hi"], color=col, lw=0.8, alpha=0.7)
            ax.plot(r["mean"], yv, "o", ms=2.4, color=col)
        yt.append(yy + 0.22 * (len(cohorts) - 1) / 2)
        yl.append(("Macro-F1" if met == "macro_f1" else "Accuracy") + "\n" + ("prevalence" if comp.startswith("prev") else "residual"))
        yy += 0.22 * len(cohorts) + 0.55
    ax.axvline(0, color=S.INK, lw=0.7)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.invert_yaxis()
    ax.set_xlabel("Placebo component (should be 0)")
    ax.set_title(f"Same-cohort placebo: {n_pass}/{len(pl)} intervals include 0", loc="left", fontsize=7)
    only_bottom(ax)
    ax = axes[1]
    co = co.sort_values(["fold", "metric"], ascending=[True, False]).reset_index(drop=True)
    y = np.arange(len(co))
    for yi, (_, r) in zip(y, co.iterrows()):
        c = S.BLUE if r["metric"] == "macro_f1" else S.GREY
        ax.hlines(yi, r["diff_ci_lo"], r["diff_ci_hi"], color=c, lw=0.8, alpha=0.7)
        ax.plot(r["diff"], yi, "o", ms=2.8, color=c)
    ax.axvline(0, color=S.INK, lw=0.7)
    ax.set_yticks(y[::2] + 0.5); ax.set_yticklabels([cl[int(f_)] for f_ in co["fold"][::2]])
    ax.invert_yaxis()
    ax.set_xlabel("Cross-site minus within-site estimate", labelpad=1)
    ax.set_title(f"Cross-site vs within-site: {n_agree}/{len(co)} agree", loc="left", fontsize=7)
    ax.legend(handles=[Line2D([], [], marker="o", color=S.BLUE, ms=2.8, lw=0.8, label="Macro-F1"),
                       Line2D([], [], marker="o", color=S.GREY, ms=2.8, lw=0.8, label="Accuracy")],
              loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
    only_bottom(ax)
    f.subplots_adjust(left=0.13, right=0.99, bottom=0.2, top=0.92, wspace=0.42)
    out = pd.concat([pl.assign(check="placebo"), co.assign(check="consistency")], ignore_index=True)
    return f, out


@fig(26)
def p26():
    cl = fold_cohort()
    d = pd.read_csv(B8 / "per_fold_decomposition.csv")
    d = d[d["metric"] == "macro_f1"].sort_values("fold").reset_index(drop=True)
    pc, rc = d["prevalence_component_shapley"], d["residual_component_shapley"]
    print(f"   p26 check: gap {d.total_gap.min():+.3f}..{d.total_gap.max():+.3f}; prevalence mean {pc.mean():+.4f} ({pc.min():+.3f}..{pc.max():+.3f}); residual mean {rc.mean():+.4f} ({rc.min():+.3f}..{rc.max():+.3f}); |res|>|prev| in {int((rc.abs() > pc.abs()).sum())}/8; prevalence CI excl 0 in {list(d.loc[(d.prevalence_ci_lo > 0) | (d.prevalence_ci_hi < 0), 'test_cohort'])}")
    f, ax = plt.subplots(figsize=(S.TEXTW, 2.9))
    y = np.arange(len(d))
    off = 0.17
    ax.hlines(y - off, d["prevalence_ci_lo"], d["prevalence_ci_hi"], color=S.RED, lw=1.0)
    ax.plot(pc, y - off, "o", ms=4, color=S.RED, label="Prevalence component")
    ax.hlines(y + off, d["residual_ci_lo"], d["residual_ci_hi"], color=S.GREY, lw=1.0)
    ax.plot(rc, y + off, "o", ms=4, color=S.GREY_D, label="Residual component")
    ax.plot(d["total_gap"], y, "|", ms=11, mew=1.6, color=S.BLUE, label="Total gap (calibration minus test)")
    ax.axvline(0, color=S.INK, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([cl[k] for k in d["fold"]])
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1 gap and its components (95% bootstrap interval)")
    only_bottom(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, columnspacing=1.0)
    return f, d
