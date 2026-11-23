"""End-to-end check: GEO re-run integration outputs -> notebook Part 2 -> package; compare with deposit.
STRING download is blocked here, so STRING-format files are synthesised from the deposited edge list
(this tests graph.string_subgraph's formatting/topology code, not STRING's content)."""
import sys, io, json, gzip, hashlib, zipfile
from pathlib import Path
import numpy as np, pandas as pd
W = Path(sys.argv[1]); DEP = Path(sys.argv[2]); GT = Path(sys.argv[3]); OUT = Path(sys.argv[4])
sys.path.insert(0, str(Path.cwd()))
from coda_preproc import config as C, reference, graph, features, folds, export
st = W / 'stage5'
bulk_all = pd.read_parquet(st/'bulk_all.parquet'); meta_all = pd.read_parquet(st/'meta_all.parquet')
bulk_sup = pd.read_parquet(st/'bulk_sup.parquet'); sup = pd.read_parquet(st/'sup.parquet')
healthy = pd.read_parquet(st/'healthy.parquet'); bulk_raw = pd.read_parquet(st/'bulk_raw.parquet')

# synthetic STRING files from the deposited edges, plus decoys that must be filtered out
E = pd.read_csv(DEP/'shared/graph_edges.csv')
sd = OUT/'string'; sd.mkdir(parents=True, exist_ok=True)
genes = sorted(set(E.src) | set(E.dst) | set(pd.read_csv(DEP/'shared/panel_genes.csv').gene) | {'DECOY1'})
pid = {g: f'9606.ENSP{i:011d}' for i, g in enumerate(genes)}
with gzip.open(sd/'string_info.txt.gz', 'wt') as f:
    f.write('#string_protein_id\tpreferred_name\tprotein_size\tannotation\n')
    for g in genes: f.write(f'{pid[g]}\t{g}\t100\tx\n')
rng = np.random.default_rng(0)
with gzip.open(sd/'string_links.txt.gz', 'wt') as f:
    f.write('protein1 protein2 combined_score\n')
    for r in E.itertuples(): f.write(f'{pid[r.src]} {pid[r.dst]} {r.string_score}\n')
    pg = list(pd.read_csv(DEP/'shared/panel_genes.csv').gene)
    for _ in range(500):   # below-threshold decoys between panel genes, and a non-panel decoy
        a, b = rng.choice(pg, 2, replace=False); f.write(f'{pid[a]} {pid[b]} {rng.integers(150, 700)}\n')
    f.write(f'{pid[pg[0]]} {pid["DECOY1"]} 999\n')

# ---- notebook Part 2, verbatim apart from paths ----
gtex = reference.load_gtex_liver(GT/'gene_tpm.gct.gz', GT/'SampleAttributesDS.txt')
disp = reference.rank_displacement(bulk_all, gtex)
dev_pg = reference.z_in_sample(disp)
dev_targets, dev_info = reference.deviation_variants(disp, meta_all)
panel, dropped = graph.select_panel(bulk_sup)
edges, topo = graph.string_subgraph(sd/'string_links.txt.gz', sd/'string_info.txt.gz', panel)
order = [s for s in sup.index]
T, order, z = features.build_tensor_AB(bulk_sup[order], sup.loc[order], panel, topo, gtex, dev_pg)
F = folds.make_folds(sup.loc[order], C.T1_COHORTS)
tensors = {f['fold']: folds.fold_tensors(f, T, order, z, sup.loc[order], panel) for f in F}
export.write_package(OUT/'handoff', bulk_sup=bulk_sup[order], bulk_raw=bulk_raw[order],
                     healthy=healthy, sup=sup.loc[order], dev_targets=dev_targets,
                     panel_tab=graph.panel_table(panel), edges=edges, topo=topo,
                     order=order, folds=F, tensors=tensors, meta_all=meta_all,
                     head3=reference.head3_config(dev_targets, meta_all))

# ---- compare every deposited file ----
H = OUT/'handoff'; rows = []
def num_diff(a, b):
    try: return float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
    except Exception: return np.nan
for p in sorted(DEP.rglob('*')):
    if p.is_dir() or p.name in ('README.md', 'SHA256SUMS.txt'): continue
    rel = p.relative_to(DEP); q = H/rel
    r = dict(file=str(rel), exists=q.exists())
    if q.exists():
        r['bytes_identical'] = p.read_bytes() == q.read_bytes()
        if not r['bytes_identical']:
            if p.suffix == '.parquet':
                a, b = pd.read_parquet(p), pd.read_parquet(q)
                r['same_shape'] = a.shape == b.shape; r['same_index'] = list(a.index) == list(b.index) and list(a.columns) == list(b.columns)
                r['max_abs_diff'] = num_diff(a.select_dtypes('number').values, b.loc[a.index, a.columns].select_dtypes('number').values) if r['same_shape'] else np.nan
                r['non_numeric_equal'] = a.select_dtypes(exclude='number').equals(b.loc[a.index, a.columns].select_dtypes(exclude='number')) if r['same_shape'] else False
            elif p.suffix == '.npy':
                a, b = np.load(p), np.load(q); r['same_shape'] = a.shape == b.shape
                r['max_abs_diff'] = float(np.abs(a - b).max()) if a.shape == b.shape else np.nan
            elif p.suffix == '.csv':
                a, b = pd.read_csv(p), pd.read_csv(q); r['same_shape'] = a.shape == b.shape
                r['same_index'] = list(a.columns) == list(b.columns)
                if r['same_shape'] and r['same_index']:
                    num = a.select_dtypes('number').columns
                    r['max_abs_diff'] = num_diff(a[num].values, b[num].values) if len(num) else 0.0
                    r['non_numeric_equal'] = a.drop(columns=num).equals(b.drop(columns=num))
            elif p.suffix == '.json':
                a, b = json.loads(p.read_text()), json.loads(q.read_text())
                r['same_keys'] = sorted(a) == sorted(b)
                r['note'] = ','.join(k for k in a if a.get(k) != b.get(k))
    rows.append(r)
rep = pd.DataFrame(rows); rep.to_csv(OUT/'e2e_report.csv', index=False)
extra = sorted(str(x.relative_to(H)) for x in H.rglob('*') if x.is_file() and not (DEP/x.relative_to(H)).exists())
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(rep.drop(columns=[c for c in ['note'] if c in rep]).to_string())
print('json differences:', rep[rep.file.str.endswith('.json')][['file','note']].to_string() if 'note' in rep else '')
print('files written but not in deposit:', extra)
print('missing:', rep[~rep.exists].file.tolist())
