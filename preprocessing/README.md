# CoDA-GAT preprocessing

This folder rebuilds the processed dataset from public data (GEO, ArrayExpress/BioStudies,
GTEx v8, STRING v12.0) and checks it against the deposited matrices (the 5 Aug 2026 archive).

| File | What it is |
|---|---|
| `preprocessing_clean.ipynb` | Part 1: the acquisition, label and per-cohort expression code of the 30 Aug 2026 notebook, the revision that produced the archive. Part 2: calls `coda_preproc`. Outputs cleared. |
| `coda_preproc/` | Every stage from batch integration onwards (integration, GTEx reference, panel and graph, 26 node features, folds, export), plus `symbols.py`, the five-gene rename map. |
| `verify_against_archive.py` | Rebuilds the downstream outputs from the deposited inputs and compares them with the deposit. |
| `verification_report.csv` | The result of that comparison: 36/36 checks pass. |
| `NOTEBOOK_CLEANUP.md` | Cell by cell: what was kept, moved or removed from the August notebook, and why. |
| `original/` | The October 2026 notebook as supplied, unchanged. It does not reproduce the deposit (10,732 genes); see `original/README.md`. |

Run the notebook from this folder (`cd preprocessing`). Set `CODAGAT_ROOT` to a working folder
with about 10 GB free; downloads and outputs go there, not into the repository.

## Re-run from GEO (23 Nov 2026)

`preprocessing_clean.ipynb` was run top to bottom from GEO in a fresh working folder, through
integration, and compared with the deposited matrices:

| Output | Re-run vs deposit |
|---|---|
| Genes | 10,735, same order |
| Samples | 1,366 supervised + 148 controls, same IDs |
| ComBat matrix (`bulk_expression.parquet`) | max abs diff 5.2e-6 |
| No-ComBat matrix (`bulk_expression_nocombat.parquet`) | max abs diff 1.5e-6 (GSE240729 only; the other cohorts are identical) |
| Controls (`healthy_reference.parquet`) | max abs diff 5.7e-8 |

The small differences are floating-point noise from package versions (here: GEOparse 2.0.4,
mygene 3.2.2, inmoose 0.9.1; the August versions were not recorded).

**The five-gene rename map is required.** HGNC has renamed five genes since August
(C12orf54 → LYLDT1, IPP → KLHL27, TBRG1 → FYRD1, TBRG4 → FASTKD4, TMEM19 → FAABTP).
MyGene.info now returns the new names for the five cohorts mapped through it, while the seven
cohorts whose platform files carry symbols keep the old names, so without the map the shared
gene set drops to 10,730. `coda_preproc/symbols.py` maps the new names back before integration.
Identifier services can change again, so the deposited matrices remain the data of record for
every reported result.

Not re-run here: the STRING download in Part 2 (this environment's network blocks
stringdb-downloads.org). The STRING graph, panel and feature files were checked instead with
`verify_against_archive.py`, below.

## Downstream check against the deposit

`verify_against_archive.py` starts from the deposited ComBat matrix, the 148 controls, the
labels and the STRING edge list, rebuilds the rest with `coda_preproc`, and compares:

| Output | Result |
|---|---|
| GTEx liver reference | 226 samples. 10,734 genes are shared with the expression matrix (GCOM1 is missing from GTEx). |
| Deviation targets: `deviation_score` (= `directional`) and `raw` | Identical (max abs diff 6e-17). |
| 72-gene panel and its order | Identical. |
| Graph topology from the edge list | Identical; 321 undirected pairs. |
| Fold membership (train / calibration / test, all 8 folds) | Identical. |
| All 26 node features, every split of every fold | Identical (max abs diff 6e-8). |

Group C Cohen's d uses the archive formula (per-class variance with ddof=0, pooled with n−1
weights): `folds.COHENS_D_MODE = "archive"`.

## Behaviour that is kept, but must be described in Methods

- ComBat is fitted within each platform on all cohorts, including the held-out test cohorts and the 148 controls. It uses no biological covariates, and there is no cross-platform correction.
- The panel is the first 72 of 74 available pathway genes in alphabetical order. **VIM and XBP1 are excluded by sort order.** HNF4A, TTR, TNF, IL6, CCN2, TGFB2, CCL5, CXCL9, TLR4, AHSG and APOH are absent because GSE213621 lacks them, and that cohort's missing genes shrink the shared gene intersection.
- `expression_z` is a within-cohort z-score. `expression_variability` and `cross_platform_consistency` use all 1,366 samples. None of these uses labels, but they are transductive.
- Group C is fitted on the whole proper-train split, which later supplies the 15% early-stopping split.
- The deviation variants were explored using stage labels from all cohorts. The default target, `directional`, is the variant that *failed* the gate, so it was not selected by that gate.
- `pw_*` features are leave-one-out pathway means. Masking a gene therefore leaves its signal inside its neighbours' `pw_*` features.

## Run the downstream check

```bash
python verify_against_archive.py --archive <unzipped Archive_RSOS.zip>/CoDA-GAT_data \
  --gtex-tpm GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz \
  --gtex-attr GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
```

It needs numpy, pandas, scipy, scikit-learn, networkx and pyarrow (`environment.yml`). Expect
about 5 minutes, most of it reading the 1.6 GB GTEx file.
