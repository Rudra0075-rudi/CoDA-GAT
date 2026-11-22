# CoDA-GAT preprocessing: cleaned code and verification

Re-verified on 22 Nov 2026: `verify_against_archive.py` again passes 36/36 checks against `Archive.zip` (5 Aug 2026 handoff) with GTEx v8 downloaded fresh. The 12 `*.parquet` files in this repository's `shared/` and `folds/` (Git LFS) have the same SHA-256 as the archive, and the other 97 files in `shared/` and `folds/` are byte-identical. To run the check on a clone, run `git lfs pull` first.

This folder holds a cleaned version of `preprocessing.ipynb`, as supplied on 21 Nov 2026. The original notebook is not modified.

| File | What it is |
|---|---|
| `coda_preproc/` | Every stage from batch integration onwards, as a package of about 600 lines. In the notebook these stages took about 2,700 lines. |
| `preprocessing_clean.ipynb` | Part 1 is the original acquisition, label and expression-loading cells, minus dead cells. Part 2 calls `coda_preproc`. All outputs are cleared. |
| `verify_against_archive.py` | Rebuilds the downstream outputs from the archived inputs and compares them with the archived handoff. |
| `verification_report.csv` | The result of that comparison: 36/36 checks pass. |
| `NOTEBOOK_CLEANUP.md` | Cell by cell: what was kept, removed or moved, and why. |

## What was verified, and how

I ran `verify_against_archive.py` against `Archive.zip` and the public GTEx v8 files. It starts from the archived inputs:
- the ComBat matrix (`shared/bulk_expression.parquet`);
- the 148 GEO controls;
- the labels;
- the archived STRING edge list.

From those it rebuilds the rest with the cleaned code and compares each result with the archive.

| Output | Result |
|---|---|
| GTEx liver reference | 226 samples, as claimed. 10,734 genes are shared with the expression matrix (GCOM1 is missing from GTEx). |
| Deviation targets: `deviation_score` (= `directional`) and `raw` | Identical (max \|diff\| 6e-17). |
| 72-gene panel and its order | Identical. |
| Graph topology recomputed from the edge list | Identical. The graph has 321 undirected pairs. |
| Fold membership (train / calibration / test, all 8 folds) | Identical. |
| All 26 node features, every split of every fold | Identical (max \|diff\| 6e-8). |

`graph_edges.csv`, `graph_topology.csv`, `panel_genes.csv` and `feature_schema_AB.csv` produced by the notebook run also have the same SHA-256 as the archived files. That confirms STRING v12.0 at a score of 700 or more.

## What could not be verified here, and why

- **Per-cohort loading, identifier mapping and ComBat.** These need GEO, BioStudies and MyGene.info, which this environment cannot reach. The archive also does not hold the per-cohort matrices.
- **The notebook does not reproduce the archived matrices.** The supplied notebook run (Colab, 8 Oct 2026) differs from the archived package (5 Aug 2026) in three ways:
  1. It produced 10,732 common genes; the archive has 10,735. The likely cause is MyGene.info symbol mapping changing between the two runs.
  2. GSE240729 was loaded differently. Its no-ComBat median is 4.17 in the notebook run and 4.62 in the archive.
  3. Group C Cohen's d was edited after the archive was made, from per-class variances with ddof=0 to ddof=1. `folds.COHENS_D_MODE = "archive"` keeps the original formula.
- **Fix for the first two.** Recover the August revision of the notebook (Colab: File → Revision history; or Drive version history), or deposit the August `interim/expr_cache/`. Without either, describe the deposited notebook as "a later revision that reproduces every downstream step exactly from the archived matrices".

## Behaviour that is kept, but must be described in Methods

- ComBat is fitted within each platform on all cohorts, including the held-out test cohorts and the 148 controls. It uses no biological covariates, and there is no cross-platform correction.
- The panel is the first 72 of 74 available pathway genes in alphabetical order. **VIM and XBP1 are excluded by sort order.** HNF4A, TTR, TNF, IL6, CCN2, TGFB2, CCL5, CXCL9, TLR4, AHSG and APOH are absent because GSE213621 lacks them, and that cohort's missing genes shrink the shared gene intersection.
- `expression_z` is a within-cohort z-score. `expression_variability` and `cross_platform_consistency` use all 1,366 samples. None of these uses labels, but they are transductive.
- Group C is fitted on the whole proper-train split, which later supplies the 15% early-stopping split.
- The deviation variants were explored using stage labels from all cohorts. The default target, `directional`, is the variant that *failed* the gate, so it was not selected by that gate.
- `pw_*` features are leave-one-out pathway means. Masking a gene therefore leaves its signal inside its neighbours' `pw_*` features.

## Run

```bash
python -I verify_against_archive.py --archive <unzipped Archive.zip> \
  --gtex-tpm GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz \
  --gtex-attr GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt
```

The script needs numpy, pandas, scipy, scikit-learn, networkx and pyarrow. `integrate.py` also needs inmoose. Expect about 5 minutes, most of it spent reading the 1.6 GB GTEx file.
