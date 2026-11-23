# How `preprocessing_clean.ipynb` was built from the 30 Aug 2026 notebook

Source: `CoDAGAT_preprocessing_notebook_2026-08-30.ipynb`, the revision that produced the
archived matrices (5 Aug 2026 handoff). Labels such as [CELL 15] are the notebook's own cell
titles. The processing logic of every kept cell is unchanged; the edits are listed in full.

## Part 1: kept, in this order

| Clean notebook | August cell | Edit |
|---|---|---|
| CELL 1 | [CELL 1] environment check | `!pip install` lines removed; packages come from `environment.yml`. |
| CELL 2 | [CELL 2] imports, paths, config | Colab Drive mount removed. `ROOT` is `$CODAGAT_ROOT`, default `./CoDAGAT_work`. |
| CELL 3 to CELL 14 | [CELL 3] to [CELL 14] registry, guards, download, BioStudies, label parsers, label tables, tier promotion, SDRF labels | None. |
| CELL 14b | [CELL 14b] loaders + the definitions from [CELL 17] | The cache wipe at the end of [CELL 14b] (`shutil.rmtree` of `interim/expr_cache`) is removed: it deleted the per-cohort cache on every run. [CELL 15] calls `reconcile_ids()` and `load_expression()`, which only [CELL 17] defines, so [CELL 17]'s definitions now sit here. [CELL 14b]'s own `load_gct`, `load_per_gsm` and `load_expression` were replaced by [CELL 17]'s before any call, so only [CELL 17]'s versions are kept. |
| CELL 15 | [CELL 15] expression loading v3 | The `!pip install mygene` fallback is removed. |
| CELL 17 | [CELL 17] retry pass | Its definitions moved to CELL 14b, except `_gpl_map()`, which stays here: [CELL 15] defines its own `_gpl_map()`, and [CELL 17]'s multi-GPL version replaces it for the retry pass, as in the August run order. |
| CELL 18, 19, 20 | [CELL 18] ID verification v2, [CELL 19] scale fixes, [CELL 20] sample QC | None. |
| CELL 22, 24 | [CELL 22] identifier remap, [CELL 24] GSE48452 symbol repair | None. |
| (new) | — | Hash of every per-cohort matrix in `expr_cache` (`reports/expr_cache_manifest.json`), and an optional label comparison against the deposited `labels.parquet`. |

## Part 1: removed

| August cell | Why |
|---|---|
| Markdown cells 0, 1 | Stray text. |
| [CELL 16] positional ID check v1 | Read-only, and superseded by [CELL 18] v2. |
| [CELL 21] integration v1 | Superseded by [CELL 25]; nothing reads its output. |
| [CELL 25] integration v2.1, [CELL 26] diagnostics and save | Replaced by `coda_preproc.integrate` (same logic; checked against the archive). |
| [CELL 27] onwards | Replaced by `coda_preproc` (reference, graph, features, folds, export); see below. |

## Part 2: `coda_preproc`

| Step | August cells | Replacement |
|---|---|---|
| Symbol renames | — | `symbols.apply_to_all`: maps five genes renamed by HGNC since August back to the archived names (see `symbols.py`). |
| Integration | [CELL 25], [CELL 26] | `integrate.integrate_all`, `integrate.split_supervised`, `integrate.uncorrected_like` |
| GTEx and deviation | [CELL 27, 27b, 28] | `reference.*`. The gate table is exploratory and uses labels from every cohort. |
| Panel and STRING | [CELL 29] | `graph.select_panel`, `graph.string_subgraph` |
| Node tensor | [CELL 30, 30b, 31] | `features.build_tensor_AB` |
| Folds and Group C | [CELL 32] | `folds.make_folds`, `folds.fold_tensors` (archive Cohen's d) |
| Head-3 record | [CELL 27b, 30b, 31] | `reference.head3_config` (the `head3` block of `handoff.json`) |
| Fold table | [CELL 32] | `folds.fold_table` (`fold_assignment_lodo.csv`) |
| Export | [CELL 33, 33b, 34] | `export.write_package`: all 14 `shared/` files, and per fold `train/`, `calibration/`, `test/` and `fold_info.json`. Not written: the flat fold-0 copy and the README notes that Cell 34 added for inspection. |

## Still redundant, kept on purpose

The label parser is built up in steps ([CELL 7] v1, [CELL 10] v3, [CELL 12] v3.1 patch; [CELL 9]
builds a table that [CELL 11] rebuilds), and the cohort repairs ([CELL 19], [CELL 22], [CELL 24])
patch the loader output in place. Collapsing them would change what runs. They are kept as
they ran in August, because that order is what reproduces the archive.
