# Cell-by-cell cleanup of preprocessing.ipynb

Cell numbers are positions in the `.ipynb` file. The notebook's own labels ("CELL 25" and so on) are given in brackets.

## Removed (dead in a top-to-bottom run)

| File cell | Label | Why it is safe to remove |
|---|---|---|
| 0, 1 | markdown "install", "xd" | Stray text |
| 7 | GSE240729 retry | One-off network retry; rerun Cell 5 if a download fails |
| 19 | BACKUP | Copies folders to Drive; nothing reads the copy |
| 20 | — | Debug print |
| 21 | [CELL 16] positional ID check v1 | Superseded by [CELL 18] v2, which checks the same cohorts and treats NaN as unverified |
| all after 2 | `%pip` / `!pip install` lines | Installs happen once, in Cell 1 |

## Replaced by `coda_preproc` (verified against the archive)

| File cells | Labels | Replacement |
|---|---|---|
| 30 | [CELL 25] integration | `integrate.integrate_all` |
| 31 | [CELL 26] diagnostics and dual save | `integrate.split_supervised`, `integrate.uncorrected_like`. The canary and ComBat-effect printouts are reporting only. |
| 32, 33, 34 | [CELL 27, 27b, 28] GTEx and deviation | `reference.*`. The gate table is exploratory and uses labels from every cohort (`reference.describe_variant`). |
| 35 | [CELL 29] gene-set audit and STRING | `graph.select_panel`, `graph.string_subgraph`. The audit (Part A) is reporting only. |
| 36, 37, 38 | [CELL 30, 30b, 31] stability, tensor, Head-3 arms | `features.build_tensor_AB`; `reference.deviation_variants` sets `deviation_score = directional`. |
| 39 | [CELL R] restore state | Not needed once the stages are functions |
| 40 | [CELL 32] folds and Group C | `folds.make_folds`, `folds.group_C` (archive Cohen's d), `folds.fold_tensors` |
| 41, 43, 44 | [CELL 33, 33b, 34] verification and export | `export.write_package` (archive-compatible file names and columns) |
| 42 | [CELL F1] PROGENy footprint | Not part of the archived package. This is an exploratory analysis, not preprocessing, so it is left in the original notebook only. |

## Kept verbatim, but still redundant

These cells patch earlier cells in place. Collapsing them changes what runs, and the only way to prove the result is unchanged is a fresh GEO download followed by a label comparison. The optional cell at the end of Part 1 does that comparison, but it could not run here because GEO is not reachable from this environment.

- **Label parser.** [CELL 7] parser v1, then [CELL 10] v3 redefines `parse_stage`, then [CELL 12] v3.1 patches `GROUPED_PATTERNS`.
  - [CELL 9] builds a label table that [CELL 11] rebuilds.
  - [CELL 9] cannot be dropped: it defines `f2plus`, `early_adv` and `ordinal`, which later cells use.
- **Expression loading.** [CELL 14b] loaders, then [CELL 15] v3, then [CELL 17] reloads failed cohorts. Then [CELL 19], [CELL 22], [CELL 24] and [CELL 24b] apply cohort-specific repairs.
- **Suggested consolidation, after one verified rerun:**
  - Move `f2plus`, `early_adv` and `ordinal` into the v3 parser cell.
  - Delete parser v1 and the first label build.
  - Fold the cohort repairs into one loader table keyed by accession.
