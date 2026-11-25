# Batch-correction sensitivity of the tabular baselines

In the main analysis, ComBat was fitted within each platform on all cohorts, including the held-out test cohort. The healthy reference for the deviation features also included 110 GEO controls from test cohorts. This folder measures how much that matters for the tabular baselines. CoDA-GAT was not retrained.

## Versions compared

| Version | ComBat fit | Healthy (control) reference |
|---|---|---|
| `deposited` | original matrix, all cohorts | all 148 GEO controls |
| `transductive_refit` | refitted with `code/combat_fw.py`, all cohorts | all 148 controls |
| `deposited_trainctrl` | original matrix | controls from training-side cohorts only |
| `inductive` | refitted per fold without the held-out cohort, which is then adjusted as a new batch from its own unlabelled samples | controls from training-side cohorts only |
| `no_combat` | none (`bulk_expression_nocombat.parquet`) | uncorrected controls from training-side cohorts |

Everything else is taken unchanged from `scripts/baselines_b3.py`:
- folds and splits;
- the train-fitted scaler;
- the top-1,000 gene ranking;
- the three feature views and four classifiers, with the same hyperparameters;
- the calibration-cohort threshold rule.

Seeds 42 to 46.

## Method notes

- `combat_fw.py` follows `inmoose.pycombat.pycombat_norm` (parametric, no covariates, batch = cohort). On identical input it reproduces it to within 1e-13.
- Without covariates, cohorts interact in ComBat only through the per-gene grand mean and pooled variance. The fold-wise fit estimates these two from training-side cohorts. The held-out cohort's own location and scale are then estimated with the same empirical-Bayes step.
- The deposit holds only ComBat-corrected values for the 148 controls. `prep.py` recovers their uncorrected values by inverting the per-gene, per-cohort linear ComBat map, estimated from the 1,366 samples present in both matrices (maximum residual 5e-14).
- Controls are assigned to cohorts by GEO accession range (`results/control_cohorts.csv`).
- Samples without usable labels entered the original ComBat fit but are not in the deposit. `transductive_refit` is therefore the like-for-like comparator for `inductive`.
- The held-out cohort is still adjusted from its own unlabelled samples as a group. Processing of a single sample in isolation is not tested here.

## Results (five-seed mean AUROC over the 8 held-out cohorts)

| Model | deposited | transductive_refit | inductive | no_combat |
|---|---|---|---|---|
| CatBoost, panel + deviation + pathway | 0.841 | 0.839 | 0.843 | 0.771 |
| CoDA-GAT, primary configuration (not refitted) | 0.822 | | | |

Across the 12 view-by-classifier combinations:
- `inductive` minus `transductive_refit` ranges from -0.003 to +0.017 (mean +0.005).
- `no_combat` lowers the mean by 0.033 relative to `deposited`.
- Restricting the control reference alone changes no tree model and changes logistic regression by at most 0.0001.

All numbers: `results/batch_correction_sensitivity_summary.csv` and `results/batch_correction_sensitivity_per_cohort.csv`. Each raw run is in `results/raw_runs/`.

## Reproduce

Requirements: Python 3.13, pandas 3.0.3, pyarrow, scikit-learn 1.9.1, xgboost 3.4.1, catboost 1.2.10, scipy, inmoose 0.9.1.

```bash
code/run_all.sh <unzipped Archive_RSOS.zip>/CoDA-GAT_data <this repository> <output folder>
```

`prep.py` writes `recovered_controls_raw.parquet` and `control_cohorts.csv` into the data folder.
