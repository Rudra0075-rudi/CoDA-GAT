# Tabular baselines re-run with five seeds (42–46)

Behind the manuscript table of the five-seed refit (T1). Each seed was run in its own copy
of the repository; the committed baseline results in `baselines/` were not changed.

How: `scripts/baselines_b3.py --only tabular --seed S` (unchanged script) for S = 42..46,
each in its own copy with the Zenodo data restored (`run_seeds.sh`). Environment: Python 3.13,
pandas 3.0.3, scikit-learn 1.9.1, xgboost 3.4.1, catboost 1.2.10, boruta 0.4.3 (the original
versions were not recorded).

| File | Content |
|---|---|
| `run_seeds.sh` | Runs one seed. It deletes `baselines/per_sample_predictions/` first, so run it only in a separate copy of the repository |
| `log_42.txt` … `log_46.txt` | Console output of each seed |
| `per_sample_seed42/` … `per_sample_seed46/` | Per-sample predictions and per-fold metrics of each seed, as written by `baselines_b3.py` |
| `summarise_seeds.py` | Builds `summary.csv` from the five seeds (run from the repository root: `python paper_analyses/baseline_5seed/summarise_seeds.py paper_analyses/baseline_5seed`) |
| `summary.csv` | Per model: saved single-seed AUROC; mean over the 8 held-out cohorts of the five-seed mean AUROC; between-cohort SD (ddof=1); min, max and SD of the eight-cohort mean across seeds; difference from CoDA-GAT (primary configuration, mean per seed); cohorts where the baseline is higher; two-sided Wilcoxon p over the 8 cohorts |

Reproduction: at seed 42, logistic regression reproduces the saved predictions (mean AUROC
equal to 4 decimals). The tree models do not reproduce exactly (per-sample probabilities
differ), which points to library-version differences; mean AUROC moves by up to 0.016
(panel72_full__xgboost) and by 0.001 for panel72_full__catboost.
