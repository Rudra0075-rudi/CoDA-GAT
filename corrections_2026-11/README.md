# CoDA-GAT — 2026-11 retrospective corrections

This is a retrospective audit of the archived study (`Archive.zip`, which is byte-identical to this working tree at commit `e06a055`). It fixes implementation and analysis errors using only saved outputs.

- **Not done:** no model was trained or retrained, no hyperparameter or dataset changed, and no new experimental comparison was added.
- **Not preregistered:** none of these corrections were preregistered. `preregistration_b6.*` and every `PREREG.txt` are unchanged.
- **Not overwritten:** every historical output directory is untouched. The run hashes 8,121 tracked files before and after, and fails if any of them changed.

| File | Contents |
|---|---|
| `CHANGELOG.md` | Each issue mapped to its code change, its result change and the manuscript claim it affects |
| `MANUSCRIPT_CHANGES.md` | The required changes to claims and numbers (the manuscript itself is not edited) |
| `UNRESOLVED.md` | What cannot be corrected from this archive, and why |
| `patches/scripts_2026-11.diff` | A readable diff of every source change in `scripts/` |
| `outputs/VALIDATION.md` | Validation results, written by the run |
| `analysis/` | The analysis-only correction code |
| `run_analysis_only.sh` | The single reproduction command |

## Reproduce (analysis only)

Restore `shared/` and `folds/` from the Zenodo deposit first (top-level README, Data). The run needs
pandas, scipy, matplotlib and torch (CPU is enough; only the CORN check uses it).

```bash
corrections_2026-11/run_analysis_only.sh
# optional: CORAL_PYTORCH_PATH=<dir containing coral_pytorch>  to check CORN against the installed package
# optional: PYTHON=<interpreter>   (default: python3 on PATH)
```

**Why it cannot train.** `analysis/guard.py` is installed before any analysis import. It makes these unimportable:

- `torch`, `torch_geometric`, `catboost` and `xgboost`;
- every training or inference script.

Only `consensus_failures.py` and the new `stage_labels.py` may be imported from `scripts/`. The CORN check needs torch, so it runs in a subprocess that allows torch but still blocks every training module. It compiles only the two CORN functions out of `train_fold0_b1.py` and never imports that module.

The run deletes and rebuilds `outputs/` only. It takes about 1 minute.

The patched training and inference scripts (`h2_decision_b4.py`, `test_a_adjacency.py`, `test_b_rate.py`, `flagging_v2/3/4.py`) were **not executed**. They were compiled and imported, and the shared enrichment code they call was checked against the independent recomputation (12/12 flagging-v2 variants match). The inference scripts now also refuse to overwrite their historical output directories.

## Corrected outputs, inputs and aggregation

| Output | Inputs | Aggregation / definition |
|---|---|---|
| `outputs/corn_check/` | `scripts/train_fold0_b1.py` (two functions, compiled via `ast`), `coral_pytorch` 1.4.0, `h2_decision/per_fold.csv` | 200,000 random logit rows plus worked examples. No model involved. |
| `outputs/consensus_failures/` (`error_matrix`, `pairwise_agreement`, `family_agreement_summary`, `consensus_failure_samples`, `adjacency_test.txt`, `overlap_with_flags.txt`, `summary.txt`) | `results_b7/*/mode_*/seed_*/probs_test.npy`, `ablations_b9/*/*/seed_*/probs_test.npy`, `baselines/per_sample_predictions/*/*.csv`, `shared/labels.parquet`, `test_a_adjacency/per_sample_flags.csv` | **GAT models (9):** decision = mean of 5 seeds' P(F2+) ≥ 0.5 (the original rule). **B3 baselines (14):** decision = the saved `pred_test` column (single seed, calibration-tuned threshold), checked against `p_test >= threshold` under round-trip parsing. **Permutation null:** 1,000 within-fold permutations, seed 0. **Group means:** given two ways, `weighting=pair` (every model pair counts once) and `weighting=equal_group_pair` (every family pair counts once). **Stage tests:** exact-stage records only (800). |
| `outputs/consensus_failures/changed_decisions.csv`, `consensus_counts_historical_vs_corrected.csv`, `consensus_membership_changes.csv`, `historical_vs_corrected.txt` | The historical `consensus_failures/` files, plus an independent rebuild of the historical decision rule | Lists the exact records whose decisions changed. |
| `outputs/stage_enrichment/adjacency_enrichment_by_coding.csv` | Saved flags from Test A, flagging v2 (12 variants), flagging v3 (CL pruned, real and empty graph), and the corrected consensus sets | One-sided Fisher test, F1/F2 vs F0/F3/F4. Four codings (below). |
| `outputs/stage_enrichment/graph_comparison_by_coding.csv` | `flagging_v2/per_patient_flags.csv` | The flagging-v2 paired bootstrap Δlog-OR (functions copied verbatim; 2,000 resamples; seed 20260922; within-cohort resampling), as stored and exact-only. |
| `outputs/stage_enrichment/exact_stage_subset_composition.csv`, `planted_error_eligibility*.csv` | `shared/labels.parquet`, `folds/*/test/y.csv` | Counts only. |
| `outputs/aggregation/auroc_aggregation_summary.csv`, `per_cohort_gat.csv`, `per_cohort_baselines.csv` | The saved per-sample predictions listed above, plus `results_b7_m1`, `results_b7_m3` and the 6 B9 arms | See the AUROC aggregations below. |
| `outputs/aggregation/per_run_metrics_recomputed.csv`, `primary_metric_invariance.csv` | Saved `probs_test.npy` / `predictions_test.npy` against the `master_results.csv` tables | All 440 runs. |
| `outputs/aggregation/ablation_summary_recomputed.csv` | `ablations_b9/master_ablation_results.csv`, `results_b7/master_results.csv` | Per-fold seed means, ablation minus full model, two-sided Wilcoxon over 8 folds (as in `ablations_b9.py`). |
| `outputs/descriptions/` | `shared/graph_edges.csv`, `learned_graph/edges/*.csv`, `learned_graph/edge_budget_audit.csv`, `shared/deviation_targets.parquet`, `shared/handoff.json`, the `results_b7_m*` directories | Graphs rebuilt with numpy using the same RNG calls as `ablations_b9.build_edge_index_and_perm`. The rebuilds are validated against the saved audit. |
| `outputs/figures/fig17–fig21`, `figS_flagging_v2_stage_distribution_exact` (PNG + PDF + CSV of the plotted values) | The corrected outputs above | Replace historical figures 17–21 and `flagging_v2/stage_distribution_per_variant.*`. |

### The four enrichment codings

- **`exact_only` (primary):** exact-stage records only.
- **`as_stored`:** the historical coding, in which grouped F0–F1 records are counted as F0 and grouped F3–F4 records as F3. It reproduces all 15 historical summaries exactly.
- **`grouped_F0F1_as_F1`:** the opposite extreme for the 137 records whose stage is unknown. This coding and `as_stored` are two extreme assignments, not confidence bounds.
- **`exact_excluding_GSE213621`:** a descriptive sensitivity check. Every exact GSE213621 record is F2.

### The AUROC aggregations

These are different quantities. Report them under different names.

- **`per_seed_auc_mean`:** AUROC for each seed, averaged within each cohort, then averaged across cohorts.
- **`seed_ensemble_auc_mean`:** predictions averaged across the 5 seeds first, then AUROC within each cohort, then averaged across cohorts. This describes a 5-model ensemble.
- **`single_seed_auc_mean`:** the baselines, which have one seed.

SD is the between-cohort SD over the 8 cohort values (ddof=1); the ddof=0 value is also given.

Treat the exact-stage subset as a subset, not as the cohort. It contains:

- no GSE31803 records;
- only GSE213621's 107 exact-F2 records (F2+ prevalence 1.00, versus 0.68 for all 299 GSE213621 records).
