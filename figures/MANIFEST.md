# Figure manifest

Every number is read from the listed source file(s). Nothing is hardcoded.
APS values come from `verification_checks/` (corrected randomized APS), not the
degenerate `aps_*` columns in `results_b7/master_results.csv`.

## fig01_samples_per_cohort
- **files**: `fig01_samples_per_cohort.png`, `fig01_samples_per_cohort.pdf`
- **sources**: `shared/labels.parquet`, `shared/fold_assignment_lodo.csv`
- **caption**: Supervised samples per cohort, coloured by whether the cohort is a held-out evaluation fold (T1) or contributes to training only (T2).

## fig02_f2plus_share_per_cohort
- **files**: `fig02_f2plus_share_per_cohort.png`, `fig02_f2plus_share_per_cohort.pdf`
- **sources**: `shared/labels.parquet`, `shared/fold_assignment_lodo.csv`
- **caption**: Share of patients with significant fibrosis (F2+) in each cohort; the spread across cohorts is the case-mix shift the decomposition later quantifies.

## fig03_label_precision_per_cohort
- **files**: `fig03_label_precision_per_cohort.png`, `fig03_label_precision_per_cohort.pdf`
- **sources**: `shared/labels.parquet`
- **caption**: Biopsy label precision per cohort: samples with an exact fibrosis stage versus samples whose stage is only known as a grouped range.

## fig04_all_models_auc_ranked
- **files**: `fig04_all_models_auc_ranked.png`, `fig04_all_models_auc_ranked.pdf`
- **sources**: `results_b7/master_results.csv`, `baselines/per_sample_predictions/`, `baselines/*/all_folds.csv`
- **caption**: Every model ranked by mean cross-cohort AUC. Baseline values are recomputed from the saved per-sample predictions over all 8 held-out cohorts; models for which only one cohort was ever evaluated are hatched and daggered and are never averaged in with the 8-cohort means. Shaded band is the 0.87-0.90 label-noise ceiling.

## fig05_per_fold_auc_heatmap
- **files**: `fig05_per_fold_auc_heatmap.png`, `fig05_per_fold_auc_heatmap.pdf`
- **sources**: `results_b7/master_results.csv`, `baselines/per_sample_predictions/`, `ablations_b9/master_ablation_results.csv`, `baselines/*/all_folds.csv`
- **caption**: Per-cohort AUC for each model, recomputed from saved per-sample predictions. Rows above the red line are complete 8-cohort LODO evaluations; rows below were evaluated on a single cohort and their unevaluated cells are grey, not imputed.

## fig06_roc_per_cohort
- **files**: `fig06_roc_per_cohort.png`, `fig06_roc_per_cohort.pdf`
- **sources**: `results_b7/fold*/mode_off/seed_*/probs_test.npy`, `folds/*/test/y.csv`
- **caption**: ROC curve for CoDA-GAT on each held-out cohort (seed-averaged probabilities), plus the pooled curve over all 1,064 evaluation samples.

## fig07_adversary_modes_macrof1
- **files**: `fig07_adversary_modes_macrof1.png`, `fig07_adversary_modes_macrof1.pdf`
- **sources**: `results_b7/master_results.csv`, `results_b7/primary_contrast_report.txt`
- **caption**: Pre-registered primary contrast: per-cohort macro-F1 with the cohort adversary off, marginal and conditional; points are cohorts, annotation is the registered test.

## fig08_conformal_coverage_vs_promised
- **files**: `fig08_conformal_coverage_vs_promised.png`, `fig08_conformal_coverage_vs_promised.pdf`
- **sources**: `verification_checks/aps_diagnostic_per_run.csv`
- **caption**: Achieved versus promised coverage for LAC and the corrected randomized APS; the original APS is shown for contrast, pinned at 1.000 by the omitted randomisation term.

## fig09_conformal_set_size_vs_alpha
- **files**: `fig09_conformal_set_size_vs_alpha.png`, `fig09_conformal_set_size_vs_alpha.pdf`
- **sources**: `verification_checks/aps_diagnostic_per_run.csv`
- **caption**: Mean prediction-set size by α. A size of 2.0 on a two-class task means the full label set, i.e. no information; the corrected APS produces informative sets.

## fig10_coverage_per_cohort_alpha10
- **files**: `fig10_coverage_per_cohort_alpha10.png`, `fig10_coverage_per_cohort_alpha10.pdf`
- **sources**: `verification_checks/aps_diagnostic_per_run.csv`, `shared/fold_assignment_lodo.csv`
- **caption**: Coverage per held-out cohort at α = 0.10: whether the 90% promise survives each individual cohort shift, not just on average.

## fig11_masking_real_vs_empty
- **files**: `fig11_masking_real_vs_empty.png`, `fig11_masking_real_vs_empty.pdf`
- **sources**: `graph_utility/masking_curves.csv`
- **caption**: Robustness to missing genes, mean of per-cohort values. The STRING graph degrades FASTER than the no-graph control on AUC and SLOWER on macro-F1 and balanced accuracy; both are shown with the signed advantage and its CI, because the sign of the 'robustness advantage' depends on which metric is chosen.

## fig12_masking_all_conditions
- **files**: `fig12_masking_all_conditions.png`, `fig12_masking_all_conditions.pdf`
- **sources**: `graph_utility/masking_curves.csv`, `learned_graph/masking_curves_all_conditions.csv`, `learned_graph/checkpoints/*/run_metrics.json`
- **caption**: All graph conditions under masking. Conditions with a completed masking sweep are drawn as full curves with CI bands; conditions that are trained but not yet swept appear as their 0%-masked point only, marked as such.

## fig13_advantage_over_empty
- **files**: `fig13_advantage_over_empty.png`, `fig13_advantage_over_empty.pdf`
- **sources**: `graph_utility/masking_curves.csv`, `results_b7/`, `graph_utility/control_checkpoints/`, `learned_graph/checkpoints/`
- **caption**: Advantage over the no-graph control for every graph condition: flatness advantage under masking (left, AUC and macro-F1) and clean-data AUC advantage paired within cohort (right), each with bootstrap CIs.

## fig14_learned_vs_string_topology
- **files**: `fig14_learned_vs_string_topology.png`, `fig14_learned_vs_string_topology.pdf`
- **sources**: `learned_graph/edges/`, `shared/graph_edges.csv`, `shared/panel_genes.csv`
- **caption**: Learned-vs-STRING edge overlap (Jaccard, against a computed chance level) and cross-fold stability of each learned topology, for both learned variants.

## fig15_top_attended_genes
- **files**: `fig15_top_attended_genes.png`, `fig15_top_attended_genes.pdf`
- **sources**: `attention_analysis_b9/per_fold_gene_attention.csv`
- **caption**: The fifteen genes receiving most attention, averaged across cohorts, with the spread across folds shown as error bars.

## fig16_attention_stability_heatmap
- **files**: `fig16_attention_stability_heatmap.png`, `fig16_attention_stability_heatmap.pdf`
- **sources**: `attention_analysis_b9/cross_fold_attention_correlation.csv`, `shared/fold_assignment_lodo.csv`
- **caption**: Agreement between cohorts on which genes matter: Spearman correlation of the attention ranking between every pair of held-out folds.

## fig17_error_agreement_heatmap
- **files**: `fig17_error_agreement_heatmap.png`, `fig17_error_agreement_heatmap.pdf`
- **sources**: `consensus_failures/pairwise_agreement.csv`
- **caption**: Do different models fail on the same patients? Pairwise Jaccard overlap of error sets across all models, ordered by architecture family.

## fig18_family_agreement_vs_chance
- **files**: `fig18_family_agreement_vs_chance.png`, `fig18_family_agreement_vs_chance.pdf`
- **sources**: `consensus_failures/family_agreement_summary.csv`
- **caption**: Within-family versus between-family error agreement against the chance expectation; between-family agreement above chance is the informative quantity.

## fig19_consensus_failure_stages
- **files**: `fig19_consensus_failure_stages.png`, `fig19_consensus_failure_stages.pdf`
- **sources**: `consensus_failures/error_matrix.csv`, `consensus_failures/consensus_failure_samples.csv`
- **caption**: Recorded fibrosis stage of the patients that nearly every model gets wrong, versus everyone else: concentration at the F1/F2 boundary points at the labels.

## fig20_error_rate_all_models
- **files**: `fig20_error_rate_all_models.png`, `fig20_error_rate_all_models.pdf`
- **sources**: `consensus_failures/error_matrix.csv`
- **caption**: Error rate of every model on the pooled held-out samples; the tight clustering is the empirical signature of a shared ceiling.

## fig21_testA_flagged_stage_distribution
- **files**: `fig21_testA_flagged_stage_distribution.png`, `fig21_testA_flagged_stage_distribution.pdf`
- **sources**: `test_a_adjacency/stage_distribution_5class.csv`
- **caption**: Test A: recorded stage of samples the detector flags versus those it does not, as proportions; flags concentrate next to the F2 decision boundary.

## fig22_testB_operating_curve
- **files**: `fig22_testB_operating_curve.png`, `fig22_testB_operating_curve.pdf`
- **sources**: `test_b_rate/threshold_sweep_extended.csv`
- **caption**: Test B operating curve: how the boundary-adjacency enrichment holds as the flagging rate rises, with the independently predicted 18.6% label-error rate marked.

## fig23_testC_cleaned_vs_random
- **files**: `fig23_testC_cleaned_vs_random.png`, `fig23_testC_cleaned_vs_random.pdf`
- **sources**: `test_c_cleaning/per_fold_deltas.json`, `shared/fold_assignment_lodo.csv`
- **caption**: Test C: per-cohort change in held-out macro-F1 when flagged samples are removed from training, against removing the same number at random.

## fig24_ablation_deltas
- **files**: `fig24_ablation_deltas.png`, `fig24_ablation_deltas.pdf`
- **sources**: `ablations_b9/master_ablation_results.csv`, `results_b7/master_results.csv`
- **caption**: Ablations: change in held-out AUC and macro-F1 when each component is removed, relative to the full model.

## fig25_safety_checks
- **files**: `fig25_safety_checks.png`, `fig25_safety_checks.pdf`
- **sources**: `safety_checks_b5/placebo_summary.csv`, `safety_checks_b5/consistency_detail.csv`
- **caption**: Estimator safety checks: placebo components should sit at zero, and the two independent prevalence estimates should agree (difference CI spanning zero).

## fig26_decomposition_per_cohort
- **files**: `fig26_decomposition_per_cohort.png`, `fig26_decomposition_per_cohort.pdf`
- **sources**: `decomposition_b8/per_fold_decomposition.csv`, `shared/fold_assignment_lodo.csv`
- **caption**: Where cross-cohort performance goes: the case-mix (prevalence) component and the inseparable residual per cohort, as counterfactual contrasts.
