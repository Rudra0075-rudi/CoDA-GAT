# Analyses and figures added for the journal paper

This folder holds the scripts and outputs behind results that the journal paper reports and
that the older folders do not produce, together with the code that draws the paper's
figures. Nothing in the older folders was changed. All scripts read the deposited data
(Zenodo, see the main README) and the committed outputs, and are run from the repository root.

```bash
# everything that needs no training (seconds)
bash paper_analyses/additional_analyses/run_all.sh
# the context-free experiment (160 training runs, about 40 minutes on 4 CPU cores)
bash paper_analyses/additional_analyses/run_ctx_free.sh
# the positive control (120 training runs, about 40 minutes on 4 CPU cores)
bash paper_analyses/additional_analyses/run_positive_control.sh
# the result figures (about 2 minutes)
python paper_analyses/result_figures/results_figures.py . paper_analyses/additional_analyses figures_out
```

## Contents

| Path | What it produces |
|---|---|
| `additional_analyses/ctx_free.py`, `ctx_summary.py`, `run_ctx_free.sh` | Node features without network context (T2): STRING and disconnected encoders, with and without pathway and topology features, 8 folds × seeds 42–46. Outputs: `ctxfree_runs/` (per-sample probabilities and metrics of all 160 runs), `ctxfree_runs.csv`, `ctxfree_summary.txt` |
| `additional_analyses/positive_control.py`, `positive_control_summary.py`, `run_positive_control.sh` | Positive control for T2: the label is replaced by a planted signal carried by STRING neighbourhoods, and STRING, disconnected and feature-permuted graphs are trained as in T2 (8 folds × seeds 42–46, 120 runs). Outputs: `positive_control_runs/`, `positive_control_runs.csv`, `positive_control_summary.txt` |
| `additional_analyses/edge_pathway_overlap.py` | Share of STRING edges that join genes of a shared pathway (209 of 321, against 455 of 2,556 gene pairs). Output: `edge_pathway_overlap.txt` |
| `additional_analyses/attention_degree.py` | Summed and per-edge attention against STRING degree (T2, attention). Output: `attention_degree.txt` |
| `additional_analyses/positive_control_reference.py` | Reference AUROCs for the positive control: logistic regression on per-gene terms from the true, no or permuted neighbourhoods, scored against the noisy planted labels and against noise-free labels (T2, positive control). Output: `positive_control_reference.txt` |
| `additional_analyses/attention_uniform_null.py` | Observed attention against the uniform-attention null, and the genes whose attention the model raised above it (T2, attention). Output: `attention_uniform_null.txt` |
| `additional_analyses/ensemble_vs_catboost.py` | CatBoost against CoDA-GAT, per-seed mean and five-seed ensemble, paired over cohorts (T1). Output: `ensemble_vs_catboost.txt` |
| `additional_analyses/conformal_classcond.py` | LAC coverage at 95, 90 and 80% nominal, by cohort and by true class (T4). Outputs: `conformal_classcond.csv`, `conformal_classcond.txt` |
| `additional_analyses/conformal_mondrian.py` | Class-conditional (Mondrian) sets (T4). Outputs: `conformal_mondrian.csv`, `conformal_mondrian.txt` |
| `additional_analyses/conformal_betabinom.py` | Mondrian coverage at 90% nominal against its beta-binomial distribution under exchangeability: expected and observed cohort–class cells below 85% per seed, and lower-tail probabilities of F2+ coverage in GSE84044 (T4). Output: `conformal_betabinom.txt` |
| `additional_analyses/derived_numbers.py` | Chance overlap of a random edge set with STRING (40.3 edges, Jaccard 0.067); random-shortlist baselines for planted-error recall (18.0% and 40.9%); Benjamini–Hochberg q values of the ablations (smallest 0.19). Output: `derived_numbers.txt` |
| `exact_stage_cohort_sensitivity.py` | Cohort-stratified (Mantel–Haenszel) and GSE213621-excluded boundary enrichment (T5). Output: `exact_stage_cohort_sensitivity.txt` |
| `baseline_5seed/` | Five-seed refit of the fourteen tabular models (T1). See its README |
| `result_figures/` | Code for the paper's result figures, and the numbers behind each figure. See its README |
| `methodology_figures/` | TikZ sources of the four methodology figures. See its README |

The context-free runs were trained in a later software environment than the deposited
runs (Python 3.13, PyTorch 2.15 on CPU, PyTorch Geometric 2.8.1, scikit-learn 1.9.1), so
both full-feature arms were retrained and the experiment is compared within its own arms.
Re-running `run_ctx_free.sh` in that environment reproduced all 160 runs to within 1e-15.

## Where each reported result comes from

Paths are relative to the repository root. T1–T5 are the framework tests of the paper.
The summaries print rounded values, and the paper rounds the unrounded ones, so a mean
printed as 0.8105 (unrounded 0.810497) appears in the paper as 0.810.

| Paper section | Result | Source |
|---|---|---|
| Evaluation set | Cohorts, samples, exact and grouped stages, label audit | `shared/` and `folds/` (Zenodo), `preprocessing/`, `corrections_2026-11/outputs/stage_enrichment/` |
| T1 | CoDA-GAT 0.822, ensemble 0.830, tabular AUROCs and ranking | `corrections_2026-11/outputs/aggregation/summary.txt` |
| T1 | Tabular macro-F1 at calibration thresholds | `corrections_2026-11/outputs/aggregation/per_cohort_baselines.csv` |
| T1 | Paired CatBoost differences (0.019 and 0.011) | `paper_analyses/additional_analyses/ensemble_vs_catboost.txt` |
| T1 | Five-seed refit table | `paper_analyses/baseline_5seed/summary.csv` |
| T2 | Control graphs | `corrections_2026-11/outputs/aggregation/summary.txt`, `ablations_b9/`, `graph_utility/`; unmasked AUROCs of the masking experiment, which retrained the controls, in `learned_graph/summary.txt` |
| T2 | Learned graphs (overlap, stability, hubs) | `learned_graph/summary.txt`; hubs by fold in `learned_graph/topology_comparison.csv`, STRING degree in `shared/graph_topology.csv`; chance overlap in `derived_numbers.txt` |
| T2 | Gene masking | `review_fixes/summary.txt` (task 1), `graph_utility/`; contrast of every graph with the disconnected model in `result_figures/figure_data/fig_p13_data.csv` |
| T2 | Ablations | `corrections_2026-11/outputs/aggregation/ablation_summary_recomputed.csv`, `ablations_b9/ablation_report.txt`; q values in `derived_numbers.txt` |
| T2 | Ordinal head (five folds), augmentation arms | `h2_decision/per_fold.csv`, `results_b7_m1/`, `results_b7_m3/` |
| T2 | Context-free node features | `paper_analyses/additional_analyses/ctxfree_summary.txt` |
| T2 | Positive control (planted graph signal) | `paper_analyses/additional_analyses/positive_control_summary.txt`, `paper_analyses/additional_analyses/positive_control_reference.txt` |
| T2 | Single-fold checks | `baselines/summary_graph_controls.csv`, `baselines/summary_hybrids.csv` |
| T2 | Attention | `attention_analysis_b9/`, `paper_analyses/additional_analyses/attention_degree.txt`, `paper_analyses/additional_analyses/attention_uniform_null.txt` |
| T3 | Adversary modes | `results_b7/primary_contrast_report.txt` |
| T3 | Degradation decomposition, placebo and cross-site versus within-site checks | `decomposition_b8/b8_summary.txt`, `safety_checks_b5/` |
| T4 | Pooled LAC and corrected APS coverage, set size and share of two-class sets | `review_fixes/summary.txt` (task 3), `verification_checks/aps_diagnostic.txt`; unrounded means in `result_figures/figure_data/fig_p8_data.csv` |
| T4 | F2+ calls at the 0.5 threshold against recorded prevalence, by cohort | `flagging_v4/summary.txt` (task 3) |
| T4 | Coverage by cohort, level and class; Mondrian sets | `conformal_classcond.txt`, `conformal_mondrian.txt`; unrounded values in the matching `.csv` files |
| T4 | Mondrian coverage expected under exchangeability (beta-binomial reference) | `paper_analyses/additional_analyses/conformal_betabinom.txt` |
| T5 | Shared errors and consensus failures | `corrections_2026-11/outputs/consensus_failures/summary.txt` |
| T5 | Boundary enrichment | `corrections_2026-11/outputs/stage_enrichment/summary.txt`, `paper_analyses/exact_stage_cohort_sensitivity.txt` |
| T5 | Reference-standard ceiling | `review_fixes/summary.txt` (task 4), `review_fixes/task4_latent_noise_ceiling.csv` |
| T5 | Review flags, planted errors, Confident Learning, split-half stability | `test_a_adjacency/`, `flagging_v4/summary.txt`; Confident Learning intervals in `flagging_v3/noise_rate_estimates.csv`; unrounded split-half and planted-error values in `flagging_v4/task1_split_half_stability.csv` and `flagging_v4/task2_planted_summary.csv`; random baselines in `derived_numbers.txt` |
| T5 | Removing flagged training samples | `test_c_cleaning/summary.txt` |
| Label audit | Original coding against exact stages | `corrections_2026-11/outputs/stage_enrichment/summary.txt` |
| Robustness | Batch-correction check | `sensitivity_batch_correction/` |
| Discussion | Edge–pathway overlap; random-graph pair counts | `edge_pathway_overlap.txt`; `corrections_2026-11/outputs/descriptions/summary.txt` |
| Figures | All result figures | `paper_analyses/result_figures/` |
