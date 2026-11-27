#!/bin/bash
# Recompute every output in this folder that needs no training (seconds).
# Run from the repository root with the Zenodo data restored.
# The context-free experiment needs training: see run_ctx_free.sh.
set -e
A=paper_analyses/additional_analyses
python $A/edge_pathway_overlap.py shared > $A/edge_pathway_overlap.txt
python $A/derived_numbers.py > $A/derived_numbers.txt
python $A/attention_degree.py attention_analysis_b9/consensus_gene_ranking.csv \
    shared/graph_topology.csv shared/panel_genes.csv shared/graph_edges.csv > $A/attention_degree.txt
python $A/attention_uniform_null.py attention_analysis_b9/consensus_gene_ranking.csv \
    shared/graph_edges.csv shared/panel_genes.csv > $A/attention_uniform_null.txt
python $A/ensemble_vs_catboost.py results_b7 folds \
    corrections_2026-11/outputs/aggregation/per_cohort_baselines.csv > $A/ensemble_vs_catboost.txt
python $A/conformal_classcond.py results_b7 folds $A/conformal_classcond.csv > $A/conformal_classcond.txt
python $A/conformal_mondrian.py results_b7 folds $A/conformal_classcond.csv \
    $A/conformal_mondrian.csv > $A/conformal_mondrian.txt
python paper_analyses/exact_stage_cohort_sensitivity.py . > paper_analyses/exact_stage_cohort_sensitivity.txt
python paper_analyses/baseline_5seed/summarise_seeds.py paper_analyses/baseline_5seed > /dev/null
echo "done"
