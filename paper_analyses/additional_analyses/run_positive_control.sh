#!/bin/bash
# Positive control for T2: 8 folds x 3 graphs (STRING, disconnected, feature-permuted) x
# seeds 42-46 (120 runs). Run from the repository root with the Zenodo data restored.
# The four fold groups below run in parallel. Finished runs are skipped.
set -e
A=paper_analyses/additional_analyses
OUT=$A/positive_control_runs
for g in fold0_GSE130970,fold1_GSE135251 fold2_GSE162694,fold3_GSE193066 \
         fold4_GSE213621,fold5_GSE240729 fold6_GSE31803,fold7_GSE84044; do
  python -W ignore $A/positive_control.py . $OUT $g 42,43,44,45,46 string,empty,shuffled &
done
wait
python $A/positive_control_summary.py $OUT $A/positive_control_runs.csv > $A/positive_control_summary.txt
