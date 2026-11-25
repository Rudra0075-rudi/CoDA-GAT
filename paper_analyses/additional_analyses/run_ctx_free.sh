#!/bin/bash
# Context-free node-feature experiment: 8 folds x 4 arms x seeds 42-46 (160 runs).
# Run from the repository root with the Zenodo data restored. About 1 minute per run on one
# CPU core; the four fold groups below run in parallel. Finished runs are skipped.
set -e
A=paper_analyses/additional_analyses
OUT=$A/ctxfree_runs
ARMS=string_full,empty_full,string_ctxfree,empty_ctxfree
for g in fold0_GSE130970,fold1_GSE135251 fold2_GSE162694,fold3_GSE193066 \
         fold4_GSE213621,fold5_GSE240729 fold6_GSE31803,fold7_GSE84044; do
  python -W ignore $A/ctx_free.py . $OUT $g 42,43,44,45,46 $ARMS &
done
wait
python $A/ctx_summary.py $OUT $A/ctxfree_runs.csv > $A/ctxfree_summary.txt
