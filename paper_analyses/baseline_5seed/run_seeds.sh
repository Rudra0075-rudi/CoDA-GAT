#!/bin/bash
# Usage: run_seeds.sh SEED
# Run from the root of a SEPARATE copy of the repository with the Zenodo data restored:
# the next line deletes the committed per-sample baseline predictions in that copy.
seed=$1
rm -rf baselines/per_sample_predictions
python -W ignore scripts/baselines_b3.py --only tabular --seed $seed > log_$seed.txt 2>&1
echo "EXIT $?" >> log_$seed.txt
