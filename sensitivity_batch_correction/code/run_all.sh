#!/bin/bash
# usage: code/run_all.sh <DATA> <REPO> <OUT> [PYTHON]
#   DATA: unzipped deposit folder that contains shared/ and folds/
#   REPO: this repository (scripts/baselines_b3.py is read from it)
#   OUT:  folder for the per-run CSVs
set -e
DATA=$1; REPO=$2; OUT=$3; PY=${4:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
"$PY" "$HERE/prep.py" "$DATA" "$HERE"
for seed in 42 43 44 45 46; do
  for arm in deposited deposited_trainctrl transductive_refit inductive no_combat; do
    "$PY" "$HERE/run_arms.py" "$DATA" "$REPO" "$OUT" "$arm" "$seed"
  done
done
"$PY" "$HERE/analyse.py" "$OUT" "$REPO" "$OUT"
