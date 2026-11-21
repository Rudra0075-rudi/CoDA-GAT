#!/usr/bin/env bash
# Analysis-only reproduction of the 2026-11 corrections. Cannot launch training:
# see analysis/guard.py. Writes only to corrections_2026-11/outputs/.
#   PYTHON=<interpreter>            default: the project's dagat env
#   CORAL_PYTORCH_PATH=<site dir>   optional: check CORN against the installed
#                                   coral_pytorch package instead of the vendored copy
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-/opt/anaconda3/envs/dagat/bin/python}"
exec "$PY" -I corrections_2026-11/analysis/run_all.py "$@"
