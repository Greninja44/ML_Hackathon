#!/usr/bin/env bash
# Full development pipeline on the train + student_test partitions.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY scripts/run_eda.py --maps 3
$PY -m src.models.train
$PY -m src.models.optimize
$PY -m src.experiments.compare_decoders --tune
$PY -m pytest -q
