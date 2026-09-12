#!/usr/bin/env bash
# Full post-training sequence: judge every clip, score it, and build the console.
set -euo pipefail
PY="${PY:-.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "== judging with the surveyed camera calibration =="
$PY -m vmax_vision.cli run --mode surveyed

echo "== re-judging with the video-only self-calibration =="
$PY -m vmax_vision.cli run --mode self

echo "== scoring against the labels =="
$PY -m vmax_vision.cli evaluate

echo "== zero-shot COCO baseline =="
$PY -m vmax_vision.cli baseline || echo "(baseline skipped)"

echo "== assembling the console =="
$PY -m vmax_vision.cli report
$PY dashboard/build.py
