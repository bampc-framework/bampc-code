#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
m=${1:-full}
v=balance_fr3/model_mismatch/rolling_friction_truth_10hz_v1
uv run python -m experiments.model_mismatch.run --version $v --mode $m --resume
uv run python -m experiments.model_mismatch.plot_truth_grid --version $v --mode $m
uv run python -m experiments.model_mismatch.plot_truth_grid_summary --task balance_fr3 --param rolling_friction --mode $m
