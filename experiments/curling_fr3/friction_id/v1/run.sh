#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
m=${1:-full}
v=curling_fr3/friction_id/v1
uv run python -m experiments.friction_id.run --version $v --mode $m --resume
uv run python -m experiments.friction_id.analysis --version $v --mode $m
uv run python -m experiments.friction_id.plot_paper --version $v --mode $m
