#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
m=${1:-full}
v=push_fr3/state_uncertainty/t-ou/oracle-warmup50
uv run python -m experiments.state_uncertainty.run --version $v --mode $m --resume
uv run python -m experiments.state_uncertainty.analysis --version $v --mode $m
