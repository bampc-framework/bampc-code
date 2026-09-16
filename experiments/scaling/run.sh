#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
m=${1:-full}
uv run python -m experiments.scaling.run --mode $m
uv run python -m experiments.scaling.analysis --mode $m
