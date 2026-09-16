#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
m=${1:-full}
v=push_fr3/state_uncertainty/t-ou/scale2.4-tau2.0-warmup50
s=; [ $m = full ] || s=/$m
o=$PWD/experiments/push_fr3/state_uncertainty/t-ou/oracle-warmup50/results$s
r=experiments/$v/results$s
for a in ps cem; do mkdir -p $r/$a; ln -sfn $o/$a/oracle $r/$a/oracle; done
uv run python -m experiments.state_uncertainty.run --version $v --mode $m --resume
uv run python -m experiments.state_uncertainty.analysis --version $v --mode $m
