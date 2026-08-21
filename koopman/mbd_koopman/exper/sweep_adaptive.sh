#!/usr/bin/env bash
# Fixed temperature against the adaptive rule MPPI-DK uses, alpha = c * std(J).
#
# The fixed alpha is what flattens the softmax once the cost spread collapses
# near the goal. The adaptive rule rescales to that spread at every stage, so it
# should hold N_eff away from N and lower the settled error, unless the sharper
# weights let the learned rollout exploit its own bias instead.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/home/home/anaconda3/envs/playground/bin/python}"
COND="oracle split bilinear"
BASE="--config exp_b --skip-open-loop --conditions $COND"

run () { local tag="$1"; shift
  echo "=================== $tag ==================="
  "$PY" -m exper.run_b $BASE --set run.tag="$tag" "$@" || echo "FAILED $tag"; }

# kinematic testbed: comparable to the fixed-alpha sweep already run (b_kin)
run kAD05 --set plant.kinematic=true --set planner.alpha_adaptive=true --set planner.alpha_scale=0.5
run kAD02 --set plant.kinematic=true --set planner.alpha_adaptive=true --set planner.alpha_scale=0.2
run kAD10 --set plant.kinematic=true --set planner.alpha_adaptive=true --set planner.alpha_scale=1.0

# MuJoCo, the setting Table II reports, at the best scale found above
run AD05 --set planner.alpha_adaptive=true --set planner.alpha_scale=0.5

echo "ALL DONE"
