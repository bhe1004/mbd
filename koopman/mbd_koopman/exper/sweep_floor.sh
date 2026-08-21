#!/usr/bin/env bash
# Where does the settled error come from, if not the terminal noise level?
#
# Two candidates, both independent of sigma:
#   (A) temperature: near the goal the cost spread across candidates falls below
#       alpha, the softmax flattens and the update loses its direction.
#   (B) control penalty: w_ctrl pulls u toward zero and sets an equilibrium
#       offset from the goal.
#
# Run on the kinematic testbed so the true-dynamics oracle is affordable; the
# oracle carries no model bias, so whatever moves it is planner-side.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/home/home/anaconda3/envs/playground/bin/python}"
COND="oracle split bilinear"
BASE="--config exp_b --skip-open-loop --conditions $COND --set plant.kinematic=true"

run () {  # run <tag> <extra --set args...>
  local tag="$1"; shift
  echo "=================== $tag ==================="
  "$PY" -m exper.run_b $BASE --set run.tag="$tag" "$@" || echo "FAILED $tag"
}

# (A) temperature
run kA10  --set planner.alpha=0.1
run kA025 --set planner.alpha=0.025

# (B) control penalty
run kC0   --set planner.w_ctrl=0.0

# (A+B) both, to see whether the floor closes entirely
run kA025C0 --set planner.alpha=0.025 --set planner.w_ctrl=0.0

echo "ALL DONE"
