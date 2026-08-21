#!/usr/bin/env bash
# Re-measure the schedule study and the stall probe on the planner random stream
# that run_b already used, so the condition shared with Table II (bilinear rollout
# under the annealed schedule) carries one number across both tables.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/home/home/anaconda3/envs/playground/bin/python}"
"$PY" -m exper.run_c     --config exp_c --set run.tag=seed1000 || echo "FAILED run_c"
"$PY" -m exper.run_probe --config exp_c --set run.tag=seed1000 || echo "FAILED run_probe"
echo "ALL DONE"
