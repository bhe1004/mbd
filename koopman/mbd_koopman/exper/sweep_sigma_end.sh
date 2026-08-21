#!/usr/bin/env bash
# Terminal-precision sweep: does lowering the final noise level close the gap
# between entering the 1 cm band (min_err) and holding it (final_err)?
#
# Same platform, model checkpoints and targets as exp_b; only sigma_end moves.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/home/home/anaconda3/envs/playground/bin/python}"

for SE in 0.20 0.15 0.10 0.05; do
  TAG="sig${SE/./}"
  echo "=================== sigma_end=${SE} -> out/b_${TAG} ==================="
  "$PY" -m exper.run_b --config exp_b --skip-open-loop \
      --conditions bilinear split \
      --set planner.sigma_end="${SE}" \
      --set run.tag="${TAG}" \
    || echo "FAILED sigma_end=${SE}"
done
echo "ALL DONE"
