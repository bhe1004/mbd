#!/usr/bin/env bash
# Re-run the two headline studies with per-step error curves recorded, so the
# error column can report what the run does after it first enters the tolerance
# band instead of where it happens to sit when the budget runs out.
#
# Identical settings to b_sweep and c_sweep; only record_curve changes, and the
# tags are new so nothing already measured is overwritten.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-/home/home/anaconda3/envs/playground/bin/python}"

echo "=================== schedules (Table IV) -> out/c_curve ==================="
"$PY" -m exper.run_c --config exp_c --set run.tag=curve || echo "FAILED run_c"

echo "=================== rollout classes (Table II) -> out/b_curve ==================="
"$PY" -m exper.run_b --config exp_b --skip-open-loop --set run.tag=curve \
  || echo "FAILED run_b"

echo "ALL DONE"
