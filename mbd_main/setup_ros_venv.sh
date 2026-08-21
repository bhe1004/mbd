#!/usr/bin/env bash
# Build the interpreter the ROS stages need (`run --plant ros2`, `home`, `view`).
#
# rclpy is compiled for ROS's own Python -- 3.10 on Humble -- while the rest of
# the project runs happily on any modern one. Rather than force everything into
# the system interpreter, this makes a venv that INHERITS the system packages
# (so rclpy and the cho_interfaces messages resolve) and adds the numeric stack
# on top. mujoco is needed here only as the kinematics library behind the robot
# description; no simulation is stepped in the ROS stages.
#
#   source /opt/ros/humble/setup.bash
#   source <your_ws>/install/setup.bash
#   ./setup_ros_venv.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/usr/bin/python3.10}
command -v "$PY" >/dev/null || { echo "no $PY -- set PY=/path/to/python3.10"; exit 1; }

"$PY" -m venv --system-site-packages .venv-ros
# --ignore-installed: without it pip considers an old system numpy "already
# satisfied" and the pinned version never lands in the venv.
.venv-ros/bin/pip install -q --upgrade pip
.venv-ros/bin/pip install -q --ignore-installed \
    "numpy==1.26.4" "mujoco==3.9.0" "PyYAML==6.0.2" "tqdm==4.67.1"

echo "--- checking ---"
.venv-ros/bin/python - <<'PYEOF'
import importlib
missing = []
for m in ("rclpy", "cho_interfaces", "torch", "numpy", "mujoco", "yaml"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:
        missing.append(m); print(f"  {m:16s} MISSING ({exc})")
if missing:
    raise SystemExit("\nsource the ROS setup files and the workspace, then re-run")
PYEOF
echo "ready:  .venv-ros/bin/python main.py run -c real.yaml"
