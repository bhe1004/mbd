"""A window onto a robot someone else owns.

With ``run.plant: ros2`` the arm is driven through ROS, and whatever is showing
it -- the bringup's MuJoCo view, or nothing at all on real hardware -- is not
ours to draw into. So this stage mirrors it: subscribe to ``/joint_states``,
pose our own copy of the model, and draw the overlays the simulator plant draws
directly. The target, its reach shell, the obstacles the planner is charged
for, the planned tool path, and the executed trail all appear in one place.

It is read-only. It never commands the robot, never plans, and can be started,
killed and restarted at any point during a run.

    python main.py view -c real.yaml       # in its own terminal

RViz2 is the alternative and needs nothing from this file: the same markers are
published on ``/mbd/markers``, so a MarkerArray display plus RobotModel shows
the target too. This exists because it also draws what RViz would not know
about without extra configuration -- and because it is the same picture the
simulator gives, which makes the two directly comparable.
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from config import Config
from env.robot import NUM_JOINTS, SCENE_XML
from pipeline import viewer as vz
from pipeline.build import build_environment


def run(cfg: Config) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from visualization_msgs.msg import MarkerArray
    except ImportError as exc:               # pragma: no cover
        raise SystemExit(f"ROS 2 imports failed ({exc}) -- run under the ROS "
                         "interpreter with the workspace sourced") from exc

    env = build_environment(cfg)
    names = list(cfg.ros2.joint_names)

    if not rclpy.ok():
        rclpy.init()
    node = Node("mbd_view")
    state = {"q": None, "goal": None, "prediction": None, "stamp": 0.0}

    def on_joint_state(msg) -> None:
        index = {n: i for i, n in enumerate(msg.name)}
        try:
            order = [index[n] for n in names]
        except KeyError:
            return
        state["q"] = np.asarray([msg.position[i] for i in order], dtype=np.float64)
        state["stamp"] = time.perf_counter()

    def on_markers(msg) -> None:
        for m in msg.markers:
            if m.ns == "target" and m.id == 0:
                p = m.pose.position
                state["goal"] = np.array([p.x, p.y, p.z])
            elif m.ns == "prediction":
                state["prediction"] = (np.array([[p.x, p.y, p.z] for p in m.points])
                                       if m.points else None)

    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(JointState, cfg.ros2.joint_state_topic, on_joint_state, qos)
    node.create_subscription(MarkerArray, cfg.ros2.marker_topic, on_markers, 1)

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    tcp = model.site("tcp").id
    trail: list = []
    dt = cfg.task.control_dt

    print(f"mirroring {cfg.ros2.joint_state_topic}, targets from {cfg.ros2.marker_topic}")
    print("close the window to stop")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        scn = viewer.user_scn
        last_warn = 0.0
        while viewer.is_running():
            frame = time.perf_counter()
            rclpy.spin_once(node, timeout_sec=0.0)

            if state["q"] is None:
                if frame - last_warn > 2.0:
                    print("waiting for joint states ...")
                    last_warn = frame
            else:
                data.qpos[:NUM_JOINTS] = state["q"]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                tool = data.site_xpos[tcp].copy()
                trail.append(tool)
                if len(trail) > 400:
                    del trail[0]
                goal = state["goal"] if state["goal"] is not None else tool
                vz.draw_overlay(scn, goal=goal, threshold=cfg.task.strict_threshold,
                                trail=trail, prediction=state["prediction"],
                                obstacles=env.obstacles,
                                show_trail=cfg.viewer.trail,
                                show_prediction=cfg.viewer.prediction)
            viewer.sync()
            slack = dt - (time.perf_counter() - frame)
            if slack > 0:
                time.sleep(slack)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
