"""ROS 2 plant: drive a real FR3 through cho_robot_project's VLA controller.

This is the concrete counterpart of :mod:`env.real_plant`'s template. It
implements :class:`~env.interface.Plant` on top of two ROS 2 endpoints:

* subscribe ``/joint_states`` for the measured configuration,
* publish ``cho_interfaces/ActionChunk`` on ``/vla/action/ee_pose`` with
  ``action_space: "joint"``, which ``VLAController`` (run with
  ``control_mode:=velocity``) interpolates at 1 kHz and differentiates into
  joint-velocity commands.

Nothing above this file changes: the planner, the cost, the model and the
control loop are the same objects the simulator uses.

Why a *position* waypoint carries a velocity command
----------------------------------------------------
The controller's output stage is

    dq_cmd = (q_ref - q_ref_prev) / period + kp * (q_ref - q_meas)

so it recovers velocity by differentiating its reference. Between two waypoints
the reference is linear, and the derivative of a straight line through
``q_k -> q_k + u*dt`` over ``dt`` is exactly ``u``. Publishing one waypoint per
control period therefore reproduces the commanded velocity, and the ``kp`` term
becomes a tracking correction on top rather than a second command.

That only holds if the reference is INTEGRATED here rather than re-anchored on
the measurement each period. Anchoring on measurement pins the tracking error at
``u*dt`` forever, and the P term turns it into a constant ``kp*u*dt`` overspeed:
with the shipped ``kp_joint_vel = 20`` and ``dt = 0.05`` the arm would run at
``(1 + kp*dt) = 2x`` the commanded speed. So ``_q_ref`` is seeded once from the
measurement and integrated from then on; ``ref_error_stop`` is the guard that
catches it drifting away from reality.

Safety
------
One waypoint per period means a stalled planner leaves the controller holding a
target only ``control_dt`` ahead: the arm stops within one period instead of
playing out a whole plan. That is deliberate, and it is the property this plant
buys by *not* streaming the full 25-step chunk. The planner keeps the rest of
its horizon as lookahead, where it belongs.

This still is not a safety system. Keep the controller's own limits, a
workspace fence, and a reachable e-stop.

Running it
----------
``rclpy`` lives in ROS's Python (3.10 for Humble), so the MBD process must run
under that interpreter with ``torch``, ``numpy``, ``mujoco`` and ``PyYAML``
importable there -- ``mujoco`` is needed only as the kinematics library behind
the robot description, no simulation is stepped. Source the workspace first so
``cho_interfaces`` resolves.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Sequence

import numpy as np

from .interface import Observation, Plant
from .robot import NUM_JOINTS, FrankaRobot


class Ros2VlaPlant(Plant):
    """Execute MBD's commands on a real FR3 via the VLA controller's chunk topic."""

    def __init__(self, robot: FrankaRobot, *, control_dt: float, action_limit: float,
                 joint_names: Sequence[str],
                 chunk_topic: str = "/vla/action/ee_pose",
                 joint_state_topic: str = "/joint_states",
                 goal_action: str = "/controller_action_server/vla_controller",
                 model_name: str = "mbd",
                 state_timeout: float = 0.2,
                 start_tolerance: float = 0.05,
                 ref_error_stop: float = 0.35,
                 use_measured_ee: bool = False,
                 ee_topic: str = "/ee_state/pose",
                 goal_timeout_s: float = 5.0,
                 marker_topic: str = "/mbd/markers",
                 marker_frame: str = "fr3_link0",
                 reach_threshold: float = 0.025,
                 obstacles=None) -> None:
        try:
            import rclpy
            from rclpy.action import ActionClient
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from cho_interfaces.action import VisionLanguageAction
            from cho_interfaces.msg import ActionChunk
            from sensor_msgs.msg import JointState
            from visualization_msgs.msg import Marker, MarkerArray
        except ImportError as exc:                       # pragma: no cover
            raise SystemExit(
                f"ROS 2 imports failed ({exc}).\n"
                "Run this under the ROS interpreter with the workspace sourced:\n"
                "  source /opt/ros/humble/setup.bash && "
                "source ~/ros2_ws/install/setup.bash\n"
                "and make sure torch / numpy / mujoco / PyYAML are importable there."
            ) from exc

        self.robot = robot
        self.control_dt = float(control_dt)
        self.action_dim = NUM_JOINTS
        self.action_low = -float(action_limit) * np.ones(NUM_JOINTS)
        self.action_high = float(action_limit) * np.ones(NUM_JOINTS)

        self.joint_names = list(joint_names)
        if len(self.joint_names) != NUM_JOINTS:
            raise SystemExit(f"ros2.joint_names must list {NUM_JOINTS} names")
        self.state_timeout = float(state_timeout)
        self.start_tolerance = float(start_tolerance)
        self.ref_error_stop = float(ref_error_stop)
        self.use_measured_ee = bool(use_measured_ee)
        self._ActionChunk = ActionChunk

        self._lock = threading.Lock()
        self._q: Optional[np.ndarray] = None
        self._qd = np.zeros(NUM_JOINTS)
        self._ee_measured: Optional[np.ndarray] = None
        self._state_stamp = 0.0
        self._q_ref: Optional[np.ndarray] = None
        self._seq = 0
        self._fault: Optional[str] = None

        self._t0 = time.perf_counter()
        self._next_tick = self._t0
        self._overrun = 0.0

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.node = Node("mbd_plant")

        # Commands are reliable; state is best-effort so it matches either QoS
        # a broadcaster might publish with.
        self._chunk_pub = self.node.create_publisher(ActionChunk, chunk_topic, 10)
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.node.create_subscription(JointState, joint_state_topic,
                                      self._on_joint_state, sensor_qos)
        if self.use_measured_ee:
            from geometry_msgs.msg import PoseStamped
            self.node.create_subscription(PoseStamped, ee_topic,
                                          self._on_ee_pose, sensor_qos)

        # Markers: the only way the target is visible when the window belongs to
        # someone else (the bringup's MuJoCo view) or does not exist at all (the
        # real arm). RViz renders these directly; `main.py view` reads the same
        # topic and draws them into a mirror of the robot.
        self._Marker, self._MarkerArray = Marker, MarkerArray
        self._marker_frame = marker_frame
        self._reach_threshold = float(reach_threshold)
        self._obstacles = obstacles
        self._marker_pub = self.node.create_publisher(MarkerArray, marker_topic, 1)
        self._marker_seq = 0

        self._goal_client = ActionClient(self.node, VisionLanguageAction, goal_action)
        self._VisionLanguageAction = VisionLanguageAction
        self._goal_handle = None
        self._goal_timeout_s = float(goal_timeout_s)
        self._model_name = model_name

        # Callbacks run on their own thread, not inside advance(). The control
        # loop blocks for long stretches that have nothing to do with ROS -- the
        # operator's "press Enter", the warm-up plans, a slow first plan -- and
        # servicing subscriptions only while advance() runs would let the cached
        # state age through all of them, then trip the staleness fault the moment
        # the loop resumes.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    # ------------------------------------------------------------ subscribers
    def _on_joint_state(self, msg) -> None:
        index = {name: i for i, name in enumerate(msg.name)}
        try:
            order = [index[name] for name in self.joint_names]
        except KeyError:                     # a partial state (e.g. gripper only)
            return
        q = np.asarray([msg.position[i] for i in order], dtype=np.float64)
        qd = (np.asarray([msg.velocity[i] for i in order], dtype=np.float64)
              if len(msg.velocity) >= len(msg.name) else np.zeros(NUM_JOINTS))
        with self._lock:
            self._q, self._qd = q, qd
            self._state_stamp = time.perf_counter()

    def _on_ee_pose(self, msg) -> None:
        p = msg.pose.position
        with self._lock:
            self._ee_measured = np.array([p.x, p.y, p.z], dtype=np.float64)

    def _wait_for_state(self, timeout: float = 5.0) -> np.ndarray:
        """Block until the spin thread has delivered a usable joint state."""

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._q is not None:
                    return self._q.copy()
            time.sleep(0.01)
        raise SystemExit(f"no joint state within {timeout:.0f} s -- is the "
                         "controller running and are the joint names right?")

    # ---------------------------------------------------------------- Plant API
    def reset(self, q0: Optional[np.ndarray] = None) -> Observation:
        """Verify the arm is already at the start pose, then open a VLA goal.

        Moving to the start pose is deliberately NOT done here. MBD is a local
        sampler with no guarantee about the path it takes to a far-away
        configuration; bring the arm there with the point-to-point controller
        (``joint_space_velocity_controller``'s JointSpace action) and switch
        controllers, then start this.
        """

        # Any failure below must still tear the node down: the spin thread is
        # already running, and leaving it alive past an early exit crashes the
        # interpreter on the way out instead of printing the reason.
        try:
            q = self._wait_for_state()
            if q0 is not None:
                error = float(np.max(np.abs(q - np.asarray(q0)[:NUM_JOINTS])))
                if error > self.start_tolerance:
                    raise SystemExit(
                        f"the arm is {error:.3f} rad from the configured start pose "
                        f"(tolerance {self.start_tolerance}). Move it there first "
                        "with the point-to-point controller, then switch to "
                        "vla_controller.")

            if not self._goal_client.wait_for_server(timeout_sec=self._goal_timeout_s):
                raise SystemExit("vla_controller action server not available -- "
                                 "launch with control_mode:=velocity use_vla:=true")
            goal = self._VisionLanguageAction.Goal()
            goal.model_name = self._model_name
            goal.inference_frequency = float(1.0 / self.control_dt)
            future = self._goal_client.send_goal_async(goal)
            deadline = time.perf_counter() + self._goal_timeout_s
            while not future.done() and time.perf_counter() < deadline:
                time.sleep(0.01)
            if not future.done() or not future.result().accepted:
                raise SystemExit("vla_controller rejected the goal")
            self._goal_handle = future.result()
        except BaseException:
            self.close()
            raise

        with self._lock:
            self._q_ref = q.copy()           # seed once; integrated from here on
            self._fault = None
        self._t0 = time.perf_counter()
        self._next_tick = self._t0
        self._overrun = 0.0
        return self.observe()

    def observe(self) -> Observation:
        with self._lock:
            q, qd = self._q.copy(), self._qd.copy()
            age = time.perf_counter() - self._state_stamp
            ee = None if self._ee_measured is None else self._ee_measured.copy()
        if age > self.state_timeout:
            self._fail(f"joint state is {age * 1000:.0f} ms stale")
        return Observation(q=q, qd=qd,
                           ee=ee if (self.use_measured_ee and ee is not None)
                           else self.robot.ee_of_q(q),
                           t=time.perf_counter() - self._t0)

    def send(self, u: np.ndarray) -> None:
        """Publish one waypoint: the integrated reference advanced by u * dt."""

        command = self.clip(u)
        with self._lock:
            if self._q_ref is None:
                raise SystemExit("send() before reset()")
            self._q_ref = self._q_ref + command * self.control_dt
            q_ref = self._q_ref.copy()
            q_meas = self._q.copy()
            self._seq += 1

        drift = float(np.max(np.abs(q_ref - q_meas)))
        if drift > self.ref_error_stop:
            self._fail(f"reference ran {drift:.3f} rad ahead of the measured "
                       "configuration -- the arm is not following (limit, "
                       "collision, or a controller fault)")

        msg = self._ActionChunk()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.action_space = "joint"
        msg.relative = False
        msg.rotation_type = ""              # unused for joint space
        msg.chunk_size = 1
        msg.control_dt = self.control_dt    # one waypoint spans one control period
        msg.arm_actions = [float(v) for v in q_ref]
        msg.gripper_actions = []
        self._chunk_pub.publish(msg)

    # ------------------------------------------------------------- markers
    def _marker(self, ns: str, mid: int, mtype, scale, rgba, *, pos=None, points=None):
        m = self._Marker()
        m.header.frame_id = self._marker_frame
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, self._Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = [float(v) for v in scale]
        m.color.r, m.color.g, m.color.b, m.color.a = [float(v) for v in rgba]
        m.pose.orientation.w = 1.0
        if pos is not None:
            m.pose.position.x, m.pose.position.y, m.pose.position.z = [float(v) for v in pos]
        if points is not None:
            from geometry_msgs.msg import Point
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
        return m

    def publish_debug(self, *, goal=None, prediction=None) -> None:
        """Publish the target, its reach shell, the planned path, and obstacles."""

        if goal is None:
            return
        M = self._Marker
        array = self._MarkerArray()
        array.markers.append(self._marker("target", 0, M.SPHERE, (0.024,) * 3,
                                          (0.1, 0.8, 0.1, 1.0), pos=goal))
        d = 2.0 * self._reach_threshold
        array.markers.append(self._marker("target", 1, M.SPHERE, (d, d, d),
                                          (0.1, 0.8, 0.1, 0.18), pos=goal))
        if prediction is not None and len(prediction) > 1:
            array.markers.append(self._marker("prediction", 0, M.LINE_STRIP,
                                              (0.004, 0.0, 0.0), (0.84, 0.15, 0.16, 0.7),
                                              points=prediction))
        # Obstacles never move; re-sending them once a second is enough to catch
        # a viewer that started late.
        if self._obstacles is not None and self._marker_seq % 20 == 0:
            i = 0
            for c, r in self._obstacles.spheres_draw:
                array.markers.append(self._marker("obstacle", i, M.SPHERE,
                                                  (2 * r,) * 3,
                                                  (0.85, 0.15, 0.15, 0.45), pos=c))
                i += 1
            for c, h in self._obstacles.boxes_draw:
                array.markers.append(self._marker("obstacle", i, M.CUBE,
                                                  (2 * h[0], 2 * h[1], 2 * h[2]),
                                                  (0.85, 0.15, 0.15, 0.45), pos=c))
                i += 1
        self._marker_seq += 1
        self._marker_pub.publish(array)

    def advance(self, substep_callback: Optional[Callable[[int], None]] = None,
                *, resync: bool = False) -> None:
        """Spin ROS callbacks until the next control tick."""

        self._next_tick = (time.perf_counter() if resync
                           else self._next_tick) + self.control_dt
        slack = self._next_tick - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        overshoot = time.perf_counter() - self._next_tick
        if overshoot > 0:
            self._overrun = max(self._overrun, overshoot)

    def start_clock(self) -> None:
        """Zero the pacing clock at the first control boundary.

        Setup -- opening the goal, warm-up plans, waiting for the operator --
        happens between ``reset()`` and here, so the first ``advance()`` must not
        consider itself that far behind schedule.
        """

        self._t0 = time.perf_counter()
        self._next_tick = self._t0
        self._overrun = 0.0

    @property
    def last_overrun(self) -> float:
        return self._overrun

    def is_running(self) -> bool:
        return self._fault is None

    def close(self) -> None:
        """Stop the arm, release the goal, and shut the node down -- always."""

        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            with self._lock:
                q = None if self._q is None else self._q.copy()
            if q is not None and self._chunk_pub is not None:
                # A waypoint at the measured configuration = zero commanded
                # velocity; the controller's P term then just holds it.
                msg = self._ActionChunk()
                msg.header.stamp = self.node.get_clock().now().to_msg()
                msg.action_space = "joint"
                msg.chunk_size = 1
                msg.control_dt = self.control_dt
                msg.arm_actions = [float(v) for v in q]
                self._chunk_pub.publish(msg)
                time.sleep(0.05)
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
                time.sleep(0.1)
        finally:
            self._executor.shutdown()
            self._spin_thread.join(timeout=2.0)
            self.node.destroy_node()
            if self._rclpy.ok():
                self._rclpy.shutdown()

    # ------------------------------------------------------------------ faults
    def _fail(self, reason: str) -> None:
        """Latch a fault: the runner sees ``is_running() == False`` and stops."""

        if self._fault is None:
            self._fault = reason
            print(f"PLANT FAULT: {reason} -- stopping")

    # ------------------------------------------------------------ no viewport
    def user_scene(self):
        return None

    def sync(self) -> None:
        pass
