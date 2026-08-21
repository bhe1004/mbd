"""Move the arm to the run's start configuration, safely, before planning.

MBD is a local sampler: it has no guarantee about the path it takes to a
configuration far from where the arm currently is, so it must never be used to
"go home". The robot stack already ships the right tool -- a point-to-point
joint controller that interpolates a cubic trajectory -- and this stage drives
it:

1. switch the command interfaces from the planner's controller to the
   point-to-point one (both claim the same joint velocity interfaces, so only
   one can be active at a time) and wait until it actually reports ``active``,
2. clear any goal stranded by an earlier interrupted move,
3. send the JointSpace goal and wait for its result,
4. switch back, leaving the stack as it was found.

Two ordering rules are load-bearing, both learned the hard way:

* **Never switch away from a controller with a goal in flight.** Its ``update()``
  stops running, the goal can never reach a terminal phase, and the action
  server then rejects every later goal as busy -- a wedge that survives until
  the stack is restarted. Step 4 always cancels first.
* **The switch is not synchronous.** Sending a goal immediately after a
  successful ``switch_controller`` gets rejected with "controller is not
  active", so step 1 polls the controller state instead of sleeping a guess.

Success also takes longer than the commanded duration: the server only succeeds
once ``elapsed > duration`` *and* the joint error is under its threshold, and in
velocity mode the reference-tracking gain needs a moment to close that gap
(measured: ~2x the duration). The result timeout accounts for that.
"""

from __future__ import annotations

import time

import numpy as np

from config import Config
from pipeline.build import build_environment


def run(cfg: Config) -> None:
    try:
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from action_msgs.srv import CancelGoal
        from cho_interfaces.action import JointSpace
        from controller_manager_msgs.srv import ListControllers, SwitchController
        from sensor_msgs.msg import JointState
    except ImportError as exc:               # pragma: no cover
        raise SystemExit(f"ROS 2 imports failed ({exc}) -- run under the ROS "
                         "interpreter with the workspace sourced") from exc

    env = build_environment(cfg)
    target = np.asarray(env.start_q, dtype=np.float64)
    r = cfg.ros2
    action_ns = f"/controller_action_server/{r.point_to_point_controller}"
    print(f"start pose: {env.start_note}")
    print(f"target joints: {np.round(target, 4)}")

    if not rclpy.ok():
        rclpy.init()
    node = Node("mbd_home")

    def call(client, request, timeout: float):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        return future.result()

    def client_for(srv_type, name):
        cli = node.create_client(srv_type, name)
        if not cli.wait_for_service(timeout_sec=r.goal_timeout_s):
            raise SystemExit(f"{name} not available -- is the stack running?")
        return cli

    switch_cli = client_for(SwitchController, r.controller_manager + "/switch_controller")
    list_cli = client_for(ListControllers, r.controller_manager + "/list_controllers")

    def switch(activate, deactivate) -> None:
        request = SwitchController.Request()
        request.activate_controllers = activate
        request.deactivate_controllers = deactivate
        request.strictness = SwitchController.Request.STRICT
        result = call(switch_cli, request, r.goal_timeout_s)
        if result is None or not result.ok:
            raise SystemExit(f"controller switch failed "
                             f"(activate={activate}, deactivate={deactivate})")

    def wait_active(name: str, timeout: float) -> None:
        """switch_controller returns before the controller is really active."""

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            listing = call(list_cli, ListControllers.Request(), r.goal_timeout_s)
            if listing is not None:
                state = {c.name: c.state for c in listing.controller}.get(name)
                if state == "active":
                    return
            time.sleep(0.1)
        raise SystemExit(f"{name} did not become active within {timeout:.0f} s")

    def cancel_all() -> int:
        """Cancel every goal on the point-to-point server (zero UUID = all)."""

        cli = node.create_client(CancelGoal, f"{action_ns}/_action/cancel_goal")
        if not cli.wait_for_service(timeout_sec=r.goal_timeout_s):
            return 0
        result = call(cli, CancelGoal.Request(), r.goal_timeout_s)
        return 0 if result is None else len(result.goals_canceling)

    handle = None
    switched = False
    try:
        switch([r.point_to_point_controller], [r.planner_controller])
        switched = True
        wait_active(r.point_to_point_controller, r.goal_timeout_s)
        print(f"controller: {r.planner_controller} -> {r.point_to_point_controller}")

        stranded = cancel_all()
        if stranded:
            print(f"cleared {stranded} stranded goal(s) from an earlier move")
            time.sleep(0.5)

        client = ActionClient(node, JointSpace, action_ns)
        if not client.wait_for_server(timeout_sec=r.goal_timeout_s):
            raise SystemExit("JointSpace action server not available")

        goal = JointSpace.Goal()
        goal.target_joints = JointState(name=list(r.joint_names),
                                        position=[float(v) for v in target])
        goal.duration = float(r.home_duration)
        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send, timeout_sec=r.goal_timeout_s)
        handle = send.result()
        if handle is None or not handle.accepted:
            raise SystemExit("the point-to-point controller rejected the goal "
                             "(not active, duration <= 0, or another goal is live)")

        print(f"moving over {goal.duration:.1f} s ...")
        result = handle.get_result_async()
        # Success needs elapsed > duration AND the joint error under threshold;
        # closing that error takes roughly the duration again in velocity mode.
        rclpy.spin_until_future_complete(
            node, result, timeout_sec=2.0 * goal.duration + r.goal_timeout_s)
        if result.result() is None:
            raise SystemExit("the point-to-point move did not report a result")
        print(f"completed: {result.result().result.is_completed}")
        handle = None                        # terminal: nothing to cancel
        time.sleep(0.3)
    finally:
        # Order matters: a goal still in flight must be cancelled BEFORE the
        # controller loses its command interfaces, or the server wedges.
        if handle is not None:
            print("cancelling the in-flight goal before handing the arm back")
            cancel_all()
            time.sleep(0.5)
        if switched:
            switch([r.planner_controller], [r.point_to_point_controller])
            print(f"controller: {r.point_to_point_controller} -> {r.planner_controller}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
