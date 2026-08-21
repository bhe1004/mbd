"""Template for driving a real FR3 instead of the simulator.

The execution loop, the planner, the cost, and the trained model are all
unchanged when you move to hardware: the *only* thing that differs is who
executes the joint-velocity command and who reports the state. That is this
file. Fill in the four marked methods, set ``run.plant: real`` in the config,
and the same ``python main.py run`` drives the arm.

What the rest of the stack assumes, and what you must guarantee here:

1. **Same command semantics.** ``send(u)`` takes joint velocities [rad/s] in
   the robot's own joint order, held until the next call. The Koopman model was
   trained against a first-order velocity servo; if your controller's tracking
   differs noticeably, retrain on data recorded from the real arm
   (``pipeline.collect`` writes the format, see its module docstring).
2. **Same control period.** ``advance()`` must return after exactly
   ``control_dt`` of wall time -- block on the robot's control tick, do not
   sleep a fixed amount and hope. The planner's staleness indexing
   (``u = U[k]``, k = plan age in periods) is only meaningful if this holds.
3. **Same frames.** ``ee`` is the tool point in the robot base frame, the same
   point the model's ``tcp`` site marks. Obstacles, targets, and the trained
   model all live in that frame.
4. **Safety is yours.** The planner charges soft penalties; it does not
   guarantee collision-free motion. Keep an external velocity/torque limit, a
   workspace fence, and a reachable e-stop. ``send`` should refuse commands it
   cannot verify rather than pass them through.

A minimal ``franky`` / ``libfranka`` style skeleton is sketched in comments.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .interface import Observation, Plant
from .robot import NUM_JOINTS, FrankaRobot


class RealFrankaPlant(Plant):
    """Adapter to a physical FR3 over your own robot driver."""

    def __init__(self, robot: FrankaRobot, *, control_dt: float, action_limit: float,
                 **driver_kwargs) -> None:
        self.robot = robot
        self.control_dt = float(control_dt)
        self.action_dim = NUM_JOINTS
        self.action_low = -float(action_limit) * np.ones(NUM_JOINTS)
        self.action_high = float(action_limit) * np.ones(NUM_JOINTS)
        self._t0 = time.perf_counter()
        self._next_tick = self._t0
        self._overrun = 0.0

        # TODO(hardware): open the connection, e.g.
        #   import franky
        #   self._robot = franky.Robot(driver_kwargs["ip"])
        #   self._robot.relative_dynamics_factor = 0.1
        #   self._robot.recover_from_errors()
        raise NotImplementedError(
            "RealFrankaPlant is a template: implement reset/observe/send/advance "
            "against your robot driver, then set run.plant: real in the config.")

    # ------------------------------------------------------------- Plant API
    def reset(self, q0: Optional[np.ndarray] = None) -> Observation:
        """Move to the start configuration under YOUR safe motion primitive.

        Do not use the MBD planner for this: it is a local sampler with no
        guarantee about the path it takes to a far-away configuration. Use the
        driver's own joint-motion command with a low velocity factor, confirm
        arrival, then hand control over.
        """

        # TODO(hardware): self._robot.move(franky.JointMotion(q0)); wait for done
        self._t0 = time.perf_counter()
        self._next_tick = self._t0
        return self.observe()

    def observe(self) -> Observation:
        """Read the latest joint state and tool position.

        Prefer the controller's own tool pose if it is published; otherwise the
        robot description's forward kinematics is exact for this arm:
        ``self.robot.ee_of_q(q)``.
        """

        # TODO(hardware): state = self._robot.current_joint_state
        q = np.zeros(NUM_JOINTS)   # state.position
        qd = np.zeros(NUM_JOINTS)  # state.velocity
        return Observation(q=q, qd=qd, ee=self.robot.ee_of_q(q),
                           t=time.perf_counter() - self._t0)

    def send(self, u: np.ndarray) -> None:
        """Command joint velocities, clipped to the configured limit."""

        command = self.clip(u)
        # TODO(hardware): self._robot.move(franky.JointVelocityMotion(command,
        #                     duration=franky.Duration(self.control_dt * 1000)),
        #                     asynchronous=True)
        del command

    def advance(self, substep_callback: Optional[Callable[[int], None]] = None,
                *, resync: bool = False) -> None:
        """Block until the next control tick.

        The fixed-rate version below is the fallback. If your driver exposes a
        control callback or a hardware-synchronised tick, wait on that instead:
        it keeps the loop phase-locked to the robot rather than to this process.
        """

        self._next_tick = (time.perf_counter() if resync
                           else self._next_tick) + self.control_dt
        slack = self._next_tick - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            self._overrun = max(self._overrun, -slack)

    def start_clock(self) -> None:
        self._t0 = time.perf_counter()
        self._next_tick = self._t0
        self._overrun = 0.0

    @property
    def last_overrun(self) -> float:
        return self._overrun

    def close(self) -> None:
        """Stop motion and release the connection -- always, including on error."""

        # TODO(hardware): self._robot.join_motion(); self._robot.stop()
