"""MuJoCo plant: the simulated FR3 the control loop drives in wall-clock time.

Implements :class:`~env.interface.Plant`. It owns the visualization scene, the
``MjData`` being stepped, the passive viewer, and the real-time pacing -- and
nothing else. It never sees the planner, the cost, or the Koopman model.

Pacing is the point of this class. ``advance()`` steps physics for one control
period and then sleeps out whatever wall time is left, so a slow planner shows
up as *stale actions* rather than as a simulation that politely waits. Two
pacing bases:

``global`` (async execution)
    deadlines are counted from the start of the run, so the simulation cannot
    silently drift late -- exactly what a real robot's control tick does.
``resync`` (lockstep execution)
    the base is reset to now, so the motion segment between two blocking plans
    still plays back at 1x instead of fast-forwarding.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import mujoco
import mujoco.viewer
import numpy as np

from .interface import Observation, Plant
from .robot import NUM_JOINTS, SCENE_XML, FrankaRobot


class MujocoPlant(Plant):
    """The FR3 in MuJoCo, driven at a fixed control period in real time."""

    def __init__(self, robot: FrankaRobot, *, control_dt: float, action_limit: float,
                 use_viewer: bool = True,
                 key_callback: Optional[Callable[[int], None]] = None,
                 xml_path=SCENE_XML) -> None:
        self.robot = robot
        self.control_dt = float(control_dt)
        self.action_dim = NUM_JOINTS
        self.action_low = -float(action_limit) * np.ones(NUM_JOINTS)
        self.action_high = float(action_limit) * np.ones(NUM_JOINTS)

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        if self.model.nq != NUM_JOINTS or self.model.nu != NUM_JOINTS:
            raise RuntimeError("scene model does not match the 7-DoF robot model")
        self.data = mujoco.MjData(self.model)
        self.tcp_site = self.model.site("tcp").id

        self.substeps = int(round(self.control_dt / self.model.opt.timestep))
        self.substep_dt = self.control_dt / self.substeps
        self._sync_every = max(1, self.substeps // 5)

        self._t0 = time.perf_counter()
        self._boundary = 0
        self._overrun = 0.0
        self._command = np.zeros(NUM_JOINTS)

        self._viewer_ctx = None
        self.viewer = None
        if use_viewer:
            self._viewer_ctx = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=key_callback)
            self.viewer = self._viewer_ctx.__enter__()

    # ------------------------------------------------------------- Plant API
    def reset(self, q0: Optional[np.ndarray] = None) -> Observation:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        if q0 is not None:
            self.data.qpos[:NUM_JOINTS] = np.asarray(q0, dtype=np.float64)[:NUM_JOINTS]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._command = np.zeros(NUM_JOINTS)
        self._t0 = time.perf_counter()
        self._boundary = 0
        self._overrun = 0.0
        return self.observe()

    def start_clock(self) -> None:
        self._t0 = time.perf_counter()
        self._boundary = 0
        self._overrun = 0.0

    def observe(self) -> Observation:
        return Observation(
            q=self.data.qpos[:NUM_JOINTS].copy(),
            qd=self.data.qvel[:NUM_JOINTS].copy(),
            ee=self.data.site_xpos[self.tcp_site].copy(),
            t=time.perf_counter() - self._t0,
        )

    def send(self, u: np.ndarray) -> None:
        self._command = self.clip(u)
        self.data.ctrl[:] = self._command

    def advance(self, substep_callback: Optional[Callable[[int], None]] = None,
                *, resync: bool = False) -> None:
        """Step one control period and sleep out the rest of its wall time."""

        base = time.perf_counter() if resync else self._t0
        for i in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if substep_callback is not None:
                substep_callback(i)
            if self.viewer is not None and i % self._sync_every == self._sync_every - 1:
                self.viewer.sync()
            if resync:
                deadline = base + (i + 1) * self.substep_dt
            else:
                deadline = base + (self._boundary * self.substeps + i + 1) * self.substep_dt
            slack = deadline - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                self._overrun = max(self._overrun, -slack)
        self._boundary += 1

    @property
    def last_overrun(self) -> float:
        return self._overrun

    def is_running(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def close(self) -> None:
        if self._viewer_ctx is not None:
            self._viewer_ctx.__exit__(None, None, None)
            self._viewer_ctx = None
            self.viewer = None

    # -------------------------------------------------------------- viewport
    def user_scene(self):
        """The viewer's user scene for overlays, or None when headless."""

        return None if self.viewer is None else self.viewer.user_scn

    def sync(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def hold(self, seconds: Optional[float] = None) -> None:
        """Keep stepping at zero velocity so the viewer stays interactive."""

        if self.viewer is None:
            return
        end = None if seconds is None else time.perf_counter() + seconds
        self.data.ctrl[:] = 0.0
        try:
            while self.viewer.is_running() and (end is None or time.perf_counter() < end):
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(self.substep_dt)
        except KeyboardInterrupt:
            pass
