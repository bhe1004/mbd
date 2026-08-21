"""Where the target comes from -- fixed, hand-driven, or moving on a path.

The planner does not care whether its goal stands still. It is handed a target
for every step of its horizon and asked to reach it; a goal source is just the
thing that answers "where should the tool be at time t".

    fixed      the configured target list, one segment at a time (the default)
    keyboard   driven from the MuJoCo viewer's key callback (simulator only:
               with the ROS stack the viewer belongs to the bringup process)
    terminal   driven from raw terminal keys -- works with any plant, including
               the real robot
    circle     a target sweeping a circle at constant speed

Two methods matter:

``goal(t)``
    where the target is now, used for reporting and for the adaptive noise
    schedule.
``horizon(t, steps, dt)``
    where the target will be at each step the planner is about to plan for.
    This is the part that makes tracking work. Scoring every candidate against
    a target frozen at its present position leaves a standing lag of roughly
    (target speed) x (planning + actuation delay) -- about 3 cm at 0.2 m/s with
    the measured ~0.15 s loop delay. Feeding the target's own future removes
    most of it, and costs nothing: the cost function already broadcasts over a
    per-step target.

A hand-driven source cannot predict its operator, so it repeats the current
position across the horizon and accepts that lag; a parametric path knows
exactly where it is going.
"""

from __future__ import annotations

import os
import select
import sys
import threading
from typing import Optional, Sequence

import numpy as np

# GLFW key codes, as delivered by mujoco.viewer's key_callback.
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_E, KEY_Q = 69, 81
KEY_KP_ADD, KEY_KP_SUB = 334, 333
KEY_BRACKET_LEFT, KEY_BRACKET_RIGHT = 91, 93
KEY_R = 82

HELP = """MOVING TARGET -- drive it while the arm tracks in real time:
  W / S  (or Up/Down)    : +x / -x   (away from / toward the base)
  A / D  (or Left/Right) : +y / -y
  E / Q                  : +z / -z
  [ / ]  : smaller / larger step     R : reset target     Ctrl-C : stop"""


class GoalSource:
    """Base class: a static target that never moves."""

    #: dynamic sources never "complete", so the run ends on time, not on reach.
    dynamic = False

    def start(self, tool0: np.ndarray, t0: float) -> None:
        """Place the target relative to where the tool actually starts."""

    def activate(self) -> None:
        """Take over any input device. Called after the operator prompt.

        Kept separate from :meth:`start` because grabbing the terminal earlier
        would swallow the "press Enter to start" keystroke: a raw-mode reader
        thread and a line-buffered ``input()`` cannot share one stdin.
        """

    def goal(self, t: float) -> np.ndarray:
        raise NotImplementedError

    def horizon(self, t: float, steps: int, dt: float) -> np.ndarray:
        """Targets for steps 1..steps ahead, shape (steps, 3)."""

        return np.repeat(self.goal(t)[None], steps, axis=0)

    def label(self) -> str:
        return type(self).__name__

    def on_key(self, keycode: int) -> None:
        """Viewer key callback; ignored by sources that do not read keys."""

    def close(self) -> None:
        """Release any input device."""


class FixedGoal(GoalSource):
    """One configured target, held still. The runner drives the segment list."""

    def __init__(self, target: Sequence[float]) -> None:
        self._p = np.asarray(target, dtype=np.float64).copy()

    def set(self, target: Sequence[float]) -> None:
        self._p = np.asarray(target, dtype=np.float64).copy()

    def goal(self, t: float) -> np.ndarray:
        return self._p.copy()

    def label(self) -> str:
        return f"fixed at {np.round(self._p, 3)}"


class _ManualGoal(GoalSource):
    """Shared state for the hand-driven sources.

    Thread note: the reader (viewer thread or tty thread) only mutates this
    object's own array while the control loop only reads a copy, so a stale read
    is at worst one control period old -- exactly how a physical input device
    behaves.
    """

    dynamic = True

    #: Raw-mode terminals do not echo, and the viewer shows only the target
    #: sphere, so a press with no visible effect is indistinguishable from a
    #: press that did not register. Subclasses set this to echo each change.
    echo = False

    def __init__(self, *, step: float, low, high,
                 min_step: float = 0.002, max_step: float = 0.1) -> None:
        self.step = float(step)
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.min_step, self.max_step = float(min_step), float(max_step)
        self._home = np.zeros(3)
        self._p = np.zeros(3)

    def start(self, tool0: np.ndarray, t0: float) -> None:
        self._home = np.clip(np.asarray(tool0, dtype=np.float64), self.low, self.high)
        self._p = self._home.copy()

    def goal(self, t: float) -> np.ndarray:
        return self._p.copy()

    def _apply(self, keycode: int) -> None:
        p, s = self._p, self.step
        if keycode == KEY_UP:
            p[0] += s
        elif keycode == KEY_DOWN:
            p[0] -= s
        elif keycode == KEY_LEFT:
            p[1] += s
        elif keycode == KEY_RIGHT:
            p[1] -= s
        elif keycode in (KEY_E, KEY_KP_ADD):
            p[2] += s
        elif keycode in (KEY_Q, KEY_KP_SUB):
            p[2] -= s
        elif keycode == KEY_BRACKET_RIGHT:
            self.step = min(self.max_step, s * 1.5)
        elif keycode == KEY_BRACKET_LEFT:
            self.step = max(self.min_step, s / 1.5)
        elif keycode == KEY_R:
            p[:] = self._home
        else:
            return
        np.clip(p, self.low, self.high, out=p)
        if self.echo:
            print(f"  target {np.round(p, 3)}  step {self.step * 1000:.0f} mm",
                  flush=True)

    def label(self) -> str:
        return f"{HELP}\n  step = {self.step * 1000:.0f} mm/press"


class ViewerKeyboardGoal(_ManualGoal):
    """Hand-driven from the MuJoCo viewer window (simulator plant only)."""

    def on_key(self, keycode: int) -> None:
        self._apply(keycode)


class TerminalKeyboardGoal(_ManualGoal):
    """Hand-driven from raw terminal keys -- works with any plant.

    Reads stdin in cbreak mode on a background thread. Arrow keys arrive as
    escape sequences, which are mapped onto the same GLFW codes the viewer
    source uses so both share one key map.
    """

    echo = True
    _ESCAPE = {"A": KEY_UP, "B": KEY_DOWN, "D": KEY_LEFT, "C": KEY_RIGHT}
    #: WASD duplicates the arrows deliberately. Arrows arrive as multi-byte
    #: escape sequences, which a terminal, multiplexer or remote session is free
    #: to mangle; the single-byte keys always survive.
    _PLAIN = {"w": KEY_UP, "W": KEY_UP, "s": KEY_DOWN, "S": KEY_DOWN,
              "a": KEY_LEFT, "A": KEY_LEFT, "d": KEY_RIGHT, "D": KEY_RIGHT,
              "e": KEY_E, "E": KEY_E, "q": KEY_Q, "Q": KEY_Q,
              "[": KEY_BRACKET_LEFT, "]": KEY_BRACKET_RIGHT,
              "r": KEY_R, "R": KEY_R, "+": KEY_KP_ADD, "-": KEY_KP_SUB}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._restore = None

    def activate(self) -> None:
        if not sys.stdin.isatty():
            print("terminal goal: stdin is not a terminal, the target will not move")
            return
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._restore = (fd, termios.tcgetattr(fd))
        tty.setcbreak(fd)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        """Read raw bytes and dispatch them.

        ``os.read`` on the descriptor, never ``sys.stdin.read``: stdin is a
        buffered text stream, so reading one character of an arrow key pulls the
        whole three-byte escape sequence into Python's buffer, after which
        ``select`` on the descriptor reports no data and the rest of the
        sequence is silently dropped. Single-byte keys survived that; arrows did
        not. Reading a chunk and parsing it whole avoids the split entirely.
        """

        fd = sys.stdin.fileno()
        pending = ""
        while not self._stop.is_set():
            if not select.select([fd], [], [], 0.1)[0]:
                pending = ""              # a lone ESC was not a sequence
                continue
            try:
                chunk = os.read(fd, 64).decode("utf-8", "ignore")
            except OSError:
                return
            if not chunk:
                return
            pending = self._dispatch(pending + chunk)

    def _dispatch(self, buf: str) -> str:
        """Consume complete keys from ``buf``; return the incomplete tail."""

        i = 0
        while i < len(buf):
            ch = buf[i]
            if ch == "\x1b":
                if i + 2 >= len(buf):
                    return buf[i:]         # sequence split across reads
                if buf[i + 1] == "[":
                    code = self._ESCAPE.get(buf[i + 2])
                    if code is not None:
                        self._apply(code)
                i += 3
                continue
            code = self._PLAIN.get(ch)
            if code is not None:
                self._apply(code)
            i += 1
        return ""

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._restore is not None:
            import termios
            fd, saved = self._restore
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            self._restore = None


class CircleGoal(GoalSource):
    """A target sweeping a circle at constant speed.

    The circle is placed so that it passes through the tool position at the
    start, which removes the step change a fixed centre would otherwise produce
    at t = 0. The angular rate ramps linearly over ``lead_in`` seconds so the
    target also starts from rest -- a target that jumps to full speed instantly
    is not a tracking test, it is a step response.
    """

    dynamic = True

    _AXES = {"xy": (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
             "yz": (np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
             "zx": (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))}

    def __init__(self, *, radius: float, period: float, plane: str = "yz",
                 center: Optional[Sequence[float]] = None, lead_in: float = 2.0,
                 clockwise: bool = False) -> None:
        if plane not in self._AXES:
            raise SystemExit(f"unknown goal.plane {plane!r} (use xy, yz or zx)")
        if radius <= 0 or period <= 0:
            raise SystemExit("goal.radius and goal.period must be positive")
        self.radius = float(radius)
        self.period = float(period)
        self.lead_in = max(float(lead_in), 0.0)
        self.e1, self.e2 = self._AXES[plane]
        if clockwise:
            self.e2 = -self.e2
        self._center = None if center is None else np.asarray(center, dtype=np.float64)
        self._plane = plane
        self.omega = 2.0 * np.pi / self.period

    def start(self, tool0: np.ndarray, t0: float) -> None:
        if self._center is None:
            # Put phase 0 exactly at the current tool position.
            self._center = np.asarray(tool0, dtype=np.float64) - self.radius * self.e1

    def _phase(self, t: float) -> float:
        """Angle at time t, with the rate ramping in over ``lead_in``."""

        t = max(t, 0.0)
        if self.lead_in > 0.0 and t < self.lead_in:
            return self.omega * t * t / (2.0 * self.lead_in)
        return self.omega * (t - self.lead_in / 2.0)

    def goal(self, t: float) -> np.ndarray:
        a = self._phase(t)
        return self._center + self.radius * (np.cos(a) * self.e1 + np.sin(a) * self.e2)

    def horizon(self, t: float, steps: int, dt: float) -> np.ndarray:
        return np.stack([self.goal(t + (k + 1) * dt) for k in range(steps)])

    def speed(self) -> float:
        return self.omega * self.radius

    def label(self) -> str:
        centre = "tool position at start" if self._center is None else np.round(self._center, 3)
        return (f"circle r={self.radius:.3f} m in the {self._plane} plane, "
                f"{self.period:.1f} s/rev ({self.speed():.3f} m/s), centre {centre}")


def build(cfg, task) -> GoalSource:
    """Create the goal source named by ``goal.mode`` in the config."""

    g = cfg.goal
    if g.mode == "fixed":
        return FixedGoal(task.goal_for(cfg.run.target_ids[0], cfg.scene.target))
    if g.mode in ("keyboard", "terminal"):
        cls = ViewerKeyboardGoal if g.mode == "keyboard" else TerminalKeyboardGoal
        return cls(step=g.step, low=g.low, high=g.high)
    if g.mode == "circle":
        return CircleGoal(radius=g.radius, period=g.period, plane=g.plane,
                          center=g.center, lead_in=g.lead_in, clockwise=g.clockwise)
    raise SystemExit(f"unknown goal.mode {g.mode!r}")
