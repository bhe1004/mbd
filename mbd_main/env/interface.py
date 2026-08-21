"""The boundary between the planner and whatever actually moves.

:class:`Plant` is the *only* thing the execution loop drives. It is deliberately
tiny -- observe, send, advance -- so that replacing MuJoCo with a real FR3 means
writing one class (see :mod:`env.real_plant`) and changing one line in the
config file. The planner never sees a plant at all; it sees features and a cost.

Timing contract for ``advance()``: the call returns after exactly one control
period of *wall* time. A simulator sleeps out the remainder after stepping
physics; a real robot simply blocks until its next control tick. Overrun (the
loop being late) is reported through :attr:`Plant.last_overrun` so the runner
can report real-time violations instead of silently drifting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Observation:
    """One measurement at a control boundary."""

    q: np.ndarray            # joint positions (n,)
    qd: np.ndarray           # joint velocities (n,)
    ee: np.ndarray           # tool position in the base frame (3,)
    t: float                 # seconds since the run started


class Plant(ABC):
    """Something that executes joint-velocity commands and reports its state."""

    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    control_dt: float

    @abstractmethod
    def reset(self, q0: Optional[np.ndarray] = None) -> Observation:
        """Bring the plant to its starting configuration and report it."""

    @abstractmethod
    def observe(self) -> Observation:
        """Measure the current state (cheap; called every control boundary)."""

    @abstractmethod
    def send(self, u: np.ndarray) -> None:
        """Command a joint-velocity vector, held until the next call."""

    @abstractmethod
    def advance(self, substep_callback: Optional[Callable[[int], None]] = None) -> None:
        """Let one control period of wall time pass, executing the last command."""

    def start_clock(self) -> None:
        """Zero the pacing clock: the next ``advance`` deadline starts here.

        Call it immediately before the control loop. Setup work -- warm-up
        plans, an operator pressing Enter -- happens between ``reset`` and the
        first boundary, and without this the plant would consider itself that
        much behind schedule and race to catch up.
        """

    @property
    def last_overrun(self) -> float:
        """Worst amount by which ``advance`` missed its deadline [s]."""

        return 0.0

    def publish_debug(self, *, goal=None, prediction=None) -> None:
        """Expose what the planner is aiming at, for whatever can render it.

        A plant that owns a window draws directly and ignores this; a plant on
        the other side of a network has no window at all, and this is how the
        target becomes visible. No-op by default so the control loop can call it
        unconditionally.
        """

    def is_running(self) -> bool:
        """False once the operator has asked to stop (e.g. closed the viewer)."""

        return True

    def clip(self, u: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(u, dtype=np.float64), self.action_low, self.action_high)

    def close(self) -> None:
        """Release resources (viewer, sockets, ...)."""
