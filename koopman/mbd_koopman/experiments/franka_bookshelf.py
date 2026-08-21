"""Shelf reaching on the FR3: insert the TCP between two bookshelf boards to
reach a book, in the MuJoCo viewer, under real-time BK-MBD planning.

This is the *concrete* version of the abstract window task (envs `franka` +
the window spike): instead of an invisible plane-with-a-hole cost, the obstacle
is the actual bookshelf mesh from
``assets/franka_description/robots/object/bookshelf.xml`` (re-exported into
``envs/assets/franka_fr3/scene_bookshelf_velocity.xml``). The FR3 base sits
``BASE_HEIGHT`` (0.70 m) above the floor on a pedestal, the bookshelf stands
on the floor to the robot's left (``SHELF_YAW``), and the LOW compartment
between ``shelf_00`` and ``shelf_01`` (z 0.03..0.35 in the base frame) lands
in the workspace; a book stands in it and the arm must thread the gap.

The boards are a SOFT keep-out penalty (the geoms are visual/non-colliding):
the planner charges a quadratic penalty for any candidate whose predicted arm
enters a board box (+radius+margin), and success is a strict reach with zero
*executed* board penetrations.

Three things the raw Koopman state (b = [q, ee-position]) does not carry are
recovered from the decoded joints via model-consistent forward kinematics, so
they can enter the candidate cost without any MuJoCo in the rollout loop:

  * ``--collision arm`` -- WHOLE-ARM keep-out: FK gives every link-frame origin
    (verified against MuJoCo) plus interpolated samples, each a collision
    sphere, so the elbow/forearm/wrist avoid the boards, not only the TCP.
  * ``--approach-axis`` -- the gripper approach axis (TCP-frame +z) is aimed +x
    to lay the hand horizontal, into the shelf.
  * ``--finger-axis`` -- the finger-opening axis (TCP-frame +y) is aimed +y to
    ROLL the hand 90 deg to a sideways grasp, matching a book turned 90 deg in
    world yaw. Together these two axes fix the full grasp orientation.

The tight (~0.3 m) compartment is made feasible whole-arm without changing the
shelf shape, using only relative-pose levers: the whole shelf is rigidly lowered
a little (SHELF_DZ) so the compartment straddles the arm's working height, and
the FR3 *starts* from a non-aligned 'tucked' pose (hand pointing down, off to the
side and ~10 cm below the book) so the grasp is a genuine reorient-and-insert,
not a pre-aligned push. With the defaults, BK-MBD reorients ~90 deg to the
sideways grasp and reaches the book strictly (~24 mm) with zero executed board
penetrations, at ~46 ms/plan; the linear DK-MBD rollout stalls on the same task.

Examples::

    python experiments/franka_bookshelf.py --realtime             # default: tucked start, sideways grasp, viewer
    python experiments/franka_bookshelf.py --no-viewer            # headless benchmark (PASS ~46 ms)
    python experiments/franka_bookshelf.py --method dk_mbd        # linear baseline (stalls)
    python experiments/franka_bookshelf.py --method bk_qp_sqp --w-orient 0 --no-viewer
                                                                  # convexified QP-SQP baseline (same bilinear model)
    python experiments/franka_bookshelf.py --collision tcp        # TCP-only keep-out (ablation)
    python experiments/franka_bookshelf.py --finger-axis 0 0 0    # free the roll (approach axis only)
    python experiments/franka_bookshelf.py --start preinsert      # easy: pre-aligned, translation only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import mujoco
import mujoco.viewer
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.config import MBDConfig, MethodName, UpdateRule  # noqa: E402
from bk_mbd.mbd import MBDOptimizer  # noqa: E402
from bk_mbd.train import load_checkpoint  # noqa: E402
from envs.franka import (  # noqa: E402
    NUM_JOINTS, FrankaTask, FrankaTaskConfig, _axis_angle_mat)

SCENE_XML = (
    PROJECT_ROOT / "envs" / "assets" / "franka_fr3" / "scene_bookshelf_velocity.xml"
)

# ============================================================================
# SHELF PLACEMENT -- the one knob to move the whole bookshelf. Edit this vector
# (dx, dy, dz) in metres to slide the shelf in front/back (x), left/right (y),
# up/down (z). It is a RIGID translation: the shelf shape (board sizes and
# spacing) never changes. Everything that must agree follows this one constant:
#   - the planner's keep-out boxes (BOARDS + SHELF_OFFSET),
#   - the visual bookshelf body, the book, and the floor (set at scene load),
#   - the default grasp goal (GRASP0 + SHELF_OFFSET).
# After a LARGE move you should also re-pick a --start pose (the tucked/preinsert
# joint poses are IK'd for the default placement) and sanity-check reachability.
#
# The FR3 base is mounted BASE_HEIGHT above the floor (pedestal): the world
# frame stays the BASE frame (origin at the base, as the planner, model, and
# FK all assume), the visual floor drops to z = -BASE_HEIGHT, and the
# bookshelf STANDS ON THE FLOOR: SHELF_OFFSET.z = -BASE_HEIGHT -
# SHELF_ASSET_BOTTOM (= +0.02 for 0.70), so the shelf bottom rests exactly on
# the floor. This raises the shelf 12 cm relative to the old base-level
# layout, which moves the mid compartment out of reach for the high start;
# the TASK therefore targets the LOW compartment (shelf_00..shelf_01,
# z 0.03..0.35 in base frame) -- comfortable reach (goal r~0.62).
# The 5 cm lateral pull (with SHELF_YAW=90, shelf on the left) keeps the lip
# at y=0.55, the closest the start poses stay clear of the boards.
BASE_HEIGHT = 0.70
SHELF_OFFSET = np.array([0.0, -0.05, -BASE_HEIGHT - (-0.72)])

# The shelf is also yawed rigidly about world z: SHELF_YAW = 90 stands the
# bookshelf to the robot's LEFT (+y) with the compartment opening facing the
# arm. Because FR3 joint 1 rotates about world z, the start poses ride along
# exactly (q1 += SHELF_YAW), so the task geometry RELATIVE to the arm is
# preserved; multiples of 90 deg keep the keep-out boxes axis-aligned. The
# point of the left placement is workspace: the home pose occupies the forward
# corridor, so a front shelf cannot come closer than lip x ~ 0.60 (the reach
# edge), whereas on the left the shelf can be pulled toward the base via
# SHELF_OFFSET to make DEEP insertions feasible. SHELF_OFFSET stays in the
# WORLD frame and is applied after the yaw.
SHELF_YAW = 90.0


def _shelf_rot() -> np.ndarray:
    """World rotation of the whole shelf task (multiple of 90 deg about z)."""

    assert abs(SHELF_YAW % 90.0) < 1e-9, "keep-out boxes must stay axis-aligned"
    th = np.radians(SHELF_YAW)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yawed_qpos(q: np.ndarray) -> np.ndarray:
    """Rotate a start pose with the shelf: joint 1 turns about world z."""

    q = np.asarray(q, dtype=np.float64).copy()
    q[0] += np.radians(SHELF_YAW)
    return q


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])

# Book pose and default grasp goal, given in the UNSHIFTED shelf frame; the world
# values are R @ these + SHELF_OFFSET, so they ride along when the shelf moves.
# The book stands in the LOW compartment: shelf_00 top is at asset z=0.01, and
# the book centre/grasp sit 0.17/0.20 above the board (as in the original
# mid-compartment layout).
BOOK_POS0 = np.array([0.64, 0.0, 0.18])   # visual book centre
GRASP0 = np.array([0.62, 0.0, 0.21])      # default TCP grasp goal
# Lowest point of the shelf in the asset frame (bottom board underside); the
# cosmetic floor is dropped to here + SHELF_OFFSET.z so the shelf rests on it
# instead of clipping through it.
SHELF_ASSET_BOTTOM = -0.72

# Bookshelf board keep-out boxes, in the *unshifted* asset frame
# (name, center xyz, half-size xyz); SHELF_DZ is added at load. We keep the full
# set so the executed-violation check is honest about every board.
BOARDS = [
    ("left_side", (0.7475, -0.385, 0.165), (0.1475, 0.010, 0.885)),
    ("right_side", (0.7475, 0.385, 0.165), (0.1475, 0.010, 0.885)),
    ("back_panel", (0.885, 0.0, 0.165), (0.010, 0.395, 0.885)),
    ("bottom_board", (0.7375, 0.0, -0.705), (0.1375, 0.375, 0.015)),
    ("lower_closed", (0.7375, 0.0, -0.355), (0.1375, 0.375, 0.335)),
    ("shelf_00", (0.7375, 0.0, -0.005), (0.1375, 0.375, 0.015)),
    ("shelf_01", (0.7375, 0.0, 0.345), (0.1375, 0.375, 0.015)),
    ("shelf_02", (0.7375, 0.0, 0.695), (0.1375, 0.375, 0.015)),
    ("top_board", (0.7375, 0.0, 1.035), (0.1375, 0.375, 0.015)),
]

# Boards the planner scores against (kept small for latency). The two boards
# bounding the target compartment are what a straight-in insertion can hit; the
# side/back/far panels stay in the honest *executed* collision check. Widen this
# (or pass a different book) if the arm swings toward the sides.
PLAN_BOARD_NAMES = {"shelf_00", "shelf_01"}

# The 'side' start needs EVERY board, not just the compartment pair. Its detour
# has room to go around the shelf in any direction, and the planner will take
# whichever route is unmodelled: with only the side panel added it dives UNDER
# the shelf and scrapes lower_closed/shelf_00, which the honest check counts.
# Scoring all nine boards costs ~85 ms/plan instead of ~46.
START_PLAN_BOARDS = {"side": {b[0] for b in BOARDS}}

# Pre-insertion start pose (torch IK): hand laid horizontal (approach axis +x)
# with the TCP at (0.50, 0, 0.21), just in front of the low-compartment lip at
# the grasp height; the planner then only translates the pre-oriented hand
# straight in (easy case, no reorientation swing).
PREINSERT_QPOS = np.array(
    [0.0, 0.4609, 0.0, -2.7382, 0.0, 4.4669, -0.7853], dtype=np.float64)

# 'tucked' start (torch IK): hand pointing DOWN, TCP in front of the low
# compartment and off to the side at (0.47, 0.12, 0.13) -- ~10 cm below the
# grasp, so the hand-down -> sideways reorientation lands at the book height.
# A genuine reorient-and-insert task, not a pre-aligned push.
TUCKED_QPOS = np.array(
    [0.1257, 0.2011, 0.1196, -2.3654, -0.0211, 2.4492, -0.5585], dtype=np.float64)

# 'high' start (torch IK): the hand is INSERTED IN THE WRONG COMPARTMENT -- the
# one ABOVE the target, between shelf_01 (top z 0.38) and shelf_02 (bottom
# z 0.70). The TCP sits at (0.70, 0, 0.50) in the shelf frame, i.e. 0.15 m past
# the front lip (x=0.55), with the hand laid roughly horizontal into the shelf
# so the forearm runs along the compartment and clears shelf_02 (a hand-down
# pose cannot go deeper than x~0.61 before the elbow hits the top board).
# The straight descent to the book is blocked: through shelf_01's z-range every
# arm sphere must stay outside the lip plane, so the planner must first BACK OUT
# of the wrong compartment, descend below shelf_01, and re-insert into the mouth
# -- a non-monotone detour that recreates the window task's forcing structure on
# real furniture. A branch-following convexification gets trapped pressing down
# on shelf_01 (its per-step faces keep saying 'stay above'), which is the
# discriminating case.
HIGH_QPOS = np.array(
    [0.0068, 0.0563, -0.0306, -1.8802, -1.7187, 2.8179, 0.1392], dtype=np.float64)

# 'side' start (torch IK): the hand is BESIDE the shelf, already at the grasp
# depth and height but outside the left side panel -- TCP at (0.60, -0.50, 0.23)
# in the shelf frame, ~10 cm clear of ``left_side`` (y -0.395..-0.375, spanning
# the shelf's whole x and z extent). The straight line to the book crosses that
# panel, so the planner must RETREAT in x to in front of the shelf (x < 0.55),
# slide laterally to y ~ 0, and only then insert -- the same non-monotone
# forcing structure as 'high', but in the horizontal plane.
#
# Two reasons this discriminates better than the vertical detour: (1) the
# lateral slide is mostly joint 1 (base yaw), which sweeps the TCP a long way
# per radian, so the detour fits the 0.75 s horizon that the vertical one did
# not; (2) the forcing obstacle sits 0.375 m from the goal instead of the
# 0.12 m of shelf_01, so a keep-out weight large enough to actually stop the
# arm no longer bleeds into the terminal approach and wreck its accuracy.
SIDE_QPOS = np.array(
    [-0.7596, 1.4115, -0.5295, -0.7824, -0.764, 4.3734, -0.5585], dtype=np.float64)

METHOD_COLORS = {
    "vanilla_mbd_true": (0.13, 0.65, 0.22),
    "dk_mbd": (0.12, 0.47, 0.71),
    "dk_mbd_split": (1.00, 0.50, 0.05),
    "bk_mbd": (0.84, 0.15, 0.16),
    "bk_qp_sqp": (0.55, 0.34, 0.64),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=[MethodName.BK_MBD.value, MethodName.DK_MBD.value,
                 MethodName.DK_MBD_SPLIT.value, "bk_qp_sqp"],
        default=MethodName.BK_MBD.value,
        help="'bk_qp_sqp' = convexified bilinear Koopman MPC (window-spike "
        "run_qp ported to the board boxes): same bk checkpoint, cost, and "
        "keep-out geometry, but a branch-following QP-SQP instead of the "
        "sampler",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--start", choices=["home", "tucked", "preinsert", "high", "side"],
                   default="tucked",
                   help="initial FR3 pose. 'tucked' (default): hand pointing "
                   "down, off to the side and below the book, so the grasp needs "
                   "a real ~90 deg reorient-and-insert. 'home': rest keyframe "
                   "(hardest -- reorients right next to the boards). 'preinsert': "
                   "already horizontal at the lip (easy, translation only). "
                   "'high': hand down ABOVE the compartment's top board, so the "
                   "straight descent is blocked and the planner must retreat-"
                   "descend-insert (the forced non-monotone detour). 'side': "
                   "beside the shelf at grasp depth and height, outside the "
                   "left side panel, so the detour is horizontal (retreat, "
                   "slide across, insert) and rides joint 1.")
    p.add_argument(
        "--book", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="TCP grasp goal, world xyz [m]. Default: GRASP0 + SHELF_OFFSET, "
        "i.e. it rides along when you move the shelf. Pass explicit coords to "
        "override.",
    )
    p.add_argument(
        "--w-wall", type=float, default=45.0,
        help="soft keep-out weight per board (matches the window spike)",
    )
    p.add_argument(
        "--margin", type=float, default=0.004,
        help="extra planning inflation of each board box [m], on top of the "
        "per-sphere radius (safety buffer for model error)",
    )
    p.add_argument(
        "--horizon", type=int, default=None,
        help="planning horizon in control steps (default: the task config's 15 "
        "= 0.75 s). The 'high' start needs a longer lookahead: its detour "
        "(back out, descend past shelf_01, re-insert) is ~0.6 m of TCP path, "
        "which does not fit in 0.75 s, so the sampler sees only the blocked "
        "straight descent",
    )
    p.add_argument(
        "--plan-boards", nargs="*", default=None,
        help="boards the planner scores against (default: shelf_00 shelf_01, "
        "the two bounding the target compartment). Starts INSIDE another "
        "compartment (--start high) also need the boards they must back out "
        "past, e.g. --plan-boards shelf_00 shelf_01 shelf_02",
    )
    p.add_argument(
        "--board-weight", nargs="*", default=[], metavar="NAME=W",
        help="per-board keep-out weight override, e.g. left_side=1500. Use it "
        "to make the board that FORCES the detour hard while the compartment "
        "boards stay soft (a single --w-wall cannot do both)",
    )
    p.add_argument(
        "--board-margin", nargs="*", default=[], metavar="NAME=M",
        help="per-board planning inflation override [m], same idea as "
        "--board-weight (a global --margin also shrinks the insertion slot)",
    )
    p.add_argument(
        "--collision", choices=["arm", "tcp"], default="arm",
        help="'arm' = whole-arm keep-out (FK collision spheres over every "
        "link); 'tcp' = tool-center-point only (the original behaviour)",
    )
    p.add_argument(
        "--approach-axis", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="desired world direction of the gripper approach axis at the "
        "book: default is the shelf-frame +x (straight into the compartment), "
        "rotated by SHELF_YAW",
    )
    p.add_argument(
        "--finger-axis", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="desired world direction of the gripper finger-opening axis "
        "(TCP local +y). Default is the shelf-frame +y rotated by SHELF_YAW, "
        "which rolls the hand 90 deg to a sideways/'laid down' grasp matching "
        "the turned book. Set 0 0 0 to leave the roll free (constrain the "
        "approach axis only).",
    )
    p.add_argument("--w-orient", type=float, default=4.0,
                   help="orientation cost weight (0 disables orientation)")
    p.add_argument("--orient-tol-deg", type=float, default=25.0,
                   help="free cone half-angle [deg]: the approach axis is only "
                   "penalized beyond this, so the hand is 'roughly horizontal "
                   "enough to grasp' rather than exactly aligned")
    p.add_argument("--arm-radius", type=float, default=0.04,
                   help="collision-sphere radius for the arm links [m]")
    p.add_argument("--tcp-radius", type=float, default=0.04,
                   help="collision-sphere radius at the TCP / hand [m]")
    p.add_argument("--link-samples", type=int, default=1,
                   help="interpolated collision spheres per link segment")
    p.add_argument("--first-link", type=int, default=4,
                   help="first arm frame included in the collision cloud "
                   "(0=base .. 7=link7); proximal links never reach the shelf")
    p.add_argument("--drop-boards", nargs="*", default=[],
                   help="board names to remove (e.g. 'shelf_01' to open a tall "
                   "slot between shelf_00 and shelf_02); must match the scene")
    p.add_argument("--sqp-iters", type=int, default=4,
                   help="bk_qp_sqp only: SQP re-freezes of the bilinear term "
                   "along the nominal rollout per control step (matches the "
                   "window spike)")
    p.add_argument("--qp-act-dist", type=float, default=0.10,
                   help="bk_qp_sqp only: a keep-out face constraint is added "
                   "only when the nominal sphere is within this distance of "
                   "the inflated board [m]; farther constraints are inactive "
                   "anyway and are pruned for QP size")
    p.add_argument("--num-samples", type=int, default=800)
    p.add_argument("--num-diffusion-steps", type=int, default=5)
    p.add_argument("--sigma-start", type=float, default=1.4)
    p.add_argument("--sigma-end", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--settle-steps", type=int, default=8,
                   help="hold after a strict reach before stopping")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--warmup-plans", type=int, default=3)
    p.add_argument("--debug-violations", action="store_true",
                   help="print which board each executed penetration hits")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--realtime", action="store_true",
                   help="pace the viewer to ~1x wall-clock (default: as fast as it renders)")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "out" / "franka" / "bookshelf")
    return p.parse_args()


# -------------------------------------------------------------- arm kinematics
class ArmKinematics:
    """Batched forward kinematics for a *set of collision spheres* covering the
    whole arm, not just the TCP.

    The Koopman rollout decodes the joint angles q at every horizon step, so we
    can run the same model-consistent FK the task uses for the TCP, but capture
    every joint-frame origin along the chain (verified to match MuJoCo's body
    xpos) plus interpolated samples along each link. Each point carries a radius
    so a link is approximated as a chain of spheres. This keeps the whole-arm
    keep-out inside the cheap torch rollout -- no MuJoCo in the candidate loop.
    """

    def __init__(self, task: FrankaTask, device, link_samples: int,
                 arm_radius: float, tcp_radius: float, first_link: int = 3):
        segs, tail = task._fk_params
        self.fixed = [torch.as_tensor(f, dtype=torch.float32, device=device)
                      for f, _ in segs]
        self.axes = [a for _, a in segs]
        self.tail = torch.as_tensor(tail, dtype=torch.float32, device=device)
        self.device = device
        self.link_samples = int(link_samples)
        # Origin order: base(0,0,0), link1..link7 frame origins, TCP. Proximal
        # links never approach the front shelf, so keep only origins from
        # `first_link` onward (0=base .. 7=link7, 8=TCP) to cut the cost tensor.
        self.first_link = int(first_link)
        origin_r = ([arm_radius] * (1 + NUM_JOINTS) + [tcp_radius])[self.first_link:]
        m = len(origin_r)                                   # kept origins
        radii = list(origin_r)
        for _ in range(self.link_samples):                  # midpoints per segment
            radii += [arm_radius] * (m - 1)
        self.radii = torch.as_tensor(radii, dtype=torch.float32, device=device)
        self.num_points = len(radii)

    def _origins(self, qf: torch.Tensor) -> torch.Tensor:
        """qf: (N, 7) -> (N, m, 3) frame origins from `first_link` to TCP."""

        n = qf.shape[0]
        T = torch.eye(4, dtype=torch.float32, device=self.device).expand(n, 4, 4)
        outs = [torch.zeros(n, 3, dtype=torch.float32, device=self.device)]  # base
        for i in range(NUM_JOINTS):
            T = T @ self.fixed[i]
            T = T @ _axis_angle_mat(self.axes[i], qf[:, i], torch.float32, self.device)
            outs.append(T[:, :3, 3])
        T = T @ self.tail
        outs.append(T[:, :3, 3])                             # TCP
        return torch.stack(outs[self.first_link:], dim=1)

    def points(self, q: torch.Tensor) -> torch.Tensor:
        """q: (..., 7) -> (..., P, 3) whole-arm collision spheres' centers."""

        shape = q.shape[:-1]
        o = self._origins(q.reshape(-1, NUM_JOINTS))         # (N, m, 3)
        pts = [o]
        for s in range(1, self.link_samples + 1):
            f = s / (self.link_samples + 1)
            pts.append(o[:, :-1] * (1 - f) + o[:, 1:] * f)   # (N, m-1, 3)
        p = torch.cat(pts, dim=1)                            # (N, P, 3)
        return p.reshape(*shape, p.shape[1], 3)

    def points_and_jacobians(self, q: torch.Tensor, eps: float = 1e-3):
        """q: (N, 7) -> (points (N, P, 3), J (N, P, 3, 7)) with J the central
        finite-difference Jacobian of every collision-sphere center w.r.t. the
        joints. Used by the QP-SQP baseline to linearize the whole-arm
        keep-out around the nominal joint trajectory; one batched FK call of
        size 14N keeps it cheap."""

        n = q.shape[0]
        p0 = self.points(q)                                   # (N, P, 3)
        eye = eps * torch.eye(NUM_JOINTS, dtype=q.dtype, device=self.device)
        qp = (q[:, None] + eye[None]).reshape(-1, NUM_JOINTS)  # (N*7, 7)
        qm = (q[:, None] - eye[None]).reshape(-1, NUM_JOINTS)
        pp = self.points(torch.cat([qp, qm], dim=0))           # (2*N*7, P, 3)
        d = pp[: n * NUM_JOINTS] - pp[n * NUM_JOINTS:]
        J = d.reshape(n, NUM_JOINTS, self.num_points, 3).permute(0, 2, 3, 1)
        return p0, J / (2.0 * eps)

    def tcp_axes(self, q: torch.Tensor):
        """q: (..., 7) -> (approach, finger), each (..., 3) world unit vectors.

        The Koopman state carries no orientation, but FK recovers the full TCP
        frame from the decoded joints, so the planner can target the gripper's
        approach axis (TCP-frame local +z, out between the fingers) AND its roll
        via the finger-opening axis (local +y, the direction the two fingers
        separate along -- verified from the hand model: fingers sit at +-0.02 in
        local y). Constraining both fixes the full grasp orientation (up to the
        parallel-jaw 180 deg symmetry handled in the cost)."""

        shape = q.shape[:-1]
        qf = q.reshape(-1, NUM_JOINTS)
        n = qf.shape[0]
        T = torch.eye(4, dtype=torch.float32, device=self.device).expand(n, 4, 4)
        for i in range(NUM_JOINTS):
            T = T @ self.fixed[i]
            T = T @ _axis_angle_mat(self.axes[i], qf[:, i], torch.float32, self.device)
        T = T @ self.tail
        approach = T[:, :3, 2].reshape(*shape, 3)
        finger = T[:, :3, 1].reshape(*shape, 3)
        return approach, finger


# ---------------------------------------------------------------- keep-out cost
class ShelfKeepout:
    """Soft keep-out penalty of a set of collision spheres against the board
    boxes. With a single zero-radius TCP point this reduces to the TCP-only
    penalty; with the ArmKinematics point cloud it avoids whole-arm collision.
    """

    def __init__(self, boards, margin: float, weight: float, device,
                 weight_by_board=None, margin_by_board=None):
        """``weight_by_board`` / ``margin_by_board`` map a board name to a value
        that overrides the scalar default for that board alone.

        Per-board values matter because one scalar cannot serve both roles the
        penalty plays: a board that FORCES a detour (the side panel the arm has
        to come around) needs a weight big enough to beat the reach cost, while
        the boards merely BOUNDING the target compartment must stay gentle --
        cranking those, or inflating them via the margin, chokes the insertion
        slot and the arm stalls in front of the mouth instead of reaching in.
        """

        c = np.array([b[1] for b in boards], dtype=np.float32)
        h = np.array([b[2] for b in boards], dtype=np.float32)
        self.centers = torch.as_tensor(c, device=device)          # (K, 3)
        self.halfs = torch.as_tensor(h, device=device)            # (K, 3)
        self.names = [b[0] for b in boards]
        wb, mb = weight_by_board or {}, margin_by_board or {}
        self.weights = torch.as_tensor(                           # (K,)
            [float(wb.get(n, weight)) for n in self.names],
            dtype=torch.float32, device=device)
        self.margins = torch.as_tensor(                           # (K,)
            [float(mb.get(n, margin)) for n in self.names],
            dtype=torch.float32, device=device)
        self.margin = float(margin)
        self.weight = float(weight)

    def penalty(self, points: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """points: (B, T, P, 3), radii: (P,) -> (B,) summed keep-out penalty.

        Each sphere inside a board box inflated by (radius + margin) is charged
        ``weight * depth**2`` where depth is the distance to the nearest face,
        with weight and margin taken per board.
        """

        # inflate[p,k,3] = halfs[k] + radii[p] + margin[k]
        inflate = (self.halfs[None] + radii[:, None, None]
                   + self.margins[None, :, None])                          # (P,K,3)
        diff = torch.abs(points[:, :, :, None, :] - self.centers[None, None, None])
        slack = inflate[None, None] - diff                                 # (B,T,P,K,3)
        inside = (slack > 0).all(dim=-1)                                   # (B,T,P,K)
        depth = slack.min(dim=-1).values.clamp(min=0.0)                    # (B,T,P,K)
        charged = inside.float() * depth ** 2 * self.weights               # (B,T,P,K)
        return charged.sum(dim=(1, 2, 3))

    def violates(self, points: np.ndarray, radii: np.ndarray, names=False):
        """True keep-out test (no margin) for one executed arm point cloud.

        points: (P, 3), radii: (P,). Returns the number of spheres penetrating
        a board (each link sphere inflated only by its own radius). With
        names=True also returns the set of penetrated board names.
        """

        c = self.centers.cpu().numpy()
        h = self.halfs.cpu().numpy()
        diff = np.abs(points[:, None, :] - c[None])                        # (P,K,3)
        slack = (h[None] + radii[:, None, None]) - diff                    # (P,K,3)
        inside = (slack > 0).all(axis=2)                                   # (P,K)
        n = int(inside.any(axis=1).sum())
        if not names:
            return n
        hit = {self.names[k] for k in range(len(self.names)) if inside[:, k].any()}
        return n, hit


# ------------------------------------------------------------ QP-SQP baseline
class QPSQPPlanner:
    """Convexified bilinear Koopman MPC, ported from the window spike's
    ``run_qp`` to the bookshelf boxes.

    Per control step it repeats ``sqp_iters`` times: (1) roll the TRUE
    bilinear model along the nominal controls, (2) freeze the bilinear input
    map at the nominal lifted states (B_k = B0 + Bs . zbar_k), (3) solve a QP
    with the exact quadratic task cost and the board keep-outs convexified by
    the branch the nominal selects. The branch logic is the box analogue of
    the window's: for every collision sphere near a board, the nominal picks
    ONE separating face (the axis of maximum signed separation) and the QP
    constrains the linearized sphere center to that face's outer half-space.
    It cannot re-select the face/homotopy by itself -- the honest per-step
    convexification.

    Whole-arm sphere centers are nonlinear in q, so they enter through the
    first-order FK expansion p ~= p_nom + J (q - q_nom) around the nominal
    joints (ArmKinematics.points_and_jacobians); with ``--collision tcp`` the
    TCP is a LINEAR readout of the lifted state and is constrained exactly,
    which reduces to the window spike's formulation.
    """

    def __init__(self, task: FrankaTask, model, armk: ArmKinematics, boards,
                 radii: np.ndarray, margin: float, whole_arm: bool,
                 horizon: int, sqp_iters: int, act_dist: float, device):
        import cvxpy as cp  # local: only the QP baseline needs it

        self.cp = cp
        self.task, self.model, self.armk, self.device = task, model, armk, device
        self.A = model.A.detach().cpu().numpy().astype(np.float64)
        self.B0 = model.B0.detach().cpu().numpy().astype(np.float64)
        self.Bs = model.Bs.detach().cpu().numpy().astype(np.float64)
        self.centers = np.array([b[1] for b in boards], dtype=np.float64)
        self.halfs = np.array([b[2] for b in boards], dtype=np.float64)
        self.radii = np.asarray(radii, dtype=np.float64)
        self.margin = float(margin)
        self.whole_arm = bool(whole_arm)
        self.H = int(horizon)
        self.sqp_iters = int(sqp_iters)
        self.act_dist = float(act_dist)
        self.u_lim = float(task.config.action_limit)
        self.w = task.cost_weights
        self.n_solve_fail = 0
        self.n_cons_last = 0

    def _nominal(self, z0: np.ndarray, U: np.ndarray):
        """True bilinear rollout under U: lifted states BEFORE each step, the
        decoded joints/tips AFTER each step, and the frozen input maps."""

        zbars = np.empty((self.H, z0.shape[0]), dtype=np.float64)
        qn = np.empty((self.H, NUM_JOINTS), dtype=np.float64)
        tn = np.empty((self.H, 3), dtype=np.float64)
        with torch.no_grad():
            z = torch.as_tensor(z0, dtype=torch.float32)
            for k in range(self.H):
                zbars[k] = z.numpy().astype(np.float64)
                z = self.model.step(z, torch.as_tensor(U[k], dtype=torch.float32))
                dec = self.model.decode(z).numpy().astype(np.float64)
                qn[k] = dec[:NUM_JOINTS]
                tn[k] = dec[NUM_JOINTS:NUM_JOINTS + 3]
            if self.whole_arm:
                qt = torch.as_tensor(qn, dtype=torch.float32, device=self.device)
                p, J = self.armk.points_and_jacobians(qt)
                pts = p.numpy().astype(np.float64)          # (H, P, 3)
                jac = J.numpy().astype(np.float64)          # (H, P, 3, 7)
            else:
                pts = tn[:, None, :]                        # (H, 1, 3)
                jac = None
        B_list = [self.B0 + np.einsum("mij,j->im", self.Bs, zb) for zb in zbars]
        return B_list, qn, pts, jac

    def _keepout_rows(self, k: int, qn: np.ndarray, pts: np.ndarray, jac):
        """Branch-selected separating-face rows for step k: G @ q_e >= h."""

        G_rows, h_rows = [], []
        for i in range(pts.shape[1]):
            p_nom = pts[k, i]
            for b in range(self.centers.shape[0]):
                c, hh = self.centers[b], self.halfs[b]
                infl = hh + self.radii[i] + self.margin
                d = np.abs(p_nom - c) - infl        # signed separation per axis
                if d.max() > self.act_dist:
                    continue                        # far from this board
                j = int(np.argmax(d))               # nominal's separating face
                s = 1.0 if p_nom[j] >= c[j] else -1.0
                if jac is None:
                    # exact linear TCP readout:
                    # s*(tip[j]-c[j]) >= infl[j]  ->  s*tip[j] >= infl[j]+s*c[j]
                    row = np.zeros(3)
                    row[j] = s
                    G_rows.append(("tip", row))
                    h_rows.append(infl[j] + s * c[j])
                else:
                    Jij = jac[k, i, j]              # (7,)
                    # s*(p_nom[j] + Jij.(q_e - qn_k) - c[j]) >= infl[j]
                    G_rows.append(("arm", s * Jij))
                    h_rows.append(
                        infl[j] - s * (p_nom[j] - c[j]) + s * float(Jij @ qn[k]))
        return G_rows, h_rows

    def plan(self, x: np.ndarray, U_nom: np.ndarray, goal: np.ndarray) -> np.ndarray:
        cp = self.cp
        b0 = self.task.state_to_base_torch(x, self.device)
        z0 = self.model.lift(b0).detach().numpy().astype(np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        U = np.asarray(U_nom, dtype=np.float64).copy()
        n_cons = 0
        for _ in range(self.sqp_iters):
            B_list, qn, pts, jac = self._nominal(z0, U)
            U_var = cp.Variable((self.H, NUM_JOINTS))
            cons = [U_var >= -self.u_lim, U_var <= self.u_lim]
            cost = 0
            z_expr = z0
            for k in range(self.H):
                z_expr = self.A @ z_expr + B_list[k] @ U_var[k]
                tip_e = z_expr[NUM_JOINTS:NUM_JOINTS + 3]
                q_e = z_expr[:NUM_JOINTS]
                cost = cost + self.w.ee * cp.sum_squares(tip_e - goal) \
                    + self.w.control * cp.sum_squares(U_var[k])
                G_rows, h_rows = self._keepout_rows(k, qn, pts, jac)
                if not G_rows:
                    continue
                n_cons += len(G_rows)
                arm_rows = [(g, h) for (kind, g), h in zip(G_rows, h_rows)
                            if kind == "arm"]
                tip_rows = [(g, h) for (kind, g), h in zip(G_rows, h_rows)
                            if kind == "tip"]
                if arm_rows:
                    G = np.stack([g for g, _ in arm_rows])
                    hv = np.array([h for _, h in arm_rows])
                    cons.append(G @ q_e >= hv)
                if tip_rows:
                    G = np.stack([g for g, _ in tip_rows])
                    hv = np.array([h for _, h in tip_rows])
                    cons.append(G @ tip_e >= hv)
            cost = cost + self.w.terminal_ee * cp.sum_squares(
                z_expr[NUM_JOINTS:NUM_JOINTS + 3] - goal)
            prob = cp.Problem(cp.Minimize(cost), cons)
            try:
                prob.solve(solver=cp.OSQP, warm_start=True)
            except cp.error.SolverError:
                pass
            if U_var.value is None:
                try:
                    prob.solve(solver=cp.CLARABEL)
                except cp.error.SolverError:
                    pass
            if U_var.value is not None:
                U = np.clip(np.asarray(U_var.value), -self.u_lim, self.u_lim)
            else:
                self.n_solve_fail += 1  # keep the previous nominal (spike behaviour)
        self.n_cons_last = n_cons // max(1, self.sqp_iters)
        return U


# --------------------------------------------------------------------- backends
def make_backend(task: FrankaTask, method: str, model, device):
    """Return (rollout_tips_and_bs) closures for the chosen rollout model."""

    if method == MethodName.DK_MBD_SPLIT.value:
        def rollout(x, U_t):
            q0 = torch.as_tensor(np.asarray(x, np.float64)[:NUM_JOINTS],
                                 dtype=torch.float32, device=device)
            z0 = model.lift(q0).expand(U_t.shape[0], -1)
            q_hat = model.decode(model.rollout(z0, U_t))
            bs = torch.cat([q_hat, task.forward_kinematics_torch(q_hat)], dim=-1)
            return bs
        return rollout

    def rollout(x, U_t):
        b0 = task.state_to_base_torch(x, device)
        z0 = model.lift(b0).expand(U_t.shape[0], -1)
        return model.decode(model.rollout(z0, U_t))
    return rollout


def run(args) -> None:
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")

    task = (FrankaTask() if args.horizon is None
            else FrankaTask(FrankaTaskConfig(horizon=args.horizon)))
    period = task.config.control_dt
    horizon = task.config.horizon
    strict = task.config.strict_threshold
    reach = task.config.reach_threshold

    short = {MethodName.BK_MBD.value: "bk", MethodName.DK_MBD.value: "dk",
             MethodName.DK_MBD_SPLIT.value: "dk", "bk_qp_sqp": "bk"}[args.method]
    ckpt = args.checkpoint or (
        PROJECT_ROOT / "out" / "franka" / "models" / f"{short}_seed{args.seed}.pt")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    model, _ = load_checkpoint(ckpt, device=device)
    model.eval()

    drop = set(args.drop_boards)
    R = _shelf_rot()
    shifted = [(n, tuple(R @ np.asarray(c) + SHELF_OFFSET),
                tuple(np.abs(R) @ np.asarray(h))) for n, c, h in BOARDS]
    all_boards = [b for b in shifted if b[0] not in drop]
    plan_names = (START_PLAN_BOARDS.get(args.start, PLAN_BOARD_NAMES)
                  if args.plan_boards is None else set(args.plan_boards))
    unknown = plan_names - {b[0] for b in shifted}
    if unknown:
        raise SystemExit(f"unknown board name(s): {sorted(unknown)}")
    plan_boards = [b for b in all_boards if b[0] in plan_names]
    def _by_board(pairs, flag):
        out = {}
        for item in pairs:
            name, _, val = item.partition("=")
            if not _ or name not in {b[0] for b in shifted}:
                raise SystemExit(f"{flag} expects NAME=VALUE with a board name, "
                                 f"got {item!r}")
            out[name] = float(val)
        return out

    w_by = _by_board(args.board_weight, "--board-weight")
    m_by = _by_board(args.board_margin, "--board-margin")
    keepout = ShelfKeepout(plan_boards, args.margin, args.w_wall, device,
                           weight_by_board=w_by, margin_by_board=m_by)  # planner
    check_keepout = ShelfKeepout(all_boards, 0.0, args.w_wall, device)     # honest check
    rollout = make_backend(task, args.method, model, device)
    goal = (np.asarray(args.book, dtype=np.float64) if args.book is not None
            else R @ GRASP0 + SHELF_OFFSET)
    w = task.cost_weights

    # FK collision/orientation kinematics are always available (orientation
    # needs FK even in TCP-only collision mode).
    armk = ArmKinematics(task, device, args.link_samples,
                         args.arm_radius, args.tcp_radius, args.first_link)
    whole_arm = args.collision == "arm"
    if whole_arm:
        collision_radii = armk.radii
    else:
        collision_radii = torch.zeros(1, device=device)  # single TCP point
    check_radii = collision_radii.cpu().numpy()

    # Orientation targets default to the SHELF frame (+x approach into the
    # compartment, +y finger opening) and ride along with SHELF_YAW; explicit
    # CLI axes are taken as world directions verbatim.
    approach = (np.asarray(args.approach_axis, np.float64)
                if args.approach_axis is not None
                else R @ np.array([1.0, 0.0, 0.0]))
    finger = (np.asarray(args.finger_axis, np.float64)
              if args.finger_axis is not None
              else R @ np.array([0.0, 1.0, 0.0]))
    a_des = torch.as_tensor(
        (approach / (np.linalg.norm(approach) + 1e-9)).astype(np.float32),
        device=device)
    use_orient = args.w_orient > 0
    cos_tol = float(np.cos(np.radians(args.orient_tol_deg)))
    f_norm = float(np.linalg.norm(finger))
    use_roll = use_orient and f_norm > 1e-6           # constrain the gripper roll too
    f_des = (torch.as_tensor((finger / (f_norm + 1e-9)).astype(np.float32),
                             device=device) if use_roll else None)

    def collision_points(bs):
        """bs: (B, T+1, 10) -> (B, T, P, 3) collision-sphere centers over the
        executed horizon (drops the fixed initial step at index 0)."""

        if whole_arm:
            return armk.points(bs[:, 1:, :NUM_JOINTS])            # (B,T,P,3)
        return bs[:, 1:, NUM_JOINTS:NUM_JOINTS + 3][:, :, None, :]  # (B,T,1,3)

    def make_evaluate(x, u_prev):
        def evaluate(cands):
            with torch.no_grad():
                U_t = torch.as_tensor(cands, dtype=torch.float32, device=device)
                bs = rollout(x, U_t)                       # (B, T+1, 10)
                costs = task.trajectory_cost_base_torch(bs, U_t, goal, u_prev=u_prev)
                pts = collision_points(bs)
                costs = costs + keepout.penalty(pts, collision_radii)
                if use_orient:
                    ax, fx = armk.tcp_axes(bs[:, 1:, :NUM_JOINTS])       # (B,T,3) each
                    dev_a = (cos_tol - (ax * a_des).sum(dim=-1)).clamp(min=0.0)
                    dev2 = dev_a ** 2
                    if use_roll:
                        # |.| : parallel-jaw grasp is symmetric under 180 deg roll
                        align_f = (fx * f_des).sum(dim=-1).abs()
                        dev2 = dev2 + (cos_tol - align_f).clamp(min=0.0) ** 2
                    costs = costs + args.w_orient * (0.1 * dev2.sum(1) + dev2[:, -1])
                return costs.cpu().numpy()
        return evaluate

    def executed_points(x_state):
        """Whole-arm (or TCP) collision spheres for one executed true state."""

        q = torch.as_tensor(np.asarray(x_state, np.float64)[None, :NUM_JOINTS],
                            dtype=torch.float32, device=device)
        if whole_arm:
            return armk.points(q)[0].cpu().numpy()               # (P,3)
        return task.ee_of_q(np.asarray(x_state)[:NUM_JOINTS])[None]  # (1,3)

    def predict_tips(x, U):
        with torch.no_grad():
            U_t = torch.as_tensor(U[None], dtype=torch.float32, device=device)
            bs = rollout(x, U_t)
            return bs[0, :, NUM_JOINTS:NUM_JOINTS + 3].cpu().numpy()

    use_qp = args.method == "bk_qp_sqp"
    if use_qp and use_orient:
        print("WARNING: bk_qp_sqp does not convexify the orientation cost; "
              "it plans position-only. Run BOTH methods with --w-orient 0 "
              "for a matched comparison.")
    qp_planner = None
    optimizer = None
    if use_qp:
        qp_planner = QPSQPPlanner(
            task, model, armk, plan_boards, check_radii, args.margin,
            whole_arm, horizon, args.sqp_iters, args.qp_act_dist, device)
    else:
        mbd_config = MBDConfig(
            num_samples=args.num_samples, num_diffusion_steps=args.num_diffusion_steps,
            sigma_start=args.sigma_start, sigma_end=args.sigma_end, alpha=args.alpha,
            eta=args.eta, update_rule=UpdateRule.SCORE_LANGEVIN, seed=args.seed,
            add_langevin_noise=False,
        )
        optimizer = MBDOptimizer(mbd_config, task.action_bounds[0], task.action_bounds[1])

    # Viewer scene: the bookshelf scene (its FR3 matches the task model). The
    # boards are non-colliding, so stepping this sim with the same controls as
    # the planner's true_step keeps the two in lockstep.
    scene = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    if scene.nq != NUM_JOINTS or scene.nu != NUM_JOINTS:
        raise SystemExit("bookshelf scene does not match the 7-DoF task model")
    # Place the shelf, the book, and the floor from SHELF_OFFSET so the single
    # constant drives both the planner boxes and everything visual. The floor
    # drops to the shelf's underside so it rests on it (no clipping); a pedestal
    # in the XML fills the gap under the arm.
    yaw_quat = np.array([np.cos(np.radians(SHELF_YAW) / 2), 0.0, 0.0,
                         np.sin(np.radians(SHELF_YAW) / 2)])
    scene.body("bookshelf").pos[:] = SHELF_OFFSET
    scene.body("bookshelf").quat[:] = yaw_quat  # geoms live in the body frame
    scene.body("target_book").pos[:] = R @ BOOK_POS0 + SHELF_OFFSET
    scene.body("target_book").quat[:] = _quat_mul(
        yaw_quat, scene.body("target_book").quat.copy())
    # The base is BASE_HEIGHT above the floor: floor at -BASE_HEIGHT (the
    # shelf offset is chosen so the shelf bottom rests exactly on it), and the
    # pedestal spans the gap under the base.
    scene.geom("floor").pos[2] = -BASE_HEIGHT
    scene.geom("pedestal").pos[2] = -BASE_HEIGHT / 2
    scene.geom("pedestal").size[2] = BASE_HEIGHT / 2
    nsub = int(round(period / scene.opt.timestep))
    sub_dt = period / nsub
    sim = mujoco.MjData(scene)
    mujoco.mj_resetDataKeyframe(scene, sim, scene.key("home").id)
    # Start poses ride along with the shelf yaw: joint 1 turns about world z.
    start_qpos = _yawed_qpos({"preinsert": PREINSERT_QPOS, "tucked": TUCKED_QPOS,
                              "high": HIGH_QPOS, "side": SIDE_QPOS}
                             .get(args.start, task.home_qpos))
    sim.qpos[:NUM_JOINTS] = start_qpos
    sim.qvel[:] = 0.0
    mujoco.mj_forward(scene, sim)
    tcp = scene.site("tcp").id
    color = METHOD_COLORS.get(args.method, (0.6, 0.6, 0.6))

    start_tip = task.ee_of_q(start_qpos)
    print(f"method={args.method}  start={args.start}  book(goal)={np.round(goal, 3)}  "
          f"start_tip={np.round(start_tip, 3)}  |goal-start|={np.linalg.norm(goal-start_tip):.3f} m")
    pb = sorted((b for b in shifted if b[0] in {"shelf_00", "shelf_01"}),
                key=lambda b: b[1][2])
    lo_b, hi_b = pb[0], pb[-1]
    open_dir = R @ np.array([-1.0, 0.0, 0.0])       # asset opening faces -x
    ax = int(np.argmax(np.abs(open_dir)))
    lip = lo_b[1][ax] + np.sign(open_dir[ax]) * lo_b[2][ax]
    print(f"compartment: {lo_b[0]} top z={lo_b[1][2] + lo_b[2][2]:.3f} .. "
          f"{hi_b[0]} bottom z={hi_b[1][2] - hi_b[2][2]:.3f}; "
          f"front lip {'xyz'[ax]}={lip:.3f} (yaw {SHELF_YAW:.0f} deg, "
          f"base {BASE_HEIGHT:.2f} m above floor); "
          f"margin={args.margin} weight={args.w_wall}")
    if whole_arm:
        print(f"collision: WHOLE-ARM keep-out, {armk.num_points} spheres "
              f"(arm r={args.arm_radius}, tcp r={args.tcp_radius}, "
              f"{args.link_samples} samples/link)")
    else:
        print("collision: TCP-only keep-out (single point)")
    if use_qp:
        print(f"planner: convexified QP-SQP, {args.sqp_iters} re-freezes/step, "
              f"constraint activation distance {args.qp_act_dist} m, "
              f"planning boards {sorted(plan_names)}")
    if use_orient:
        roll_txt = (f", finger -> {np.round(f_des.cpu().numpy(), 2)}" if use_roll
                    else ", roll free")
        print(f"orientation: approach axis -> {np.round(a_des.cpu().numpy(), 2)}{roll_txt} "
              f"(w_orient={args.w_orient}, free cone {args.orient_tol_deg:.0f} deg); "
              f"drop_boards={sorted(drop) or 'none'}")

    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(scene, sim)

    # torch/threadpool (or cvxpy-compile) warm-up, excluded from latency stats.
    if use_qp:
        x_warm = np.concatenate([start_qpos, np.zeros(NUM_JOINTS)]).astype(np.float64)
        for _ in range(args.warmup_plans):
            qp_planner.plan(x_warm, np.zeros((horizon, NUM_JOINTS)), goal)
        qp_planner.n_solve_fail = 0
    else:
        warm_eval = make_evaluate(
            np.concatenate([task.home_qpos, np.zeros(NUM_JOINTS)]), np.zeros(NUM_JOINTS))
        warm_rng = np.random.default_rng(10**6 + args.seed)
        for _ in range(args.warmup_plans):
            optimizer.optimize(np.zeros((horizon, NUM_JOINTS)), warm_eval, rng=warm_rng)

    rng = np.random.default_rng(args.seed)
    x = np.concatenate([sim.qpos[:NUM_JOINTS], sim.qvel[:NUM_JOINTS]]).astype(np.float64)
    U = np.zeros((horizon, NUM_JOINTS))
    u_prev = np.zeros(NUM_JOINTS)
    trail: List[np.ndarray] = []
    trail_q: List[np.ndarray] = []          # executed joints, for diagnostics
    errors: List[float] = []
    latencies: List[float] = []
    violations = 0
    min_err = np.inf
    t_reach = None
    reached_at = None

    def draw():
        if viewer is None:
            return
        scn = viewer.user_scn
        scn.ngeom = 0

        def sphere(pos, size, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                [size, 0, 0], np.asarray(pos, np.float64),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            scn.ngeom += 1

        def path(pts, radius, rgba):
            pts = np.asarray(pts)
            for a, b in zip(pts[:-1], pts[1:]):
                if scn.ngeom >= scn.maxgeom:
                    return
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                    np.zeros(3), np.eye(3).flatten(),
                                    np.asarray(rgba, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, a, b)
                scn.ngeom += 1

        sphere(goal, 0.012, [0.1, 0.85, 0.1, 1.0])
        sphere(goal, strict, [0.1, 0.85, 0.1, 0.15])
        if len(pred) >= 2:
            path(pred, 0.0015, [*color, 0.4])
        if len(trail) >= 2:
            path(trail, 0.002, [*color, 0.9])

    pred = np.empty((0, 3))
    t0 = time.perf_counter()
    for step in range(args.max_steps):
        if viewer is not None and not viewer.is_running():
            break
        ee = sim.site_xpos[tcp].copy()
        err = float(np.linalg.norm(ee - goal))
        min_err = min(min_err, err)
        errors.append(err)
        trail.append(ee)
        trail_q.append(sim.qpos[:NUM_JOINTS].copy())
        n_pen, hit = check_keepout.violates(executed_points(x), check_radii, names=True)
        if n_pen:
            violations += 1
            if args.debug_violations:
                print(f"[step {step:3d}] VIOLATION boards={sorted(hit)} "
                      f"tcp_z={ee[2]:.3f}")
        if t_reach is None and err < strict:
            t_reach = step
            reached_at = step
            print(f"[step {step:3d}] strict reach ({err*1000:.1f} mm)")
        if reached_at is not None and step - reached_at >= args.settle_steps:
            break

        tp = time.perf_counter()
        if use_qp:
            U = qp_planner.plan(x, U, goal)
        else:
            result = optimizer.optimize(U, make_evaluate(x, u_prev), rng=rng)
            U = result.controls
        latencies.append(time.perf_counter() - tp)
        u0 = np.clip(U[0], task.action_bounds[0], task.action_bounds[1])
        pred = predict_tips(x, U)

        # Advance the true task state (planner) and mirror it in the viewer sim.
        x = task.true_step(x, u0)
        sim.ctrl[:] = u0
        base = time.perf_counter()
        for i in range(nsub):
            mujoco.mj_step(scene, sim)
            if viewer is not None and i % max(1, nsub // 4) == 0:
                viewer.sync()
            if args.realtime:
                slack = base + (i + 1) * sub_dt - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
        if viewer is not None:
            draw()
            viewer.sync()

        u_prev = u0
        U = np.roll(U, -1, axis=0)
        U[-1] = U[-2]
    wall = time.perf_counter() - t0

    # ---- final state / verdict
    ee = sim.site_xpos[tcp].copy()
    final_err = float(np.linalg.norm(ee - goal))
    final_R = sim.site_xmat[tcp].reshape(3, 3)
    a_des_np = a_des.cpu().numpy()
    orient_deg = float(np.degrees(np.arccos(np.clip(final_R[:, 2] @ a_des_np, -1, 1))))
    roll_deg = None
    if use_roll:
        f_des_np = f_des.cpu().numpy()
        roll_deg = float(np.degrees(np.arccos(np.clip(abs(final_R[:, 1] @ f_des_np), 0, 1))))
    lat = np.asarray(latencies) * 1000.0
    reached = t_reach is not None            # touched the strict shell at least once
    ok = reached and final_err < reach and violations == 0
    if reached and violations == 0 and final_err >= reach:
        why = "reached then drifted out of the reach shell"
    elif not reached:
        why = "never reached the strict shell (stall)"
    elif violations > 0:
        why = f"{violations} executed board penetration(s)"
    else:
        why = ""
    print("\n" + "=" * 64)
    print(f"BOOKSHELF REACH  ({args.method})")
    print("=" * 64)
    print(f"final error: {final_err*1000:.1f} mm  (min {min_err*1000:.1f} mm; "
          f"strict<{strict*1000:.0f} reach<{reach*1000:.0f})")
    if use_orient:
        line = f"approach-axis error: {orient_deg:.1f} deg from {np.round(a_des_np, 2)}"
        if roll_deg is not None:
            line += f" | roll(finger) error: {roll_deg:.1f} deg from {np.round(f_des.cpu().numpy(), 2)}"
        print(line)
    print(f"executed board violations: {violations}")
    print(f"strict reach at step: {t_reach if t_reach is not None else 'NONE'}")
    if len(lat):
        print(f"plan latency: mean {lat.mean():.1f} ms | p95 "
              f"{np.percentile(lat,95):.1f} | max {lat.max():.1f}  ({len(lat)} plans)")
    if use_qp:
        print(f"qp: solver failures {qp_planner.n_solve_fail} "
              f"(kept previous nominal), ~{qp_planner.n_cons_last} active "
              f"keep-out rows/iter at the last plan")
    print(f"verdict: {'PASS - reached the book, no board hit' if ok else 'FAIL - ' + why}")
    print("=" * 64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_seed{args.seed}"
    np.savez(args.output_dir / f"{stem}.npz",
             trail=np.stack(trail) if trail else np.zeros((0, 3)),
             trail_q=np.stack(trail_q) if trail_q else np.zeros((0, NUM_JOINTS)),
             errors=np.asarray(errors), plan_latencies=np.asarray(latencies),
             goal=goal, violations=violations,
             boards_center=keepout.centers.cpu().numpy(),
             boards_half=keepout.halfs.cpu().numpy(),
             final_error=final_err, wall_time=wall)
    print(f"saved={args.output_dir / (stem + '.npz')}")

    if viewer is not None:
        # Loop a kinematic replay of the executed episode (with the TCP trail
        # growing along) until the viewer is closed.
        print("replaying the episode in a loop -- close the viewer (ESC) to exit")
        full_q = list(trail_q)
        full_trail = list(trail)
        pred = np.empty((0, 3))
        try:
            while viewer.is_running() and full_q:
                for i, qr in enumerate(full_q):
                    if not viewer.is_running():
                        break
                    sim.qpos[:NUM_JOINTS] = qr
                    sim.qvel[:] = 0.0
                    mujoco.mj_forward(scene, sim)
                    trail[:] = full_trail[: i + 1]
                    draw()
                    viewer.sync()
                    time.sleep(period)
                time.sleep(0.8)
        except KeyboardInterrupt:
            pass
        viewer.close()


if __name__ == "__main__":
    run(parse_args())
