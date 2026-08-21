"""Visualize the convex approximation the SQP-MPC actually optimizes against.

The whole point of paper section C: an obstacle box carves a NON-convex free
space (outside the box = the union of six half-spaces). A convexified planner
cannot express that union, so each SQP iterate the nominal picks, per arm
sphere near a box, ONE separating face and replaces "stay outside the box" with
the single linear half-space "stay on the outer side of THIS face's plane". The
QP then optimizes as if everything on the outer side of that plane were free --
including the region straight through the rest of the wall -- and it is locked
to that branch; it cannot re-select the face within the solve.

This module draws those selected face planes as translucent quads: the visible
boundary of the linearized (convex) region the QP believes in. Watching them,
you see the QP commit to one homotopy branch and press into it. Kept separate
from :mod:`secy.viewer` so the feature is optional and self-contained -- a
planner exposes ``last_linearization`` (a list of :class:`FacePlane`); the
runtime hands it here to draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import mujoco

#: translucent blue, matching the SQP planner's overlay color
FACE_RGBA = (0.12, 0.47, 0.71, 0.25)
#: how far past the box the drawn plane extends, to read as a half-space cut
PAD = 0.30
#: drawn thickness of the plane along its normal
THICKNESS = 0.004
#: green arrows point to the ALLOWED (free) side the QP believes in
ARROW_RGBA = (0.15, 0.90, 0.30, 0.95)
ARROW_LEN = 0.10
ARROW_WIDTH = 0.006
#: arrows are placed on an ARROW_GRID x ARROW_GRID lattice across each face,
#: so the free side reads from any camera angle
ARROW_GRID = 2
#: faint green volume filling the allowed half-space near the boundary
FILL_RGBA = (0.9, 0.1, 0.1, 0.55)
FILL_DEPTH = 0.15


@dataclass(frozen=True)
class FacePlane:
    """One branch-selected separating face the QP linearized against.

    The keep-out constraint is ``sign * (p[axis] - coord) >= 0`` on the
    linearized sphere center -- i.e. stay on ``sign`` side of the axis-aligned
    plane at ``coord``. ``center``/``half`` are the source box, used only to
    place and size the drawn quad on the two in-plane axes.
    """

    axis: int            # 0/1/2: the separating axis
    sign: float          # +1 -> outer side is +axis, -1 -> -axis
    coord: float         # world coordinate of the (inflated) face along axis
    center: tuple        # source box center (3,)
    half: tuple          # source box half-extents (3,)


def _add_arrow(scn, frm, to, width, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=np.zeros(3), pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(frm, dtype=np.float64),
                         np.asarray(to, dtype=np.float64))
    scn.ngeom += 1


def _add_box(scn, pos, half, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=np.asarray(half, dtype=np.float64),
                        pos=np.asarray(pos, dtype=np.float64),
                        mat=np.eye(3).flatten(),
                        rgba=np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def draw_faces(scn, planes: Optional[List[FacePlane]], rgba=FACE_RGBA,
               pad: float = PAD, thickness: float = THICKNESS,
               arrows: bool = True, arrow_rgba=ARROW_RGBA,
               arrow_len: float = ARROW_LEN, arrow_width: float = ARROW_WIDTH,
               arrow_grid: int = ARROW_GRID, fill: bool = False,
               fill_rgba=FILL_RGBA, fill_depth: float = FILL_DEPTH) -> None:
    """Append the selected-face planes to an existing MuJoCo user scene.

    Every face draws its translucent boundary quad. Two optional cues show
    which side the QP treats as free (the ``sign`` half-space):

    - ``arrows``: green normal arrows across the face pointing to the free side.
    - ``fill``: a faint green slab filling the allowed half-space to
      ``fill_depth`` -- the free region drawn as a volume.

    Both can be on together. Idempotent w.r.t. content: call once per frame
    after the base overlay. Does nothing if ``planes`` is empty/None.
    """

    if not planes:
        return
    for pl in planes:
        c = np.asarray(pl.center, dtype=np.float64).copy()
        h = np.asarray(pl.half, dtype=np.float64).copy()
        c[pl.axis] = pl.coord
        others = [i for i in range(3) if i != pl.axis]

        # free-side volume first (drawn behind the boundary quad)
        if fill:
            fc = c.copy()
            fc[pl.axis] = pl.coord + pl.sign * fill_depth / 2.0
            fh = h + pad
            fh[pl.axis] = fill_depth / 2.0
            _add_box(scn, fc, fh, fill_rgba)

        # boundary quad
        hp = h + pad
        hp[pl.axis] = thickness
        _add_box(scn, c, hp, rgba)

        # arrows across the face pointing along sign*axis (the allowed side)
        if arrows:
            fracs = ([0.0] if arrow_grid < 2
                     else np.linspace(-0.6, 0.6, arrow_grid))
            for f0 in fracs:
                for f1 in fracs:
                    base = c.copy()
                    base[others[0]] += f0 * h[others[0]]
                    base[others[1]] += f1 * h[others[1]]
                    tip = base.copy()
                    tip[pl.axis] += pl.sign * arrow_len
                    _add_arrow(scn, base, tip, arrow_width, arrow_rgba)


def dedupe(planes: List[FacePlane]) -> List[FacePlane]:
    """Collapse identical (box, axis, sign) selections so each committed face
    is drawn once, however many arm spheres chose it."""

    seen = {}
    for pl in planes:
        key = (round(float(pl.center[0]), 4), round(float(pl.center[1]), 4),
               round(float(pl.center[2]), 4), pl.axis, pl.sign)
        seen[key] = pl
    return list(seen.values())
