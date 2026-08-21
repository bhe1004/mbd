"""Viewer overlays -- pure drawing, no planning, no state.

What is drawn is exactly what is checked: the obstacles at their true sizes,
the goal with its strict-reach shell, the executed tool trail, the planner's
own predicted tool path, and -- at the start pose -- the whole-arm collision
spheres. Nothing here is cosmetic-only geometry that the planner does not see.

Every function takes the viewer's ``user_scn`` and respects its geom budget, so
a long trail can never crowd out the obstacles.
"""

from __future__ import annotations

from typing import Optional, Sequence

import mujoco
import numpy as np

#: Robot visual meshes live in geom group 2 (group 3 is collision), so a ghost
#: copies only these and looks like the robot rather than its collision hulls.
_VISUAL_GEOM_GROUP = 2
_GEOM_COPY_FIELDS = (
    "type", "dataid", "objtype", "objid", "category", "matid", "texcoord",
    "segid", "size", "pos", "mat", "rgba", "emission", "specular", "shininess",
    "reflectance", "camdist", "modelrbound", "transparent",
)

PLANNER_COLOR = (0.84, 0.15, 0.16)
GOAL_COLOR = (0.1, 0.8, 0.1)
OBSTACLE_RGBA = (0.85, 0.15, 0.15, 0.5)
SPHERE_RGBA = (0.95, 0.55, 0.10, 0.4)


def add_sphere(scn, pos, radius, rgba) -> None:
    if scn is None or scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[float(radius), 0, 0], pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(), rgba=np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def add_box(scn, pos, half, rgba) -> None:
    if scn is None or scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.asarray(half, dtype=np.float64), pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(), rgba=np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def add_path(scn, points, radius, rgba, budget: int) -> None:
    """Connect points with capsules, thinned to fit ``budget`` geoms."""

    pts = np.asarray(points)
    if scn is None or len(pts) < 2:
        return
    stride = max(1, int(np.ceil((len(pts) - 1) / max(budget, 1))))
    prev = pts[0]
    for j in range(stride, len(pts), stride):
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                            size=np.zeros(3), pos=np.zeros(3),
                            mat=np.eye(3).flatten(),
                            rgba=np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, prev, pts[j])
        scn.ngeom += 1
        prev = pts[j]


def draw_obstacles(scn, obstacles) -> None:
    if scn is None or obstacles is None:
        return
    for c, r in obstacles.spheres_draw:
        add_sphere(scn, c, r, OBSTACLE_RGBA)
    for c, h in obstacles.boxes_draw:
        add_box(scn, c, h, OBSTACLE_RGBA)


def draw_overlay(scn, *, goal, threshold: float, trail: Sequence[np.ndarray],
                 prediction: Optional[np.ndarray], obstacles=None,
                 color=PLANNER_COLOR, show_trail: bool = True,
                 show_prediction: bool = True, extra=()) -> None:
    """Redraw the whole overlay for one frame."""

    if scn is None:
        return
    scn.ngeom = 0
    draw_obstacles(scn, obstacles)
    add_sphere(scn, goal, 0.012, (*GOAL_COLOR, 1.0))
    add_sphere(scn, goal, threshold, (*GOAL_COLOR, 0.15))
    for fn in extra:
        fn(scn)
    if prediction is not None and show_prediction:
        add_path(scn, prediction, 0.0015, (*color, 0.4), budget=len(prediction))
    if trail is not None and len(trail) and show_trail:
        add_path(scn, trail, 0.002, (*color, 0.9), budget=scn.maxgeom - scn.ngeom - 2)


def draw_start_pose(scn, *, goal, threshold: float, obstacles=None, robot=None,
                    q=None, color=PLANNER_COLOR) -> None:
    """The overlay plus, optionally, the collision spheres at the start pose.

    Drawn only before motion: translucent balls at exactly the centres and radii
    the planner charges and the referee tests, so you can see what is checked.
    """

    draw_overlay(scn, goal=goal, threshold=threshold, trail=(), prediction=None,
                 obstacles=obstacles, color=color)
    if robot is not None and q is not None:
        for p, r in zip(robot.spheres_np(q), robot.sphere_radii_np):
            add_sphere(scn, p, float(r), SPHERE_RGBA)


# ------------------------------------------------------------------- ghosts
def build_ghost(model, qpos):
    """A rendered scene of the robot frozen at ``qpos``, for the ghost trail."""

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(model, data)
    scn = mujoco.MjvScene(model, 2 * model.ngeom + 100)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(),
                           int(mujoco.mjtCatBit.mjCAT_ALL), scn)
    return scn


def _copy_geom(dst, src) -> None:
    for field in _GEOM_COPY_FIELDS:
        try:
            value = getattr(src, field)
        except AttributeError:
            continue
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            getattr(dst, field)[:] = value
        else:
            setattr(dst, field, value)


def draw_ghosts(scn, ghosts, model, alpha: float, fade: bool = True) -> None:
    """Blit stored ghost scenes in as translucent decor, oldest first."""

    n = len(ghosts)
    if scn is None or not n:
        return
    for gi, gscn in enumerate(ghosts):
        a = alpha if not (fade and n > 1) else alpha * (0.3 + 0.7 * gi / (n - 1))
        for i in range(gscn.ngeom):
            src = gscn.geoms[i]
            if int(src.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM):
                continue
            if int(model.geom_group[int(src.objid)]) != _VISUAL_GEOM_GROUP:
                continue
            if scn.ngeom >= scn.maxgeom:
                return
            dst = scn.geoms[scn.ngeom]
            _copy_geom(dst, src)          # keeps each mesh's own colour
            dst.rgba[3] = a               # only the opacity is overridden
            dst.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            scn.ngeom += 1
