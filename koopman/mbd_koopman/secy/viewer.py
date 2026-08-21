"""MuJoCo viewer overlays -- pure drawing, no planning.

What is drawn is exactly what the planner and referee check: the keep-out
obstacles (their true boxes/spheres), the goal and its reach shell, the actual
TCP trail, the planner's predicted EE path, and -- at the start pose only --
the whole-arm collision spheres.
"""

from __future__ import annotations

import numpy as np

import mujoco

# Robot visual meshes live in geom group 2 (group 3 = collision) -- verified for
# the FR3 scene. The ghost copies only these so it looks like the robot, not its
# collision hulls.
_VISUAL_GEOM_GROUP = 2
# mjvGeom fields copied verbatim so mesh geoms keep the render state the OpenGL
# renderer needs (orientation mat, mesh dataid, material). Ported from divas.
_GEOM_COPY_FIELDS = (
    "type", "dataid", "objtype", "objid", "category", "matid", "texcoord",
    "segid", "size", "pos", "mat", "rgba", "emission", "specular", "shininess",
    "reflectance", "camdist", "modelrbound", "transparent",
)


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


def build_ghost(model, qpos):
    """A fully-rendered MjvScene of the robot frozen at joint config ``qpos``.

    Built once per snapshot on a scratch MjData (the pose is static thereafter);
    ``draw_ghosts`` blits its visual geoms into the live user scene each frame."""

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(model, data)
    scn = mujoco.MjvScene(model, 2 * model.ngeom + 100)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None,
                           mujoco.MjvCamera(),
                           int(mujoco.mjtCatBit.mjCAT_ALL), scn)
    return scn


def draw_ghosts(scn, ghosts, model, alpha, fade=True) -> None:
    """Blit the visual geoms of every stored ghost scene into ``scn`` as
    translucent decor, keeping each mesh's OWN colour and only overriding the
    opacity. Oldest first; with ``fade`` the older ones are fainter. Respects
    the geom budget (stops when the scene is full)."""

    n = len(ghosts)
    if not n:
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
            _copy_geom(dst, src)          # copies the mesh's original rgba
            dst.rgba[3] = a               # keep RGB, override only opacity
            dst.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            scn.ngeom += 1


def _add_sphere(scn, pos, size, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[size, 0, 0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _add_box(scn, pos, half, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=np.asarray(half, dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def _add_path(scn, points, radius, rgba, budget):
    pts = np.asarray(points)
    if len(pts) < 2:
        return
    stride = max(1, int(np.ceil((len(pts) - 1) / max(budget, 1))))
    prev = pts[0]
    for j in range(stride, len(pts), stride):
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.zeros(3), pos=np.zeros(3), mat=np.eye(3).flatten(),
            rgba=np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, prev, pts[j])
        scn.ngeom += 1
        prev = pts[j]


def draw_overlay(viewer, target, threshold, trail, pred, color, obstacle=None,
                 overlays=None, show_trail=True, show_pred=True):
    """Obstacles + goal + reach shell + actual TCP trail + planned EE path.

    ``overlays`` is an optional list of ``fn(scn)`` callbacks invoked after the
    obstacles/goal/prediction but BEFORE the trail (so they never lose the geom
    budget to a long trail). The runtime uses it to draw the SQP planner's
    linearized-region faces without coupling this module to that feature.

    ``show_trail`` / ``show_pred`` (from ``config.json``'s ``path_line``) drop
    the planner-coloured lines; everything else still draws.
    """

    scn = viewer.user_scn
    scn.ngeom = 0
    if obstacle is not None:
        for c, r in obstacle.spheres_draw:
            _add_sphere(scn, c, r, [0.8, 0.25, 0.2, 0.45])
        for c, h in obstacle.boxes_draw:
            _add_box(scn, c, h, [0.9, 0.1, 0.1, 0.55])   # walls in red
    _add_sphere(scn, target, 0.012, [0.1, 0.8, 0.1, 1.0])
    _add_sphere(scn, target, threshold, [0.1, 0.8, 0.1, 0.15])
    for fn in (overlays or []):
        fn(scn)
    if pred is not None and show_pred:
        _add_path(scn, pred, 0.0015, [*color, 0.4], budget=len(pred))
    if trail and show_trail:
        _add_path(scn, trail, 0.002, [*color, 0.9],
                  budget=scn.maxgeom - scn.ngeom - 2)


def draw_start_pose(viewer, target, threshold, color, obstacle, fk, q):
    """The overlay plus the whole-arm collision spheres, drawn ONLY at the
    start pose (translucent orange balls at exactly the centres and radii the
    planner and referee check) -- so you see what is checked before motion.

    Pass ``fk=None`` to skip the spheres; the runtime does that when
    ``config.json``'s ``collision_view.spheres`` is false."""

    draw_overlay(viewer, target, threshold, [], None, color, obstacle=obstacle)
    if fk is not None:
        scn = viewer.user_scn
        for p, r in zip(fk.spheres_np(q), fk.radii_np):
            _add_sphere(scn, p, float(r), [0.95, 0.55, 0.10, 0.4])
    viewer.sync()
