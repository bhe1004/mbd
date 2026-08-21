"""Typed configuration for the secy experiment.

Every tunable lives in ``config.json`` (no CLI flags). This module reads that
file once and hands back a validated, typed :class:`Config` tree so the rest
of the code touches attributes, not raw dict keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"

# One obstacle entry: ("box", (x,y,z), (hx,hy,hz)) or ("sphere", center, r),
# where a sphere center may be the string "auto" (placed by the Scene).
Vec3 = Tuple[float, float, float]
Obstacle = Tuple[str, Union[str, Vec3], Union[float, Vec3]]


@dataclass(frozen=True)
class EnvCfg:
    obstacles: List[Obstacle]
    start_tcp: Optional[Vec3]
    target: Optional[Vec3]
    horizon: Optional[int]
    max_joint_velocity: Optional[float]
    wait_for_start: bool


@dataclass(frozen=True)
class ArmCfg:
    first_link: int
    link_samples: int
    gripper_fingers: bool


@dataclass(frozen=True)
class CollisionCfg:
    margin: float
    weight: float
    hard: bool
    substeps: int
    floor_z: Optional[float]   # keep every arm sphere above this height; None = off


@dataclass(frozen=True)
class JointLimitCfg:
    weight: float
    margin: float


@dataclass(frozen=True)
class MbdCfg:
    num_samples: int
    num_diffusion_steps: int
    sigma_start: float
    sigma_end: float
    alpha: float
    eta: float
    update_rule: str
    seed: int
    langevin_noise: bool
    tube_mode: str          # "none" | "plain" | "cost-sens"
    beta_e: float
    data_seed: int
    checkpoint: Optional[Path]


@dataclass(frozen=True)
class AdaptiveCfg:
    enabled: bool
    err_full: float
    floor: float


@dataclass(frozen=True)
class SqpCfg:
    max_iters: int          # SQP re-freezes per plan
    act_dist: float         # a keep-out face is added within this distance
    osqp_eps: float         # OSQP abs/rel tolerance (looser = faster)
    osqp_max_iter: int      # OSQP iteration cap
    trust_region: float     # max |q - q_nom| per SQP iterate [rad] (linearization validity)
    slack_weight: float     # exact-penalty weight on keep-out slack (recursive feasibility)


@dataclass(frozen=True)
class LinearizationCfg:
    enabled: bool           # draw the SQP linearized-region faces at all
    arrows: bool            # green normal arrows to the allowed side
    fill: bool              # green translucent slab filling the allowed side
    fill_depth: float       # how deep the slab extends into the allowed side [m]


@dataclass(frozen=True)
class CollisionViewCfg:
    spheres: bool           # draw the whole-arm collision spheres at the start pose


@dataclass(frozen=True)
class PathLineCfg:
    trail: bool             # draw the actual TCP trail (the solid planner-colour line)
    prediction: bool        # draw the planner's predicted EE path (the faint one)


@dataclass(frozen=True)
class GhostCfg:
    enabled: bool           # leave translucent robot 'ghosts' of past poses
    interval_s: float       # sim-time between snapshots
    alpha: float            # opacity; the robot keeps its own mesh colours
    max_ghosts: int         # oldest dropped past this many
    fade: bool              # older ghosts more transparent


@dataclass(frozen=True)
class RuntimeCfg:
    mode: str               # "async" | "lockstep"
    target_ids: List[int]
    max_time: float
    settle_time: float
    warmup_plans: int
    min_plan_ms: float
    torch_threads: int
    device: str
    viewer: bool


@dataclass(frozen=True)
class Config:
    env: EnvCfg
    arm: ArmCfg
    collision: CollisionCfg
    joint_limit: JointLimitCfg
    mbd: MbdCfg
    adaptive: AdaptiveCfg
    sqp: SqpCfg
    linearization: LinearizationCfg
    collision_view: CollisionViewCfg
    path_line: PathLineCfg
    ghost: GhostCfg
    runtime: RuntimeCfg
    source: Path = field(default=DEFAULT_CONFIG)


def _vec3(x) -> Optional[Vec3]:
    if x is None:
        return None
    v = tuple(float(c) for c in x)
    if len(v) != 3:
        raise ValueError(f"expected a length-3 vector, got {x!r}")
    return v


def _parse_obstacle(e: Sequence) -> Obstacle:
    if len(e) != 3:
        raise ValueError(
            f"obstacle must be [kind, center, size], got {e!r}")
    kind, center, size = e
    if kind not in ("box", "sphere"):
        raise ValueError(f"unknown obstacle kind {kind!r} (use box/sphere)")
    if isinstance(center, str):
        if center != "auto":
            raise ValueError(f"sphere center string must be 'auto', got {center!r}")
        c = "auto"
    else:
        c = _vec3(center)
    s = float(size) if kind == "sphere" else _vec3(size)
    return (kind, c, s)


def load_config(path: Union[str, Path, None] = None) -> Config:
    """Read ``config.json`` (or ``path``) into a validated :class:`Config`."""

    src = Path(path) if path is not None else DEFAULT_CONFIG
    with open(src) as fh:
        raw = json.load(fh)

    env = raw["env"]
    arm = raw["arm"]
    col = raw["collision"]
    jl = raw["joint_limit"]
    mbd = raw["mbd"]
    adp = raw["adaptive_noise"]
    sqp = raw["sqp"]
    lin = raw.get("linearization", {})
    cv = raw.get("collision_view", {})
    pl = raw.get("path_line", {})
    gh = raw.get("ghost_trail", {})
    run = raw["runtime"]

    if mbd["update_rule"] not in ("score_langevin", "weighted_mean"):
        raise ValueError(f"unknown update_rule {mbd['update_rule']!r}")
    if mbd["tube_mode"] not in ("none", "plain", "cost-sens"):
        raise ValueError(f"unknown tube_mode {mbd['tube_mode']!r}")
    if run["mode"] not in ("async", "lockstep"):
        raise ValueError(f"unknown runtime.mode {run['mode']!r}")
    if not mbd["sigma_start"] > 0 or not mbd["sigma_end"] > 0:
        raise ValueError("sigma_start/sigma_end must be positive")

    ckpt = mbd.get("checkpoint")
    checkpoint = Path(ckpt) if ckpt else None

    return Config(
        env=EnvCfg(
            obstacles=[_parse_obstacle(e) for e in env["obstacles"]],
            start_tcp=_vec3(env["start_tcp"]),
            target=_vec3(env["target"]),
            horizon=(int(env["horizon"]) if env["horizon"] is not None else None),
            max_joint_velocity=(float(env["max_joint_velocity"])
                                if env["max_joint_velocity"] is not None else None),
            wait_for_start=bool(env["wait_for_start"]),
        ),
        arm=ArmCfg(
            first_link=int(arm["first_link"]),
            link_samples=int(arm["link_samples"]),
            gripper_fingers=bool(arm["gripper_fingers"]),
        ),
        collision=CollisionCfg(
            margin=float(col["margin"]),
            weight=float(col["weight"]),
            hard=bool(col["hard"]),
            substeps=int(col["substeps"]),
            floor_z=(None if col.get("floor_z", 0.0) is None
                     else float(col.get("floor_z", 0.0))),
        ),
        joint_limit=JointLimitCfg(
            weight=float(jl["weight"]),
            margin=float(jl["margin"]),
        ),
        mbd=MbdCfg(
            num_samples=int(mbd["num_samples"]),
            num_diffusion_steps=int(mbd["num_diffusion_steps"]),
            sigma_start=float(mbd["sigma_start"]),
            sigma_end=float(mbd["sigma_end"]),
            alpha=float(mbd["alpha"]),
            eta=float(mbd["eta"]),
            update_rule=mbd["update_rule"],
            seed=int(mbd["seed"]),
            langevin_noise=bool(mbd["langevin_noise"]),
            tube_mode=mbd["tube_mode"],
            beta_e=float(mbd["beta_e"]),
            data_seed=int(mbd["data_seed"]),
            checkpoint=checkpoint,
        ),
        adaptive=AdaptiveCfg(
            enabled=bool(adp["enabled"]),
            err_full=float(adp["err_full"]),
            floor=float(adp["floor"]),
        ),
        sqp=SqpCfg(
            max_iters=int(sqp["max_iters"]),
            act_dist=float(sqp["act_dist"]),
            osqp_eps=float(sqp["osqp_eps"]),
            osqp_max_iter=int(sqp["osqp_max_iter"]),
            trust_region=float(sqp.get("trust_region", 0.35)),
            slack_weight=float(sqp.get("slack_weight", 5000.0)),
        ),
        linearization=LinearizationCfg(
            enabled=bool(lin.get("enabled", True)),
            arrows=bool(lin.get("arrows", True)),
            fill=bool(lin.get("fill", False)),
            fill_depth=float(lin.get("fill_depth", 0.15)),
        ),
        collision_view=CollisionViewCfg(
            spheres=bool(cv.get("spheres", True)),
        ),
        path_line=PathLineCfg(
            trail=bool(pl.get("trail", True)),
            prediction=bool(pl.get("prediction", True)),
        ),
        ghost=GhostCfg(
            enabled=bool(gh.get("enabled", False)),
            interval_s=float(gh.get("interval_s", 0.5)),
            alpha=float(gh.get("alpha", 0.4)),
            max_ghosts=int(gh.get("max_ghosts", 12)),
            fade=bool(gh.get("fade", True)),
        ),
        runtime=RuntimeCfg(
            mode=run["mode"],
            target_ids=[int(i) for i in run["target_ids"]],
            max_time=float(run["max_time"]),
            settle_time=float(run["settle_time"]),
            warmup_plans=int(run["warmup_plans"]),
            min_plan_ms=float(run["min_plan_ms"]),
            torch_threads=int(run["torch_threads"]),
            device=run["device"],
            viewer=bool(run["viewer"]),
        ),
        source=src,
    )
