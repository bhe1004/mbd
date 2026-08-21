"""Every tunable in one place: the YAML config file, parsed into typed objects.

There are no tuning constants inside the source files -- no module-level
"TUNE HERE" blocks, no defaults hidden in argparse. A run is fully described by
its config file, which is copied into the output directory so a result can
always be traced back to the settings that produced it.

The tree mirrors the pipeline stages::

    paths     where datasets, checkpoints and outputs live
    robot     the arm's collision-sphere cover
    task      control period, horizon, limits, cost weights
    scene     obstacles, start pose, targets
    collision penalty margins and weights
    collect   how the training dataset is recorded
    koopman   model shape
    train     optimizer settings for learning the model
    mbd       the sampler
    adaptive  the across-plans noise shrink
    goal      where the target comes from and how it moves
    run       execution: which plant, which mode, how long
    ros2      endpoints and limits for the ROS 2 plant
    viewer    what gets drawn

Single-value overrides are supported from the CLI without editing the file::

    python main.py run -s mbd.num_samples=256 -s run.mode=lockstep
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from env.collision import CollisionConfig
from env.robot import ArmBodyConfig
from env.simulator import CollectConfig
from env.task import CostWeights, TaskConfig
from mbd.koopman import KoopmanArchitecture
from mbd.optimizer import MBDSettings
from mbd.training import TrainSettings

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

Vec3 = Tuple[float, float, float]
#: ("box", (x,y,z), (hx,hy,hz)) or ("sphere", (x,y,z) | "auto", radius)
Obstacle = Tuple[str, Any, Any]


@dataclass(frozen=True)
class PathsConfig:
    dataset: Path = ROOT / "data" / "fr3_snippets.npz"
    checkpoint: Path = ROOT / "models" / "bk_koopman.pt"
    output_dir: Path = ROOT / "out"


@dataclass(frozen=True)
class SceneConfig:
    obstacles: List[Obstacle] = field(default_factory=list)
    start_tcp: Optional[Vec3] = None
    target: Optional[Vec3] = None
    targets: Optional[List[Vec3]] = None


@dataclass(frozen=True)
class AdaptiveConfig:
    enabled: bool = True
    err_full: float = 0.4
    floor: float = 0.05


@dataclass(frozen=True)
class GoalConfig:
    """Where the target comes from, and how it moves (see pipeline/goals.py)."""

    mode: str = "fixed"          # fixed | keyboard | terminal | circle
    predict: bool = True         # feed the target's future to the planner
    # hand-driven modes
    step: float = 0.02           # [m] per key press
    low: Vec3 = (0.25, -0.45, 0.15)
    high: Vec3 = (0.75, 0.45, 0.95)
    # circle
    radius: float = 0.12         # [m]
    period: float = 8.0          # [s] per revolution
    plane: str = "yz"            # xy | yz | zx
    center: Optional[Vec3] = None    # null = placed so the circle starts at the tool
    lead_in: float = 2.0         # [s] the angular rate ramps up over this
    clockwise: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "keyboard", "terminal", "circle"):
            raise ValueError(f"unknown goal.mode {self.mode!r}")


@dataclass(frozen=True)
class Ros2Config:
    """Endpoints and safety limits for the ROS 2 plant (``run.plant: ros2``)."""

    joint_names: List[str] = field(default_factory=lambda: [
        f"fr3_joint{i}" for i in range(1, 8)])
    chunk_topic: str = "/vla/action/ee_pose"
    joint_state_topic: str = "/joint_states"
    goal_action: str = "/controller_action_server/vla_controller"
    model_name: str = "mbd"
    state_timeout: float = 0.2       # [s] a staler measurement stops the run
    start_tolerance: float = 0.05    # [rad] max deviation from the start pose
    ref_error_stop: float = 0.35     # [rad] reference-vs-measured gap that faults
    use_measured_ee: bool = False    # else the tool position comes from FK
    ee_topic: str = "/ee_state/pose"
    goal_timeout_s: float = 5.0
    # Markers published so RViz (or `main.py view`) can show the target.
    marker_topic: str = "/mbd/markers"
    marker_frame: str = "fr3_link0"
    # `main.py home`: the point-to-point move to the start pose.
    controller_manager: str = "/controller_manager"
    planner_controller: str = "vla_controller"
    point_to_point_controller: str = "joint_space_velocity_controller"
    home_duration: float = 5.0       # [s] the point-to-point move takes this long


@dataclass(frozen=True)
class RunConfig:
    plant: str = "mujoco"        # mujoco | ros2 | real
    mode: str = "async"          # async | lockstep
    target_ids: List[int] = field(default_factory=lambda: [0])
    cycle: bool = False
    max_time: float = 30.0
    settle_time: float = 0.4
    warmup_plans: int = 3
    min_plan_ms: float = 0.0
    wait_for_start: bool = True
    torch_threads: int = 4
    device: str = "cpu"
    save_replay: str = "ask"     # ask | always | never

    def __post_init__(self) -> None:
        if self.plant not in ("mujoco", "ros2", "real"):
            raise ValueError(f"unknown run.plant {self.plant!r}")
        if self.mode not in ("async", "lockstep"):
            raise ValueError(f"unknown run.mode {self.mode!r}")
        if self.save_replay not in ("ask", "always", "never"):
            raise ValueError(f"unknown run.save_replay {self.save_replay!r}")


@dataclass(frozen=True)
class GhostConfig:
    enabled: bool = False
    interval_s: float = 0.3
    alpha: float = 0.4
    max_ghosts: int = 40
    fade: bool = True


@dataclass(frozen=True)
class ViewerConfig:
    enabled: bool = True
    trail: bool = True
    prediction: bool = True
    collision_spheres: bool = False
    ghosts: GhostConfig = field(default_factory=GhostConfig)


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    robot: ArmBodyConfig
    task: TaskConfig
    scene: SceneConfig
    collision: CollisionConfig
    collect: CollectConfig
    koopman: KoopmanArchitecture
    train: TrainSettings
    mbd: MBDSettings
    adaptive: AdaptiveConfig
    goal: GoalConfig
    run: RunConfig
    ros2: Ros2Config
    viewer: ViewerConfig
    source: Path = DEFAULT_CONFIG
    raw: Dict[str, Any] = field(default_factory=dict)

    def dump(self, path: Path) -> None:
        """Write the config as loaded (with overrides applied) next to results."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True))


# ------------------------------------------------------------------- parsing
def _vec3(value, name: str) -> Optional[Vec3]:
    if value is None:
        return None
    v = tuple(float(c) for c in value)
    if len(v) != 3:
        raise ValueError(f"{name} must be a length-3 vector, got {value!r}")
    return v


def _obstacle(entry: Sequence, index: int) -> Obstacle:
    if len(entry) != 3:
        raise ValueError(f"scene.obstacles[{index}] must be [kind, center, size]")
    kind, center, size = entry
    if kind not in ("box", "sphere"):
        raise ValueError(f"scene.obstacles[{index}]: unknown kind {kind!r} (box/sphere)")
    if isinstance(center, str):
        if center != "auto":
            raise ValueError(f"scene.obstacles[{index}]: only 'auto' is allowed as a "
                             "string centre")
        c: Any = "auto"
    else:
        c = _vec3(center, f"scene.obstacles[{index}] centre")
    s = float(size) if kind == "sphere" else _vec3(size, f"scene.obstacles[{index}] size")
    return (kind, c, s)


def _build(cls, raw: Dict[str, Any], section: str, **extra):
    """Instantiate a dataclass from a config section, rejecting unknown keys."""

    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(raw) - known - set(extra)
    if unknown:
        raise ValueError(f"unknown key(s) in [{section}]: {sorted(unknown)}\n"
                         f"  known keys: {sorted(known)}")
    values = {k: v for k, v in raw.items() if k in known}
    values.update(extra)
    return cls(**values)


def _apply_override(tree: Dict[str, Any], assignment: str) -> None:
    """Apply one ``a.b.c=value`` override in place; the value is YAML-parsed."""

    if "=" not in assignment:
        raise ValueError(f"override must look like section.key=value, got {assignment!r}")
    dotted, _, literal = assignment.partition("=")
    keys = dotted.strip().split(".")
    node = tree
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise ValueError(f"override {assignment!r}: no section {k!r}")
        node = node[k]
    node[keys[-1]] = yaml.safe_load(literal)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; scalars and lists in ``override`` win outright."""

    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_with_extends(src: Path, seen: List[Path]) -> Dict[str, Any]:
    """Read a config, applying ``extends:`` so variants stay short.

    A variant names its base and overrides only what differs, which keeps the
    difference between two experiments visible instead of buried in two
    near-identical files.
    """

    src = src.resolve()
    if src in seen:
        chain = " -> ".join(p.name for p in seen + [src])
        raise SystemExit(f"circular extends: {chain}")
    raw = yaml.safe_load(src.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{src}: top level must be a mapping of sections")
    parent = raw.pop("extends", None)
    if parent is None:
        return copy.deepcopy(raw)
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = src.parent / parent_path
    if not parent_path.exists():
        raise SystemExit(f"{src}: extends target not found: {parent_path}")
    return _deep_merge(_read_with_extends(parent_path, seen + [src]), raw)


def _resolve(path_value, default: Path) -> Path:
    """Resolve a configured path; relative paths are relative to mbd_main/."""

    if path_value is None:
        return default
    p = Path(str(path_value)).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def load(path: Path | str | None = None,
         overrides: Sequence[str] = ()) -> Config:
    """Read a YAML config, apply CLI overrides, and validate every section."""

    src = Path(path) if path is not None else DEFAULT_CONFIG
    if not src.exists():
        candidate = ROOT / "configs" / src.name
        if not candidate.exists():
            raise SystemExit(f"config not found: {src} (also tried {candidate})")
        src = candidate
    raw = _read_with_extends(src, seen=[])
    for assignment in overrides:
        _apply_override(raw, assignment)

    section = lambda name: dict(raw.get(name) or {})  # noqa: E731

    paths_raw = section("paths")
    paths = PathsConfig(
        dataset=_resolve(paths_raw.get("dataset"), PathsConfig.dataset),
        checkpoint=_resolve(paths_raw.get("checkpoint"), PathsConfig.checkpoint),
        output_dir=_resolve(paths_raw.get("output_dir"), PathsConfig.output_dir),
    )

    task_raw = section("task")
    weights = _build(CostWeights, dict(task_raw.pop("weights", {}) or {}), "task.weights")
    task = _build(TaskConfig, task_raw, "task", weights=weights)

    scene_raw = section("scene")
    scene = SceneConfig(
        obstacles=[_obstacle(e, i) for i, e in enumerate(scene_raw.get("obstacles") or [])],
        start_tcp=_vec3(scene_raw.get("start_tcp"), "scene.start_tcp"),
        target=_vec3(scene_raw.get("target"), "scene.target"),
        targets=([_vec3(t, "scene.targets") for t in scene_raw["targets"]]
                 if scene_raw.get("targets") else None),
    )

    viewer_raw = section("viewer")
    ghosts = _build(GhostConfig, dict(viewer_raw.pop("ghosts", {}) or {}), "viewer.ghosts")
    viewer = _build(ViewerConfig, viewer_raw, "viewer", ghosts=ghosts)

    koopman_raw = section("koopman")
    koopman = _build(KoopmanArchitecture, koopman_raw, "koopman",
                     feature_dim=koopman_raw.get("feature_dim", 10),
                     action_dim=koopman_raw.get("action_dim", 7))

    cfg = Config(
        paths=paths,
        robot=_build(ArmBodyConfig, section("robot"), "robot"),
        task=task,
        scene=scene,
        collision=_build(CollisionConfig, section("collision"), "collision"),
        collect=_build(CollectConfig, section("collect"), "collect"),
        koopman=koopman,
        train=_build(TrainSettings, section("train"), "train"),
        mbd=_build(MBDSettings, section("mbd"), "mbd"),
        adaptive=_build(AdaptiveConfig, section("adaptive"), "adaptive"),
        goal=_build(GoalConfig, _goal_section(raw), "goal"),
        run=_build(RunConfig, section("run"), "run"),
        ros2=_build(Ros2Config, section("ros2"), "ros2"),
        viewer=viewer,
        source=src,
        raw=raw,
    )
    _cross_check(cfg)
    return cfg


def _goal_section(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Read [goal], normalising the vector fields."""

    g = dict(raw.get("goal") or {})
    for key in ("low", "high", "center"):
        if key in g:
            g[key] = _vec3(g[key], f"goal.{key}")
    return g


def _cross_check(cfg: Config) -> None:
    """Catch the mismatches that would otherwise only surface mid-run."""

    if cfg.train.rollout_horizon > cfg.collect.snippet_horizon:
        raise SystemExit(
            f"train.rollout_horizon ({cfg.train.rollout_horizon}) exceeds "
            f"collect.snippet_horizon ({cfg.collect.snippet_horizon}): the recorded "
            "snippets are shorter than one training window")
    if cfg.koopman.feature_dim != 10 or cfg.koopman.action_dim != 7:
        raise SystemExit("koopman.feature_dim/action_dim are fixed at 10/7 for the "
                         "FR3 reaching task ([q, ee] and 7 joint velocities)")
    if cfg.goal.mode != "fixed" and cfg.run.mode != "async":
        raise SystemExit(f"goal.mode {cfg.goal.mode!r} needs run.mode: async -- a "
                         "moving target cannot be tracked by a planner that blocks "
                         "the plant while it thinks")
    if cfg.goal.mode == "keyboard" and not cfg.viewer.enabled:
        raise SystemExit("goal.mode keyboard reads the MuJoCo viewer's keys, so it "
                         "needs viewer.enabled: true and run.plant: mujoco. With the "
                         "ROS stack the viewer belongs to the bringup -- use "
                         "goal.mode: terminal instead")
    if cfg.goal.mode == "keyboard" and cfg.run.plant != "mujoco":
        raise SystemExit("goal.mode keyboard only works with run.plant: mujoco "
                         "(use goal.mode: terminal elsewhere)")
    if not cfg.run.target_ids:
        raise SystemExit("run.target_ids must list at least one target")
    if cfg.run.plant != "mujoco" and cfg.viewer.enabled:
        raise SystemExit(f"run.plant is {cfg.run.plant!r}: set viewer.enabled false "
                         "(the MuJoCo viewer has no simulation to show)")
    if cfg.run.plant != "mujoco" and cfg.run.mode != "async":
        raise SystemExit(f"run.plant is {cfg.run.plant!r}: use run.mode async -- "
                         "lockstep would stall a real robot while it plans")


def describe(obj: Any, indent: int = 0) -> str:
    """Readable dump of a config subtree (used by ``main.py show``)."""

    pad = "  " * indent
    if is_dataclass(obj):
        lines = []
        for f in obj.__dataclass_fields__.values():
            if f.name == "raw":       # the unparsed tree; noise in a summary
                continue
            value = getattr(obj, f.name)
            if is_dataclass(value):
                lines.append(f"{pad}{f.name}:")
                lines.append(describe(value, indent + 1))
            else:
                lines.append(f"{pad}{f.name}: {json.dumps(value, default=str)}")
        return "\n".join(lines)
    return f"{pad}{obj}"
