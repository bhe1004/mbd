"""Typed configuration for the FR3 experiments.

Every tunable lives in a JSON file under ``exper/configs``. A run picks one with
``--config`` and may override single leaves with ``--set section.key=value``, so
the JSON stays the record of what was run and the CLI stays short.

    from exper.config import load_config, build_parser
    cfg = load_config(**vars(build_parser().parse_args()))
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


# --------------------------------------------------------------------- sections
@dataclass(frozen=True)
class PlantCfg:
    """Plant and control interface."""

    control_dt: float = 0.05
    action_limit: float = 1.5
    rollout_threads: int = 16
    kinematic: bool = False    # True: analytic velocity integrator, no physics engine


@dataclass(frozen=True)
class DataCfg:
    """Offline excitation used to identify the rollout model."""

    num_snippets: int = 6000
    snippet_horizon: int = 15
    joint_margin: float = 0.15
    coherent_frac: float = 0.6      # per-snippet velocity bias
    jitter_frac: float = 0.5        # per-step jitter around that bias
    white: bool = False             # True: drop the bias, draw each step alone


@dataclass(frozen=True)
class ModelCfg:
    """Lifted rollout model."""

    variant: str = "bilinear"       # bilinear | linear | mlp
    lift_dim: int = 20              # r, including the d observed coordinates
    lift_dim_large: int = 50        # r for the over-parametrized linear control
    hidden: int = 96
    layers: int = 2


@dataclass(frozen=True)
class TrainCfg:
    """Multi-step identification."""

    horizon: int = 15               # H
    latent_weight: float = 0.1      # gamma
    epochs: int = 300
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    one_step: bool = False          # True: fit one step at a time instead of H


@dataclass(frozen=True)
class PlannerCfg:
    """Annealed sampling optimizer."""

    num_samples: int = 800          # N per stage
    stages: int = 5                 # S
    sigma_start: float = 1.2
    sigma_end: float = 0.3
    alpha: float = 0.4
    # Adaptive temperature, as MPPI-DK uses it: alpha = alpha_scale * std(costs),
    # recomputed at every stage. This tracks the cost spread instead of holding
    # a fixed scale, so the softmax does not flatten as the spread collapses
    # near the goal.
    alpha_adaptive: bool = False
    alpha_scale: float = 0.5
    horizon: int = 15               # T
    eta: float = 1.0
    w_ee: float = 1.0
    w_ctrl: float = 0.002
    w_term: float = 10.0


@dataclass(frozen=True)
class TaskCfg:
    """Reaching task and success criteria."""

    steps: int = 120                # closed-loop cap
    reach: float = 0.05
    strict: float = 0.01
    num_targets: int = 10
    target_seed: int = 20260804


@dataclass(frozen=True)
class RunCfg:
    """Bookkeeping for one sweep."""

    model_seeds: List[int] = None    # filled by __post_init__ of Config
    # Base of the planner's random stream; a trial draws from rng_base + target
    # index. Every runner shares it so that a condition measured in two studies
    # sees the same sampling noise.
    rng_base: int = 1000
    torch_threads: int = 4
    out_dir: str = "out"
    tag: str = "run"

    def __post_init__(self):  # pragma: no cover - dataclass default plumbing
        if self.model_seeds is None:
            object.__setattr__(self, "model_seeds", [0, 1, 2, 3, 4])


@dataclass(frozen=True)
class Config:
    plant: PlantCfg = PlantCfg()
    data: DataCfg = DataCfg()
    model: ModelCfg = ModelCfg()
    train: TrainCfg = TrainCfg()
    planner: PlannerCfg = PlannerCfg()
    task: TaskCfg = TaskCfg()
    run: RunCfg = RunCfg()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def replace(self, **sections: Any) -> "Config":
        """Return a copy with whole sections swapped."""
        merged = {f.name: sections.get(f.name, getattr(self, f.name))
                  for f in fields(self)}
        return Config(**merged)


_SECTIONS = {
    "plant": PlantCfg, "data": DataCfg, "model": ModelCfg, "train": TrainCfg,
    "planner": PlannerCfg, "task": TaskCfg, "run": RunCfg,
}


# ------------------------------------------------------------------- construction
def _coerce(current: Any, text: str) -> Any:
    """Parse a --set value against the type of the leaf it replaces."""
    if isinstance(current, bool):
        return text.lower() in ("1", "true", "yes", "on")
    if isinstance(current, list):
        return json.loads(text) if text.strip().startswith("[") else \
            [int(v) for v in text.replace(",", " ").split()]
    if isinstance(current, int) and not isinstance(current, bool):
        return int(text)
    if isinstance(current, float):
        return float(text)
    return text


def _build_section(name: str, raw: Dict[str, Any]):
    cls = _SECTIONS[name]
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise SystemExit(f"config section '{name}': unknown keys {sorted(unknown)}")
    return cls(**raw)


def load_config(config: str | Path | None = None,
                set_: Sequence[str] = (),
                **_ignored: Any) -> Config:
    """Read a JSON config and apply ``section.key=value`` overrides."""
    raw: Dict[str, Any] = {}
    if config is not None:
        path = Path(config)
        if not path.exists() and not path.is_absolute():
            path = CONFIG_DIR / path
            if path.suffix != ".json":
                path = path.with_suffix(".json")
        if not path.exists():
            raise SystemExit(f"config not found: {config}")
        raw = json.loads(path.read_text())

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise SystemExit(f"config: unknown sections {sorted(unknown)}")

    sections = {name: _build_section(name, raw.get(name, {}))
                for name in _SECTIONS}
    cfg = Config(**sections)

    for item in set_ or ():
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise SystemExit(f"--set expects section.key=value, got '{item}'")
        dotted, text = item.split("=", 1)
        sect, key = dotted.split(".", 1)
        if sect not in _SECTIONS:
            raise SystemExit(f"--set: unknown section '{sect}'")
        current = getattr(cfg, sect)
        if not hasattr(current, key):
            raise SystemExit(f"--set: unknown key '{sect}.{key}'")
        value = _coerce(getattr(current, key), text)
        updated = type(current)(**{**asdict(current), key: value})
        cfg = cfg.replace(**{sect: updated})

    return cfg


def build_parser(description: str = "") -> argparse.ArgumentParser:
    """Argument parser shared by every runner in this package."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None,
                    help="JSON config name under exper/configs, or a path")
    ap.add_argument("--set", dest="set_", action="append", default=[],
                    metavar="SECTION.KEY=VALUE",
                    help="override one config leaf (repeatable)")
    return ap


def describe(cfg: Config) -> str:
    """One-line digest for run logs."""
    return (f"model={cfg.model.variant} r={cfg.model.lift_dim} "
            f"N={cfg.planner.num_samples} S={cfg.planner.stages} "
            f"sigma={cfg.planner.sigma_start}->{cfg.planner.sigma_end} "
            f"T={cfg.planner.horizon} seeds={cfg.run.model_seeds}")


def dump(cfg: Config, path: Path) -> None:
    """Write the resolved config next to the results it produced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2))


class RunLock:
    """A pid lock so two processes never append to one output directory.

    The trials files are append-only and each runner reads its resume set once
    at startup, so a second live writer silently corrupts the sweep. This lock
    makes the second start fail loudly instead.
    """

    def __init__(self, out_dir: Path) -> None:
        self.path = Path(out_dir) / ".lock"

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            pid = self.path.read_text().strip()
            if pid and Path(f"/proc/{pid}").exists():
                raise SystemExit(
                    f"another run is writing {self.path.parent} (pid {pid}); "
                    f"wait for it or remove {self.path}")
        import atexit
        import os
        self.path.write_text(str(os.getpid()))
        atexit.register(self.__exit__)   # release on normal exit too
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
