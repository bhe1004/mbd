"""Metrics and result summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrialResult:
    """One closed-loop evaluation result."""

    task: str
    method: str
    seed: int
    case_id: str
    success: bool
    final_error: float
    total_cost: float
    planning_time_ms: float
    train_loss: float = float("nan")
    val_multistep_error: float = float("nan")
    koopman_open_loop_error: float = float("nan")
    tube_cost: float = float("nan")
    max_tube: float = float("nan")


def summarize_success(results: Iterable[TrialResult]) -> dict[str, float]:
    """Return compact success/error summary for a set of trials."""

    items = list(results)
    if not items:
        return {
            "num_trials": 0,
            "success_rate": float("nan"),
            "final_error_mean": float("nan"),
            "final_error_std": float("nan"),
        }
    errors = np.asarray([r.final_error for r in items], dtype=np.float64)
    successes = np.asarray([r.success for r in items], dtype=np.float64)
    return {
        "num_trials": float(len(items)),
        "success_rate": float(np.mean(successes)),
        "final_error_mean": float(np.mean(errors)),
        "final_error_std": float(np.std(errors)),
    }

