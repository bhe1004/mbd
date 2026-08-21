"""Aggregate a sweep into the numbers the paper reports.

    python -m exper.stats out/b_run
    python -m exper.stats out/c_run
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "trials.jsonl"
    if not path.exists():
        raise SystemExit(f"no trials at {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fmt_rate(hits: int, n: int) -> str:
    return f"{hits}/{n}"


def report_rollout_class(rows: list[dict]) -> None:
    """Experiment B: accuracy by condition, then latency."""
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)

    print(f"{'condition':14s} {'reach':>9s} {'strict':>9s} "
          f"{'final err [m]':>16s} {'steps':>7s}")
    for name, rs in by.items():
        n = len(rs)
        fe = [r["final_err"] for r in rs]
        print(f"{name:14s} {_fmt_rate(sum(r['reached'] for r in rs), n):>9s} "
              f"{_fmt_rate(sum(r['reached_strict'] for r in rs), n):>9s} "
              f"{np.mean(fe):>9.3f} +- {np.std(fe):<4.3f} "
              f"{int(np.median([r['steps'] for r in rs])):>7d}")

    print(f"\n{'condition':14s} {'median ms':>10s} {'worst ms':>10s} "
          f"{'miss steps':>11s}")
    for name, rs in by.items():
        med = np.median([r["ms_per_step"] for r in rs])
        worst = max(r["worst_ms"] for r in rs)
        # total control steps that overran the period, across every trial
        miss_steps = sum(r.get("deadline_misses", 0) for r in rs)
        print(f"{name:14s} {med:>10.1f} {worst:>10.1f} {miss_steps:>11d}")


def report_schedules(rows: list[dict]) -> None:
    """Experiment C: strict reach by schedule across the error levels."""
    lams = sorted({r["lam"] for r in rows})
    scheds = sorted({r["schedule"] for r in rows})
    counts: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        counts[r["schedule"]][r["lam"]].append(r)

    head = "".join(f"{('lam=' + format(l, '.2f')):>14s}" for l in lams)
    print(f"{'schedule':16s}{head}")
    for s in scheds:
        cells = []
        for l in lams:
            rs = counts[s][l]
            cells.append(_fmt_rate(sum(r["reached_strict"] for r in rs), len(rs))
                         if rs else "-")
        print(f"{s:16s}" + "".join(f"{c:>14s}" for c in cells))

    print(f"\n{'schedule':16s}{'median steps':>14s}{'median final err':>18s}")
    for s in scheds:
        rs = [r for r in rows if r["schedule"] == s]
        print(f"{s:16s}{int(np.median([r['steps'] for r in rs])):>14d}"
              f"{np.median([r['final_err'] for r in rs]):>18.3f}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run_dir = Path(sys.argv[1])
    if not run_dir.is_absolute():
        run_dir = Path(__file__).resolve().parent / run_dir
    rows = load(run_dir)
    print(f"{run_dir.name}: {len(rows)} trials\n")

    if any(r.get("lam") for r in rows) or len({r["schedule"] for r in rows}) > 1:
        report_schedules(rows)
        diag = run_dir / "diagnostics.json"
        if diag.exists():
            print("\ndiagnostics:")
            for k, v in json.loads(diag.read_text()).items():
                print(f"  {k}: " + "  ".join(f"{a}={b:.4f}" for a, b in v.items()))
    else:
        report_rollout_class(rows)
        ol = run_dir / "open_loop.json"
        if ol.exists():
            print("\nheld-out horizon error [m]:")
            for k, v in json.loads(ol.read_text()).items():
                print(f"  {k:18s} {v:.4f}")


if __name__ == "__main__":
    main()
