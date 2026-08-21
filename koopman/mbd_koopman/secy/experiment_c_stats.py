"""Aggregate the Section C batch trials into the numbers the paper reports.

    python secy/experiment_c_stats.py

Reads ``out/exp_c/trials.jsonl`` and prints, per planner, the reach counts at
the 5 cm and 2.5 cm (strict) tolerances, executed-violation counts, and
safe-stall counts, then writes ``out/exp_c/summary.json``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "out" / "exp_c"
TRIALS_PATH = OUT_DIR / "trials.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"


def main() -> None:
    if not TRIALS_PATH.exists():
        sys.exit(f"no trials file: {TRIALS_PATH} (run experiment_c.py first)")

    by_planner = defaultdict(list)
    n_error = 0
    for line in TRIALS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("error") is not None:
            n_error += 1
            continue
        by_planner[r["planner"]].append(r)

    summary = {}
    print(f"\nSection C results  ({TRIALS_PATH})")
    print("=" * 72)
    for planner in sorted(by_planner):
        rows = by_planner[planner]
        n = len(rows)
        reach5 = sum(r["reached_5cm"] for r in rows)
        strict = sum(r["reached_strict"] for r in rows)   # first entry < strict
        strict25 = sum(r["reached_2p5cm"] for r in rows)  # closest approach < 2.5cm
        viol = sum(r["executed_violation"] for r in rows)
        stall = sum(r["safe_stall"] for r in rows)
        errs = [r["min_err"] for r in rows]
        mean_err = sum(errs) / n if n else float("nan")
        summary[planner] = {
            "n_trials": n,
            "reach_5cm": reach5,
            "reach_strict_2p5cm": strict,
            "reach_min_err_2p5cm": strict25,
            "executed_violations": viol,
            "safe_stalls": stall,
            "mean_min_err_m": mean_err,
        }
        print(f"\n{planner}  (n={n})")
        print(f"  reach  (<5 cm)         : {reach5}/{n}")
        print(f"  strict (first <2.5 cm) : {strict}/{n}")
        print(f"  strict (min_err<2.5cm) : {strict25}/{n}")
        print(f"  executed violations    : {viol}/{n}")
        print(f"  safe stalls            : {stall}/{n}")
        print(f"  mean closest error     : {mean_err * 100:.2f} cm")
    if n_error:
        print(f"\n(!) {n_error} trial(s) recorded an error; excluded from counts.")

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("=" * 72)
    print(f"summary written: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
