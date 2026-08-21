"""Read the temperature grid and decide between the two accounts of the wide end.

    precision floor (Remark 2)  alpha down -> N_eff down -> dispersion up
                                -> the wide level should get WORSE
    small effective step        alpha down -> the mean commits harder
                                -> the wide level should get BETTER

The manipulation check comes first: unless alpha actually moves N_eff, nothing
below carries any weight. Then the outcome table, then the post-arrival motion
that Sec. V-C claims the wide level suffers from.

    python -m exper.analyze_d
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

ORDER = ["fixed0.3", "fixed0.5", "fixed0.8", "fixed1.2", "anneal"]
TOL = 0.01


def load(pattern="exper/out/d_a_s*/trials.jsonl"):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def post_arrival(rows):
    """Spread of the tracking error after the run first enters the band."""
    out = []
    for r in rows:
        if not r["reached_strict"] or not r.get("err_curve"):
            continue
        e = np.asarray(r["err_curve"], float)
        tail = e[int(r["steps"]):]
        if tail.size >= 10:
            out.append(tail.std())
    return float(np.median(out)) if out else float("nan")


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit("no trials found; run exper.run_d first")
    g = defaultdict(list)
    for r in rows:
        g[(r["condition"], r["schedule"])].append(r)
    alphas = sorted({c for c, _ in g}, key=lambda c: -float(c[1:]))
    print(f"{len(rows)} trials, {len(g)} cells\n")

    def cell(a, s, fn):
        rs = g.get((a, s))
        return fn(rs) if rs else float("nan")

    def show(title, fn, fmt="{:>9.4f}"):
        print(title)
        print("  " + "sched".ljust(10) + "".join(a.rjust(11) for a in alphas))
        for s in ORDER:
            line = "  " + s.ljust(10)
            for a in alphas:
                v = cell(a, s, fn)
                line += (fmt.format(v) if v == v else "        --").rjust(11)
            print(line)
        print()

    show("MANIPULATION CHECK  N_eff of the terminal stage (of N=800)",
         lambda rs: float(np.median([r["n_eff"] for r in rs])), "{:>9.1f}")
    show("REACH 1cm  fraction of trials",
         lambda rs: float(np.mean([r["reached_strict"] for r in rs])), "{:>9.2f}")
    show("STEPS to first entry  median over reachers",
         lambda rs: float(np.median([r["steps"] for r in rs if r["reached_strict"]]
                                    or [np.nan])), "{:>9.1f}")
    show("FINAL error [m]  median",
         lambda rs: float(np.median([r["final_err"] for r in rs])))
    show("POST-ARRIVAL motion  std of the error after entry [m]", post_arrival,
         "{:>9.5f}")
    show("STEP SIZE  |dU| per control step",
         lambda rs: float(np.median([r["dU"] for r in rs])))
    show("NULL FRACTION  share of the commanded step the TCP cannot see",
         lambda rs: float(np.median([r["null_disp"] /
                                     max(np.hypot(r["null_disp"], r["task_disp"]), 1e-12)
                                     for r in rs])), "{:>9.3f}")

    # the verdict, read off the widest fixed level
    base, low = alphas[0], alphas[-1]
    s_hi = cell(base, "fixed1.2", lambda rs: np.mean([r["reached_strict"] for r in rs]))
    s_lo = cell(low, "fixed1.2", lambda rs: np.mean([r["reached_strict"] for r in rs]))
    n_hi = cell(base, "fixed1.2", lambda rs: np.median([r["n_eff"] for r in rs]))
    n_lo = cell(low, "fixed1.2", lambda rs: np.median([r["n_eff"] for r in rs]))
    print(f"VERDICT at sigma=1.2: alpha {base[1:]} -> {low[1:]} moves N_eff "
          f"{n_hi:.0f} -> {n_lo:.0f} and reach {s_hi:.2f} -> {s_lo:.2f}")
    if n_lo < n_hi * 0.8:
        print("  the temperature did sharpen the weights, so the test is live.")
        print("  wide improves -> small-step account; wide degrades -> precision floor.")
    else:
        print("  alpha did NOT move N_eff; the test is void as designed.")


if __name__ == "__main__":
    main()
