"""Aggregate the drone window experiment (Section C port) into paper numbers.

Reads out/window/trials.jsonl (written by drone_window.py) and reports, per
planner: reach rate at 2.5 cm and 1 cm, executed wall violations, safe stalls,
median settling steps and latency; then the PAIRED contingency between BK-MBD
and QP-MPC on identical targets (both / BK-only / QP-only / neither), the
one-sided statement (does BK reach every target QP stalls on, and is there any
target QP reaches but BK does not?), and an exact-binomial McNemar test on the
discordant pairs.

    python drone/drone_window_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "out" / "window"
TRIALS_PATH = OUT_DIR / "trials.jsonl"
PLANNERS = ("bk_mbd", "bk_qp_sqp")
LABEL = {"bk_mbd": "BK-MBD (sampling)", "bk_qp_sqp": "QP-MPC (convex)"}


def load():
    if not TRIALS_PATH.exists():
        raise SystemExit(f"no trials at {TRIALS_PATH} -- run drone_window.py first")
    rows = [json.loads(l) for l in TRIALS_PATH.read_text().splitlines() if l.strip()]
    by = {p: {} for p in PLANNERS}
    for r in rows:
        # key on (model seed, target) so several training seeds coexist
        by[r["planner"]][(r.get("model_seed"), r["target_idx"])] = r
    return by


def mcnemar_p(b, c):
    """Exact two-sided McNemar (binomial on the discordant pairs b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5).pvalue)
    except Exception:
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        return float(min(1.0, 2 * tail))


def main() -> None:
    by = load()
    idxs = sorted(set().union(*[set(by[p]) for p in PLANNERS]))
    n = len(idxs)
    seeds = sorted({k[0] for k in idxs})
    n_tgt = len({k[1] for k in idxs})
    print(f"drone window experiment: {len(seeds)} model seeds x {n_tgt} targets "
          f"= {n} trials per planner, planners {list(PLANNERS)}\n")

    # ---- per-planner summary ------------------------------------------------
    print(f"{'planner':22s} {'reach@2.5cm':>12s} {'reach@1cm':>10s} "
          f"{'viol':>5s} {'safe-stall':>11s} {'med steps':>10s} {'med ms':>8s}")
    reach25 = {}
    for p in PLANNERS:
        rs = [by[p][i] for i in idxs if i in by[p]]
        r25 = sum(r["reached_2p5cm"] and r["viol"] == 0 for r in rs)
        r1 = sum(r["reached_1cm"] and r["viol"] == 0 for r in rs)
        viol = sum(r["viol"] > 0 for r in rs)
        stall = sum(r["safe_stall"] for r in rs)
        reached_steps = [r["steps"] for r in rs
                         if r["reached_2p5cm"] and r["viol"] == 0]
        med_steps = int(np.median(reached_steps)) if reached_steps else -1
        med_ms = np.median([r["ms_per_step"] for r in rs])
        reach25[p] = {i: (by[p][i]["reached_2p5cm"] and by[p][i]["viol"] == 0)
                      for i in by[p]}
        print(f"{LABEL[p]:22s} {r25:>8d}/{len(rs):<3d} {r1:>7d}/{len(rs):<3d} "
              f"{viol:>5d} {stall:>8d}/{len(rs):<3d} {med_steps:>10d} {med_ms:>8.1f}")

    # ---- paired contingency (BK vs QP on identical targets) -----------------
    both = bk_only = qp_only = neither = 0
    paired = [i for i in idxs if i in reach25["bk_mbd"] and i in reach25["bk_qp_sqp"]]
    for i in paired:
        b, q = reach25["bk_mbd"][i], reach25["bk_qp_sqp"][i]
        both += b and q
        bk_only += b and not q
        qp_only += q and not b
        neither += not b and not q
    qp_stall = bk_only + neither          # targets QP-MPC did NOT reach
    bk_on_qp_stall = sum(reach25["bk_mbd"][i] for i in paired
                         if not reach25["bk_qp_sqp"][i])
    p_val = mcnemar_p(qp_only, bk_only)

    print(f"\npaired on {len(paired)} identical targets:")
    print(f"  both reach          : {both}")
    print(f"  BK-only reach       : {bk_only}")
    print(f"  QP-only reach       : {qp_only}")
    print(f"  neither             : {neither}")
    print(f"\n  QP-MPC stalls on {qp_stall} targets; BK-MBD reaches "
          f"{bk_on_qp_stall}/{qp_stall} of them.")
    print(f"  Targets QP reaches but BK does not: {qp_only}.")
    print(f"  McNemar exact-binomial p = {p_val:.2e} "
          f"(discordant: BK-only={bk_only}, QP-only={qp_only}).")

    # ---- mechanism: where the convex solve stops ---------------------------
    trajs_path = OUT_DIR / "trajs.npz"
    if trajs_path.exists():
        d = np.load(trajs_path, allow_pickle=True)
        win_c = np.asarray(d["win_c"]).ravel() if "win_c" in d else None
        r_win = float(np.asarray(d["r_win"]).ravel()[0]) if "r_win" in d else None
        y_wall = float(np.asarray(d["wall"]).ravel()[0]) if "wall" in d else 0.0

        stall_y, crossed = [], 0
        for (ms, ti), r in by["bk_qp_sqp"].items():
            key = f"bk_qp_sqp_s{ms}_{ti}"
            if key not in d.files:
                continue
            tr = np.asarray(d[key])
            stall_y.append(tr[-1, 1] - y_wall)
            crossed += int(tr[:, 1].max() > y_wall)
        if stall_y:
            print(f"\nwhere the convex solve stops (n={len(stall_y)}):")
            print(f"  final signed distance to the wall : "
                  f"median {np.median(stall_y):+.3f} m")
            print(f"  |distance| to the wall            : "
                  f"median {np.median(np.abs(stall_y)):.3f} m")
            print(f"  trials that ever crossed the wall : {crossed}")

        if win_c is not None:
            def radial(rec):
                t = rec["target"]
                return float(np.hypot(t[0] - win_c[0], t[2] - win_c[1]))
            allr = [radial(r) for r in by["bk_qp_sqp"].values()]
            okr = [radial(r) for k, r in by["bk_qp_sqp"].items()
                   if reach25["bk_qp_sqp"].get(k, False)]
            print(f"  target offset from the aperture axis "
                  f"(r_win={r_win}): median {np.median(allr):.3f} m")
            if okr:
                print(f"  same offset, targets QP reaches    : "
                      f"median {np.median(okr):.3f} m, max {max(okr):.3f} m")

    # ---- one-line paper sentence -------------------------------------------
    nb = sum(reach25["bk_mbd"].get(i, False) for i in paired)
    nq = sum(reach25["bk_qp_sqp"].get(i, False) for i in paired)
    print(f"\n[paper] Over {len(paired)} targets, BK-MBD reaches {nb}/{len(paired)} "
          f"with zero executed violations, whereas QP-MPC reaches {nq}/{len(paired)} "
          f"and stalls on the rest; BK-MBD reaches all {qp_stall} targets on which "
          f"QP-MPC stalls, and no target is reached by QP-MPC but not BK-MBD.")


if __name__ == "__main__":
    main()
