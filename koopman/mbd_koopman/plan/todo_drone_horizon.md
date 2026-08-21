# Drone planning horizon vs training horizon — RESOLVED

**Status: closed 2026-08-07.** No open work.

## What was inconsistent

The drone identified its bilinear model over 15-step snippets while the planner
rolled that model across a 25-step horizon, whereas the manipulator had `H = T =
15`. Table I reported both, so a reader comparing the two columns could see that
the stated motivation for the multi-step loss ("the planner rolls the model over
the whole planning horizon") did not hold on one of the two platforms.

## What was done

`drone/drone_window.py` now ties the training snippet length to the planning
horizon rather than fixing it:

```python
K = CFG["planner_mbd"]["horizon"]   # was: K = 15
```

so a change of horizon carries into identification automatically and the two
cannot drift apart again. The checkpoint name now carries `K`
(`drone_window_bk_K{K}_seed{seed}.pt`), because the previous name did not, and a
run at a new horizon would otherwise have reloaded the old fit in silence.

The five drone models were retrained at `K = 25` and Sec. IV-D was re-run over
the same 5 model seeds x 10 targets x 2 planners.

## Result: every reported number is unchanged

| | K = 15 | K = 25 |
|---|---|---|
| BK-MBD reaches 1 cm | 50/50 | 50/50 |
| convexified controller reaches 1 cm | 10/50 | 10/50 |
| constraint violations | 0 / 0 | 0 / 0 |
| safe stalls | 0 / 40 | 0 / 40 |
| paired outcome | 40 gained, 0 given up | 40 gained, 0 given up |
| BK-MBD mean final error | 0.0075 m | 0.0073 m |
| median ms per control step | 29.1 / 17.6 | 28.9 / 17.8 |

All 100 trials differ in their individual values, which confirms the new models
were actually used; the aggregates simply do not move.

## Paper changes that followed

- Table I, drone column: `Loss (15)` `H = 15` -> **`H = 25`**.
- Sec. III-B restored to the direct statement, now true on both platforms:
  *"we set H to the planning horizon of each platform and train every model on
  length-H trajectory snippets by the same loss."*

## Data

- `drone/out/window/` — current, `K = 25`
- `drone/out/window_K15/` — preserved, the run the earlier draft reported
