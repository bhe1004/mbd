# fast/ — the same controller, a cheaper implementation

`exper/` is the reference implementation and every number in the paper comes from
it. Nothing here changes it. This package re-implements only the hot path of one
control step and is meant to be checked against `exper/` rather than trusted.

The algorithm is untouched: same schedule, same temperature rule, same cost, same
update, same trained checkpoints. Only the arithmetic layout changes, so the
outputs agree to float32 rounding and the accuracy tables should not move.

## What changed and what it bought

Measured on an Intel Core Ultra 5 225F (ten cores, no GPU), N=800, S=5, T=15,
`phys_bilinear_r20_seed0`. Per component, one control step:

| | reference | fast | saved |
|---|---|---|---|
| bilinear rollout | 11.0 ms | 6.2 ms | 4.8 ms |
| candidate noise | 3.2 ms | 0.8 ms | 2.4 ms |
| torch->numpy hop + cost | 1.3 ms | ~0 | 1.3 ms |

1. **Fused input channels.** `LiftedModel.step` adds the bilinear term with a
   Python loop over the `m` input channels, so one lifted step issues `m` small
   `(N, r) @ (r, r)` products. Concatenating `{B_i}` into one `(r, m*r)` matrix
   turns that into a single product per step: 525 small matrix products per
   control step become 75. Same FLOPs, same weights, better BLAS occupancy.

2. **Noise drawn in torch.** A stage draws `N*T*m = 84,000` normals. NumPy's
   PCG64 spends 0.64 ms on that against 0.16 ms for `torch.randn`.

3. **No round trip through NumPy.** The reference backend decodes the rollout,
   copies `(N, T, d)` to NumPy, and scores it there, then the planner returns the
   next batch to torch. Everything here stays in torch.

Closed loop over the 50 trials of the paper's Table IV (5 training seeds x 10
goals), paired against `exper/out/b_sweep` by (seed, goal):

| | reference | `exact` | `torch` |
|---|---|---|---|
| median | 16.2 ms | **10.9 ms** | 7.7 ms |
| worst | 21.9 ms | **17.2 ms** | 18.5 ms |
| deadline misses | 0 | 0 | 0 |
| reach 1 cm | 50/50 | **50/50** | 47/50 |
| paired with the reference | -- | yes | no |

`exact` is the mode the paper's numbers should come from: the largest
disagreement in the closed-loop final error over all 50 trials is 1.1e-08, and
the 1 cm verdict is identical on every trial. `torch` is faster but draws its
candidates from a torch generator, so its trials are a different sample of the
same distribution -- 47/50 sits inside the 47--50 spread the reference itself
shows across streams, but it cannot be checked goal by goal.

Two things that did **not** help and are left alone: raising the torch thread
count (4, 6, 8 and 10 threads all give the same time -- these matrices are small
enough to be bandwidth-bound, which is what Sec. III-B of the paper observes),
and moving the lifted rollout to the GPU (N=800 does not fill a device, and the
transfers are charged on top).

## Files

    rollout.py       FusedBilinear: wraps a trained LiftedModel, no retraining
    planner.py       FusedNumpyBackend  fused rollout behind the reference
                                        planner -- the `exact` mode
                     TorchMBDPlanner    the update of exper/planner.py in torch
                     FusedBilinearBackend  its all-torch cost -- the `torch` mode
    bench.py         equivalence check against exper/ plus the timing table
    run_latency.py   closed-loop latency and accuracy, to re-measure Table IV

## Use

    # prove it computes the same thing, then time it
    PYTHONPATH=. python -m fast.bench --config exp_b

    # closed-loop, same goals and planner stream as run_b
    PYTHONPATH=. python -m fast.run_latency --config exp_b            # exact
    PYTHONPATH=. python -m fast.run_latency --config exp_b --mode torch

`bench.py` is the gate: it reports the largest disagreement with `exper/` on
identical inputs. Treat a jump in that number as a bug in this package, not as a
new result.
