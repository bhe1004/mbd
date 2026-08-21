# mbd_main — BK-MBD on the Franka FR3

Model-Based Diffusion planning over a learned bilinear Koopman model, from
data collection to real-time execution. Self-contained: nothing outside this
directory is imported, and every tunable lives in a config file.

```
python main.py collect      # 1. record the training dataset      -> data/*.npz
python main.py train        # 2. learn the bilinear Koopman model -> models/*.pt
python main.py run          # 3. plan and execute in real time
```

## The one structural idea

The algorithm and the environment never import each other.

```
mbd/        the planner. numpy + torch only. Does not know what a joint,
            a gripper, or a simulator is.
env/        the robot, the task, the obstacles, and the plants that execute.
            Does not know what diffusion or a Koopman model is.
pipeline/   the only place the two meet, wired from the config file.
```

They meet through two small protocols (`mbd/types.py`):

| protocol | supplied by | meaning |
|---|---|---|
| `PredictiveModel` | the trained checkpoint | roll a batch of candidate control sequences into predicted *features* |
| `CandidateCost` | `env/task.py` | score those feature trajectories against a goal |

A "feature" here is `b = [q (7), ee (3)]` — joint angles and tool position, both
measurable on real hardware. The planner rolls candidates in the lifted Koopman
space, decodes to features, and asks the environment what they cost. That is the
entire coupling.

Consequences, which is the point:

* **Swap the simulator for the real arm** — implement `env/interface.py:Plant`
  (a template with the exact contract is in `env/real_plant.py`), set
  `run.plant: real`. The planner, the cost, the model and the loop are untouched.
* **Swap the task** — supply a different `candidate_cost` and a matching feature
  definition. `mbd/` does not change.
* **Swap the model** — anything with a `rollout(features0, controls)` works;
  the bilinear Koopman one is just the default.

## Layout

```
main.py                CLI: collect | train | run | replay | show
config.py              YAML -> typed dataclasses, validated, with `extends`
configs/
  default.yaml         the full parameter set, commented
  interactive.yaml     keyboard-driven target (extends default)
  benchmark.yaml       headless lockstep latency run (extends default)

mbd/
  types.py             the two protocols + data containers
  optimizer.py         MBD sampler; sigma anneal; adaptive across-plan shrink
  koopman.py           bilinear deep Koopman model + the PredictiveModel adapter
  training.py          multi-step lifted loss, training loop, checkpoints
  planner.py           BKMBDPlanner: model + cost + optimizer

env/
  interface.py         Observation, Plant (the execution boundary)
  robot.py             FR3 kinematics + whole-arm collision spheres, read out
                       of the MuJoCo model tree (so torch FK == simulator FK)
  task.py              features(q) and candidate_cost(...) — the reaching task
  collision.py         obstacles: planner penalty + margin-free referee
  simulator.py         batched MuJoCo rollouts — the data source
  plant.py             MuJoCo execution plant with real-time pacing
  real_plant.py        hardware adapter template
  assets/franka_fr3/   the model and meshes (mujoco_menagerie, see its LICENSE)

pipeline/
  build.py             the wiring: config -> environment, planner, plant
  collect.py  train.py  run.py  replay.py
  teleop.py            the keyboard goal
  viewer.py            overlays (draws exactly what is checked)
```

## Configuration

One YAML tree drives every stage; there are no tuning constants in the source.
Override single values without editing a file, and build variants with
`extends:`:

```bash
python main.py run -c interactive.yaml
python main.py run -s run.mode=lockstep -s mbd.num_samples=256
python main.py show                      # print the resolved config and exit
```

Sections: `paths`, `robot`, `task`, `scene`, `collision`, `collect`, `koopman`,
`train`, `mbd`, `adaptive`, `run`, `viewer`. Unknown keys are an error,
and cross-section mismatches (e.g. training windows longer than the recorded
snippets, or a dataset recorded at a different control period) are caught before
anything runs.

## The three stages

**collect** rolls thousands of short snippets of *coherent* random joint
velocities — a random constant drift per snippet plus per-step jitter — through
the velocity-servo model, and writes `features (N, H+1, 10)` and
`controls (N, H, 7)`. Coherence is what puts real displacement in the data:
independent per-step noise averages out and the arm never goes anywhere. The
printed tool displacement per snippet is the check.

**train** fits the bilinear Koopman model with the multi-step lifted loss. The
number that matters is the open-loop RMSE at the *end* of a horizon-length
window — the planner rolls the full horizon with no feedback, so that is the
error it actually plans against.

**run** executes. Two modes:

* `async` (default, deployment-like): the plant advances in wall-clock real time
  no matter what the planner is doing. A background thread replans from the
  newest snapshot; each boundary applies the newest finished plan *indexed by
  its age* (`u = U[k]`, `k = plan age in control periods`). Slow planning shows
  up as stale actions, or as expired plans if it is hopeless. Nothing waits for
  the planner — the only honest way to ask if it is fast enough.
* `lockstep`: plan → apply → step, serialized. Measures the serial replanning
  rate.

Collisions are counted by a referee that is independent of the planner: it tests
the *executed* pose against the true obstacle sizes, with no margin. The
planner's own margin can never flatter that number.

## Keyboard target

```bash
python main.py run -c interactive.yaml
```

`W`/`S` move the target in x, `A`/`D` in y, `E`/`Q` in z; `[` / `]` shrink and
grow the step and `R` resets it. Arrow keys do the same as WASD, but only WASD
is guaranteed: arrows arrive as multi-byte escape sequences that a multiplexer
or a remote session may mangle.

Which window takes the keys depends on the mode. `goal.mode: keyboard` reads the
MuJoCo viewer's key callback, so the **viewer window** must have focus and the
plant must be `mujoco`. `goal.mode: terminal` reads raw terminal input, so the
**terminal running the command** must have focus — and it is the only option
once the robot is driven over ROS, where the window belongs to the bringup.

## Measured on this machine (10-core CPU, `torch_threads: 4`)

With the shipped config (horizon 25, 512 samples, 5 diffusion steps, one wall
between the start pose and the target):

| | latency / plan | replanning | outcome |
|---|---|---|---|
| async, obstacle | 79 ms | 12.7 Hz | reached in 9.7 s; applied plans 1–4 periods stale; 5/203 boundaries grazed the wall |
| lockstep, obstacle | 81 ms | 12.3 Hz serial | reached in 7.6 s; **0** boundaries in collision |
| async, free space (interactive) | 26 ms | 38 Hz | tracks a hand-driven target within ~20 mm |

The async grazing is not a discrepancy — it is what async mode exists to expose.
The same plans, executed 100–150 ms after the state they were computed from,
clip a wall that lockstep clears. Widen `collision.margin`, shorten the horizon,
or lower `task.action_limit` if you need the executed motion to stay clear.

## Running on a real robot (ROS 2)

`env/ros2_plant.py` implements `Plant` against
[cho_robot_project](https://github.com/)'s `VLAController`: it subscribes to
`/joint_states` and publishes one `cho_interfaces/ActionChunk` waypoint per
control period on `/vla/action/ee_pose`. Select it with `run.plant: ros2` and
run `configs/real.yaml`. Nothing above the plant changes.

**The command is a position waypoint carrying a velocity.** The controller's
output stage differentiates its own reference, and the derivative of a straight
line through `q -> q + u*dt` is exactly `u`. The reference must be *integrated*
in the plant, not re-anchored on the measurement each period: anchoring pins the
tracking error at `u*dt`, and the controller's `kp_joint_vel` turns that into a
constant `(1 + kp*dt)` overspeed -- 2x with the stock gains.

**One waypoint, not the whole chunk.** A stalled planner leaves the controller
holding a target one control period ahead, so the arm stops instead of playing
out a 1.25 s plan. Measured on the simulator, only 1-4 steps of each 25-step
plan are ever executed anyway; the rest is lookahead and belongs in the planner.

**Seeing the target.** With `run.plant: ros2` nothing of ours owns a window, so
the plant publishes the target, its reach shell, the planned path and the
obstacles as a `visualization_msgs/MarkerArray` on `/mbd/markers` in the
`fr3_link0` frame. Either add a MarkerArray display in RViz2, or run
`python main.py view`, which mirrors `/joint_states` into a copy of the model and
draws the same overlays the simulator plant draws — read-only, and safe to start
and stop at any point during a run.

**Going to the start pose.** `python main.py home` switches to the stack's
point-to-point controller, sends a JointSpace goal, and switches back. Two
ordering rules are load-bearing: the switch is not synchronous (poll the
controller state, do not sleep a guess), and a goal in flight must be cancelled
*before* its controller loses the command interfaces -- otherwise the action
server wedges as busy until the stack restarts.

**Interpreter.** `rclpy` is built for ROS's Python (3.10 on Humble), so the MBD
process must run there. `.venv-ros` (created with
`python3.10 -m venv --system-site-packages`) inherits `rclpy` and the workspace
and carries its own `numpy` / `mujoco` / `PyYAML`. `mujoco` is needed purely as
the kinematics library behind the robot description -- no simulation is stepped.

**Measured against the MuJoCo bringup** (`control_mode:=velocity use_vla:=true`):

| check | result |
|---|---|
| our FK vs `/ee_state/pose`, same `q` | 0.00 mm (after removing the FT-sensor spacer from the URDF) |
| MBD reach, no obstacles | 23.4 mm in 2.0 s, 22.6 ms/plan, 44 Hz replanning |
| MBD reach, `chunk_ema_factor: 0.2` (stock) | 2.00 s, velocity ratio 0.61-0.64 |
| MBD reach, `chunk_ema_factor: 1.0` | 1.65 s, velocity ratio 0.90 |
| constant-velocity step, displacement over 3 s | 0.986 of commanded |
| coast after a zero command, EMA 0.2 -> 1.0 | 0.14 rad/s for ~0.25 s -> 0.005 rad/s within 0.1 s |
| whole-arm obstacle run, referee | 0/300 boundaries in collision |

## Things worth knowing

* **`torch_threads`.** The library default (one thread per core) produces
  10–30× straggler spikes in plan latency on a loaded desktop. A small fixed
  pool gives flat plans.
* **Viewer exit.** MuJoCo 3.9's `launch_passive` segfaults during its own GL
  teardown on this platform — reproducible in ten lines with no project code.
  `main.py` therefore flushes and `os._exit(0)`s after a viewer run, so the exit
  status stays usable. Remove `_exit_after_viewer` once upstream is fixed.
* **Checkpoints.** `mbd/training.py:load_checkpoint` also reads the older
  `bk_mbd.train` bilinear checkpoints, so an already-trained model can be
  dropped in without re-running stages 1–2.
* **Safety.** The planner charges soft penalties; it does not certify
  collision-free motion. On hardware, keep an independent limit and an e-stop —
  `env/real_plant.py` says where.

## Setting this up on another machine

The repository is self-contained -- the robot description and meshes are in it,
and no source file refers to a path outside it -- so a clone plus an interpreter
is enough for everything except the ROS stages.

**Simulator only.** Any Python with the pinned packages:

```bash
git clone <this repo> && cd mbd_main
pip install -r requirements.txt
python main.py run            # the committed checkpoint runs immediately
```

**With the ROS 2 stack.** `rclpy` is compiled for ROS's own Python (3.10 on
Humble), so the ROS stages need an interpreter that can import it *and* torch:

```bash
source /opt/ros/humble/setup.bash
source <your_ws>/install/setup.bash     # for cho_interfaces
./setup_ros_venv.sh                     # builds .venv-ros and verifies imports
.venv-ros/bin/python main.py view -c real.yaml
```

**What is and is not in the repository.** The trained checkpoint is committed
(64 KB) so a clone can plan straight away; the 12 MB dataset is not, because
`main.py collect` regenerates it in seconds. `.venv-ros/` is excluded — it holds
absolute paths and must be rebuilt per machine. To retrain from scratch:

```bash
python main.py collect && python main.py train    # ~1 minute total
```

A checkpoint carries the `control_dt` and `action_limit` it was trained with,
and `run` refuses to use one that no longer matches the config rather than
silently planning against a plant that does not exist.

**What the repository cannot carry.** Two things live elsewhere and must be
reproduced on the new machine:

* the ROS workspace (`cho_robot_project`) *including the local changes this
  integration needs* — the FT-sensor spacer removed from the three bringups, and
  `chunk_ema_factor: 1.0` in both `controllers.yaml`. Commit and push that
  repository too, or re-apply the same five edits;
* machine-tuned numbers: `run.torch_threads` (4 suits a 10-core desktop; 0 lets
  the library decide, which caused 10-30x latency spikes here) and
  `collect.num_threads`.

## Install

```bash
pip install -r requirements.txt
```
