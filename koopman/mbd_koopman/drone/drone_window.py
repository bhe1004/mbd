"""Section C (drone port): BK-MBD vs convexified bilinear QP-MPC on a WINDOW pass.

A 3D velocity-controlled drone (the bilinear "3D unicycle" of drone3d_control.py)
must fly from the near side of a wall (plane y = Y_WALL) to a target on the far
side. The only passage is a CIRCULAR window offset ABOVE the straight chord, so
reaching needs a non-monotone detour: rise through the window, then descend.

This is the point-mass restoration of the paper's window experiment. On the FR3
the whole arm could not thread a hole cleanly, so the passage had to degrade to a
solid wall the arm lifts over; a drone is effectively a point, so the clean
circular-window contrast comes back. Geometry / penalty structure are ported
verbatim from ../../version 2/experiments/franka_window_spike.py, with the TCP
forward-kinematics replaced by the drone's position (the first three lifted
coordinates z[:3]).

Both planners share ONE fixed, well-trained bilinear model, so the comparison
isolates the optimizer (global annealed sampling that re-selects the passage
homotopy vs per-step convexification that commits to the nominal branch), not
model-training variance. BK-MBD ingests the wall as a cost penalty on plane
crossings outside the window; the QP-SQP gets the honest branch-following
convexification and cannot re-select the homotopy by itself.

    python drone/drone_window.py                 # both planners, all targets
    python drone/drone_window.py --planner bk_mbd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]           # .../mbd_koopman
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.config import MBDConfig, UpdateRule               # noqa: E402
from bk_mbd.mbd import MBDOptimizer                           # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
SAVED = Path(__file__).resolve().parent / "saved"
SAVED.mkdir(exist_ok=True)

# ---- experiment conditions loaded from config.json ----------------------------
# All tunable parameters live in config.json (next to this file). Edit that file
# to change conditions for BOTH the batch runner and the live viewer; the live
# viewer can also override any dw.<GLOBAL> or geo.<attr> at runtime.
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(path=CONFIG_PATH):
    with open(path) as fh:
        return json.load(fh)


CFG = load_config()

# ---- drone dynamics / observation (verbatim from drone3d_control.py) ----------
DT = 0.05
M = 4                      # u = [vx, vy, vz, w] (body-frame velocities + yaw rate)
ND = 5                     # b = [x, y, z, sin(yaw), cos(yaw)]
K = CFG["planner_mbd"]["horizon"]   # training snippet length, tied to the
#                            planning horizon so the model is identified over
#                            the same window the planner rolls it across


def stepn(s, u):
    x, y, z, ya = s
    vx, vy, vz, w = u
    c, sn = np.cos(ya), np.sin(ya)
    return np.array([x + (c * vx - sn * vy) * DT, y + (sn * vx + c * vy) * DT,
                     z + vz * DT, ya + w * DT])


def obsn(s):
    return np.array([s[0], s[1], s[2], np.sin(s[3]), np.cos(s[3])], np.float32)


# ---- deep Koopman model (verbatim class from drone3d_control.py) --------------
class DK(nn.Module):
    def __init__(self, extra=8, hid=64, bilinear=False):
        super().__init__()
        self.N = ND + extra
        self.bilinear = bilinear
        self.enc = nn.Sequential(nn.Linear(ND, hid), nn.Tanh(),
                                 nn.Linear(hid, hid), nn.Tanh(),
                                 nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N) + .01 * torch.randn(self.N, self.N))
        self.B0 = nn.Parameter(.01 * torch.randn(self.N, M))
        if bilinear:
            self.B1 = nn.Parameter(torch.zeros(M, self.N, self.N))

    def lift(self, b):
        return torch.cat([b, self.enc(b)], -1)

    def fstep(self, z, u):
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            for i in range(M):
                zn = zn + u[..., i:i + 1] * (z @ self.B1[i].T)
        return zn

    def decode(self, z):
        return z[..., :ND]


def gen(N, seed):
    r = np.random.default_rng(seed)
    B = np.zeros((N, K + 1, ND), np.float32)
    U = np.zeros((N, K, M), np.float32)
    for i in range(N):
        s = np.array([r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1),
                      r.uniform(-np.pi, np.pi)])
        B[i, 0] = obsn(s)
        bias = np.concatenate([r.uniform(-0.7, 0.7, 3), r.uniform(-1.2, 1.2, 1)])
        for k in range(K):
            u = bias + np.concatenate([r.uniform(-0.4, 0.4, 3), r.uniform(-0.6, 0.6, 1)])
            U[i, k] = u
            s = stepn(s, u)
            B[i, k + 1] = obsn(s)
    return torch.tensor(B), torch.tensor(U)


def train_model(seed, epochs=200, bs=512, n_traj=4000):
    """One fixed, well-trained bilinear model shared by both planners."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    Btr, Utr = gen(n_traj, 100 + seed)
    m = DK(bilinear=True)
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    n = Btr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b = Btr[idx]
            u = Utr[idx]
            z = m.lift(b[:, 0])
            loss = 0.
            for k in range(K):
                z = m.fstep(z, u[:, k])
                loss = loss + ((m.decode(z) - b[:, k + 1]) ** 2).mean() \
                    + .1 * ((z - m.lift(b[:, k + 1]).detach()) ** 2).mean()
            opt.zero_grad()
            (loss / K).backward()
            opt.step()
    m.eval()
    return m


def get_model(seed):
    ckpt = SAVED / f"drone_window_bk_K{K}_seed{seed}.pt"
    m = DK(bilinear=True)
    if ckpt.exists():
        m.load_state_dict(torch.load(ckpt))
        m.eval()
        print(f"loaded {ckpt}", flush=True)
        return m
    print("training bilinear model (one fixed model for both planners)...", flush=True)
    m = train_model(seed)
    torch.save(m.state_dict(), ckpt)
    print(f"saved {ckpt}", flush=True)
    return m


# ---- window geometry + task (values from config.json; overridable at runtime) -
# These stay module-level so run_mbd / run_qp / DroneQP read them unchanged and
# the live viewer can sweep a single knob via `dw.<NAME> = ...`.
H = CFG["planner_mbd"]["horizon"]        # planning horizon
STEPS = CFG["task"]["steps"]             # closed-loop cap
SEED = CFG["experiment"]["model_seed"]
REACH = CFG["task"]["reach"]             # task success tolerance
SUCCESS_TOL = CFG["task"]["success_tol"]  # primary reach tolerance
STRICT = CFG["task"]["strict"]           # strict success tolerance
W_WALL = CFG["wall_penalty"]["w_wall"]
SLAB = CFG["wall_penalty"]["slab"]       # wall half-thickness used by the penalty
MARGIN = CFG["wall_penalty"]["margin"]   # planning margin inside the window / sides
WA = 1                      # wall axis (y); the window lives in the (x, z) plane
WIN = [0, 2]                # in-plane axes (x, z)
POS = [0, 1, 2]             # drone position = first three lifted coordinates
U_LIM = np.array(CFG["limits"]["u_lim"])  # per-channel velocity / yaw-rate limits

# cost weights (paper-style: per-step + control + terminal)
W_EE = CFG["cost"]["w_ee"]
W_CTRL = CFG["cost"]["w_ctrl"]
W_TERM = CFG["cost"]["w_term"]

# BK-MBD optimizer hyperparameters
MBD_SAMPLES = CFG["planner_mbd"]["num_samples"]
MBD_STAGES = CFG["planner_mbd"]["stages"]
MBD_SIG_START = CFG["planner_mbd"]["sigma_start"]
MBD_SIG_END = CFG["planner_mbd"]["sigma_end"]
MBD_ALPHA = CFG["planner_mbd"]["alpha"]
QP_SQP_ITERS = CFG["qp"]["sqp_iters"]


class Geometry:
    """Wall plane y = y_wall with a circular window in the (x, z) plane. Reads
    the geometry block of config.json; attributes can be overridden per instance
    (e.g. geo.win_c = ...) for live experimentation."""

    def __init__(self, cfg=None):
        g = (cfg or CFG)["geometry"]
        self.start = np.array(g["start"], dtype=np.float64)  # [x, y, z, yaw]
        self.y_wall = float(g["y_wall"])
        self.win_c = np.array(g["win_c"], dtype=np.float64)  # (x, z) of the hole
        self.r_win = float(g["r_win"])
        self.target_x = tuple(g["target_x"])
        self.target_y = tuple(g["target_y"])
        self.target_z = tuple(g["target_z"])

    def sample_targets(self, n=8, seed=11):
        """Targets behind the wall, spanning heights from well below the window
        up to near its centre. Low targets sit below the straight start->target
        chord's wall crossing, so reaching needs a non-monotone rise-through-
        then-descend detour (QP-MPC commits to the near-side branch and stalls);
        high targets are level with the window, so the straight chord already
        threads it (QP reaches those). This mirrors the franka window protocol,
        where the convex baseline reaches the no-homotopy-switch subset and
        stalls on the rest, while sampling reaches all of them."""

        rng = np.random.default_rng(seed)
        xs = rng.uniform(*self.target_x, n)
        ys = rng.uniform(*self.target_y, n)
        zs = rng.uniform(*self.target_z, n)
        return [np.array([x, y, z], np.float64) for x, y, z in zip(xs, ys, zs)]

    def executed_violation(self, traj) -> int:
        """Count executed plane crossings outside the TRUE window."""

        pos = [np.asarray(s)[:3] for s in traj]
        n_bad = 0
        for a, b in zip(pos[:-1], pos[1:]):
            da, db = a[WA] - self.y_wall, b[WA] - self.y_wall
            if da * db < 0:
                t = da / (da - db)
                xz = (1 - t) * a[WIN] + t * b[WIN]
                if np.linalg.norm(xz - self.win_c) > self.r_win:
                    n_bad += 1
        return n_bad


# ---- BK-MBD (annealed sampling; wall enters only as a cost penalty) -----------
def run_mbd(model, geo: Geometry, target, seed, on_step=None):
    config = MBDConfig(num_samples=MBD_SAMPLES, num_diffusion_steps=MBD_STAGES,
                       sigma_start=MBD_SIG_START, sigma_end=MBD_SIG_END,
                       alpha=MBD_ALPHA, eta=1.0,
                       update_rule=UpdateRule.SCORE_LANGEVIN,
                       add_langevin_noise=False)
    opt = MBDOptimizer(config, -U_LIM, U_LIM)
    s = geo.start.copy()
    goal = np.asarray(target, dtype=np.float64)
    U = np.zeros((H, M))
    rng = np.random.default_rng(seed)
    tgt = torch.as_tensor(goal, dtype=torch.float32)
    wc = torch.as_tensor(geo.win_c, dtype=torch.float32)
    traj = [s.copy()]
    t_plan, n_steps = 0.0, 0
    for step in range(STEPS):
        def evaluate(cands):
            with torch.no_grad():
                Ut = torch.as_tensor(cands, dtype=torch.float32)
                b = torch.as_tensor(obsn(s), dtype=torch.float32).expand(Ut.shape[0], -1)
                z = model.lift(b)
                cost = torch.zeros(Ut.shape[0])
                prev_p = b[:, POS]
                for k in range(Ut.shape[1]):
                    z = model.fstep(z, Ut[:, k])
                    p = model.decode(z)[:, POS]
                    cost = cost + W_EE * ((p - tgt) ** 2).sum(-1) \
                        + W_CTRL * (Ut[:, k] ** 2).sum(-1)
                    # penalty 1: sitting inside the wall slab, outside the hole
                    in_slab = (torch.abs(p[:, WA] - geo.y_wall) < SLAB).float()
                    rad = (p[:, WIN] - wc).norm(dim=-1)
                    excess = torch.clamp(rad - (geo.r_win - MARGIN), min=0.0)
                    cost = cost + W_WALL * in_slab * excess ** 2
                    # penalty 2: crossing the plane outside the hole
                    da = prev_p[:, WA] - geo.y_wall
                    db = p[:, WA] - geo.y_wall
                    crossed = (da * db < 0).float()
                    t_frac = (da / (da - db + 1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
                    xz = (1 - t_frac) * prev_p[:, WIN] + t_frac * p[:, WIN]
                    rad_c = (xz - wc).norm(dim=-1)
                    exc_c = torch.clamp(rad_c - (geo.r_win - MARGIN), min=0.0)
                    cost = cost + W_WALL * crossed * exc_c ** 2
                    prev_p = p
                cost = cost + W_TERM * ((p - tgt) ** 2).sum(-1)
                return cost.numpy()

        t0 = time.perf_counter()
        result = opt.optimize(U, evaluate, rng=rng)
        t_plan += time.perf_counter() - t0
        U = result.controls
        u0 = np.clip(U[0], -U_LIM, U_LIM)
        s = stepn(s, u0)
        traj.append(s.copy())
        U = np.roll(U, -1, axis=0)
        U[-1] = U[-2]
        n_steps = step + 1
        if on_step is not None:
            on_step(step, s, traj)
        if np.linalg.norm(s[:3] - goal) < STRICT:
            break
    return s, goal, traj, n_steps, 1e3 * t_plan / n_steps


# ---- convexified bilinear QP-MPC (honest per-step branch following) -----------
class DroneQP:
    """Convexified bilinear-Koopman MPC for the drone, compiled ONCE as a
    DPP-parametrized problem (cvxpy Parameters for the frozen input maps, the
    goal, the initial lifted state, and the per-step keep-out row); each SQP
    iterate and each plan only updates parameter *values*, skipping cvxpy's
    canonicalization. Ported from secy/sqp_mpc.py, simplified because the drone
    position is a linear lifted coordinate (p = z[:3]), so the keep-out is linear
    in the state and needs no forward-kinematics Jacobian.

    Per iterate it freezes the bilinear input map at the nominal lifted states
    (B_k = B0 + Bs . zbar_k) so z_{k+1} = A z_k + B_k u_k is linear in u, then
    constrains p_k to the single half-space branch the nominal selects (near
    side / far side / inside the window cylinder). It cannot re-select the
    branch within the solve -- the honest per-step convexification, and exactly
    why it stalls at the non-convex window while MBD re-selects by global
    weighting. The keep-out is a HARD constraint (no slack), so an infeasible
    branch simply fails the solve and the previous plan is kept -> a safe stall,
    never an executed wall crossing."""

    def __init__(self, model, geo: Geometry, sqp_iters=QP_SQP_ITERS):
        import cvxpy as cp

        self.cp = cp
        self.model = model
        self.geo = geo
        self.sqp_iters = sqp_iters
        self.A = model.A.detach().numpy().astype(np.float64)
        self.B0 = model.B0.detach().numpy().astype(np.float64)
        self.B1 = model.B1.detach().numpy().astype(np.float64)   # (M, N, N)
        self.L = self.A.shape[0]
        self._solve_kw = dict(solver=cp.OSQP, warm_start=True,
                              eps_abs=1.5e-3, eps_rel=1.5e-3, max_iter=1500)
        self._build()

    def _build(self):
        cp = self.cp
        L = self.L
        self.U_var = cp.Variable((H, M))
        self.Z_var = cp.Variable((H, L))                 # explicit lifted states
        self.z0_p = cp.Parameter(L)
        self.goal_p = cp.Parameter(3)
        self.Bk = [cp.Parameter((L, M)) for _ in range(H)]
        self.gk = [cp.Parameter(3) for _ in range(H)]    # keep-out row over p=z[:3]
        self.hk = [cp.Parameter() for _ in range(H)]     # keep-out rhs (>=)

        cons = [self.U_var >= -U_LIM, self.U_var <= U_LIM]
        cons.append(self.Z_var[0] == self.A @ self.z0_p + self.Bk[0] @ self.U_var[0])
        for k in range(1, H):
            cons.append(self.Z_var[k]
                        == self.A @ self.Z_var[k - 1] + self.Bk[k] @ self.U_var[k])
        for k in range(H):
            cons.append(self.gk[k] @ self.Z_var[k][:3] >= self.hk[k])

        cost = 0
        for k in range(H):
            p_e = self.Z_var[k][:3]
            cost = cost + W_EE * cp.sum_squares(p_e - self.goal_p) \
                + W_CTRL * cp.sum_squares(self.U_var[k])
        cost = cost + W_TERM * cp.sum_squares(self.Z_var[H - 1][:3] - self.goal_p)
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def _nominal(self, z0, U):
        """Frozen input maps + decoded positions along the nominal rollout."""

        geo = self.geo
        with torch.no_grad():
            z = torch.as_tensor(z0, dtype=torch.float32)
            zbars, poss = [], []
            for k in range(H):
                zbars.append(z.numpy().astype(np.float64))
                z = self.model.fstep(z, torch.as_tensor(U[k], dtype=torch.float32))
                poss.append(self.model.decode(z)[POS].numpy().astype(np.float64))
        B_list = [self.B0 + np.einsum("mij,j->im", self.B1, zb) for zb in zbars]
        g = np.zeros((H, 3), dtype=np.float64)
        h = np.full(H, -1e9, dtype=np.float64)           # inactive default
        for k, nom in enumerate(poss):
            rad_nom = np.linalg.norm(nom[WIN] - geo.win_c)
            if rad_nom <= geo.r_win - MARGIN:            # inside the window cylinder
                v = nom[WIN] - geo.win_c
                nn = np.linalg.norm(v)
                if nn > 1e-9:
                    nhat = v / nn
                    # nhat . p[WIN] <= r_win - MARGIN + nhat . win_c  (as a >= row)
                    g[k, WIN[0]] = -nhat[0]
                    g[k, WIN[1]] = -nhat[1]
                    h[k] = -((geo.r_win - MARGIN) + float(nhat @ geo.win_c))
            elif nom[WA] < geo.y_wall:                   # near side: p[WA] <= y_wall-M
                g[k, WA] = -1.0
                h[k] = -(geo.y_wall - MARGIN)
            else:                                        # far side: p[WA] >= y_wall+M
                g[k, WA] = 1.0
                h[k] = geo.y_wall + MARGIN
        return B_list, g, h

    def plan(self, z0, goal, U_warm):
        cp = self.cp
        self.z0_p.value = np.asarray(z0, dtype=np.float64)
        self.goal_p.value = np.asarray(goal, dtype=np.float64)
        U = np.asarray(U_warm, dtype=np.float64).copy()
        for _ in range(self.sqp_iters):
            B_list, g, h = self._nominal(z0, U)
            for k in range(H):
                self.Bk[k].value = B_list[k]
                self.gk[k].value = g[k]
                self.hk[k].value = h[k]
            try:
                self.prob.solve(**self._solve_kw)
            except cp.error.SolverError:
                pass
            if self.U_var.value is None:
                try:
                    self.prob.solve(solver=cp.CLARABEL)
                except cp.error.SolverError:
                    pass
            if self.U_var.value is not None:
                U = np.clip(np.asarray(self.U_var.value), -U_LIM, U_LIM)
        return U


def run_qp(model, geo: Geometry, target, sqp_iters=QP_SQP_ITERS, on_step=None,
           qp=None):
    if qp is None:
        qp = DroneQP(model, geo, sqp_iters)
    s = geo.start.copy()
    goal = np.asarray(target, dtype=np.float64)
    U_nom = np.zeros((H, M))
    traj = [s.copy()]
    t_plan, n_steps = 0.0, 0
    for step in range(STEPS):
        z0 = model.lift(torch.as_tensor(obsn(s), dtype=torch.float32)) \
            .detach().numpy().astype(np.float64)
        t0 = time.perf_counter()
        U_nom = qp.plan(z0, goal, U_nom)
        t_plan += time.perf_counter() - t0
        u0 = np.clip(U_nom[0], -U_LIM, U_LIM)
        s = stepn(s, u0)
        traj.append(s.copy())
        U_nom = np.roll(U_nom, -1, axis=0)
        U_nom[-1] = U_nom[-2]
        n_steps = step + 1
        if on_step is not None:
            on_step(step, s, traj)
        if np.linalg.norm(s[:3] - goal) < STRICT:
            break
    return s, goal, traj, n_steps, 1e3 * t_plan / n_steps


# ---- frozen experiment design (Section C drone port) --------------------------
N_TARGETS = CFG["experiment"]["n_targets"]
EXP_SEED = CFG["experiment"]["exp_seed"]   # target-sampling seed (config.json)
OUT_DIR = OUT / "window"
TRIALS_PATH = OUT_DIR / "trials.jsonl"
TRAJS_PATH = OUT_DIR / "trajs.npz"
PLANNERS = ("bk_mbd", "bk_qp_sqp")


def trial_metrics(traj, goal, viol, steps, ms):
    P = np.asarray(traj)[:, :3]
    d = np.linalg.norm(P - goal, axis=1)
    final_err, min_err = float(d[-1]), float(d.min())
    reached = final_err < SUCCESS_TOL and viol == 0
    return dict(final_err=final_err, min_err=min_err, viol=int(viol),
                steps=int(steps), ms_per_step=float(ms),
                reached_5cm=bool(final_err < REACH),
                reached_2p5cm=bool(final_err < SUCCESS_TOL),
                reached_1cm=bool(final_err < STRICT),
                reached=bool(reached),
                safe_stall=bool(not reached and viol == 0))


def done_keys(path):
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                seen.add((r["planner"], r.get("model_seed"), r["target_idx"]))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planner", choices=list(PLANNERS), default=None,
                    help="restrict to one planner (default: both)")
    ap.add_argument("--n-targets", type=int, default=N_TARGETS)
    ap.add_argument("--seed", type=int, default=EXP_SEED)
    ap.add_argument("--model-seed", type=int, nargs="+", default=[SEED],
                    help="one or more training seeds for the shared bilinear model")
    args = ap.parse_args()

    torch.set_num_threads(4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    geo = Geometry()
    targets = geo.sample_targets(n=args.n_targets, seed=args.seed)
    print(f"start {np.round(geo.start, 3)}  wall y={geo.y_wall:.3f}  "
          f"window c(x,z)={np.round(geo.win_c, 3)} r={geo.r_win}  "
          f"N={args.n_targets} seed={args.seed}", flush=True)

    planners = [args.planner] if args.planner else list(PLANNERS)
    models = {ms: get_model(ms) for ms in args.model_seed}
    qps = {ms: DroneQP(m, geo, sqp_iters=QP_SQP_ITERS) for ms, m in models.items()} \
        if "bk_qp_sqp" in planners else {}

    seen = done_keys(TRIALS_PATH)
    trajs = dict(np.load(TRAJS_PATH)) if TRAJS_PATH.exists() else {}
    jobs = [(p, ms, ti) for ms in args.model_seed for p in planners
            for ti in range(len(targets)) if (p, ms, ti) not in seen]
    total = len(planners) * len(args.model_seed) * len(targets)
    print(f"exp: {len(jobs)} trials to run ({total - len(jobs)} done, {total} total)"
          f" -> {TRIALS_PATH}", flush=True)

    t_start = time.perf_counter()
    with open(TRIALS_PATH, "a") as fh:
        for i, (name, mseed, ti) in enumerate(jobs, 1):
            target = targets[ti]
            rng_seed = 20 + ti
            model = models[mseed]
            if name == "bk_mbd":
                s, goal, traj, steps, ms = run_mbd(model, geo, target, rng_seed)
            else:
                s, goal, traj, steps, ms = run_qp(model, geo, target, qp=qps[mseed])
            viol = geo.executed_violation(traj)
            rec = dict(planner=name, target_idx=int(ti),
                       target=list(map(float, target)), model_seed=int(mseed),
                       rng_seed=int(rng_seed),
                       **trial_metrics(traj, goal, viol, steps, ms))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            trajs[f"{name}_s{mseed}_{ti}"] = np.stack(traj)
            print(f"[{i}/{len(jobs)}] {name} s{mseed} t{ti} {np.round(target,2)}: "
                  f"err={rec['final_err']:.4f} viol={viol} steps={steps} "
                  f"ms={ms:.0f} -> {'REACH' if rec['reached'] else 'stall'}",
                  flush=True)

    # persist every per-trial trajectory array (bk_mbd_<ti>, bk_qp_sqp_<ti>) plus
    # the scene metadata; `trajs` already carries any resumed arrays.
    meta = {k: trajs[k] for k in ("targets", "start", "wall", "win_c", "r_win")
            if k in trajs}
    meta.update(targets=np.stack(targets), start=geo.start,
                wall=np.array([geo.y_wall]), win_c=geo.win_c,
                r_win=np.array([geo.r_win]))
    traj_arrays = {k: v for k, v in trajs.items()
                   if k not in ("targets", "start", "wall", "win_c", "r_win")}
    np.savez(TRAJS_PATH, **traj_arrays, **meta)
    dt = time.perf_counter() - t_start
    print(f"\ndone: {len(jobs)} trials in {dt/60:.1f} min. "
          f"metrics -> {TRIALS_PATH}, trajs -> {TRAJS_PATH}", flush=True)


if __name__ == "__main__":
    main()
