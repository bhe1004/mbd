"""SQP-MPC planner: whole-body convexified bilinear-Koopman MPC.

The convex counterfactual to BK-MBD for paper section C. Same Scene, same
obstacles, same margins -- only the way geometry is consumed differs.

Per control step it repeats ``max_iters`` SQP iterates: (1) roll the TRUE
bilinear model along the nominal controls, (2) freeze the bilinear input map at
the nominal lifted states (``B_k = B0 + Bs . zbar_k``) so ``z_{k+1} = A z_k +
B_k u_k`` is linear in u, (3) solve a QP with the exact quadratic task cost and
the obstacle keep-outs convexified by the branch the nominal selects. For every
arm sphere near a box the nominal picks ONE separating face (the axis of
maximum signed separation) and the QP constrains the linearized sphere center
``p ~= p_nom + J (q - q_nom)`` (from ``ArmFK.spheres_and_jacobians``) to that
face's outer half-space. It cannot re-select the face/homotopy within the
solve -- the honest per-step convexification, and exactly why it stalls at a
non-convex passage while MBD re-selects the branch by global weighting.

The set of selected faces is exposed as ``last_linearization`` (a list of
:class:`secy.linearization_view.FacePlane`) so the runtime can draw the convex
region the QP actually believes in. Ported from
``experiments/franka_bookshelf.py:QPSQPPlanner``.

Speed: the QP is built ONCE as a DPP-parametrized problem (cvxpy Parameters for
the frozen input maps, the goal, the initial lifted state, and the keep-out
rows), then only the parameter *values* change each SQP iterate and each plan.
That skips cvxpy's canonicalization -- which dominated the cost when the problem
was rebuilt every iterate -- so a plan drops from hundreds of ms to tens. The
lifted state z is kept as an explicit variable (not eliminated) precisely so
every product is parameter-times-variable, which DPP allows; eliminating z would
multiply frozen-map parameters into the keep-out parameters and break DPP.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.train import load_checkpoint  # noqa: E402
from envs.franka import NUM_JOINTS  # noqa: E402

from secy.config import Config  # noqa: E402
from secy.environment import Scene  # noqa: E402
from secy.linearization_view import FacePlane, dedupe  # noqa: E402

COLOR = (0.12, 0.47, 0.71)   # overlay color for this planner
_OFF = -1e9                  # rhs of an inactive keep-out row: 0 . q >= -1e9


class SQPMPCPlanner:
    """Whole-body convexified bilinear-Koopman MPC (branch-following QP-SQP)."""

    color = COLOR
    name = "sqp_mpc"

    def __init__(self, scene: Scene, cfg: Config) -> None:
        import cvxpy as cp  # local: only the QP baseline needs it

        self.cp = cp
        self.scene = scene
        self.cfg = cfg
        self.task = scene.task
        self.device = scene.device
        self.fk = scene.fk
        self.horizon = scene.horizon
        self.whole_arm = scene.fk is not None

        # ---- the (same) Koopman model --------------------------------------
        mbd = cfg.mbd
        checkpoint = mbd.checkpoint or (
            PROJECT_ROOT / "out" / "franka" / "models" / f"bk_seed{mbd.seed}.pt")
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint}")
        self.model, _ = load_checkpoint(checkpoint, device=self.device)
        self.model.eval()
        print(f"loaded={checkpoint}")
        self.A = self.model.A.detach().cpu().numpy().astype(np.float64)
        self.B0 = self.model.B0.detach().cpu().numpy().astype(np.float64)
        self.Bs = self.model.Bs.detach().cpu().numpy().astype(np.float64)
        self.L = self.A.shape[0]                       # lift dimension

        # ---- obstacle geometry (shared with MBD, consumed convexly) --------
        obs = scene.obstacle
        self.box_centers = (obs.b_centers_np if obs is not None
                            else np.zeros((0, 3)))
        self.box_halfs = (obs.b_halfs_np if obs is not None
                          else np.zeros((0, 3)))
        self.sph_centers = (obs.s_centers_np if obs is not None
                            else np.zeros((0, 3)))
        self.sph_radii = (obs.s_radii_np if obs is not None else np.zeros(0))
        self.radii = (self.fk.radii_np if self.whole_arm else np.zeros(1))
        self.margin = cfg.collision.margin
        self.P = len(self.radii)                       # arm spheres
        self.floor_z = (obs.floor_z if obs is not None else None)
        self.use_floor = self.floor_z is not None and self.whole_arm
        # one selected-face row per (arm sphere, obstacle) + one floor row per
        # sphere; a fixed count so the parametrized problem keeps a constant
        # structure (inactive rows are switched off by value, not removed).
        self.R = self.P * (len(self.box_centers) + len(self.sph_centers)) \
            + (self.P if self.use_floor else 0)

        # ---- SQP + joint-limit settings ------------------------------------
        self.max_iters = cfg.sqp.max_iters
        self.act_dist = cfg.sqp.act_dist
        # OSQP options: the QP is re-solved every SQP iterate on shifted data, so
        # a loose tolerance (the SQP re-freeze re-corrects anyway) and a capped
        # iteration count keep each solve to a few ms without changing behaviour.
        self._solve_kw = dict(solver=cp.OSQP, warm_start=True,
                              eps_abs=cfg.sqp.osqp_eps, eps_rel=cfg.sqp.osqp_eps,
                              max_iter=cfg.sqp.osqp_max_iter)
        self.u_lim = float(self.task.config.action_limit)
        self.w = self.task.cost_weights
        self.jl_lo = self.task.joint_low + cfg.joint_limit.margin
        self.jl_hi = self.task.joint_high - cfg.joint_limit.margin
        # trust region (keeps the FK/dynamics linearization valid) + keep-out
        # slack (exact penalty -> the QP stays feasible when the nominal grazes).
        self.trust_region = float(cfg.sqp.trust_region)
        self.slack_weight = float(cfg.sqp.slack_weight)

        self.n_solve_fail = 0
        self.last_linearization = []
        self._build_problem()
        print(f"SQP-MPC: {self.max_iters} SQP re-freezes, act_dist {self.act_dist}"
              f", whole_arm={self.whole_arm}, {len(self.box_centers)} box(es), "
              f"{self.R} keep-out rows/step (compiled once, DPP)")

    # ------------------------------------------------- the parametrized QP
    def _build_problem(self) -> None:
        """Build the QP ONCE with cvxpy Parameters; solves reuse the compile."""

        cp = self.cp
        H, L = self.horizon, self.L
        self.U_var = cp.Variable((H, NUM_JOINTS))
        self.Z_var = cp.Variable((H, L))             # explicit lifted states

        self.z0_p = cp.Parameter(L)
        self.goal_p = cp.Parameter(3)
        self.Bk = [cp.Parameter((L, NUM_JOINTS)) for _ in range(H)]
        # nominal joints per step -> the trust-region centre (re-set each iterate)
        self.qnom_p = [cp.Parameter(NUM_JOINTS) for _ in range(H)]
        # keep-out rows on the joints q = Z[k][:7]; inactive -> zero row, -1e9
        self.Gk = [cp.Parameter((self.R, NUM_JOINTS)) for _ in range(H)] \
            if self.R else []
        self.hk = [cp.Parameter(self.R) for _ in range(H)] if self.R else []
        # exact-penalty slack on the keep-out (>=0), so a grazing nominal never
        # makes the QP infeasible (which would return garbage controls).
        self.slack = cp.Variable((H, self.R), nonneg=True) if self.R else None

        cons = [self.U_var >= -self.u_lim, self.U_var <= self.u_lim,
                self.Z_var[:, :NUM_JOINTS] <= self.jl_hi,
                self.Z_var[:, :NUM_JOINTS] >= self.jl_lo]
        cons.append(self.Z_var[0] == self.A @ self.z0_p + self.Bk[0] @ self.U_var[0])
        for k in range(1, H):
            cons.append(self.Z_var[k]
                        == self.A @ self.Z_var[k - 1] + self.Bk[k] @ self.U_var[k])
        tr = self.trust_region
        for k in range(H):
            q_e = self.Z_var[k][:NUM_JOINTS]
            # trust region: bound the joint step from the nominal so the FK /
            # frozen-dynamics linearization stays valid (else huge steps drive
            # the true arm through obstacles the linear model thinks are clear).
            cons += [q_e - self.qnom_p[k] <= tr, q_e - self.qnom_p[k] >= -tr]
            if self.R:
                cons.append(self.Gk[k] @ q_e + self.slack[k] >= self.hk[k])

        cost = 0
        for k in range(H):
            tip = self.Z_var[k][NUM_JOINTS:NUM_JOINTS + 3]
            cost = cost + self.w.ee * cp.sum_squares(tip - self.goal_p) \
                + self.w.control * cp.sum_squares(self.U_var[k])
        cost = cost + self.w.terminal_ee * cp.sum_squares(
            self.Z_var[H - 1][NUM_JOINTS:NUM_JOINTS + 3] - self.goal_p)
        if self.R:
            cost = cost + self.slack_weight * cp.sum(self.slack)
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    # ------------------------------------------------------------- nominal
    def _nominal(self, z0, U):
        """True bilinear rollout under U: frozen input maps, decoded joints,
        tips, and (whole-arm) sphere centers + FD Jacobians per step."""

        zbars = np.empty((self.horizon, z0.shape[0]), dtype=np.float64)
        qn = np.empty((self.horizon, NUM_JOINTS), dtype=np.float64)
        tn = np.empty((self.horizon, 3), dtype=np.float64)
        with torch.no_grad():
            z = torch.as_tensor(z0, dtype=torch.float32, device=self.device)
            for k in range(self.horizon):
                zbars[k] = z.cpu().numpy().astype(np.float64)
                z = self.model.step(
                    z, torch.as_tensor(U[k], dtype=torch.float32, device=self.device))
                dec = self.model.decode(z).cpu().numpy().astype(np.float64)
                qn[k] = dec[:NUM_JOINTS]
                tn[k] = dec[NUM_JOINTS:NUM_JOINTS + 3]
            if self.whole_arm:
                qt = torch.as_tensor(qn, dtype=torch.float32, device=self.device)
                p, J = self.fk.spheres_and_jacobians(qt)
                pts = p.cpu().numpy().astype(np.float64)      # (H, P, 3)
                jac = J.cpu().numpy().astype(np.float64)      # (H, P, 3, 7)
            else:
                pts = tn[:, None, :]
                jac = None
        B_list = [self.B0 + np.einsum("mij,j->im", self.Bs, zb) for zb in zbars]
        return B_list, qn, tn, pts, jac

    # ------------------------------------------------------- keep-out values
    def _keepout_values(self, qn, pts, jac, planes):
        """Fill the fixed (H, R, 7) keep-out coefficient / (H, R) rhs arrays for
        the current nominal, branch-selecting one face per (sphere, obstacle).
        Inactive rows are left as (0, -1e9). Appends selected box faces to
        ``planes`` for the linearization overlay."""

        H, R = self.horizon, self.R
        G = np.zeros((H, R, NUM_JOINTS), dtype=np.float64)
        h = np.full((H, R), _OFF, dtype=np.float64)
        for k in range(H):
            r = 0
            for i in range(pts.shape[1]):
                p_nom = pts[k, i]
                r_i = self.radii[i] if self.whole_arm else 0.0
                Ji = jac[k, i] if jac is not None else None      # (3,7)
                # ---- box obstacles: one separating face (the branch) --------
                for b in range(self.box_centers.shape[0]):
                    c, hh = self.box_centers[b], self.box_halfs[b]
                    infl = hh + r_i + self.margin
                    d = np.abs(p_nom - c) - infl
                    if d.max() <= self.act_dist:
                        j = int(np.argmax(d))
                        s = 1.0 if p_nom[j] >= c[j] else -1.0
                        planes.append(FacePlane(axis=j, sign=s,
                                                coord=float(c[j] + s * infl[j]),
                                                center=tuple(c), half=tuple(hh)))
                        Jij = Ji[j]
                        G[k, r] = s * Jij
                        h[k, r] = (infl[j] - s * (p_nom[j] - c[j])
                                   + s * float(Jij @ qn[k]))
                    r += 1
                # ---- sphere obstacles: single radial half-space -------------
                for b in range(self.sph_centers.shape[0]):
                    c = self.sph_centers[b]
                    rr = self.sph_radii[b] + r_i + self.margin
                    v = p_nom - c
                    dist = float(np.linalg.norm(v))
                    if dist - rr <= self.act_dist:
                        nrm = v / (dist + 1e-9)
                        g = nrm @ Ji
                        G[k, r] = g
                        h[k, r] = rr - float(nrm @ (p_nom - c)) + float(g @ qn[k])
                    r += 1
                # ---- floor: keep the sphere above z = floor_z (+ r + margin) --
                if self.use_floor:
                    Jz = Ji[2]                       # z-row of the FK Jacobian
                    # Jz . q_e >= floor_z + r_i + margin - (p_nom_z - Jz.q_nom)
                    G[k, r] = Jz
                    h[k, r] = (self.floor_z + r_i + self.margin
                               - p_nom[2] + float(Jz @ qn[k]))
                    r += 1
        return G, h

    # --------------------------------------------------------------- public
    def plan(self, x, goal, u_prev, U_warm, err, rng):
        cp = self.cp
        b0 = self.task.state_to_base_torch(x, self.device)
        z0 = self.model.lift(b0).detach().cpu().numpy().astype(np.float64)
        self.z0_p.value = z0
        self.goal_p.value = np.asarray(goal, dtype=np.float64)
        U = np.asarray(U_warm, dtype=np.float64).copy()
        planes = []
        for _ in range(self.max_iters):
            planes = []                              # keep only the last iterate
            B_list, qn, tn, pts, jac = self._nominal(z0, U)
            for k in range(self.horizon):
                self.Bk[k].value = B_list[k]
                self.qnom_p[k].value = qn[k]           # trust-region centre
            if self.R:
                G, h = self._keepout_values(qn, pts, jac, planes)
                for k in range(self.horizon):
                    self.Gk[k].value = G[k]
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
                U = np.clip(np.asarray(self.U_var.value), -self.u_lim, self.u_lim)
            else:
                self.n_solve_fail += 1   # keep the previous nominal (stall)

        self.last_linearization = dedupe(planes)
        _, _, tn, _, _ = self._nominal(z0, U)
        ee0 = np.asarray(self.task.ee_of_q(np.asarray(x)[:NUM_JOINTS]))
        ee_pred = np.concatenate([ee0[None], tn], axis=0)
        return U, ee_pred

    def warmup(self, x0, goal, n_plans: int) -> None:
        """One plan to trigger the one-time cvxpy canonicalization before the
        clock starts, so the first live plan is already fast."""

        self.plan(x0, goal, np.zeros(NUM_JOINTS),
                  np.zeros((self.horizon, NUM_JOINTS)), 1.0, None)
