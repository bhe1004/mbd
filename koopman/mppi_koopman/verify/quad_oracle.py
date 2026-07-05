"""
Step (b): get the ORACLE (true-dynamics) MPPI to fly the quadrotor to a goal.
 - Execution: MuJoCo (ground truth).
 - Oracle rollout: a BATCHED analytic rigid-body quadrotor (torch) — VERIFIED to match MuJoCo first.
If the oracle flies, the MPPI cost/setup is valid -> the earlier crash is a Koopman-model accuracy issue.
Output: out/quad_oracle.png (+ verification + fly result)
"""
import os, numpy as np, torch, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
XML = """
<mujoco><option timestep="0.01" gravity="0 0 -9.81"/>
<worldbody><body name="quad" pos="0 0 1"><freejoint/>
<geom type="box" size="0.12 0.12 0.03" mass="1"/></body></worldbody></mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model)
bid = model.body("quad").id; MG = 9.81
MASS = float(model.body_mass[bid]); INER = torch.tensor(model.body_inertia[bid], dtype=torch.float32)
G = 9.81; DT = 0.01; NSUB = 2
print(f"mass={MASS:.3f} inertia={INER.numpy()}")

def mj_ctrl(u, nsub=NSUB):
    for _ in range(nsub):
        R = data.xmat[bid].reshape(3, 3)
        data.xfrc_applied[bid, :3] = R @ np.array([0.0, 0.0, u[0]])
        data.xfrc_applied[bid, 3:] = R @ np.array([u[1], u[2], u[3]])
        mujoco.mj_step(model, data)
def reset(pos=(0, 0, 1)):
    mujoco.mj_resetData(model, data); data.qpos[:3] = pos; data.qpos[3] = 1.0; mujoco.mj_forward(model, data)

# ----- batched analytic quadrotor (torch) -----
def skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)
def exp_so3(w):
    ang = w.norm(dim=-1, keepdim=True); axis = w/(ang+1e-9); K = skew(axis)
    a = ang.unsqueeze(-1); I = torch.eye(3).expand(K.shape)
    return I + torch.sin(a)*K + (1-torch.cos(a))*(K @ K)
def step_true(P, V, R, W, u):                       # batched; u=(...,4); returns updated
    for _ in range(NSUB):
        T = u[..., 0:1]; tau = u[..., 1:4]
        acc = R[..., :, 2]*T/MASS + torch.tensor([0., 0., -G])
        Iw = INER*W; wdot = (tau - torch.cross(W, Iw, dim=-1))/INER
        W = W + wdot*DT; V = V + acc*DT; R = R @ exp_so3(W*DT); P = P + V*DT
    return P, V, R, W

# ----- verify analytic == MuJoCo on a random control sequence -----
reset((0, 0, 1.5)); r = np.random.default_rng(0); us = [np.array([MG+r.uniform(-3, 3), *r.uniform(-0.4, 0.4, 3)]) for _ in range(40)]
P = torch.tensor(data.qpos[:3], dtype=torch.float32)[None]; V = torch.tensor(data.qvel[:3], dtype=torch.float32)[None]
R = torch.tensor(data.xmat[bid].reshape(3, 3), dtype=torch.float32)[None]; W = torch.tensor(data.qvel[3:6], dtype=torch.float32)[None]
err = 0.0
for u in us:
    mj_ctrl(u); P, V, R, W = step_true(P, V, R, W, torch.tensor(u, dtype=torch.float32)[None])
    err = max(err, float(np.linalg.norm(P[0].numpy()-data.qpos[:3])))
print(f"[verify] analytic vs MuJoCo  max position drift over 40 steps = {err:.4f}  (small => valid oracle)")

# ----- oracle MPPI (analytic rollout, execute on MuJoCo) -----
GOAL = np.array([1.0, 1.0, 1.6]); gt = torch.tensor(GOAL, dtype=torch.float32); U_HOVER = torch.tensor([MG, 0, 0, 0.])
@torch.no_grad()
def mppi_oracle(T=30, Ks=1500, lam=0.6, steps=160, seed=3):
    g = torch.Generator().manual_seed(seed); reset((0, 0, 1)); U = torch.zeros(T, 4); traj = []
    sig = torch.tensor([1.0, 0.04, 0.04, 0.04])      # torque noise scaled to the tiny inertia (I~0.005)
    for t in range(steps):
        P0 = torch.tensor(data.qpos[:3], dtype=torch.float32); V0 = torch.tensor(data.qvel[:3], dtype=torch.float32)
        R0 = torch.tensor(data.xmat[bid].reshape(3, 3), dtype=torch.float32); W0 = torch.tensor(data.qvel[3:6], dtype=torch.float32)
        P = P0.repeat(Ks, 1); V = V0.repeat(Ks, 1); R = R0.repeat(Ks, 1, 1); W = W0.repeat(Ks, 1)
        eps = torch.randn(Ks, T, 4, generator=g)*sig; Vc = U[None] + eps; cost = torch.zeros(Ks)
        for k in range(T):
            uk = U_HOVER + Vc[:, k]; uk = torch.cat([uk[:, :1].clamp(0, 20), uk[:, 1:]], 1)
            P, V, R, W = step_true(P, V, R, W, uk)
            up = R[:, 0, 2]**2 + R[:, 1, 2]**2 + (R[:, 2, 2]-1)**2
            cost += ((P-gt)**2).sum(1) + 3.0*up + 0.2*(V**2).sum(1) + 0.08*(W**2).sum(1) + 0.01*(Vc[:, k]**2).sum(1)
        up = R[:, 0, 2]**2 + R[:, 1, 2]**2 + (R[:, 2, 2]-1)**2; cost += 8.0*((P-gt)**2).sum(1) + 8.0*up
        w = torch.softmax(-(cost-cost.min())/lam, 0); U = U + (w[:, None, None]*eps).sum(0)
        u0 = (U_HOVER + U[0]).numpy(); u0[0] = np.clip(u0[0], 0, 20); mj_ctrl(u0)
        Rm = data.xmat[bid].reshape(3, 3); traj.append((data.qpos[:3].copy(), Rm[:, 2].copy()))
        U = torch.roll(U, -1, 0); U[-1] = U[-2]
        if np.linalg.norm(data.qpos[:3]-GOAL) < 0.12: break
    return traj
print("running ORACLE MPPI fly-to-goal..."); tr = mppi_oracle()
print(f"   oracle: final pos={np.round(tr[-1][0],3)}  dist={np.linalg.norm(tr[-1][0]-GOAL):.3f}  steps={len(tr)}  "
      f"{'FLEW' if np.linalg.norm(tr[-1][0]-GOAL)<0.2 else 'CRASH/FAIL'}")

P = np.array([p for p, _ in tr])
fig = plt.figure(figsize=(6.5, 5.6)); ax = fig.add_subplot(111, projection="3d")
ax.plot(P[:, 0], P[:, 1], P[:, 2], "C2", lw=2); ax.scatter(*GOAL, c="k", marker="*", s=160)
ax.scatter([0], [0], [1], c="0.5", s=40); ax.set_title(f"Oracle (true-dynamics) MPPI fly-to-goal\nfinal dist {np.linalg.norm(tr[-1][0]-GOAL):.2f}")
ax.set_xlabel("x"); ax.set_ylabel("y"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "quad_oracle.png"), dpi=110)
print(f"outputs -> {OUT}/quad_oracle.png")
