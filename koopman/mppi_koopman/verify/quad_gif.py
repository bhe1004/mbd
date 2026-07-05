"""
3D MuJoCo quadrotor fly-to-goal, 3 controllers (FIXED torque scale: τ matched to inertia I~0.005):
  Ours      = deep-BILINEAR Koopman MPPI
  MPPI-true = oracle (batched analytic true dynamics, verified == MuJoCo)
  MPPI-DK   = deep-LINEAR Koopman MPPI
All EXECUTE on MuJoCo; ours/DK plan with the learned Koopman, oracle with true dynamics.
Output: out/quad_gif.gif , .png
"""
import os, numpy as np, torch, torch.nn as nn, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
XML = """
<mujoco><option timestep="0.01" gravity="0 0 -9.81"/>
<worldbody><body name="quad" pos="0 0 1"><freejoint/>
<geom type="box" size="0.12 0.12 0.03" mass="1"/></body></worldbody></mujoco>
"""
model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model); bid = model.body("quad").id
MG = 9.81; MASS = float(model.body_mass[bid]); INER = torch.tensor(model.body_inertia[bid], dtype=torch.float32)
G, DT, NSUB = 9.81, 0.01, 2; U_HOVER = torch.tensor([MG, 0, 0, 0.])
TORQ = 0.05   # control torque scale (matched to inertia)

def mj_ctrl(u, nsub=NSUB):
    for _ in range(nsub):
        R = data.xmat[bid].reshape(3, 3)
        data.xfrc_applied[bid, :3] = R @ np.array([0.0, 0.0, u[0]])
        data.xfrc_applied[bid, 3:] = R @ np.array([u[1], u[2], u[3]]); mujoco.mj_step(model, data)
def reset(pos=(0, 0, 1)):
    mujoco.mj_resetData(model, data); data.qpos[:3] = pos; data.qpos[3] = 1.0; mujoco.mj_forward(model, data)
def feat():
    R = data.xmat[bid].reshape(3, 3); return np.concatenate([data.qpos[:3], data.qvel[:3], R.flatten(), data.qvel[3:6]])
def reset_rand(r):
    mujoco.mj_resetData(model, data); data.qpos[:3] = r.uniform([-2, -2, 0.5], [2, 2, 2.5])
    ax = r.normal(size=3); ax /= np.linalg.norm(ax)+1e-9; ang = r.uniform(-0.6, 0.6)
    data.qpos[3:7] = [np.cos(ang/2), *(np.sin(ang/2)*ax)]
    data.qvel[:3] = r.uniform(-1.2, 1.2, 3); data.qvel[3:6] = r.uniform(-1.2, 1.2, 3); mujoco.mj_forward(model, data)

# ----- analytic batched true dynamics (oracle) -----
def skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1), torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)
def exp_so3(w):
    ang = w.norm(dim=-1, keepdim=True); K = skew(w/(ang+1e-9)); a = ang.unsqueeze(-1)
    return torch.eye(3).expand(K.shape) + torch.sin(a)*K + (1-torch.cos(a))*(K @ K)
def step_true(P, V, R, W, u):
    for _ in range(NSUB):
        acc = R[..., :, 2]*u[..., 0:1]/MASS + torch.tensor([0., 0., -G])
        wdot = (u[..., 1:4] - torch.cross(W, INER*W, dim=-1))/INER
        W = W + wdot*DT; V = V + acc*DT; R = R @ exp_so3(W*DT); P = P + V*DT
    return P, V, R, W

# ----- Koopman -----
K = 20; ND = 18
def gen(N, seed):
    # COHERENT control (per-snippet bias + noise) -> sustained maneuvers, matches MPPI deployment distribution
    r = np.random.default_rng(seed); B = np.zeros((N, K+1, 18)); U = np.zeros((N, K, 4))
    for i in range(N):
        reset_rand(r); B[i, 0] = feat()
        tau_c = r.uniform(-TORQ*0.6, TORQ*0.6, 3); T_c = MG + r.uniform(-2, 2)   # coherent bias
        for k in range(K):
            u = np.array([T_c + r.uniform(-1.2, 1.2), *np.clip(tau_c + r.uniform(-TORQ*0.4, TORQ*0.4, 3), -TORQ, TORQ)])
            U[i, k] = u; mj_ctrl(u); B[i, k+1] = feat()
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
print("collecting data (fixed torque scale)..."); Btr, Utr = gen(6500, 1)
class DK(nn.Module):
    def __init__(self, extra=6, hid=64, bilinear=False, m=4):
        super().__init__(); self.N = ND+extra; self.bilinear = bilinear; self.m = m
        self.enc = nn.Sequential(nn.Linear(ND, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N)+0.01*torch.randn(self.N, self.N)); self.B0 = nn.Parameter(0.01*torch.randn(self.N, m))
        if bilinear: self.B1 = nn.Parameter(torch.zeros(m, self.N, self.N))   # init 0 -> start as LINEAR (bilinear ⊇ linear)
    def lift(self, b): return torch.cat([b, self.enc(b)], -1)
    def fstep(self, z, u):
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            for i in range(self.m): zn = zn + u[..., i:i+1]*(z @ self.B1[i].T)
        return zn
    def decode(self, z): return z[..., :ND]
def train(bilinear, epochs=320, bs=512):
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0]); loss = 0.0
            for k in range(K):
                z = m.fstep(z, u[:, k]); loss = loss + ((m.decode(z)-b[:, k+1])**2).mean() + 0.1*((z-m.lift(b[:, k+1]).detach())**2).mean()
            opt.zero_grad(); (loss/K).backward(); opt.step()
    return m
print("training ours(bilinear) & MPPI-DK(linear)..."); m_bi = train(True); m_lin = train(False)

GOAL = np.array([1.0, 1.0, 1.6]); gt = torch.tensor(GOAL, dtype=torch.float32)
@torch.no_grad()
def mppi(mode, model, T=20, Ks=1500, lam=0.5, steps=170, seed=3):
    g = torch.Generator().manual_seed(seed); reset((0, 0, 1)); U = torch.zeros(T, 4); traj = []
    sig = torch.tensor([1.0, 0.04, 0.04, 0.04])
    for t in range(steps):
        eps = torch.randn(Ks, T, 4, generator=g)*sig; Vc = U[None] + eps; cost = torch.zeros(Ks)
        if mode == "oracle":
            P = torch.tensor(data.qpos[:3], dtype=torch.float32).repeat(Ks, 1); V = torch.tensor(data.qvel[:3], dtype=torch.float32).repeat(Ks, 1)
            R = torch.tensor(data.xmat[bid].reshape(3, 3), dtype=torch.float32).repeat(Ks, 1, 1); W = torch.tensor(data.qvel[3:6], dtype=torch.float32).repeat(Ks, 1)
        else:
            z = model.lift(torch.tensor(feat(), dtype=torch.float32)).repeat(Ks, 1)
        for k in range(T):
            uk = U_HOVER + Vc[:, k]; uk = torch.cat([uk[:, :1].clamp(0, 20), uk[:, 1:]], 1)
            if mode == "oracle":
                P, V, R, W = step_true(P, V, R, W, uk); pos, vel, up, om = P, V, R[:, 0, 2]**2+R[:, 1, 2]**2+(R[:, 2, 2]-1)**2, W
            else:
                z = model.fstep(z, uk); b = model.decode(z); pos, vel, up, om = b[:, 0:3], b[:, 3:6], b[:, 8]**2+b[:, 11]**2+(b[:, 14]-1)**2, b[:, 15:18]
            cost += ((pos-gt)**2).sum(1) + 3.0*up + 0.2*(vel**2).sum(1) + 0.08*(om**2).sum(1) + 0.01*(Vc[:, k]**2).sum(1)
        cost += 8.0*((pos-gt)**2).sum(1) + 8.0*up
        w = torch.softmax(-(cost-cost.min())/lam, 0); U = U + (w[:, None, None]*eps).sum(0)
        u0 = (U_HOVER + U[0]).numpy(); u0[0] = np.clip(u0[0], 0, 20); mj_ctrl(u0)
        Rm = data.xmat[bid].reshape(3, 3); traj.append((data.qpos[:3].copy(), Rm[:, 2].copy()))
        U = torch.roll(U, -1, 0); U[-1] = U[-2]
        if np.linalg.norm(data.qpos[:3]-GOAL) < 0.12: break
    return traj
print("running fly-to-goal (3 controllers)...")
runs = [("Ours: bilinear Koopman MPPI", lambda: mppi("bilinear", m_bi), "C0"),
        ("MPPI-true: oracle", lambda: mppi("oracle", None), "C2"),
        ("MPPI-DK: linear Koopman MPPI", lambda: mppi("linear", m_lin), "C3")]
results = []
for title, fn, col in runs:
    tr = fn(); d = np.linalg.norm(tr[-1][0]-GOAL)
    results.append((title, tr, col)); print(f"   {title.split(':')[0]:9s}: dist={d:.3f} steps={len(tr)} {'FLEW' if d<0.2 else 'FAIL'}")

def setup(ax, title):
    ax.set_xlim(-0.3, 1.6); ax.set_ylim(-0.3, 1.6); ax.set_zlim(0.6, 1.9); ax.set_title(title, fontsize=10.5)
    ax.scatter(*GOAL, c="k", marker="*", s=150); ax.scatter([0], [0], [1], c="0.5", s=40); ax.set_xlabel("x"); ax.set_ylabel("y")
fig = plt.figure(figsize=(16, 5.6)); axes = [fig.add_subplot(1, 3, i+1, projection="3d") for i in range(3)]
nframes = max(len(r[1]) for r in results)
def update(i):
    for ax, (title, tr, col) in zip(axes, results):
        ax.clear(); setup(ax, title); j = min(i, len(tr)-1); P = np.array([p for p, _ in tr[:j+1]])
        ax.plot(P[:, 0], P[:, 1], P[:, 2], c=col, lw=2); p, zb = tr[j]
        ax.quiver(p[0], p[1], p[2], 0.3*zb[0], 0.3*zb[1], 0.3*zb[2], color=col, lw=2)
        ax.scatter([p[0]], [p[1]], [p[2]], c=col, s=70)
        ax.text2D(0.02, 0.95, f"step {j} dist {np.linalg.norm(p-GOAL):.2f}", transform=ax.transAxes, fontsize=9, family="monospace")
    return []
FuncAnimation(fig, update, frames=nframes, interval=60, blit=False).save(os.path.join(OUT, "quad_gif.gif"), writer=PillowWriter(fps=18))
update(nframes-1); plt.tight_layout(); plt.savefig(os.path.join(OUT, "quad_gif.png"), dpi=110)
print(f"outputs -> {OUT}/quad_gif.gif , .png")
