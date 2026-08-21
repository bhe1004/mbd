"""3D velocity-drone MPPI tracking a figure-8 while yawing along the path. Ours(bilinear) tracks;
MPPI-DK(linear) cannot rotate body velocity into the world frame -> drifts off. Renders a 3D GIF."""
import os, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
from matplotlib.animation import FuncAnimation, PillowWriter
torch.manual_seed(0); np.random.seed(0)
DT = 0.05; M = 4; ND = 5
def stept(S, U):
    x, y, z, ya = S[..., 0], S[..., 1], S[..., 2], S[..., 3]; vx, vy, vz, w = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
    c, sn = torch.cos(ya), torch.sin(ya)
    return torch.stack([x+(c*vx-sn*vy)*DT, y+(sn*vx+c*vy)*DT, z+vz*DT, ya+w*DT], -1)
def obst(S): return torch.stack([S[..., 0], S[..., 1], S[..., 2], torch.sin(S[..., 3]), torch.cos(S[..., 3])], -1)
def stepn(s, u):
    x, y, z, ya = s; vx, vy, vz, w = u; c, sn = np.cos(ya), np.sin(ya)
    return np.array([x+(c*vx-sn*vy)*DT, y+(sn*vx+c*vy)*DT, z+vz*DT, ya+w*DT])
def obsn(s): return np.array([s[0], s[1], s[2], np.sin(s[3]), np.cos(s[3])], np.float32)
# ---- data + train ----
K = 15
def gen(N, seed):
    r = np.random.default_rng(seed); B = np.zeros((N, K+1, ND), np.float32); U = np.zeros((N, K, M), np.float32)
    for i in range(N):
        s = np.array([r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-np.pi, np.pi)]); B[i, 0] = obsn(s)
        bias = np.concatenate([r.uniform(-0.7, 0.7, 3), r.uniform(-1.2, 1.2, 1)])
        for k in range(K):
            u = bias + np.concatenate([r.uniform(-0.4, 0.4, 3), r.uniform(-0.6, 0.6, 1)]); U[i, k] = u
            s = stepn(s, u); B[i, k+1] = obsn(s)
    return torch.tensor(B), torch.tensor(U)
print("data+train..."); Btr, Utr = gen(4000, 1)
class DK(nn.Module):
    def __init__(self, extra=8, hid=64, bilinear=False):
        super().__init__(); self.N = ND+extra; self.bilinear = bilinear
        self.enc = nn.Sequential(nn.Linear(ND, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N)+.01*torch.randn(self.N, self.N)); self.B0 = nn.Parameter(.01*torch.randn(self.N, M))
        if bilinear: self.B1 = nn.Parameter(torch.zeros(M, self.N, self.N))
    def lift(self, b): return torch.cat([b, self.enc(b)], -1)
    def fstep(self, z, u):
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            for i in range(M): zn = zn + u[..., i:i+1]*(z @ self.B1[i].T)
        return zn
    def decode(self, z): return z[..., :ND]
def train(bilinear, epochs=200, bs=512):
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0]); loss = 0.
            for k in range(K):
                z = m.fstep(z, u[:, k]); loss = loss + ((m.decode(z)-b[:, k+1])**2).mean() + .1*((z-m.lift(b[:, k+1]).detach())**2).mean()
            opt.zero_grad(); (loss/K).backward(); opt.step()
    return m
m_bi = train(True); m_lin = train(False)
# ---- reference: figure-8 + path-tangent yaw ----
F = 0.11
def refp(step):
    t = step*DT; return np.array([1.1*np.sin(2*np.pi*F*t), 0.8*np.sin(4*np.pi*F*t), 0.6+0.35*np.sin(2*np.pi*F*t)])
def refyaw(step):
    d = refp(step+1)-refp(step); return np.arctan2(d[1], d[0])
STEPS = 175
REFP = np.array([refp(t) for t in range(STEPS+K+2)]); REFY = np.array([refyaw(t) for t in range(STEPS+K+2)])
RP = torch.tensor(REFP, dtype=torch.float32); RY = torch.tensor(REFY, dtype=torch.float32)
@torch.no_grad()
def mppi(mode, model, T=15, Ks=1024, lam=0.25, sig=(0.4, 0.4, 0.4, 0.7), gy=0.3):
    g = torch.Generator().manual_seed(7); s = torch.tensor(np.concatenate([refp(0), [refyaw(0)]]), dtype=torch.float32)
    U = torch.zeros(T, M); sig = torch.tensor(sig); traj = [s.numpy().copy()]
    for t in range(STEPS):
        eps = torch.randn(Ks, T, M, generator=g)*sig; V = U[None]+eps; cost = torch.zeros(Ks)
        if mode == "oracle": S = s.repeat(Ks, 1)
        else: z = model.lift(obst(s)).repeat(Ks, 1)
        for k in range(T):
            uk = V[:, k]
            if mode == "oracle": S = stept(S, uk); p = S[:, :3]; ya = S[:, 3]
            else: z = model.fstep(z, uk); d = model.decode(z); p = d[:, :3]; ya = torch.atan2(d[:, 3], d[:, 4])
            rp = RP[t+k+1]; cost = cost + ((p-rp)**2).sum(1) + gy*(torch.sin(ya)-np.sin(REFY[t+k+1]))**2 + 0.02*(uk**2).sum(1)
        w = torch.softmax(-(cost-cost.min())/lam, 0); U = U + (w[:, None, None]*eps).sum(0)
        u0 = U[0]; s = stept(s, u0); traj.append(s.numpy().copy())
        U = torch.roll(U, -1, 0); U[-1] = U[-2]
    return np.array(traj)
print("mppi tracking (ours/oracle/linear)...")
TR = {"ours": mppi("bi", m_bi), "oracle": mppi("oracle", None), "linear": mppi("lin", m_lin)}
for k, tr in TR.items():
    err = np.linalg.norm(tr[:, :3]-REFP[:len(tr)], axis=1).mean()
    print(f"  {k:7s}: mean tracking error = {err:.3f} m")
# ---- 3D GIF ----
def quad(ax, s, col, L=0.12):
    x, y, z, ya = s; c, sn = np.cos(ya), np.sin(ya)
    for dx, dy in [(L, 0), (-L, 0), (0, L), (0, -L)]:
        ex, ey = x+c*dx-sn*dy, y+sn*dx+c*dy; ax.plot([x, ex], [y, ey], [z, z], '-', c=col, lw=1.5)
        ax.scatter([ex], [ey], [z], c=col, s=10)
cols = {"ours": "#1f77b4", "oracle": "#2ca02c", "linear": "#d62728"}
fig = plt.figure(figsize=(6, 5.4)); ax = fig.add_subplot(111, projection='3d')
def frame(t):
    ax.clear()
    ax.plot(REFP[:STEPS, 0], REFP[:STEPS, 1], REFP[:STEPS, 2], '--', c='0.6', lw=1, label='reference')
    for name, tr in TR.items():
        j = min(t, len(tr)-1); ax.plot(tr[:j+1, 0], tr[:j+1, 1], tr[:j+1, 2], '-', c=cols[name], lw=2, label=name); quad(ax, tr[j], cols[name])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.2, 1.2); ax.set_zlim(0, 1.2)
    ax.set_title("3D velocity-drone: figure-8 tracking", fontsize=11); ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=22, azim=20+t*1.2); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    return []
FuncAnimation(fig, frame, frames=STEPS, interval=55).save(os.path.join(OUT, "drone3d.gif"), writer=PillowWriter(fps=20))
frame(STEPS//2); fig.savefig(os.path.join(OUT, "drone3d_frame.png"), dpi=120)
print("saved out/drone3d.gif + out/drone3d_frame.png")
