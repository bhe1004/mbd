"""
FAITHFUL unicycle parking with a DEEP Koopman lifting (PyTorch, CPU) — deep counterpart of
unicycle_gif.py, for consistency with pendulum_deep.py and faithfulness to MPPI-DK (deep Koopman).
True dynamics: x'=v cosθ, y'=v sinθ, θ'=ω  (RK4, dt=0.05). Input u=(v,ω).
Base lift b=[x, y, sinθ, cosθ-1]; deep encoder adds 'extra' features.
Three controllers (same MPPI cost/samples; only rollout MODEL differs):
  Ours      = deep-Koopman BILINEAR  (z+ = A z + B0 u + Σ_i u_i (B_i z))
  MPPI-DK   = deep-Koopman LINEAR    (z+ = A z + B0 u)
  MPPI-true = TRUE dynamics (oracle)
Both deep models share the SAME NN architecture/data/training (fair). Stop on goal convergence.
Outputs: out/unicycle_deep.gif , out/unicycle_deep.png  (+ multi-step prediction accuracy)
"""
import os, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
DT = 0.05

def dyn_np(s, u):
    def f(s): return np.array([u[0]*np.cos(s[2]), u[0]*np.sin(s[2]), u[1]])
    k1 = f(s); k2 = f(s+0.5*DT*k1); k3 = f(s+0.5*DT*k2); k4 = f(s+DT*k3)
    return s + (DT/6.0)*(k1+2*k2+2*k3+k4)
def dyn_t(S, U):                                  # S(...,3), U(...,2)
    def f(S):
        return torch.stack([U[..., 0]*torch.cos(S[..., 2]), U[..., 0]*torch.sin(S[..., 2]), U[..., 1]], -1)
    k1 = f(S); k2 = f(S+0.5*DT*k1); k3 = f(S+0.5*DT*k2); k4 = f(S+DT*k3)
    return S + (DT/6.0)*(k1+2*k2+2*k3+k4)
def base_t(S): return torch.stack([S[..., 0], S[..., 1], torch.sin(S[..., 2]), torch.cos(S[..., 2])-1.0], -1)

class DK(nn.Module):
    def __init__(self, extra=4, hid=64, bilinear=False, m=2):
        super().__init__(); self.N = 4+extra; self.bilinear = bilinear; self.m = m
        self.enc = nn.Sequential(nn.Linear(4, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N) + 0.01*torch.randn(self.N, self.N))
        self.B0 = nn.Parameter(0.01*torch.randn(self.N, m))
        if bilinear: self.B1 = nn.Parameter(0.01*torch.randn(m, self.N, self.N))
    def lift(self, b):                            # b(...,4)=(x,y,sinθ,cosθ-1)
        inp = torch.stack([b[..., 0], b[..., 1], b[..., 2], b[..., 3]+1.0], -1)
        return torch.cat([b, self.enc(inp)], -1)
    def fstep(self, z, u):                        # u(...,2)
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            for i in range(self.m): zn = zn + u[..., i:i+1]*(z @ self.B1[i].T)
        return zn
    def decode(self, z): return z[..., :4]

def make_snips(Nsnip, K, seed):
    r = np.random.default_rng(seed)
    S = np.zeros((Nsnip, K+1, 3))
    S[:, 0, 0] = r.uniform(-2, 2, Nsnip); S[:, 0, 1] = r.uniform(-2, 2, Nsnip); S[:, 0, 2] = r.uniform(-np.pi, np.pi, Nsnip)
    U = np.stack([r.uniform(-1.5, 1.5, (Nsnip, K)), r.uniform(-1.5, 1.5, (Nsnip, K))], -1)
    for k in range(K):
        for i in range(Nsnip): S[i, k+1] = dyn_np(S[i, k], U[i, k])
    B = np.stack([S[..., 0], S[..., 1], np.sin(S[..., 2]), np.cos(S[..., 2])-1.0], -1)
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)

K = 15
print("generating snippets..."); Btr, Utr = make_snips(8000, K, 1); Bte, Ute = make_snips(2000, K, 2)

def train(bilinear, epochs=200, bs=512, lr=1e-3):
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), lr=lr); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]
            z = m.lift(b[:, 0, :]); loss = 0.0
            for k in range(K):
                z = m.fstep(z, u[:, k, :])
                loss = loss + ((m.decode(z)-b[:, k+1, :])**2).mean() + 0.1*((z - m.lift(b[:, k+1, :]).detach())**2).mean()
            loss = loss/K; opt.zero_grad(); loss.backward(); opt.step()
    return m

print("training deep Koopman (linear & bilinear)..."); m_lin = train(False); m_bi = train(True)

@torch.no_grad()
def pred_err(m):
    z = m.lift(Bte[:, 0, :]); errs = []
    for k in range(K):
        z = m.fstep(z, Ute[:, k, :]); errs.append(((m.decode(z)-Bte[:, k+1, :])**2).sum(-1).sqrt().mean().item())
    return errs
el, eb = pred_err(m_lin), pred_err(m_bi)
print("[multi-step open-loop pred err in (x,y,sinθ,cosθ-1), TEST]:")
for h in [1, 5, 10, 15]:
    print(f"   H={h:2d}  linear={el[h-1]:.4f}  bilinear={eb[h-1]:.4f}  ratio(lin/bi)={el[h-1]/max(eb[h-1],1e-9):.2f}x")

STEPS = 170
@torch.no_grad()
def mppi(mode, model, s0, goal, T=40, Ks=1024, lam=0.5, sig=(0.6, 0.9), w_du=0.5, seed=3):
    g = torch.Generator().manual_seed(seed); s = torch.tensor(s0); U = torch.zeros(T, 2)
    traj = [s.numpy().copy()]; u_prev = torch.zeros(2); gT = torch.tensor(goal); stop = STEPS
    sig = torch.tensor(sig)
    for t in range(STEPS):
        eps = torch.randn(Ks, T, 2, generator=g)*sig; V = U[None] + eps; cost = torch.zeros(Ks)
        if mode == "true": S = s.repeat(Ks, 1)
        else: z = model.lift(base_t(s)).repeat(Ks, 1)
        for k in range(T):
            uk = V[:, k, :]
            if mode == "true": S = dyn_t(S, uk); b = base_t(S)
            else: z = model.fstep(z, uk); b = model.decode(z)
            cost += (b[:, 0]-gT[0])**2 + (b[:, 1]-gT[1])**2 + 0.3*(b[:, 2]**2 + b[:, 3]**2) + 0.05*(uk**2).sum(1)
            du = uk - (u_prev if k == 0 else V[:, k-1, :]); cost += w_du*(du**2).sum(1)
        cost += 30.0*((b[:, 0]-gT[0])**2 + (b[:, 1]-gT[1])**2) + 8.0*(b[:, 2]**2 + b[:, 3]**2)
        w = torch.softmax(-(cost-cost.min())/lam, 0)
        U = U + (w[:, None, None]*eps).sum(0)
        u0 = U[0].clamp(torch.tensor([-1.5, -1.5]), torch.tensor([1.5, 1.5]))
        s = dyn_t(s, u0); u_prev = u0; traj.append(s.numpy().copy())
        U = torch.roll(U, -1, 0); U[-1] = U[-2]
        if np.linalg.norm(s[:2].numpy()-goal[:2]) < 0.08 and abs(((s[2].item()-goal[2]+np.pi) % (2*np.pi))-np.pi) < 0.2:
            stop = len(traj)-1; break
    return np.array(traj), stop

s0 = [1.5, 1.5, -np.pi/2]; goal = [0.0, 0.0, 0.0]
print("running MPPI parking (3 controllers)...")
runs = [("Ours: deep-bilinear Koopman MPPI", lambda: mppi("bilinear", m_bi, s0, goal), "C0"),
        ("MPPI-true: oracle (true dynamics)", lambda: mppi("true", None, s0, goal), "C2"),
        ("MPPI-DK: deep-linear Koopman MPPI", lambda: mppi("linear", m_lin, s0, goal), "C3")]
results = []
for title, fn, col in runs:
    tr, stop = fn(); d = np.linalg.norm(tr[-1, :2])
    conv = f"parked@step {stop}" if stop < STEPS else "did NOT park"
    results.append((title, tr, col, stop)); print(f"   {title.split(':')[0]:9s}: final ||pos||={d:.3f}  {conv}")

lim = 2.6
def robot(ax, s, c):
    ax.arrow(s[0], s[1], 0.35*np.cos(s[2]), 0.35*np.sin(s[2]), head_width=0.16, head_length=0.16,
             fc=c, ec=c, lw=2, length_includes_head=True, zorder=5)
def setup(ax, title):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(title, fontsize=10.5)
    ax.scatter([goal[0]], [goal[1]], c="k", marker="*", s=240, zorder=4)
    ax.arrow(goal[0], goal[1], 0.3, 0.0, head_width=0.12, head_length=0.12, fc="k", ec="k", zorder=4)
    ax.scatter([s0[0]], [s0[1]], c="0.5", marker="o", s=70, zorder=4)
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.7)); nframes = max(len(r[1]) for r in results)
def update(i):
    for ax, (title, tr, col, stop) in zip(axes, results):
        ax.clear(); setup(ax, title); j = min(i, len(tr)-1)
        ax.plot(tr[:j+1, 0], tr[:j+1, 1], "-", c=col, lw=2, alpha=0.9); robot(ax, tr[j], col)
        tag = "  PARKED" if (stop < STEPS and j >= stop) else ""
        ax.text(-lim+0.1, lim-0.25, f"step {j:3d}  dist {np.linalg.norm(tr[j,:2]-np.array(goal[:2])):.2f}{tag}", fontsize=9, family="monospace")
    return []
FuncAnimation(fig, update, frames=nframes, interval=70, blit=False).save(os.path.join(OUT, "unicycle_deep.gif"), writer=PillowWriter(fps=15))
update(nframes-1); plt.tight_layout(); plt.savefig(os.path.join(OUT, "unicycle_deep.png"), dpi=110)
print(f"outputs -> {OUT}/unicycle_deep.gif , unicycle_deep.png")
