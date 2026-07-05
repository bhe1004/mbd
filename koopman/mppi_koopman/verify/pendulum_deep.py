"""
FAITHFUL pendulum swing-up & balance vs MPPI-DK, with a DEEP Koopman lifting (PyTorch, CPU).
Dynamics = MPPI-DK eq.(13): semi-implicit Euler, 3/2 rod inertia, dt=0.05, torque |u|<=2, undamped.
  theta_dot+ = theta_dot + (15 sin theta + 3 u) dt ;  theta+ = theta + theta_dot+ dt
  theta=0 = UPRIGHT (unstable). Start hanging down (theta=pi). Must PUMP (3*2=6 < 15 gravity).

Three controllers (identical MPPI cost/samples; only the rollout MODEL differs):
  Ours      = MPPI with deep-Koopman BILINEAR  surrogate  (z+ = A z + B0 u + u (B1 z))
  MPPI-DK   = MPPI with deep-Koopman LINEAR    surrogate  (z+ = A z + B0 u)   [their method]
  MPPI-true = MPPI with the TRUE dynamics (oracle)
Both deep models share the SAME NN lifting architecture (lifted dim 8) for a fair comparison.
Outputs: out/pendulum_deep.gif , out/pendulum_deep.png  (+ printed multi-step prediction accuracy)
"""
import os, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)

DT, UMAX = 0.05, 2.0
def dyn_np(s, u):
    dth = s[1] + (15*np.sin(s[0]) + 3*u)*DT; th = s[0] + dth*DT
    return np.array([th, dth])
def dyn_t(S, U):                                  # S(...,2), U(...)
    dth = S[..., 1] + (15*torch.sin(S[..., 0]) + 3*U)*DT
    th = S[..., 0] + dth*DT
    return torch.stack([th, dth], -1)
def base_t(S): return torch.stack([torch.sin(S[..., 0]), torch.cos(S[..., 0])-1.0, S[..., 1]], -1)

# ----------------------------------------------------------------- deep Koopman
class DK(nn.Module):
    def __init__(self, extra=5, hid=64, bilinear=False):
        super().__init__(); self.N = 3+extra; self.bilinear = bilinear
        self.enc = nn.Sequential(nn.Linear(3, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N) + 0.01*torch.randn(self.N, self.N))
        self.B0 = nn.Parameter(0.01*torch.randn(self.N))
        if bilinear: self.B1 = nn.Parameter(0.01*torch.randn(self.N, self.N))
    def lift(self, b):                            # b(...,3)=(sinθ,cosθ-1,θ̇)
        inp = torch.stack([b[..., 0], b[..., 1]+1.0, b[..., 2]], -1)   # (sinθ,cosθ,θ̇)
        return torch.cat([b, self.enc(inp)], -1)
    def fstep(self, z, u):
        zn = z @ self.A.T + u.unsqueeze(-1)*self.B0
        if self.bilinear: zn = zn + u.unsqueeze(-1)*(z @ self.B1.T)
        return zn
    def decode(self, z): return z[..., :3]

# ----------------------------------------------------------------- training data (snippets)
def make_snips(Nsnip, K, seed):
    r = np.random.default_rng(seed)
    S = np.zeros((Nsnip, K+1, 2))
    S[:, 0, 0] = r.uniform(-np.pi, np.pi, Nsnip); S[:, 0, 1] = r.uniform(-8, 8, Nsnip)
    U = r.uniform(-UMAX, UMAX, (Nsnip, K))
    for k in range(K):
        dth = S[:, k, 1] + (15*np.sin(S[:, k, 0]) + 3*U[:, k])*DT
        S[:, k+1, 0] = S[:, k, 0] + dth*DT; S[:, k+1, 1] = dth
    B = np.stack([np.sin(S[..., 0]), np.cos(S[..., 0])-1.0, S[..., 1]], -1)
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)

K = 15
Btr, Utr = make_snips(20000, K, 1)
Bte, Ute = make_snips(4000, K, 2)

def train(bilinear, epochs=220, bs=512, lr=1e-3):
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]
            z = m.lift(b[:, 0, :]); loss = 0.0
            for k in range(K):
                z = m.fstep(z, u[:, k])
                loss = loss + ((m.decode(z) - b[:, k+1, :])**2).mean()
                loss = loss + 0.1*((z - m.lift(b[:, k+1, :]).detach())**2).mean()
            loss = loss / K
            opt.zero_grad(); loss.backward(); opt.step()
    return m

print("training deep Koopman (linear & bilinear)...")
m_lin = train(bilinear=False)
m_bi = train(bilinear=True)

# ----------------------------------------------------------------- multi-step prediction audit
@torch.no_grad()
def pred_err(m):
    z = m.lift(Bte[:, 0, :]); errs = []
    for k in range(K):
        z = m.fstep(z, Ute[:, k]); errs.append(((m.decode(z)-Bte[:, k+1, :])**2).sum(-1).sqrt().mean().item())
    return errs
el, eb = pred_err(m_lin), pred_err(m_bi)
print("[multi-step open-loop pred err in (sinθ,cosθ-1,θ̇), TEST set]:")
for h in [1, 5, 10, 15]:
    print(f"   H={h:2d}  linear={el[h-1]:.4f}  bilinear={eb[h-1]:.4f}  ratio(lin/bi)={el[h-1]/max(eb[h-1],1e-9):.2f}x")

# ----------------------------------------------------------------- MPPI (torch)
STEPS = 140
@torch.no_grad()
def mppi(mode, model, T=50, Ks=1024, lam=1.0, sig=1.6, seed=3):
    g = torch.Generator().manual_seed(seed)
    s = torch.tensor([np.pi, 0.0]); U = torch.zeros(T); traj = [s.numpy().copy()]; stop = STEPS
    for t in range(STEPS):
        eps = torch.randn(Ks, T, generator=g)*sig; V = U[None] + eps
        cost = torch.zeros(Ks)
        if mode == "true":
            S = s.repeat(Ks, 1)
        else:
            z = model.lift(base_t(s)).repeat(Ks, 1)
        for k in range(T):
            uk = V[:, k].clamp(-UMAX, UMAX)
            if mode == "true":
                S = dyn_t(S, uk); b = base_t(S)
            else:
                z = model.fstep(z, uk); b = model.decode(z)
            cost += (b[:, 0]**2 + b[:, 1]**2) + 0.04*b[:, 2]**2 + 0.002*uk**2    # 2(1-cosθ)+...
        cost += 20.0*(b[:, 0]**2 + b[:, 1]**2) + 1.0*b[:, 2]**2                  # terminal balance
        w = torch.softmax(-(cost - cost.min())/lam, 0)
        U = U + (w[:, None]*eps).sum(0)
        u0 = float(U[0].clamp(-UMAX, UMAX)); s = dyn_t(s, torch.tensor(u0))
        traj.append(s.numpy().copy())
        U = torch.roll(U, -1); U[-1] = U[-2]
        if (1 - np.cos(s[0].item())) < 0.05 and abs(s[1].item()) < 0.6:
            stop = len(traj)-1; break
    return np.array(traj), stop

print("running MPPI swing-up (3 controllers)...")
runs = [("Ours: deep-bilinear Koopman MPPI", lambda: mppi("bilinear", m_bi), "C0"),
        ("MPPI-true: oracle (true dynamics)", lambda: mppi("true", None), "C2"),
        ("MPPI-DK: deep-linear Koopman MPPI", lambda: mppi("linear", m_lin), "C3")]
results = []
for title, fn, col in runs:
    tr, stop = fn(); up = 1 - np.cos(tr[-1, 0])
    conv = f"up@step {stop} ({stop*DT:.1f}s)" if stop < STEPS else "did NOT swing up"
    results.append((title, tr, col, stop)); print(f"   {title.split(':')[0]:9s}: final 1-cos={up:.3f} {conv}")

# ----------------------------------------------------------------- animation
def setup(ax, title):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10.5); ax.scatter([0], [1], c="k", marker="*", s=160, zorder=3)
    ax.scatter([0], [0], c="0.4", s=40, zorder=4)
def draw(ax, tr, j, col):
    th = tr[j, 0]; bx, by = np.sin(th), np.cos(th); tail = max(0, j-50)
    ax.plot(np.sin(tr[tail:j+1, 0]), np.cos(tr[tail:j+1, 0]), "-", c=col, lw=1, alpha=0.35)
    ax.plot([0, bx], [0, by], "-", c=col, lw=3, zorder=5); ax.scatter([bx], [by], c=col, s=130, zorder=6)
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8)); nframes = max(len(r[1]) for r in results)
def update(i):
    for ax, (title, tr, col, stop) in zip(axes, results):
        ax.clear(); setup(ax, title); j = min(i, len(tr)-1); draw(ax, tr, j, col)
        tag = "  UP" if (stop < STEPS and j >= stop) else ""
        ax.text(-1.32, 1.2, f"t={j*DT:4.1f}s  1-cos={1-np.cos(tr[j,0]):.2f}{tag}", fontsize=9, family="monospace")
    return []
FuncAnimation(fig, update, frames=nframes, interval=50, blit=False).save(
    os.path.join(OUT, "pendulum_deep.gif"), writer=PillowWriter(fps=20))
update(nframes-1); plt.tight_layout(); plt.savefig(os.path.join(OUT, "pendulum_deep.png"), dpi=110)
print(f"outputs -> {OUT}/pendulum_deep.gif , pendulum_deep.png")
