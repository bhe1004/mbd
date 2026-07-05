"""Honesty check: does the certified error tube hold for the DEEP bilinear Koopman model
actually used in the control experiments (not the manual EDMD model)?
Trains the deep bilinear unicycle model (same architecture as unicycle_deep.py), fits the
proportional bound c_x,c_u on its one-step residuals (linprog), then propagates the Euclidean
tube on held-out rollouts. Reports validity (violations) and tightness (||A||, e growth)."""
import os, numpy as np, torch, torch.nn as nn
from scipy.optimize import linprog
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
FIGS = "/figs"; DT = 0.05

def dyn_np(s, u):
    def f(s): return np.array([u[0]*np.cos(s[2]), u[0]*np.sin(s[2]), u[1]])
    k1 = f(s); k2 = f(s+0.5*DT*k1); k3 = f(s+0.5*DT*k2); k4 = f(s+DT*k3)
    return s + (DT/6.0)*(k1+2*k2+2*k3+k4)

class DK(nn.Module):
    def __init__(self, extra=4, hid=64, m=2):
        super().__init__(); self.N = 4+extra; self.m = m
        self.enc = nn.Sequential(nn.Linear(4, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N) + 0.01*torch.randn(self.N, self.N))
        self.B0 = nn.Parameter(0.01*torch.randn(self.N, m))
        self.B1 = nn.Parameter(0.01*torch.randn(m, self.N, self.N))
    def lift(self, b):
        inp = torch.stack([b[..., 0], b[..., 1], b[..., 2], b[..., 3]+1.0], -1)
        return torch.cat([b, self.enc(inp)], -1)
    def fstep(self, z, u):
        zn = z @ self.A.T + u @ self.B0.T
        for i in range(self.m): zn = zn + u[..., i:i+1]*(z @ self.B1[i].T)
        return zn
    def decode(self, z): return z[..., :4]

def make_snips(Nsnip, K, seed):
    r = np.random.default_rng(seed); S = np.zeros((Nsnip, K+1, 3))
    S[:, 0, 0] = r.uniform(-2, 2, Nsnip); S[:, 0, 1] = r.uniform(-2, 2, Nsnip); S[:, 0, 2] = r.uniform(-np.pi, np.pi, Nsnip)
    U = np.stack([r.uniform(-1.5, 1.5, (Nsnip, K)), r.uniform(-1.5, 1.5, (Nsnip, K))], -1)
    for k in range(K):
        for i in range(Nsnip): S[i, k+1] = dyn_np(S[i, k], U[i, k])
    B = np.stack([S[..., 0], S[..., 1], np.sin(S[..., 2]), np.cos(S[..., 2])-1.0], -1)
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)

K = 15
print("data..."); Btr, Utr = make_snips(8000, K, 1)
def train(epochs=200, bs=512, lr=1e-3):
    m = DK(); opt = torch.optim.Adam(m.parameters(), lr=lr); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0, :]); loss = 0.
            for k in range(K):
                z = m.fstep(z, u[:, k, :])
                loss = loss + ((m.decode(z)-b[:, k+1, :])**2).mean() + 0.1*((z - m.lift(b[:, k+1, :]).detach())**2).mean()
            (loss/K).backward(); opt.step(); opt.zero_grad()
    return m
print("training deep bilinear (200 epochs)..."); m = train()

A = m.A.detach().numpy(); B0 = m.B0.detach().numpy(); Bs = [m.B1[i].detach().numpy() for i in range(m.m)]
N = m.N; nA = np.linalg.norm(A, 2); nBs = [np.linalg.norm(b, 2) for b in Bs]
print(f"deep bilinear: N={N}  ||A||2={nA:.3f}  specrad(A)={max(abs(np.linalg.eigvals(A))):.3f}  ||B_i||={[round(x,3) for x in nBs]}")

# --- fit proportional bound on TRAIN one-step residuals ---
@torch.no_grad()
def resid(B, U):
    Z = m.lift(B); Zx = Z[:, :-1, :]; Zy = Z[:, 1:, :]; Zp = m.fstep(Zx, U)
    R = (Zy - Zp).reshape(-1, N).numpy(); Zxf = Zx.reshape(-1, N).numpy(); Uf = U.reshape(-1, 2).numpy()
    return R, Zxf, Uf
R, Zxf, Uf = resid(Btr, Utr)
rn = np.linalg.norm(R, axis=1); zn = np.linalg.norm(Zxf, axis=1); un = np.linalg.norm(Uf, axis=1)
sol = linprog([1, 1], A_ub=-np.vstack([zn, un]).T, b_ub=-rn, bounds=[(0, None), (0, None)], method="highs")
cx, cu = float(sol.x[0]), float(sol.x[1])
print(f"fitted bound (deep): c_x={cx:.4e}  c_u={cu:.4e}")

# --- propagate Euclidean tube on held-out TEST rollouts (horizon 40) ---
Lt = 40; Bte, Ute = make_snips(2000, Lt, 2)
@torch.no_grad()
def rollout(B, U):
    Ztrue = m.lift(B).numpy()
    zt = m.lift(B[:, 0, :]); Zhat = [zt.numpy()]
    for k in range(Lt):
        zt = m.fstep(zt, U[:, k, :]); Zhat.append(zt.numpy())
    return Ztrue, np.stack(Zhat, 1)
Ztrue, Zhat = rollout(Bte, Ute)
Un = Ute.numpy(); Ns = Bte.shape[0]
mbar = nA + np.abs(Un[:, :, 0])*nBs[0] + np.abs(Un[:, :, 1])*nBs[1]          # (Ns,Lt)
znhat = np.linalg.norm(Zhat, axis=2); unrm = np.linalg.norm(Un, axis=2)        # (Ns,Lt+1),(Ns,Lt)
E = np.zeros((Ns, Lt+1))
for k in range(Lt):
    E[:, k+1] = (mbar[:, k]+cx)*E[:, k] + cx*znhat[:, k] + cu*unrm[:, k]
dl = np.linalg.norm(Ztrue - Zhat, axis=2)
viol = int(np.sum(dl > E + 1e-6)); tot = dl.size
print(f"tube on TEST (H={Lt}): violations={viol}/{tot} ({100*viol/tot:.3f}%)")
print(f"  median true error e_H={np.median(dl[:, -1]):.3e}   median tube e_H={np.median(E[:, -1]):.3e}   ratio={np.median(E[:, -1])/max(np.median(dl[:, -1]),1e-12):.1f}x")

# representative trajectory (median final true error)
idx = int(np.argsort(dl[:, -1])[Ns//2])
fig, ax = plt.subplots(figsize=(4.7, 3.2))
ax.plot(dl[idx], lw=2.2, color="#1f77b4", label=r"true lifted error $\|z_k-\hat z_k\|$")
ax.plot(E[idx], "--", lw=2.2, color="#d62728", label=r"certified tube $e_k$")
ax.set_yscale("log"); ax.set_xlabel(r"rollout step $k$"); ax.set_ylabel("lifted-space error")
ax.grid(True, alpha=0.35); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_tube_deep.png"), dpi=200)
print("saved /figs/fig_tube_deep.png")
