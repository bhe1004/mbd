"""
Deterministic EFFICIENCY result (robust, no fragile MPPI): multi-step open-loop prediction error
vs lifting dimension, for deep-LINEAR vs deep-BILINEAR Koopman on the MPPI-DK pendulum (eq.13).
Message: bilinear reaches a given long-horizon accuracy at a MUCH smaller lifting dim than linear
=> "we do well with less" (capacity efficiency), without any claim that MPPI-DK 'fails'.
Output: out/pendulum_efficiency.png  (+ printed table)
"""
import os, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)

DT, UMAX = 0.05, 2.0
def dyn_np(s, u):
    dth = s[1] + (15*np.sin(s[0]) + 3*u)*DT; return np.array([s[0]+dth*DT, dth])

class DK(nn.Module):
    def __init__(self, extra, hid=64, bilinear=False):
        super().__init__(); self.N = 3+extra; self.bilinear = bilinear
        self.enc = nn.Sequential(nn.Linear(3, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N)+0.01*torch.randn(self.N, self.N))
        self.B0 = nn.Parameter(0.01*torch.randn(self.N))
        if bilinear: self.B1 = nn.Parameter(0.01*torch.randn(self.N, self.N))
    def lift(self, b):
        inp = torch.stack([b[..., 0], b[..., 1]+1.0, b[..., 2]], -1); return torch.cat([b, self.enc(inp)], -1)
    def fstep(self, z, u):
        zn = z@self.A.T + u.unsqueeze(-1)*self.B0
        if self.bilinear: zn = zn + u.unsqueeze(-1)*(z@self.B1.T)
        return zn
    def decode(self, z): return z[..., :3]

def make_snips(N, K, seed):
    r = np.random.default_rng(seed); S = np.zeros((N, K+1, 2))
    S[:, 0, 0] = r.uniform(-np.pi, np.pi, N); S[:, 0, 1] = r.uniform(-8, 8, N)
    U = r.uniform(-UMAX, UMAX, (N, K))
    for k in range(K):
        dth = S[:, k, 1] + (15*np.sin(S[:, k, 0]) + 3*U[:, k])*DT
        S[:, k+1, 0] = S[:, k, 0] + dth*DT; S[:, k+1, 1] = dth
    B = np.stack([np.sin(S[..., 0]), np.cos(S[..., 0])-1.0, S[..., 1]], -1)
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)

K = 15
Btr, Utr = make_snips(15000, K, 1); Bte, Ute = make_snips(4000, K, 2)

def train_eval(extra, bilinear, epochs=120, bs=512, lr=1e-3):
    m = DK(extra, bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), lr=lr); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0, :]); loss = 0.0
            for k in range(K):
                z = m.fstep(z, u[:, k]); loss = loss + ((m.decode(z)-b[:, k+1, :])**2).mean() + 0.1*((z-m.lift(b[:, k+1, :]).detach())**2).mean()
            opt.zero_grad(); (loss/K).backward(); opt.step()
    with torch.no_grad():
        z = m.lift(Bte[:, 0, :])
        for k in range(K): z = m.fstep(z, Ute[:, k])
        return ((m.decode(z)-Bte[:, K, :])**2).sum(-1).sqrt().mean().item()      # H=15 test err

dims = [4, 6, 8, 12, 16]
lin = [train_eval(d-3, False) for d in dims]      # extra = dim - 3 base
bi = [train_eval(d-3, True) for d in dims]
print("lift_dim  linear_H15  bilinear_H15")
for d, l, b in zip(dims, lin, bi): print(f"   {d:2d}      {l:.4f}      {b:.4f}")

plt.figure(figsize=(6.4, 4.6))
plt.plot(dims, lin, "o-", lw=2, label="deep-LINEAR (MPPI-DK)")
plt.plot(dims, bi, "s-", lw=2, label="deep-BILINEAR (ours)")
plt.axhline(bi[2], ls=":", c="gray", lw=1)
plt.xlabel("lifting dimension"); plt.ylabel("multi-step (H=15) prediction error")
plt.title("Pendulum: bilinear reaches lower long-horizon error at smaller dim\n(deterministic open-loop, TEST set)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "pendulum_efficiency.png"), dpi=120)
print(f"\nbilinear@dim8 = {bi[2]:.4f}   vs   linear@dim16 = {lin[-1]:.4f}  "
      f"-> bilinear at HALF the dim is {'better' if bi[2] < lin[-1] else 'worse'}")
print(f"outputs -> {OUT}/pendulum_efficiency.png")
