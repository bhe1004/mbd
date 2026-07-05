"""Multi-seed robustness of the unicycle parking result: retrain deep ours/MPPI-DK over several
seeds and report final goal distance mean+-std (ours parks, MPPI-DK does not). Data once; training seed varies."""
import os, numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)
DT = 0.05
def dyn_np(s, u):
    def f(s): return np.array([u[0]*np.cos(s[2]), u[0]*np.sin(s[2]), u[1]])
    k1 = f(s); k2 = f(s+0.5*DT*k1); k3 = f(s+0.5*DT*k2); k4 = f(s+DT*k3)
    return s + (DT/6.0)*(k1+2*k2+2*k3+k4)
def dyn_t(S, U):
    def f(S): return torch.stack([U[..., 0]*torch.cos(S[..., 2]), U[..., 0]*torch.sin(S[..., 2]), U[..., 1]], -1)
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
    def lift(self, b):
        inp = torch.stack([b[..., 0], b[..., 1], b[..., 2], b[..., 3]+1.0], -1)
        return torch.cat([b, self.enc(inp)], -1)
    def fstep(self, z, u):
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            for i in range(self.m): zn = zn + u[..., i:i+1]*(z @ self.B1[i].T)
        return zn
    def decode(self, z): return z[..., :4]
def make_snips(Nsnip, Kk, seed):
    r = np.random.default_rng(seed); S = np.zeros((Nsnip, Kk+1, 3))
    S[:, 0, 0] = r.uniform(-2, 2, Nsnip); S[:, 0, 1] = r.uniform(-2, 2, Nsnip); S[:, 0, 2] = r.uniform(-np.pi, np.pi, Nsnip)
    U = np.stack([r.uniform(-1.5, 1.5, (Nsnip, Kk)), r.uniform(-1.5, 1.5, (Nsnip, Kk))], -1)
    for k in range(Kk):
        for i in range(Nsnip): S[i, k+1] = dyn_np(S[i, k], U[i, k])
    B = np.stack([S[..., 0], S[..., 1], np.sin(S[..., 2]), np.cos(S[..., 2])-1.0], -1)
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
K = 15
print("data (once)..."); Btr, Utr = make_snips(8000, K, 1)
def train(bilinear, epochs=200, bs=512, lr=1e-3):
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), lr=lr); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0, :]); loss = 0.0
            for k in range(K):
                z = m.fstep(z, u[:, k, :]); loss = loss + ((m.decode(z)-b[:, k+1, :])**2).mean() + 0.1*((z - m.lift(b[:, k+1, :]).detach())**2).mean()
            (loss/K).backward(); opt.step(); opt.zero_grad()
    return m
STEPS = 170
@torch.no_grad()
def mppi(mode, model, s0, goal, T=40, Ks=1024, lam=0.5, sig=(0.6, 0.9), w_du=0.5, seed=3):
    g = torch.Generator().manual_seed(seed); s = torch.tensor(s0); U = torch.zeros(T, 2)
    u_prev = torch.zeros(2); gT = torch.tensor(goal); stop = STEPS; sig = torch.tensor(sig)
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
        w = torch.softmax(-(cost-cost.min())/lam, 0); U = U + (w[:, None, None]*eps).sum(0)
        u0 = U[0].clamp(torch.tensor([-1.5, -1.5]), torch.tensor([1.5, 1.5]))
        s = dyn_t(s, u0); u_prev = u0
        U = torch.roll(U, -1, 0); U[-1] = U[-2]
        if np.linalg.norm(s[:2].numpy()-goal[:2]) < 0.08 and abs(((s[2].item()-goal[2]+np.pi) % (2*np.pi))-np.pi) < 0.2:
            stop = t; break
    return float(np.linalg.norm(s[:2].numpy()-np.array(goal[:2]))), stop
s0 = [1.5, 1.5, -np.pi/2]; goal = [0.0, 0.0, 0.0]; SEEDS = [0, 1, 2, 3, 4]; PARK = 0.3
orc_d, _ = mppi("true", None, s0, goal, seed=3)
bi_d, lin_d = [], []
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    m_bi = train(True); m_lin = train(False)
    db, sb = mppi("bilinear", m_bi, s0, goal, seed=seed+3)
    dl, sl = mppi("linear", m_lin, s0, goal, seed=seed+3)
    bi_d.append(db); lin_d.append(dl)
    print(f"seed {seed}: ours final ||pos||={db:.3f} ({'parked' if db<PARK else 'FAIL'})  MPPI-DK={dl:.3f} ({'parked' if dl<PARK else 'FAIL'})")
bi_d, lin_d = np.array(bi_d), np.array(lin_d)
print(f"\nFinal goal distance over {len(SEEDS)} seeds (park if <{PARK} m):")
print(f"  ours    : {np.round(bi_d,3).tolist()}  -> {bi_d.mean():.3f} +/- {bi_d.std():.3f}  park {int((bi_d<PARK).sum())}/{len(SEEDS)}")
print(f"  MPPI-DK : {np.round(lin_d,3).tolist()}  -> {lin_d.mean():.3f} +/- {lin_d.std():.3f}  park {int((lin_d<PARK).sum())}/{len(SEEDS)}")
print(f"  oracle  : {orc_d:.3f}  park {'yes' if orc_d<PARK else 'no'}")
