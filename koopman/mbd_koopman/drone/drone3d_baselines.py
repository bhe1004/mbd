"""Baselines on the 3D velocity-drone: linear Koopman (small), linear Koopman (large, ~same FLOPs as
bilinear), bilinear Koopman (ours), and a nonlinear MLP dynamics model. Reports multi-step world-position
prediction error and per-control-step rollout cost, to answer: did we under-parametrize linear? why not an MLP?"""
import numpy as np, torch, torch.nn as nn, time
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
DT = 0.05; M = 4; ND = 5
def stepn(s, u):
    x, y, z, ya = s; vx, vy, vz, w = u; c, sn = np.cos(ya), np.sin(ya)
    return np.array([x+(c*vx-sn*vy)*DT, y+(sn*vx+c*vy)*DT, z+vz*DT, ya+w*DT])
def obsn(s): return np.array([s[0], s[1], s[2], np.sin(s[3]), np.cos(s[3])], np.float32)
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
print("data..."); Btr, Utr = gen(2500, 1); Bte, Ute = gen(500, 2)
class DK(nn.Module):                       # linear / bilinear Koopman
    def __init__(self, extra, hid=64, bilinear=False):
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
    def predict(self, b0, U):              # multi-step base prediction
        z = self.lift(b0); out = []
        for k in range(U.shape[1]): z = self.fstep(z, U[:, k]); out.append(self.decode(z))
        return torch.stack(out, 1)
class MLP(nn.Module):                      # nonlinear dynamics model b' = b + g(b,u)
    def __init__(self, hid=96):
        super().__init__(); self.net = nn.Sequential(nn.Linear(ND+M, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, ND))
    def fstep(self, b, u): return b + self.net(torch.cat([b, u], -1))
    def predict(self, b0, U):
        b = b0; out = []
        for k in range(U.shape[1]): b = self.fstep(b, U[:, k]); out.append(b)
        return torch.stack(out, 1)
def train(m, epochs=200, bs=512):
    opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; pred = m.predict(b[:, 0], u)
            loss = ((pred[..., :3]-b[:, 1:, :3])**2).mean() + ((pred-b[:, 1:])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m
models = {"linear (N=13)": DK(8, bilinear=False), "linear-large (N=29)": DK(24, bilinear=False),
          "bilinear ours (N=13)": DK(8, bilinear=True), "MLP dynamics": MLP()}
for name, m in models.items(): print("train", name); train(m)
torch.save({k: m.state_dict() for k, m in models.items()}, 'saved/drone3d_baselines.pt')
@torch.no_grad()
def perr(m):
    pred = m.predict(Bte[:, 0], Ute)
    e = ((pred[..., :3]-Bte[:, 1:, :3])**2).sum(-1).sqrt().mean(0)
    return e.numpy()
print("\n[3D drone: multi-step world-position prediction error]")
print(f"{'model':22s} {'H=1':>8s} {'H=8':>8s} {'H=15':>8s}")
res = {}
for name, m in models.items():
    e = perr(m); res[name] = e
    print(f"{name:22s} {e[0]:8.4f} {e[7]:8.4f} {e[14]:8.4f}")
# ---- rollout cost per control step (Ks samples x T horizon) ----
Ks, T = 500, 15   # capped by the 500-snippet test set: Bte[:512] silently gave 500 and crashed the batch matmul
U = torch.randn(Ks, T, M)*0.5
@torch.no_grad()
def rollout_time(m, reps=20):
    b0 = Bte[:Ks, 0]
    for _ in range(3): m.predict(b0, U)         # warmup
    t0 = time.perf_counter()
    for _ in range(reps): m.predict(b0, U)
    return (time.perf_counter()-t0)/reps*1e3
print(f"\n[rollout cost per control step, Ks={Ks} x T={T}]")
for name, m in models.items():
    print(f"  {name:22s} {rollout_time(m):6.2f} ms")
