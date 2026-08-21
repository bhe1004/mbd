"""E1: reconcile the 128x (Fig. 3b, drone3d_predict.py) vs 200x (Table 2, drone3d_baselines.py)
prediction-ratio discrepancy. Both scripts use the SAME statistic (mean over held-out trajectories
of the H=15 Euclidean world-position error, then ratio linear/bilinear) but DIFFERENT training
protocols and a single seed each:
  predict.py  : 4000 train traj, rollout loss + 0.1 latent-consistency, test=gen(1000, seed 2)
  baselines.py: 2500 train traj, rollout loss with double position weight, no latent term, test=gen(500, seed 2)
This script trains BOTH protocols for each battery seed and evaluates on ONE shared test set
gen(1000, 999) -> ratio mean+/-std per protocol. Battery seeds match drone3d_battery.py (data seed 100+s).
Outputs: out/drone3d_ratio_log.txt, out/drone3d_ratio.npz."""
import os, time, numpy as np, torch, torch.nn as nn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "drone3d_ratio_log.txt"), "w")
def log(msg=""):
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()

DT = 0.05; M = 4; ND = 5; K = 15
def stepn(s, u):
    x, y, z, ya = s; vx, vy, vz, w = u; c, sn = np.cos(ya), np.sin(ya)
    return np.array([x+(c*vx-sn*vy)*DT, y+(sn*vx+c*vy)*DT, z+vz*DT, ya+w*DT])
def obsn(s): return np.array([s[0], s[1], s[2], np.sin(s[3]), np.cos(s[3])], np.float32)
def gen(N, seed):
    r = np.random.default_rng(seed); B = np.zeros((N, K+1, ND), np.float32); U = np.zeros((N, K, M), np.float32)
    for i in range(N):
        s = np.array([r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-np.pi, np.pi)]); B[i, 0] = obsn(s)
        bias = np.concatenate([r.uniform(-0.7, 0.7, 3), r.uniform(-1.2, 1.2, 1)])
        for k in range(K):
            u = bias + np.concatenate([r.uniform(-0.4, 0.4, 3), r.uniform(-0.6, 0.6, 1)]); U[i, k] = u
            s = stepn(s, u); B[i, k+1] = obsn(s)
    return torch.tensor(B), torch.tensor(U)
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
    def predict(self, b0, U):
        z = self.lift(b0); out = []
        for k in range(U.shape[1]): z = self.fstep(z, U[:, k]); out.append(self.decode(z))
        return torch.stack(out, 1)
def train_predict_protocol(Btr, Utr, bilinear, epochs=200, bs=512):     # = drone3d_predict.py / _control.py
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0]); loss = 0.
            for k in range(K):
                z = m.fstep(z, u[:, k]); loss = loss + ((m.decode(z)-b[:, k+1])**2).mean() + .1*((z-m.lift(b[:, k+1]).detach())**2).mean()
            opt.zero_grad(); (loss/K).backward(); opt.step()
    return m
def train_baselines_protocol(Btr, Utr, bilinear, epochs=200, bs=512):   # = drone3d_baselines.py
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; pred = m.predict(b[:, 0], u)
            loss = ((pred[..., :3]-b[:, 1:, :3])**2).mean() + ((pred-b[:, 1:])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m

Bte, Ute = gen(1000, 999)   # SHARED test set (same as drone3d_battery.py)
@torch.no_grad()
def perr15(m):
    pred = m.predict(Bte[:, 0], Ute)
    return float(((pred[..., :3]-Bte[:, 1:, :3])**2).sum(-1).sqrt().mean(0)[-1])

SEEDS = list(range(10))
res = {p: {"lin": [], "bi": []} for p in ["predict", "baselines"]}
t0 = time.time()
for seed in SEEDS:
    # --- predict/control protocol (4000 traj, latent-consistency loss) ---
    torch.manual_seed(seed); np.random.seed(seed)
    Btr, Utr = gen(4000, 100+seed)
    ml = train_predict_protocol(Btr, Utr, False); mb = train_predict_protocol(Btr, Utr, True)
    el, eb = perr15(ml), perr15(mb); res["predict"]["lin"].append(el); res["predict"]["bi"].append(eb)
    # --- baselines protocol (2500 traj, double-position loss, no latent term) ---
    torch.manual_seed(seed); np.random.seed(seed)
    Btr2, Utr2 = gen(2500, 100+seed)
    ml2 = train_baselines_protocol(Btr2, Utr2, False); mb2 = train_baselines_protocol(Btr2, Utr2, True)
    el2, eb2 = perr15(ml2), perr15(mb2); res["baselines"]["lin"].append(el2); res["baselines"]["bi"].append(eb2)
    log(f"seed {seed}: predict-protocol lin={el:.4f} bi={eb:.4f} ({el/max(eb,1e-9):.0f}x)   "
        f"baselines-protocol lin={el2:.4f} bi={eb2:.4f} ({el2/max(eb2,1e-9):.0f}x)   [{time.time()-t0:.0f}s]")

log("\n================ 128x vs 200x reconciliation ================")
log("statistic (BOTH published numbers): mean over held-out trajectories of H=15 Euclidean world-position error; ratio = linear/bilinear")
log("published: 128x = drone3d_predict.py (single seed 0, train gen(4000,1), test gen(1000,2))")
log("published: 200x = drone3d_baselines.py 0.40/0.002 (single seed 0, train gen(2500,1), test gen(500,2), different loss)")
for p in ["predict", "baselines"]:
    lin = np.array(res[p]["lin"]); bi = np.array(res[p]["bi"]); r = lin/np.maximum(bi, 1e-9)
    log(f"\n[{p}-protocol, {len(SEEDS)} seeds, shared test gen(1000,999)]")
    log(f"  linear H=15   : {lin.mean():.4f}+/-{lin.std():.4f} m")
    log(f"  bilinear H=15 : {bi.mean():.4f}+/-{bi.std():.4f} m")
    log(f"  ratio per seed: " + " ".join(f"{v:.0f}" for v in r))
    log(f"  ratio         : mean={r.mean():.0f}x  std={r.std():.0f}  min={r.min():.0f}x  max={r.max():.0f}x  "
        f"ratio-of-mean-errors={lin.mean()/bi.mean():.0f}x")
np.savez(os.path.join(OUT, "drone3d_ratio.npz"),
         predict_lin=np.array(res["predict"]["lin"]), predict_bi=np.array(res["predict"]["bi"]),
         baselines_lin=np.array(res["baselines"]["lin"]), baselines_bi=np.array(res["baselines"]["bi"]),
         seeds=np.array(SEEDS))
log(f"\nsaved out/drone3d_ratio.npz  [total {time.time()-t0:.0f}s]")
LOG.close()
