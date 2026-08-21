"""E1 statistical battery for the 3D velocity-drone (see supp_experiments_design.md SE1).
10 training seeds x 3 references (figure-8 / square / random waypoints) x 2 observation-noise
levels (0, 0.02) x 3 controllers (ours bilinear / oracle / linear). Model, data, training and
MPPI configuration are copied VERBATIM from drone3d_control.py / drone3d_predict.py (not modified).
Per seed both models are trained once and shared across references/noise levels.
Also evaluates, per seed, the open-loop world-position prediction error on a shared held-out set
(same statistic as Fig. 3b / drone3d_predict.py) -> prediction-ratio mean+/-std for the 128x/200x
reconciliation. Outputs: out/drone3d_battery_log.txt, out/drone3d_battery.npz."""
import os, time, numpy as np, torch, torch.nn as nn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "drone3d_battery_log.txt"), "w")
def log(msg=""):
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()

DT = 0.05; M = 4; ND = 5; K = 15
STEPS = 175; T = 15; KS = 1024; LAM = 0.25; SIG = (0.4, 0.4, 0.4, 0.7); GY = 0.3   # = drone3d_control.py
SEEDS = list(range(10)); NOISES = [0.0, 0.02]; REFS = ["fig8", "square", "waypoints"]; CTRLS = ["ours", "oracle", "linear"]

# ---- dynamics / observation (verbatim from drone3d_control.py) ----
def stept(S, U):
    x, y, z, ya = S[..., 0], S[..., 1], S[..., 2], S[..., 3]; vx, vy, vz, w = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
    c, sn = torch.cos(ya), torch.sin(ya)
    return torch.stack([x+(c*vx-sn*vy)*DT, y+(sn*vx+c*vy)*DT, z+vz*DT, ya+w*DT], -1)
def obst(S): return torch.stack([S[..., 0], S[..., 1], S[..., 2], torch.sin(S[..., 3]), torch.cos(S[..., 3])], -1)
def stepn(s, u):
    x, y, z, ya = s; vx, vy, vz, w = u; c, sn = np.cos(ya), np.sin(ya)
    return np.array([x+(c*vx-sn*vy)*DT, y+(sn*vx+c*vy)*DT, z+vz*DT, ya+w*DT])
def obsn(s): return np.array([s[0], s[1], s[2], np.sin(s[3]), np.cos(s[3])], np.float32)

# ---- data (verbatim) ----
def gen(N, seed):
    r = np.random.default_rng(seed); B = np.zeros((N, K+1, ND), np.float32); U = np.zeros((N, K, M), np.float32)
    for i in range(N):
        s = np.array([r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-np.pi, np.pi)]); B[i, 0] = obsn(s)
        bias = np.concatenate([r.uniform(-0.7, 0.7, 3), r.uniform(-1.2, 1.2, 1)])
        for k in range(K):
            u = bias + np.concatenate([r.uniform(-0.4, 0.4, 3), r.uniform(-0.6, 0.6, 1)]); U[i, k] = u
            s = stepn(s, u); B[i, k+1] = obsn(s)
    return torch.tensor(B), torch.tensor(U)

# ---- model + training (verbatim) ----
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
def train(Btr, Utr, bilinear, epochs=200, bs=512):
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0]); loss = 0.
            for k in range(K):
                z = m.fstep(z, u[:, k]); loss = loss + ((m.decode(z)-b[:, k+1])**2).mean() + .1*((z-m.lift(b[:, k+1]).detach())**2).mean()
            opt.zero_grad(); (loss/K).backward(); opt.step()
    return m

# ---- references (length STEPS+K+2, like drone3d_control.py) ----
NREF = STEPS + K + 2
def path_from_waypoints(wps, speed, n):
    wps = np.array(wps, float); seg = np.diff(wps, axis=0); L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    S = np.minimum(np.arange(n)*speed*DT, cum[-1]-1e-9); P = np.zeros((n, 3))
    for j, s_ in enumerate(S):
        i = min(int(np.searchsorted(cum, s_, side="right"))-1, len(L)-1)
        r = (s_-cum[i])/max(L[i], 1e-9); P[j] = wps[i] + r*seg[i]
    return P
def yaw_from_path(P):
    Y = np.zeros(len(P)); prev = 0.0
    for t in range(len(P)):
        d = P[min(t+1, len(P)-1)] - P[t]
        if np.linalg.norm(d[:2]) > 1e-6: prev = np.arctan2(d[1], d[0])
        Y[t] = prev
    return Y
def make_ref(name, n=NREF):
    if name == "fig8":                                    # = drone3d_control.py reference
        F = 0.11; ts = np.arange(n)*DT
        P = np.stack([1.1*np.sin(2*np.pi*F*ts), 0.8*np.sin(4*np.pi*F*ts), 0.6+0.35*np.sin(2*np.pi*F*ts)], -1)
    elif name == "square":                                # square circuit, alternating altitude
        sq = [[0.9, 0.9, 0.5], [-0.9, 0.9, 0.9], [-0.9, -0.9, 0.5], [0.9, -0.9, 0.9]]
        P = path_from_waypoints([sq[i % 4] for i in range(10)], 0.8, n)
    elif name == "waypoints":                             # random waypoints (fixed rng -> same ref for all seeds)
        r = np.random.default_rng(11)
        wps = [[0.0, 0.0, 0.6]] + [[r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(0.3, 1.0)] for _ in range(9)]
        P = path_from_waypoints(wps, 0.8, n)
    return P, yaw_from_path(P)
REF = {name: make_ref(name) for name in REFS}

# ---- MPPI (verbatim controller from drone3d_control.py + observation noise on the PLANNER state) ----
@torch.no_grad()
def mppi(mode, model, RP, RY, mppi_seed, noise_sig, noise_seed):
    g = torch.Generator().manual_seed(mppi_seed); nr = np.random.default_rng(noise_seed)
    RPt = torch.tensor(RP, dtype=torch.float32); RYs = torch.tensor(np.sin(RY), dtype=torch.float32)
    s = torch.tensor(np.concatenate([RP[0], [RY[0]]]), dtype=torch.float32)   # start on the reference
    U = torch.zeros(T, M); sig = torch.tensor(SIG); traj = [s.numpy().copy()]
    for t in range(STEPS):
        s_obs = s + torch.tensor(nr.normal(0.0, noise_sig, 4), dtype=torch.float32) if noise_sig > 0 else s
        eps = torch.randn(KS, T, M, generator=g)*sig; V = U[None]+eps; cost = torch.zeros(KS)
        if mode == "oracle": S = s_obs.repeat(KS, 1)
        else: z = model.lift(obst(s_obs)).repeat(KS, 1)
        for k in range(T):
            uk = V[:, k]
            if mode == "oracle": S = stept(S, uk); p = S[:, :3]; ya = S[:, 3]
            else: z = model.fstep(z, uk); d = model.decode(z); p = d[:, :3]; ya = torch.atan2(d[:, 3], d[:, 4])
            cost = cost + ((p-RPt[t+k+1])**2).sum(1) + GY*(torch.sin(ya)-RYs[t+k+1])**2 + 0.02*(uk**2).sum(1)
        w = torch.softmax(-(cost-cost.min())/LAM, 0); U = U + (w[:, None, None]*eps).sum(0)
        s = stept(s, U[0]); traj.append(s.numpy().copy())          # TRUE state evolves noise-free
        U = torch.roll(U, -1, 0); U[-1] = U[-2]
    traj = np.array(traj)
    e = np.linalg.norm(traj[:, :3]-RP[:len(traj)], axis=1)          # same statistic as drone3d_control.py
    return e.mean(), e.max()

# ---- open-loop prediction statistic (verbatim from drone3d_predict.py, shared test set) ----
Bte, Ute = gen(1000, 999)
@torch.no_grad()
def pos_err(m):
    z = m.lift(Bte[:, 0]); e = []
    for k in range(K):
        z = m.fstep(z, Ute[:, k]); p = m.decode(z)[:, :3]
        e.append(((p-Bte[:, k+1, :3])**2).sum(-1).sqrt().mean().item())
    return np.array(e)

# ---- battery ----
nS, nR, nN, nC = len(SEEDS), len(REFS), len(NOISES), len(CTRLS)
err_mean = np.full((nS, nR, nN, nC), np.nan); err_max = np.full((nS, nR, nN, nC), np.nan)
pred_lin = np.zeros((nS, K)); pred_bi = np.zeros((nS, K))
t_start = time.time()
for si, seed in enumerate(SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    Btr, Utr = gen(4000, 100+seed)
    m_bi = train(Btr, Utr, True); m_lin = train(Btr, Utr, False)
    pred_lin[si] = pos_err(m_lin); pred_bi[si] = pos_err(m_bi)
    log(f"seed {seed}: trained. pred@H15 linear={pred_lin[si, -1]:.4f} bilinear={pred_bi[si, -1]:.4f} "
        f"ratio={pred_lin[si, -1]/max(pred_bi[si, -1], 1e-9):.1f}x  [{time.time()-t_start:.0f}s]")
    for ri, ref in enumerate(REFS):
        RP, RY = REF[ref]
        for ni, nz in enumerate(NOISES):
            mseed = 1000*seed + 100*ri + 10*ni + 3                 # shared across the 3 controllers
            for ci, (ctrl, mdl) in enumerate([("ours", m_bi), ("oracle", None), ("linear", m_lin)]):
                em, ex = mppi("oracle" if ctrl == "oracle" else ("bi" if ctrl == "ours" else "lin"),
                              mdl, RP, RY, mseed, nz, mseed+1)
                err_mean[si, ri, ni, ci] = em; err_max[si, ri, ni, ci] = ex
            log(f"  seed {seed} {ref:9s} noise={nz:.2f}: " +
                "  ".join(f"{CTRLS[ci]}={err_mean[si, ri, ni, ci]:.3f}" for ci in range(nC)))

# ---- summary ----
log("\n================ E1 drone3d battery summary ================")
log(f"seeds={SEEDS}  refs={REFS}  noise={NOISES}  controllers={CTRLS}")
log(f"config: STEPS={STEPS} T={T} Ks={KS} lam={LAM} sig={SIG} (identical to drone3d_control.py); beta=0 (no tube term)")
log("\n[closed-loop mean tracking error (m), mean+/-std over seeds, n=%d per cell]" % nS)
for ni, nz in enumerate(NOISES):
    for ri, ref in enumerate(REFS):
        row = "  ".join(f"{CTRLS[ci]}={err_mean[:, ri, ni, ci].mean():.3f}+/-{err_mean[:, ri, ni, ci].std():.3f}" for ci in range(nC))
        log(f"  noise={nz:.2f} {ref:9s}: {row}")
log("\n[aggregate over refs+seeds, n=%d per (noise, controller)]" % (nS*nR))
for ni, nz in enumerate(NOISES):
    row = "  ".join(f"{CTRLS[ci]}={err_mean[:, :, ni, ci].mean():.3f}+/-{err_mean[:, :, ni, ci].std():.3f}" for ci in range(nC))
    log(f"  noise={nz:.2f}: {row}")
log("\n[open-loop world-position prediction @H=15 on shared test set (n=1000 traj), per training seed]")
r15 = pred_lin[:, -1]/np.maximum(pred_bi[:, -1], 1e-9)
log(f"  linear   : {pred_lin[:, -1].mean():.4f}+/-{pred_lin[:, -1].std():.4f} m")
log(f"  bilinear : {pred_bi[:, -1].mean():.4f}+/-{pred_bi[:, -1].std():.4f} m")
log(f"  ratio    : mean={r15.mean():.1f}x  std={r15.std():.1f}  min={r15.min():.1f}x  max={r15.max():.1f}x   "
    f"ratio-of-mean-errors={pred_lin[:, -1].mean()/pred_bi[:, -1].mean():.1f}x")
log(f"  per-seed ratios: " + " ".join(f"{v:.0f}" for v in r15))
np.savez(os.path.join(OUT, "drone3d_battery.npz"), err_mean=err_mean, err_max=err_max,
         pred_lin=pred_lin, pred_bi=pred_bi, seeds=np.array(SEEDS), noises=np.array(NOISES),
         refs=np.array(REFS), ctrls=np.array(CTRLS))
log(f"\nsaved out/drone3d_battery.npz  [total {time.time()-t_start:.0f}s]")
LOG.close()
