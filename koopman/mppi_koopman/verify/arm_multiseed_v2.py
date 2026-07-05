"""Improved-training multi-seed for the 7-DOF arm: more data + cosine LR decay + grad clipping +
more epochs, to see if the bilinear arm's reach count stabilizes across seeds (v1 was 3.2+-1.7/7).
Same architecture/eval as arm_multiseed.py; only the optimizer schedule and data size change."""
import os, numpy as np, torch, torch.nn as nn, mujoco
FIGS = "/figs"
XML = """<mujoco><option timestep="0.01" gravity="0 0 0"/><worldbody>
<body name="b1" pos="0 0 0.1"><joint axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.04"/>
<body name="b2" pos="0 0 0.18"><joint axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 0.18" size="0.038"/>
<body name="b3" pos="0 0 0.18"><joint axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.035"/>
<body name="b4" pos="0 0 0.16"><joint axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 0.16" size="0.033"/>
<body name="b5" pos="0 0 0.16"><joint axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0 0 0.14" size="0.03"/>
<body name="b6" pos="0 0 0.14"><joint axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.028"/>
<body name="b7" pos="0 0 0.12"><joint axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.025"/>
<site name="ee" pos="0 0 0.12" size="0.03"/></body></body></body></body></body></body></body></worldbody></mujoco>"""
model = mujoco.MjModel.from_xml_string(XML); data = mujoco.MjData(model); eid = model.site("ee").id
def ee_mj(q): data.qpos[:] = q; mujoco.mj_forward(model, data); return data.site_xpos[eid].copy()
OFFS = torch.tensor([[0, 0, .1], [0, 0, .18], [0, 0, .18], [0, 0, .16], [0, 0, .16], [0, 0, .14], [0, 0, .12]])
AX = ['z', 'y', 'z', 'y', 'z', 'y', 'z']; EE = torch.tensor([0, 0, .12, 1.])
def rot(a, q):
    c, s, o, l = torch.cos(q), torch.sin(q), torch.zeros_like(q), torch.ones_like(q)
    R = [[c, -s, o, o], [s, c, o, o], [o, o, l, o], [o, o, o, l]] if a == 'z' else [[c, o, s, o], [o, l, o, o], [-s, o, c, o], [o, o, o, l]]
    return torch.stack([torch.stack(r, -1) for r in R], -2)
def bfk(q):
    Ks = q.shape[0]; T = torch.eye(4).expand(Ks, 4, 4)
    for i in range(7):
        Tr = torch.eye(4).clone(); Tr[:3, 3] = OFFS[i]; T = T @ (Tr.expand(Ks, 4, 4) @ rot(AX[i], q[:, i]))
    return (T @ EE)[:, :3]
DT, QV, QLIM = 0.05, 2.0, 2.6; K = 15; ND, M = 10, 7
def gen(N, seed):
    r = np.random.default_rng(seed); B = np.zeros((N, K+1, 10)); U = np.zeros((N, K, 7))
    for i in range(N):
        q = r.uniform(-2, 2, 7); qd_c = r.uniform(-QV*.6, QV*.6, 7); B[i, 0] = np.concatenate([q, ee_mj(q)])
        for k in range(K):
            qd = np.clip(qd_c + r.uniform(-QV*.5, QV*.5, 7), -QV, QV); U[i, k] = qd
            q = np.clip(q + qd*DT, -QLIM, QLIM); B[i, k+1] = np.concatenate([q, ee_mj(q)])
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)
print("data 6000 (once)..."); Btr, Utr = gen(6000, 1)
class DK(nn.Module):
    def __init__(self, extra=10, hid=96, bilinear=False):
        super().__init__(); self.N = ND+extra; self.bilinear = bilinear
        self.enc = nn.Sequential(nn.Linear(14, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, extra))
        self.A = nn.Parameter(torch.eye(self.N)+.01*torch.randn(self.N, self.N)); self.B0 = nn.Parameter(.01*torch.randn(self.N, M))
        if bilinear: self.B1 = nn.Parameter(torch.zeros(M, self.N, self.N))
    def lift(self, b):
        q = b[..., :7]; return torch.cat([b, self.enc(torch.cat([torch.sin(q), torch.cos(q)], -1))], -1)
    def fstep(self, z, u):
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            for i in range(M): zn = zn + u[..., i:i+1]*(z @ self.B1[i].T)
        return zn
    def decode(self, z): return z[..., :ND]
def train(bilinear, epochs=300, bs=512):
    m = DK(bilinear=bilinear); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs); n = Btr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; b = Btr[idx]; u = Utr[idx]; z = m.lift(b[:, 0]); loss = 0.
            for k in range(K):
                z = m.fstep(z, u[:, k]); loss = loss + ((m.decode(z)-b[:, k+1])**2).mean() + .1*((z-m.lift(b[:, k+1]).detach())**2).mean()
            opt.zero_grad(); (loss/K).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
        sched.step()
    return m
@torch.no_grad()
def mppi(mode, model, target, T=15, Ks=800, lam=0.4, steps=120, seed=3):
    g = torch.Generator().manual_seed(seed); q = np.zeros(7); U = torch.zeros(T, 7); tt = torch.tensor(target, dtype=torch.float32)
    for t in range(steps):
        eps = torch.randn(Ks, T, 7, generator=g)*1.2; V = (U[None]+eps).clamp(-QV, QV); cost = torch.zeros(Ks)
        if mode == "oracle": qb = torch.tensor(q, dtype=torch.float32).repeat(Ks, 1)
        else: z = model.lift(torch.tensor(np.concatenate([q, ee_mj(q)]), dtype=torch.float32)).repeat(Ks, 1)
        for k in range(T):
            uk = V[:, k]
            if mode == "oracle": qb = (qb+uk*DT).clamp(-QLIM, QLIM); ee = bfk(qb)
            else: z = model.fstep(z, uk); ee = model.decode(z)[:, 7:10]
            cost += ((ee-tt)**2).sum(1) + 0.002*(uk**2).sum(1)
        cost += 10.0*((ee-tt)**2).sum(1)
        w = torch.softmax(-(cost-cost.min())/lam, 0); U = U + (w[:, None, None]*eps).sum(0)
        q = np.clip(q + U[0].clamp(-QV, QV).numpy()*DT, -QLIM, QLIM)
        U = torch.roll(U, -1, 0); U[-1] = U[-2]
        if np.linalg.norm(ee_mj(q)-target) < 0.025: break
    return float(np.linalg.norm(ee_mj(q)-target))
TARGETS = [[0.4, 0.3, 0.55], [-0.3, 0.4, 0.6], [0.3, -0.35, 0.7], [-0.4, -0.2, 0.5], [0.5, 0.0, 0.65], [0.0, 0.5, 0.5], [-0.45, 0.25, 0.75]]
SEEDS = [0, 1, 2, 3, 4]; THR = 0.05; nt = len(TARGETS)
bi_err = np.zeros((len(SEEDS), nt)); lin_err = np.zeros((len(SEEDS), nt)); orc_err = np.zeros(nt)
for j, tg in enumerate(TARGETS): orc_err[j] = mppi("oracle", None, tg, seed=3)
for si, seed in enumerate(SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    m_bi = train(True); m_lin = train(False)
    for j, tg in enumerate(TARGETS):
        bi_err[si, j] = mppi("bilinear", m_bi, tg, seed=seed+3)
        lin_err[si, j] = mppi("linear", m_lin, tg, seed=seed+3)
    print(f"seed {seed}: ours {int((bi_err[si]<THR).sum())}/{nt}  MPPI-DK {int((lin_err[si]<THR).sum())}/{nt}")
bi_reach = (bi_err < THR).sum(1); lin_reach = (lin_err < THR).sum(1)
print(f"\n[v2 improved] Reach (<{THR}) over {len(SEEDS)} seeds:")
print(f"  ours    : {bi_reach.tolist()}  -> {bi_reach.mean():.1f} +/- {bi_reach.std():.1f} / {nt}")
print(f"  MPPI-DK : {lin_reach.tolist()}  -> {lin_reach.mean():.1f} +/- {lin_reach.std():.1f} / {nt}")
print(f"  ours reached-only err {bi_err[bi_err<THR].mean():.3f}   MPPI-DK err {lin_err.mean():.3f}+/-{lin_err.std():.3f}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.35, "axes.labelsize": 13, "legend.fontsize": 11})
x = np.arange(nt); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.bar(x-w, bi_err.mean(0), w, yerr=bi_err.std(0), capsize=2, label="Ours (bilinear)", color="#1f77b4", error_kw=dict(lw=1, ecolor="0.3"))
ax.bar(x,   orc_err,        w,                      label="Oracle (true dyn.)", color="#2ca02c")
ax.bar(x+w, lin_err.mean(0), w, yerr=lin_err.std(0), capsize=2, label="MPPI-DK (linear)", color="#d62728", error_kw=dict(lw=1, ecolor="0.3"))
ax.axhline(THR, ls="--", color="k", lw=1.2)
ax.annotate(f"reach threshold ({THR} m)", xy=(6.45, THR), xytext=(6.45, 0.14), ha="right", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f"T{j+1}" for j in range(nt)])
ax.set_xlabel("reach target"); ax.set_ylabel("final EE error [m]"); ax.set_ylim(0, 1.05); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_arm_bar_v2.png"), dpi=200)
print("saved fig_arm_bar_v2.png")
