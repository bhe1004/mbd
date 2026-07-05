"""
FAITHFUL MPPI-DK reproduction: DKO (Deep Koopman Operator) training exactly as their paper —
for fixed NN lifting θ, solve A,B(,C) by analytic least-squares ([A B]=Ḡ[G;U]†, C=X̄Ḡ†),
update ONLY θ by gradient on the 1-step lifted+decode loss. (Not the multi-step joint-Adam I used before.)
  linear  (their method): Ḡ ≈ A G + B U
  bilinear (ours)       : Ḡ ≈ A G + B0 U + u (B1 G)
Same encoder/data/training for both. Dynamics = MPPI-DK eq.13 swing-up (|u|≤2). Data = random + expert
(energy-shaping) demos, faithful to their expert-augmented dataset. Oracle = MPPI-true.
Outputs: out/pendulum_dko.gif , pendulum_dko.png  (+ multi-step pred + did-it-swing-up)
"""
import os, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
DT, UMAX, R = 0.05, 2.0, 8

def dyn_np(s, u):
    dth = s[1] + (15*np.sin(s[0]) + 3*u)*DT; return np.array([s[0]+dth*DT, dth])
def dyn_t(S, U):
    dth = S[..., 1] + (15*torch.sin(S[..., 0]) + 3*U)*DT
    return torch.stack([S[..., 0]+dth*DT, dth], -1)
def feat_t(S): return torch.stack([torch.sin(S[..., 0]), torch.cos(S[..., 0]), S[..., 1]], -1)

class ENC(nn.Module):
    def __init__(self, r=R, hid=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(3, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, r))
    def forward(self, f): return self.net(f)

def energy_swingup(s0, steps=220, kp=0.8):
    s = np.array(s0, float); traj = [s.copy()]; U = []
    for _ in range(steps):
        E = 0.5*s[1]**2 + 15.0*np.cos(s[0])
        u = -(3.0*np.sin(s[0]) + 1.2*s[1]) if (1-np.cos(s[0])) < 0.4 else kp*s[1]*(15.0-E)
        u = float(np.clip(u, -UMAX, UMAX)); U.append(u); s = dyn_np(s, u); traj.append(s.copy())
    return np.array(traj), np.array(U)

def gen_data(Drand, Nexp, seed):
    r = np.random.default_rng(seed)
    TH = r.uniform(-np.pi, np.pi, Drand); DTH = r.uniform(-8, 8, Drand); U = r.uniform(-UMAX, UMAX, Drand)
    SP = np.array([dyn_np([TH[i], DTH[i]], U[i]) for i in range(Drand)])
    S = np.stack([TH, DTH], 1); Us = list(U); Ss = list(S); SPs = list(SP)
    for _ in range(Nexp):                                   # expert swing-up transitions
        tr, u = energy_swingup([r.uniform(-np.pi, np.pi), r.uniform(-1, 1)])
        for k in range(len(u)):
            Ss.append(tr[k]); Us.append(u[k]); SPs.append(tr[k+1])
    S = np.array(Ss); SP = np.array(SPs); U = np.array(Us)
    F = np.stack([np.sin(S[:, 0]), np.cos(S[:, 0]), S[:, 1]], 1)
    FP = np.stack([np.sin(SP[:, 0]), np.cos(SP[:, 0]), SP[:, 1]], 1)
    return (torch.tensor(F, dtype=torch.float32), torch.tensor(U[:, None], dtype=torch.float32),
            torch.tensor(FP, dtype=torch.float32))

F, U, FP = gen_data(20000, 200, 1)
Fte, Ute, FPte = gen_data(4000, 40, 2)
print(f"DKO data: {F.shape[0]} transitions (random + expert)")

def solve(Phi, Y, ridge=1e-4):                              # min ||Phi W - Y||  -> W
    P = Phi.shape[1]
    return torch.linalg.solve(Phi.T@Phi + ridge*torch.eye(P), Phi.T@Y)

def dko_loss(enc, F, U, FP, bilinear):
    G = enc(F); Gp = enc(FP)
    Phi = torch.cat([G, U] + ([U*G] if bilinear else []), 1)
    W = solve(Phi, Gp); Cw = solve(Gp, FP)
    loss = ((Phi@W - Gp)**2).mean() + ((Gp@Cw - FP)**2).mean()
    return loss, W, Cw

def train_dko(bilinear, steps=500, lr=1e-3):
    enc = ENC(); opt = torch.optim.Adam(enc.parameters(), lr=lr)
    for s in range(steps):
        loss, _, _ = dko_loss(enc, F, U, FP, bilinear)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, W, Cw = dko_loss(enc, F, U, FP, bilinear)
        A = W[:R].T.contiguous(); B0 = W[R:R+1].T.contiguous()
        B1 = W[R+1:R+1+R].T.contiguous() if bilinear else None; C = Cw.T.contiguous()
    return enc, dict(A=A, B0=B0, B1=B1, C=C, bilinear=bilinear, enc=enc)

print("training DKO (linear & bilinear)..."); _, M_lin = train_dko(False); _, M_bi = train_dko(True)

def lift(M, S): return M["enc"](feat_t(S))
def fstep(M, z, u):
    zn = z @ M["A"].T + u.unsqueeze(-1)*M["B0"][:, 0]
    if M["bilinear"]: zn = zn + u.unsqueeze(-1)*(z @ M["B1"].T)
    return zn
def decode(M, z): return z @ M["C"].T                       # -> (sinθ, cosθ, θ̇)

# multi-step audit on fresh trajectories
@torch.no_grad()
def multistep(M, H=15, n=2000, seed=5):
    r = np.random.default_rng(seed); S0 = np.stack([r.uniform(-np.pi, np.pi, n), r.uniform(-6, 6, n)], 1)
    Us = r.uniform(-UMAX, UMAX, (n, H))
    S = torch.tensor(S0, dtype=torch.float32); z = lift(M, S); errs = []
    Strue = S.clone()
    for k in range(H):
        uk = torch.tensor(Us[:, k], dtype=torch.float32)
        z = fstep(M, z, uk); Strue = dyn_t(Strue, uk)
        b = decode(M, z); bt = feat_t(Strue)
        errs.append(((b-bt)**2).sum(-1).sqrt().mean().item())
    return errs
el, eb = multistep(M_lin), multistep(M_bi)
print("[multi-step open-loop pred err in (sinθ,cosθ,θ̇)]:")
for h in [1, 5, 10, 15]:
    print(f"   H={h:2d}  linear={el[h-1]:.4f}  bilinear={eb[h-1]:.4f}  ratio={el[h-1]/max(eb[h-1],1e-9):.2f}x")

STEPS = 160
@torch.no_grad()
def mppi(mode, M, T=55, Ks=1536, lam=1.0, sig=1.8, seed=3):
    g = torch.Generator().manual_seed(seed); s = torch.tensor([np.pi, 0.0]); Useq = torch.zeros(T)
    traj = [s.numpy().copy()]; stop = STEPS
    for t in range(STEPS):
        eps = torch.randn(Ks, T, generator=g)*sig; V = Useq[None] + eps; cost = torch.zeros(Ks)
        if mode == "true": S = s.repeat(Ks, 1)
        else: z = lift(M, s).repeat(Ks, 1)
        for k in range(T):
            uk = V[:, k].clamp(-UMAX, UMAX)
            if mode == "true": S = dyn_t(S, uk); b = feat_t(S)
            else: z = fstep(M, z, uk); b = decode(M, z)
            E = 0.5*b[:, 2]**2 + 15.0*b[:, 1]
            cost += 0.6*(b[:, 0]**2 + (b[:, 1]-1.0)**2) + 0.02*b[:, 2]**2 + 0.002*uk**2 + 0.03*(E-15.0)**2
        cost += 25.0*(b[:, 0]**2 + (b[:, 1]-1.0)**2) + 1.5*b[:, 2]**2
        w = torch.softmax(-(cost-cost.min())/lam, 0); Useq = Useq + (w[:, None]*eps).sum(0)
        u0 = float(Useq[0].clamp(-UMAX, UMAX)); s = dyn_t(s, torch.tensor(u0)); traj.append(s.numpy().copy())
        Useq = torch.roll(Useq, -1); Useq[-1] = Useq[-2]
        if (1-np.cos(s[0].item())) < 0.05 and abs(s[1].item()) < 0.6: stop = len(traj)-1; break
    return np.array(traj), stop

print("running MPPI swing-up (DKO)...")
runs = [("Ours: DKO-BILINEAR MPPI", lambda: mppi("bilinear", M_bi), "C0"),
        ("MPPI-true: oracle", lambda: mppi("true", None), "C2"),
        ("MPPI-DK: DKO-LINEAR MPPI", lambda: mppi("linear", M_lin), "C3")]
results = []
for title, fn, col in runs:
    tr, stop = fn(); up = 1-np.cos(tr[-1, 0]); conv = f"up@{stop} ({stop*DT:.1f}s)" if stop < STEPS else "did NOT swing up"
    results.append((title, tr, col, stop)); print(f"   {title.split(':')[0]:9s}: final 1-cos={up:.3f}  {conv}")

def setup(ax, title):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(title, fontsize=10.5)
    ax.scatter([0], [1], c="k", marker="*", s=160, zorder=3); ax.scatter([0], [0], c="0.4", s=40, zorder=4)
def draw(ax, tr, j, col):
    th = tr[j, 0]; bx, by = np.sin(th), np.cos(th); tail = max(0, j-55)
    ax.plot(np.sin(tr[tail:j+1, 0]), np.cos(tr[tail:j+1, 0]), "-", c=col, lw=1, alpha=0.35)
    ax.plot([0, bx], [0, by], "-", c=col, lw=3, zorder=5); ax.scatter([bx], [by], c=col, s=130, zorder=6)
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8)); nframes = max(len(r[1]) for r in results)
def update(i):
    for ax, (title, tr, col, stop) in zip(axes, results):
        ax.clear(); setup(ax, title); j = min(i, len(tr)-1); draw(ax, tr, j, col)
        tag = "  UP" if (stop < STEPS and j >= stop) else ""
        ax.text(-1.32, 1.2, f"t={j*DT:4.1f}s  1-cos={1-np.cos(tr[j,0]):.2f}{tag}", fontsize=9, family="monospace")
    return []
FuncAnimation(fig, update, frames=nframes, interval=50, blit=False).save(os.path.join(OUT, "pendulum_dko.gif"), writer=PillowWriter(fps=20))
update(nframes-1); plt.tight_layout(); plt.savefig(os.path.join(OUT, "pendulum_dko.png"), dpi=110)
print(f"outputs -> {OUT}/pendulum_dko.gif , pendulum_dko.png")
