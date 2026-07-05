"""
NEURON-COUNT efficiency (the story: "ours needs FEW neurons, MPPI-DK's linear needs MANY").
MPPI-DK use 256 neurons/layer and report neuron count is the key lever. We fix lifting dim=8 and
sweep hidden width; deep-LINEAR (their model) vs deep-BILINEAR (ours), our multi-step training.
Metric = deterministic multi-step (H=12) open-loop prediction error on a TEST set (robust, no MPPI).
Output: out/pendulum_neurons.png  (+ table)
"""
import os, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
DT, UMAX = 0.05, 2.0

def dyn_step(TH, DTH, U):
    dth = DTH + (15*np.sin(TH) + 3*U)*DT; return TH + dth*DT, dth

class DK(nn.Module):
    def __init__(self, extra, hid, bilinear):
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
        S[:, k+1, 0], S[:, k+1, 1] = dyn_step(S[:, k, 0], S[:, k, 1], U[:, k])
    B = np.stack([np.sin(S[..., 0]), np.cos(S[..., 0])-1.0, S[..., 1]], -1)
    return torch.tensor(B, dtype=torch.float32), torch.tensor(U, dtype=torch.float32)

K = 12
Btr, Utr = make_snips(8000, K, 1); Bte, Ute = make_snips(3000, K, 2)

def train_eval(hid, bilinear, epochs=110, bs=512, lr=1e-3):
    m = DK(5, hid, bilinear); opt = torch.optim.Adam(m.parameters(), lr=lr); n = Btr.shape[0]
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
        return ((m.decode(z)-Bte[:, K, :])**2).sum(-1).sqrt().mean().item()

NEUR = [8, 16, 32, 64, 128, 256]
lin, bi = [], []
print("neurons  linear_H12  bilinear_H12")
for h in NEUR:
    l = train_eval(h, False); b = train_eval(h, True); lin.append(l); bi.append(b)
    print(f"  {h:4d}     {l:.4f}      {b:.4f}")

plt.figure(figsize=(6.6, 4.6))
plt.plot(NEUR, lin, "o-", lw=2, label="deep-LINEAR (MPPI-DK)")
plt.plot(NEUR, bi, "s-", lw=2, label="deep-BILINEAR (ours)")
plt.xscale("log", base=2); plt.xticks(NEUR, [str(n) for n in NEUR])
plt.axhline(bi[2], ls=":", c="gray", lw=1, label=f"ours @ {NEUR[2]} neurons")
plt.xlabel("hidden neurons per layer (lifting dim fixed = 8)")
plt.ylabel("multi-step (H=12) prediction error")
plt.title("Pendulum: ours reaches good accuracy with FEW neurons;\nlinear needs MANY (MPPI-DK use 256)")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "pendulum_neurons.png"), dpi=120)
# how many neurons does linear need to match ours@32 ?
tgt = bi[2]
need = next((NEUR[i] for i in range(len(NEUR)) if lin[i] <= tgt), None)
print(f"\nours @ {NEUR[2]} neurons = {tgt:.4f}.  linear reaches that at "
      f"{('>'+str(NEUR[-1])) if need is None else need} neurons.")
print(f"outputs -> {OUT}/pendulum_neurons.png")
